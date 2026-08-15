"""
多智能体编排 (v6.0) — Supervisor + ResearchAgent + BrowserAgent。

Supervisor 判断任务类型（检索 / 浏览器 / 混合），将控制权交给对应 worker。
每个 worker 是独立的 ReAct 循环（独立 system prompt + 独立工具子集），
混合任务时 ResearchAgent 与 BrowserAgent 并行执行，结果归并后由 Supervisor 汇总。

  研究型任务 → ResearchAgent（fused_search + read_page + synthesize Map-Reduce）
  操作型任务 → BrowserAgent（browser_* 15 工具）
  混合型任务 → 两者并行 → Supervisor 汇总
"""

from __future__ import annotations

import asyncio
from operator import add
from typing import Annotated, AsyncGenerator, Optional, TypedDict

from langgraph.graph import StateGraph, END

from src.agent.runtime import AgentRuntime
from src.agent.tools import (
    UNIFIED_TOOL_SCHEMAS,
    BROWSER_TOOL_SCHEMAS,
    execute_unified_tool,
)
from src.browser.perception.dom_snapshot import build_state_summary
from src.config import AppConfig


# ─────────────────────────────────────────────────────────────
# Worker 工具子集与角色 prompt
# ─────────────────────────────────────────────────────────────

KNOWLEDGE_TOOL_NAMES = {
    "fused_search", "search_web", "search_papers", "search_news", "search_blogs",
    "read_page", "search_wiki", "read_wiki_page", "follow_link", "synthesize", "final_answer",
}
BROWSER_TOOL_NAMES = {
    s["function"]["name"] for s in BROWSER_TOOL_SCHEMAS
    if s["function"]["name"].startswith("browser_")
} | {"final_answer"}

KNOWLEDGE_TOOLS = [s for s in UNIFIED_TOOL_SCHEMAS if s["function"]["name"] in KNOWLEDGE_TOOL_NAMES]
BROWSER_TOOLS = [s for s in UNIFIED_TOOL_SCHEMAS if s["function"]["name"] in BROWSER_TOOL_NAMES]

RESEARCH_SYSTEM_PROMPT = """You are a rigorous research agent. You gather multi-source evidence and synthesize a cited report.

## Workflow (at most ~6 steps)
1. Call fused_search once to query all sources (web/papers/news/blogs) in parallel and get ranked results.
2. Optionally call read_page on 1-2 key URLs for depth (skip if snippets suffice).
3. Call synthesize to produce the Map-Reduce report (per-source summary → cross-source synthesis with citations).
4. Call final_answer with the synthesized report.

## Rules
- Do NOT loop on searching: after fused_search + at most 2 read_page, ALWAYS synthesize then final_answer.
- Every factual claim must have a source; never fabricate.
- Respond in the user's language (Chinese by default)."""

BROWSER_SYSTEM_PROMPT = """You are a web automation agent operating a real browser.

## How to work
1. On a NEW page call browser_read_a11y first; prefer element "index" over vague text.
2. Perform ONE logical action per step; re-read the page when state changes.
3. Use browser_extract to scrape data; browser_scroll to reveal more.
4. When done call final_answer with the result.

## Rules
- NEVER guess a URL or element — observe first.
- If an action fails, adapt: re-read, retry, or try a different locator.
- Respond in the user's language (Chinese by default)."""


# ─────────────────────────────────────────────────────────────
# 通用 ReAct 循环（供两个 worker 复用）
# ─────────────────────────────────────────────────────────────

async def _run_react(
    runtime: AgentRuntime,
    task: str,
    tools: list,
    system_prompt: str,
    max_steps: int = 12,
    perceive: bool = False,
) -> str:
    """Run a bounded ReAct loop with a role-specific prompt and tool subset."""
    runtime.set_task(task)
    history: list = []

    for step in range(max_steps):
        state_summary = ""
        if perceive and runtime.browser_alive:
            state_summary = await build_state_summary(runtime.driver)

        decision = await runtime.local.decide(
            task=task,
            subtask_goal=task,
            state_summary=state_summary,
            history=_compress(history),
            messages=[{"role": "system", "content": system_prompt}],
            tools=tools,
        )
        if decision.is_final:
            return decision.params.get("answer", "")

        obs = await execute_unified_tool(decision.action, decision.params, runtime)
        history.append(f"[{step + 1}] {decision.action}: {obs[:300]}")

    return "(worker 预算耗尽，未显式完成)"


def _compress(history: list) -> str:
    if not history:
        return "(none)"
    if len(history) <= 8:
        return "\n".join(history)
    older, recent = history[:-5], history[-5:]
    return (
        "# 早期动作摘要\n"
        + "\n".join(f"- {h.split(chr(10))[0][:120]}" for h in older)
        + "\n\n# 最近动作\n"
        + "\n".join(recent)
    )


# ─────────────────────────────────────────────────────────────
# Supervisor 任务分类
# ─────────────────────────────────────────────────────────────

async def classify_task(runtime: AgentRuntime, task: str) -> str:
    """Classify a task as research / browser / both."""
    prompt = (
        "判断以下任务的类型，返回单个词：\n"
        "- research：纯知识检索/研究/总结类（搜索、调研、对比、解释）\n"
        "- browser：纯网页操作类（登录、填表、点击、抓取动态数据）\n"
        "- both：需要同时检索和浏览器操作\n\n"
        f"任务：{task}\n\n类型："
    )
    try:
        resp = await runtime.local.client.chat.completions.create(
            model=runtime.config.llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
    except Exception:
        raw = ""
    if "browser" in raw:
        return "both" if "research" in raw or "both" in raw else "browser"
    return "research"


# ─────────────────────────────────────────────────────────────
# 多智能体图
# ─────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict, total=False):
    """Supervisor + worker 共享状态。"""
    task: str
    next_agent: str
    research_result: str
    browser_result: str
    final_answer: str
    status: str
    logs: Annotated[list, add]


def build_multiagent_graph(runtime: AgentRuntime):
    graph = StateGraph(MultiAgentState)

    async def supervisor_node(state: dict) -> dict:
        task = state["task"]
        kind = await classify_task(runtime, task)
        return {"next_agent": kind, "logs": [f"[supervisor] classify → {kind}"]}

    async def research_node(state: dict) -> dict:
        result = await _run_react(
            runtime, state["task"], KNOWLEDGE_TOOLS, RESEARCH_SYSTEM_PROMPT, perceive=False,
            max_steps=8,
        )
        return {"research_result": result, "logs": ["[research] done"]}

    async def browser_node(state: dict) -> dict:
        result = await _run_react(
            runtime, state["task"], BROWSER_TOOLS, BROWSER_SYSTEM_PROMPT, perceive=True,
        )
        return {"browser_result": result, "logs": ["[browser] done"]}

    async def both_node(state: dict) -> dict:
        # 并行执行两个 worker
        research_task = asyncio.create_task(
            _run_react(runtime, state["task"], KNOWLEDGE_TOOLS, RESEARCH_SYSTEM_PROMPT, perceive=False)
        )
        browser_task = asyncio.create_task(
            _run_react(runtime, state["task"], BROWSER_TOOLS, BROWSER_SYSTEM_PROMPT, perceive=True)
        )
        research_result, browser_result = await asyncio.gather(research_task, browser_task)
        return {
            "research_result": research_result,
            "browser_result": browser_result,
            "logs": ["[both] research + browser done in parallel"],
        }

    async def summarize_node(state: dict) -> dict:
        research = state.get("research_result", "")
        browser = state.get("browser_result", "")
        if research and browser:
            prompt = (
                f"任务：{state['task']}\n\n"
                f"检索结果：\n{research}\n\n浏览器操作结果：\n{browser}\n\n"
                "请综合这两部分结果，输出最终答案（结构化，保留引用/数据）。"
            )
            resp = await runtime.local.client.chat.completions.create(
                model=runtime.config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            answer = resp.choices[0].message.content or ""
        else:
            answer = research or browser
        return {"final_answer": answer, "status": "done"}

    def route(state: dict) -> str:
        kind = state.get("next_agent", "research")
        if kind == "browser":
            return "browser"
        if kind == "both":
            return "both"
        return "research"

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("browser", browser_node)
    graph.add_node("both", both_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", route,
        {"research": "research", "browser": "browser", "both": "both"},
    )
    for n in ("research", "browser", "both"):
        graph.add_edge(n, "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


async def run_multiagent(task: str, config: AppConfig = None, runtime: AgentRuntime = None) -> dict:
    """Run the multi-agent supervisor pipeline. Returns the final state dict."""
    if config is None:
        config = AppConfig.from_env()
    owns_runtime = runtime is None
    if runtime is None:
        runtime = AgentRuntime(config)

    try:
        graph = build_multiagent_graph(runtime)
        result = await graph.ainvoke({"task": task, "logs": []})
        final = dict(result)
        if not final.get("final_answer"):
            final["final_answer"] = "(多智能体未产出答案)"
        return final
    finally:
        if owns_runtime:
            await runtime.stop()


async def run_multiagent_stream(
    task: str, config: AppConfig = None, runtime: AgentRuntime = None,
) -> AsyncGenerator[dict, None]:
    """Streaming variant of the multi-agent pipeline."""
    if config is None:
        config = AppConfig.from_env()
    owns_runtime = runtime is None
    if runtime is None:
        runtime = AgentRuntime(config)

    try:
        graph = build_multiagent_graph(runtime)
        final_state = {"task": task, "logs": []}
        async for event in graph.astream(final_state, stream_mode="updates"):
            for node_name, update in event.items():
                if not update:
                    continue
                for k, v in update.items():
                    if k == "logs" and isinstance(v, list):
                        final_state.setdefault("logs", []).extend(v)
                    else:
                        final_state[k] = v
                for log in update.get("logs", []):
                    yield {"type": "log", "content": log, "meta": {"node": node_name}}
                if update.get("status") == "done":
                    yield {"type": "status", "content": "完成", "meta": {"node": node_name}}

        answer = final_state.get("final_answer", "(无答案)")
        yield {"type": "done", "content": answer, "meta": {"logs": final_state.get("logs", [])}}
    finally:
        if owns_runtime:
            await runtime.stop()
