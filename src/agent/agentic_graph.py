"""
Agentic RAG graph builder (v3.0).
ReAct agent loop: agentic_node ↔ tool_execution_node → persist → END.
"""

from typing import AsyncGenerator

from langgraph.graph import StateGraph, END

from src.agent.agentic_state import AgenticState
from src.agent.agentic_nodes import (
    agentic_node,
    tool_execution_node,
    agentic_wiki_persist_node,
)
from src.config import AppConfig


def build_agentic_graph(config: AppConfig) -> StateGraph:
    """
    Build the agentic ReAct agent StateGraph.

    Flow:
      START → agentic_node → (has tool_calls?)
                ├─ Yes → tool_node → agentic_node (loop)
                └─ No  → wiki_persist → END
    """
    graph = StateGraph(AgenticState)

    # Wrap nodes with config binding
    async def a_node(state: AgenticState):
        return await agentic_node(state, config)

    async def t_node(state: AgenticState):
        return await tool_execution_node(state, config)

    async def p_node(state: AgenticState):
        return await agentic_wiki_persist_node(state, config)

    graph.add_node("agentic", a_node)
    graph.add_node("tools", t_node)
    graph.add_node("wiki_persist", p_node)

    graph.set_entry_point("agentic")

    # Conditional routing after agentic node
    def route_after_agentic(state: AgenticState) -> str:
        # If evidence is sufficient, go to persist
        if state.get("evidence_sufficient"):
            return "wiki_persist"
        # If there are pending tool calls, execute them
        if state.get("pending_tool_calls"):
            # Check budget: max tool calls
            max_calls = state.get("max_tool_calls", 12)
            made = state.get("tool_calls_made", 0)
            pending_count = len(state.get("pending_tool_calls", []))
            if made + pending_count > max_calls:
                return "wiki_persist"  # budget exceeded
            return "tools"
        # No tool calls and no answer — go to persist (edge case)
        return "wiki_persist"

    graph.add_conditional_edges("agentic", route_after_agentic, {
        "tools": "tools",
        "wiki_persist": "wiki_persist",
    })

    # After tool execution, always go back to agentic for evaluation
    def route_after_tools(state: AgenticState) -> str:
        max_calls = state.get("max_tool_calls", 12)
        made = state.get("tool_calls_made", 0)

        if made >= max_calls:
            return "wiki_persist"  # budget exhausted

        # Check for empty search streak (last 3 observations all empty)
        obs = state.get("observations", [])
        if len(obs) >= 3:
            recent = obs[-3:]
            if all(
                "0 results" in o.get("content", "")
                or "No " in o.get("content", "")
                or "not found" in o.get("content", "").lower()
                for o in recent
            ):
                return "wiki_persist"  # empty search streak

        return "agentic"

    graph.add_conditional_edges("tools", route_after_tools, {
        "agentic": "agentic",
        "wiki_persist": "wiki_persist",
    })

    graph.add_edge("wiki_persist", END)

    return graph


def _make_agentic_initial_state(
    query: str,
    sources: list = None,
    max_results: int = 5,
    conversation_history: list = None,
    max_tool_calls: int = 12,
) -> AgenticState:
    """Build initial AgenticState for a new query."""
    return AgenticState(
        query=query,
        search_sources=sources or ["web", "paper"],
        max_results_per_source=max_results,
        messages=[],
        tool_calls_made=0,
        max_tool_calls=max_tool_calls,
        observations=[],
        agent_reasoning_trace=[],
        evidence_sufficient=False,
        agent_plan="",
        pending_tool_calls=[],
        agent_iteration=0,
        final_answer="",
        status="thinking",
        status_message="Agent analyzing query...",
        logs=[],
        errors=[],
        conversation_history=conversation_history or [],
    )


async def run_agentic_search(
    query: str,
    sources: list = None,
    max_results: int = 5,
    config: AppConfig = None,
    conversation_history: list = None,
    max_tool_calls: int = 12,
) -> dict:
    """Run agentic search in batch mode. Returns final state dict."""
    if config is None:
        config = AppConfig.from_env()

    graph = build_agentic_graph(config)
    compiled = graph.compile()
    initial = _make_agentic_initial_state(
        query, sources, max_results, conversation_history, max_tool_calls
    )
    return await compiled.ainvoke(initial)


async def run_agentic_search_stream(
    query: str,
    sources: list = None,
    max_results: int = 5,
    config: AppConfig = None,
    conversation_history: list = None,
    max_tool_calls: int = 12,
) -> AsyncGenerator[dict, None]:
    """
    Run agentic search in streaming mode.
    Yields events: {type: "status"|"thought"|"action"|"observation"|"answer"|"done", content, meta}
    """
    if config is None:
        config = AppConfig.from_env()

    graph = build_agentic_graph(config)
    compiled = graph.compile()
    initial = _make_agentic_initial_state(
        query, sources, max_results, conversation_history, max_tool_calls
    )

    last_reasoning_len = 0
    last_status = ""

    # Accumulate state across stream updates
    final_state = dict(initial)

    async for event in compiled.astream(initial, stream_mode="updates"):
        for node_name, update in event.items():
            if update is None:
                continue

            # Merge update into accumulated state (each key may be a list append or a replace)
            for k, v in update.items():
                if isinstance(v, list) and isinstance(final_state.get(k), list):
                    # Annotated list: extend
                    final_state[k] = final_state[k] + v
                else:
                    final_state[k] = v

            # Status messages
            msg = update.get("status_message", "")
            if msg and msg != last_status:
                last_status = msg
                if node_name == "agentic":
                    yield {"type": "thought", "content": msg, "meta": {"node": node_name}}
                elif node_name == "tools":
                    yield {"type": "action", "content": msg, "meta": {"node": node_name}}

            # Reasoning trace
            trace = update.get("agent_reasoning_trace", [])
            if trace:
                for t in trace[last_reasoning_len:]:
                    yield {"type": "thought", "content": t, "meta": {"node": "agentic"}}
                last_reasoning_len = len(trace)

            # Observations
            observations = update.get("observations", [])
            if observations:
                for obs in observations:
                    summary = obs.get("content", "")[:300]
                    yield {
                        "type": "observation",
                        "content": summary,
                        "meta": {"tool": obs.get("tool", ""), "node": node_name},
                    }

    # Final answer comes from the accumulated state, not a fresh run
    answer = final_state.get("final_answer", "")

    if not answer:
        # Fallback: build a summary from observations if the agent didn't reach final_answer
        observations = final_state.get("observations", [])
        if observations:
            tool_calls = final_state.get("tool_calls_made", 0)
            iterations = final_state.get("agent_iteration", 0)
            answer = (
                f"## 调研已完成（{iterations} 轮, {tool_calls} 次工具调用）\n\n"
                f"我已搜集到 {len(observations)} 条信息，但无法整合出最终答案。"
                f"以下是收集到的关键信息片段：\n\n"
            )
            for i, obs in enumerate(observations[:6], 1):
                tool = obs.get("tool", "tool")
                content_preview = obs.get("content", "")[:500]
                answer += f"### [{i}] {tool}\n{content_preview}\n\n"
            answer += (
                "\n---\n"
                "**建议**：尝试换一种问法，或使用 Fast 模式获得更稳定的总结结果。"
            )
        else:
            answer = (
                "搜索未能返回有效结果，请尝试换个问题或换个关键词。"
            )

    yield {
        "type": "done",
        "content": answer,
        "meta": {
            "tool_calls_made": final_state.get("tool_calls_made", 0),
            "iterations": final_state.get("agent_iteration", 0),
            "logs": final_state.get("logs", []),
        },
    }
