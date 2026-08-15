"""
统一 Agent LangGraph (v5.0) — 单图，plan → perceive → decide → execute → verify。

一个 Agent 同时调度知识检索工具与浏览器动作工具；浏览器按需（lazy）启动。
感知层按上一步动作类型智能切换：浏览器活跃才读 DOM，知识模式依赖工具观察。

路由：
  START → plan → perceive → decide → execute → verify
      ├─ continue → perceive (loop)
      ├─ replan   → plan      (drift ≥2 / 连续失败停滞)
      └─ done     → END
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from src.agent.runtime import AgentRuntime
from src.agent.state import UnifiedAgentState
from src.agent.tools import UNIFIED_TOOL_SCHEMAS, execute_unified_tool
from src.browser.exception.taxonomy import DeadLoopError
from src.browser.perception.dom_snapshot import build_state_summary
from src.browser.planning.planner import Plan
from src.config import AppConfig


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

DRIFT_REPLAN_THRESHOLD = 2   # 连续 drift 次数 → 重规划（v5.0 更敏感）
STALL_REPLAN_THRESHOLD = 4   # 连续失败步数 → 重规划（停滞检测）


def format_plan_text(plan: Plan) -> str:
    return "\n".join(f"{s.id}. {s.goal}" for s in plan.subtasks)


def _initial_state(task: str) -> UnifiedAgentState:
    return UnifiedAgentState(
        task=task,
        plan_text="",
        subtasks=[],
        plan_version=0,
        current_subtask=0,
        subtask_status=[],
        verify_result="continue",
        task_done_pending=False,
        drift_count=0,
        state_summary="",
        last_action_kind="none",
        action_history=[],
        action_trace=[],
        last_result={},
        pending_decision={},
        final_answer="",
        evidence_sufficient=False,
        status="starting",
        status_message="Starting agent",
        logs=[],
        errors=[],
        tool_calls_made=0,
        iteration=0,
        page_state_hash="",
        consecutive_repeats=0,
    )


def _history_text(state: UnifiedAgentState, limit: int = 15) -> str:
    """组装历史上下文，超长时用 scratchpad 压缩（保留最近 5 条全文）。"""
    history = state.get("action_history", [])
    if not history:
        return "(none)"
    if len(history) <= 8:
        return "\n".join(history[-limit:])
    # 早期动作压缩为一行摘要，最近动作保留全文
    older, recent = history[:-5], history[-5:]
    summaries = [h.split("\n")[0][:120] for h in older]
    return (
        "# 早期动作摘要 (scratchpad)\n"
        + "\n".join(f"- {s}" for s in summaries[-12:])
        + "\n\n# 最近动作（全文）\n"
        + "\n".join(recent)
    )


def _is_failure(obs: str) -> bool:
    """Heuristic success/failure from a tool observation string."""
    head = obs[:60].lower()
    return obs.startswith("FAIL") or "execution error" in head or obs.startswith("Error:")


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────

def build_unified_graph(runtime: AgentRuntime):
    """Build the unified agent StateGraph."""
    graph = StateGraph(UnifiedAgentState)

    # -- plan ------------------------------------------------------

    async def plan_node(state: UnifiedAgentState) -> dict:
        task = state["task"]
        runtime.set_task(task)

        context = ""
        if state.get("plan_version", 0) > 0:
            context = "已完成动作 (completed actions):\n" + _history_text(state, 10)

        plan = await runtime.planner.plan(task, context)
        subtasks = [s.to_dict() for s in plan.subtasks]
        plan_text = format_plan_text(plan)

        return {
            "subtasks": subtasks,
            "plan_text": plan_text,
            "plan_version": state.get("plan_version", 0) + 1,
            "current_subtask": 0,
            "subtask_status": ["running" if i == 0 else "pending" for i in range(len(subtasks))],
            "task_done_pending": False,
            "drift_count": 0,
            "status": "planning",
            "status_message": f"Planned {len(subtasks)} subtask(s)",
            "logs": [f"[plan] v{state.get('plan_version', 0) + 1}: " + " | ".join(s["goal"][:40] for s in subtasks)],
        }

    # -- perceive (智能切换) -----------------------------------------

    async def perceive_node(state: UnifiedAgentState) -> dict:
        if runtime.browser_alive:
            summary = await build_state_summary(runtime.driver)
            return {
                "state_summary": summary,
                "status": "perceiving",
                "status_message": "Observed browser page",
            }
        # Knowledge mode: no browser; the LLM relies on action_history.
        return {
            "status": "perceiving",
            "status_message": "Knowledge mode (no browser active)",
        }

    # -- decide -----------------------------------------------------

    async def decide_node(state: UnifiedAgentState) -> dict:
        if state.get("task_done_pending"):
            answer = await runtime.verifier.summarize(state["task"], _history_text(state, 30))
            return {
                "pending_decision": {"action": "final_answer", "params": {"answer": answer}, "reason": "task complete"},
                "logs": ["[decide] task_done_pending → final_answer"],
            }

        subtasks = state.get("subtasks", [])
        idx = state.get("current_subtask", 0)
        goal = subtasks[idx]["goal"] if 0 <= idx < len(subtasks) else state.get("plan_text", "") or state["task"]

        decision = await runtime.local.decide(
            task=state["task"],
            subtask_goal=goal,
            state_summary=state.get("state_summary", ""),
            history=_history_text(state),
            tools=UNIFIED_TOOL_SCHEMAS,
        )
        return {
            "pending_decision": {
                "action": decision.action,
                "params": decision.params,
                "reason": decision.reason,
                "subtask_goal": goal,
            },
            "logs": [f"[decide] subtask[{idx}] '{goal[:50]}' → {decision.action}"],
        }

    # -- execute (统一分发 + 升级阶梯 + confirm 门) --------------------

    async def execute_node(state: UnifiedAgentState) -> dict:
        decision = state.get("pending_decision", {})
        action = decision.get("action", "")
        params = decision.get("params", {})
        iteration = state.get("iteration", 0) + 1

        if action == "final_answer":
            return {
                "final_answer": params.get("answer", ""),
                "evidence_sufficient": True,
                "iteration": iteration,
                "status": "done",
                "status_message": "Task completed",
                "logs": [f"[execute] final_answer after {iteration} iterations"],
            }

        runtime.set_task(state["task"])
        kind = "browser" if action.startswith("browser_") else "knowledge"

        # Confirm gate for destructive browser actions.
        warn_logs: List[str] = []
        if kind == "browser":
            hint = runtime.guard.destructive_hint(action, params)
            if hint:
                approved = True
                if runtime.confirm_callback is not None:
                    approved = await runtime.confirm_callback(action, params, hint)
                if not approved:
                    return {
                        "errors": [f"Destructive action blocked: {hint}"],
                        "status": "blocked",
                        "status_message": f"Blocked: {hint}",
                        "logs": [f"[execute] BLOCKED: {hint}"],
                    }
                if runtime.confirm_callback is None:
                    warn_logs.append(f"[execute] destructive action auto-approved: {hint}")

        # Escalation ladder: attempt + bounded retries.
        obs = await execute_unified_tool(action, params, runtime)
        success = not _is_failure(obs)
        attempt = 1
        while not success and attempt <= 2:
            await asyncio.sleep(0.5 * attempt)
            obs = await execute_unified_tool(action, params, runtime)
            success = not _is_failure(obs)
            attempt += 1

        # Watchdog: dead-loop detection on identical action signatures.
        signature = action + "::" + json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
        try:
            runtime.watchdog.observe(signature)
        except DeadLoopError as e:
            return {
                "errors": [str(e)],
                "final_answer": f"(任务因检测到死循环而中止：{str(e)})",
                "evidence_sufficient": True,
                "iteration": iteration,
                "status": "done",
                "status_message": "Dead loop detected",
                "logs": [f"[execute] dead loop: {e}"],
            }

        # Refresh state summary (browser: DOM; knowledge: tool observation).
        if kind == "browser" and runtime.browser_alive:
            new_summary = await build_state_summary(runtime.driver)
        else:
            new_summary = obs

        history_entry = f"[{iteration}] {action}: {obs[:300]}"
        masked = runtime.guard.mask_params(params) if kind == "browser" else params
        trace_entry = {
            "iteration": iteration,
            "action": action,
            "params": masked,
            "success": success,
            "kind": kind,
            "message": obs[:200],
            "subtask": decision.get("subtask_goal", ""),
        }

        return {
            "action_history": [history_entry],
            "action_trace": [trace_entry],
            "last_result": {"success": success, "action": action, "message": obs[:200]},
            "state_summary": new_summary,
            "last_action_kind": kind,
            "tool_calls_made": state.get("tool_calls_made", 0) + 1,
            "iteration": iteration,
            "status": "acting",
            "status_message": f"{action}: {'OK' if success else 'FAIL'}",
            "logs": warn_logs + [f"[execute] {iteration}: {action} -> {'OK' if success else 'FAIL'}"],
        }

    # -- verify (子任务推进 + drift + 停滞 + 终止) ----------------------

    async def verify_node(state: UnifiedAgentState) -> dict:
        if state.get("final_answer") or state.get("evidence_sufficient"):
            return {"verify_result": "done", "status": "done"}

        if state.get("tool_calls_made", 0) >= runtime.watchdog.max_steps:
            return {"verify_result": "done", "status": "done", "status_message": "Budget exhausted",
                    "logs": ["[verify] budget exhausted"]}

        # 停滞检测：连续 N 步失败 → 重规划
        recent = state.get("action_trace", [])[-STALL_REPLAN_THRESHOLD:]
        if len(recent) >= STALL_REPLAN_THRESHOLD and all(not t.get("success") for t in recent):
            return {"verify_result": "replan", "drift_count": 0, "status_message": "Stall detected → re-planning",
                    "logs": [f"[verify] stall ({STALL_REPLAN_THRESHOLD} consecutive failures) → replan"]}

        subtasks = state.get("subtasks", [])
        idx = state.get("current_subtask", 0)
        statuses = list(state.get("subtask_status", []))
        drift = state.get("drift_count", 0)

        if 0 <= idx < len(subtasks):
            verdict = await runtime.verifier.evaluate(
                goal=subtasks[idx].get("goal", ""),
                expected_state=subtasks[idx].get("expected_state", ""),
                state_summary=state.get("state_summary", ""),
                history=_history_text(state, 12),
            )

            if verdict == "done":
                if idx < len(statuses):
                    statuses[idx] = "done"
                idx += 1
                if idx < len(statuses):
                    statuses[idx] = "running"
                if idx >= len(subtasks):
                    return {"verify_result": "continue", "current_subtask": idx, "subtask_status": statuses,
                            "task_done_pending": True, "drift_count": 0,
                            "status_message": "All subtasks completed",
                            "logs": ["[verify] all subtasks done → final answer"]}
                return {"verify_result": "continue", "current_subtask": idx, "subtask_status": statuses, "drift_count": 0,
                        "status_message": f"Advanced to subtask {idx}",
                        "logs": [f"[verify] subtask {idx-1} done → advance to {idx}"]}

            elif verdict == "drift":
                drift += 1
                if drift >= DRIFT_REPLAN_THRESHOLD:
                    return {"verify_result": "replan", "current_subtask": idx, "subtask_status": statuses, "drift_count": 0,
                            "status_message": "Drift detected → re-planning",
                            "logs": [f"[verify] drift x{drift} → replan"]}
                return {"verify_result": "continue", "current_subtask": idx, "subtask_status": statuses, "drift_count": drift,
                        "status_message": f"Drift warning ({drift})",
                        "logs": [f"[verify] drift warning {drift}"]}
            else:
                drift = 0

        return {"verify_result": "continue", "current_subtask": idx, "subtask_status": statuses, "drift_count": drift}

    def route_after_verify(state: UnifiedAgentState) -> str:
        vr = state.get("verify_result", "continue")
        if vr == "done":
            return "end"
        if vr == "replan":
            return "plan"
        return "perceive"

    graph.add_node("plan", plan_node)
    graph.add_node("perceive", perceive_node)
    graph.add_node("decide", decide_node)
    graph.add_node("execute", execute_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "perceive")
    graph.add_edge("perceive", "decide")
    graph.add_edge("decide", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"perceive": "perceive", "plan": "plan", "end": END},
    )

    return graph.compile()


def _make_budget_final_answer(state: UnifiedAgentState) -> str:
    steps = state.get("tool_calls_made", 0)
    return f"(预算耗尽：已完成 {steps} 步操作，未显式完成。最后状态请见 action_history。)"


async def run_agent(
    task: str,
    config: AppConfig = None,
    runtime: AgentRuntime = None,
) -> dict:
    """Run the unified agent in batch mode. Returns the final state dict."""
    if config is None:
        config = AppConfig.from_env()

    owns_runtime = runtime is None
    if runtime is None:
        runtime = AgentRuntime(config)

    try:
        graph = build_unified_graph(runtime)
        initial = _initial_state(task)
        result = await graph.ainvoke(initial)

        final = dict(result)
        if not final.get("final_answer"):
            final["final_answer"] = _make_budget_final_answer(final)
        return final
    finally:
        if owns_runtime:
            await runtime.stop()
