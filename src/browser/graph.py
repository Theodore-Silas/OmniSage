"""
Browser agent LangGraph (v4.0) — full plan → perceive → decide → execute → verify loop.

The non-serializable runtime (driver, resolver, executors, watchdog, verifier)
lives in :class:`BrowserRuntime`; LangGraph nodes close over it. State carries
only serializable fields.

Routing:
  START → plan → perceive → decide → execute → verify
      ├─ continue → perceive (loop)
      ├─ replan   → plan      (drift / repeated failure)
      └─ done     → END

The escalation ladder (retry → self-healing locate → re-perceive) is applied
inside the execute node; the verify node drives subtask advancement, drift
detection, and re-planning.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from src.browser.action.primitives import ActionExecutor
from src.browser.action.tools import _action_of, format_observation
from src.browser.driver.base import BrowserDriver
from src.browser.driver.playwright_driver import PlaywrightDriver
from src.browser.driver.selenium_driver import SeleniumDriver
from src.browser.exception.escalation import Watchdog
from src.browser.exception.taxonomy import DeadLoopError
from src.browser.locator.cache import LocatorCache
from src.browser.locator.resolver import LocatorResolver
from src.browser.perception.dom_snapshot import build_state_summary
from src.browser.perception.vlm import VLMClient
from src.browser.planning.executor import LocalExecutor
from src.browser.planning.planner import GlobalPlanner, Plan
from src.browser.planning.verifier import SubtaskVerifier
from src.browser.safety.guard import SafetyGuard
from src.browser.state import BrowserState
from src.config import AppConfig


# ─────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────

def create_driver(config: AppConfig) -> BrowserDriver:
    """Factory: build the configured driver (playwright by default)."""
    bc = config.browser
    if bc.driver == "selenium":
        return SeleniumDriver(
            browser_type=bc.browser_type,
            headless=bc.headless,
            viewport_width=bc.viewport_width,
            viewport_height=bc.viewport_height,
            default_timeout=bc.default_timeout,
            navigation_timeout=bc.navigation_timeout,
        )
    return PlaywrightDriver(
        browser_type=bc.browser_type,
        headless=bc.headless,
        viewport_width=bc.viewport_width,
        viewport_height=bc.viewport_height,
        default_timeout=bc.default_timeout,
        navigation_timeout=bc.navigation_timeout,
        user_agent=bc.user_agent,
        channel=bc.channel,
    )


class BrowserRuntime:
    """Holds all non-serializable objects the graph nodes need.

    ``confirm_callback`` (optional) is an async ``fn(action, params, hint) ->
    bool`` used as the human confirmation gate for destructive actions.
    """

    def __init__(self, config: AppConfig, confirm_callback=None):
        self.config = config
        self.confirm_callback = confirm_callback
        self.driver = create_driver(config)
        self.vlm = VLMClient(config.vlm)
        self.guard = SafetyGuard(config.browser.allowed_domains)
        self.cache = LocatorCache()
        self.resolver = LocatorResolver(self.driver, self.vlm, self.cache, task="")
        self.executor = ActionExecutor(self.driver, self.resolver, self.guard, self.vlm, task="")
        self.planner = GlobalPlanner(config.llm)
        self.local = LocalExecutor(config.llm)
        self.verifier = SubtaskVerifier(config.llm)
        self.watchdog = Watchdog(max_steps=config.browser.max_steps)

    async def start(self) -> None:
        await self.driver.launch()

    async def stop(self) -> None:
        await self.driver.close()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def format_plan_text(plan: Plan) -> str:
    return "\n".join(f"{s.id}. {s.goal}" for s in plan.subtasks)


def _initial_state(task: str) -> BrowserState:
    return BrowserState(
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
        action_history=[],
        action_trace=[],
        last_result={},
        pending_decision={},
        final_answer="",
        evidence_sufficient=False,
        status="starting",
        status_message="Starting browser agent",
        logs=[],
        errors=[],
        tool_calls_made=0,
        iteration=0,
        page_state_hash="",
        consecutive_repeats=0,
    )


def _history_text(state: BrowserState, limit: int = 15) -> str:
    return "\n".join(state.get("action_history", [])[-limit:])


# ─────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────

def build_browser_graph(runtime: BrowserRuntime):
    """Build the full browser agent StateGraph."""
    graph = StateGraph(BrowserState)

    # -- plan ------------------------------------------------------

    async def plan_node(state: BrowserState) -> dict:
        task = state["task"]
        runtime.resolver.task = task
        runtime.executor.task = task

        # On re-plan, pass completed history as context so the planner
        # can decompose the *remaining* work.
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

    # -- perceive ---------------------------------------------------

    async def perceive_node(state: BrowserState) -> dict:
        summary = await build_state_summary(runtime.driver)
        return {
            "state_summary": summary,
            "status": "perceiving",
            "status_message": "Observed page state",
        }

    # -- decide -----------------------------------------------------

    async def decide_node(state: BrowserState) -> dict:
        # All subtasks done → emit final answer.
        if state.get("task_done_pending"):
            answer = await runtime.verifier.summarize(state["task"], _history_text(state, 30))
            return {
                "pending_decision": {"action": "final_answer", "params": {"answer": answer}, "reason": "task complete"},
                "logs": ["[decide] task_done_pending → final_answer"],
            }

        subtasks = state.get("subtasks", [])
        idx = state.get("current_subtask", 0)
        goal = subtasks[idx]["goal"] if 0 <= idx < len(subtasks) else state.get("plan_text", "") or state["task"]
        statuses = state.get("subtask_status", [])
        progress = "\n".join(f"- {s.get('goal','')} [{statuses[i] if i < len(statuses) else '?'}]" for i, s in enumerate(subtasks)) if subtasks else ""

        decision = await runtime.local.decide(
            task=state["task"],
            subtask_goal=goal,
            state_summary=state.get("state_summary", ""),
            history=_history_text(state),
        )
        # Attach progress context into the decision (recorded for trace).
        return {
            "pending_decision": {
                "action": decision.action,
                "params": decision.params,
                "reason": decision.reason,
                "subtask_goal": goal,
                "plan_progress": progress,
            },
            "logs": [f"[decide] subtask[{idx}] '{goal[:50]}' → {decision.action}"],
        }

    # -- execute (with escalation ladder + confirm gate) -------------

    async def execute_node(state: BrowserState) -> dict:
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

        runtime.resolver.task = state["task"]
        runtime.executor.task = state["task"]

        # Confirm gate for destructive actions.
        hint = runtime.guard.destructive_hint(action, params) if runtime.guard else ""
        warn_logs: List[str] = []
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

        # Escalation ladder: attempt + bounded retries (resolver self-heals
        # internally by re-locating / re-perceiving).
        result = await runtime.executor.execute(_action_of(action), params)
        attempt = 1
        while not result.success and attempt <= 2:
            await asyncio.sleep(0.5 * attempt)
            result = await runtime.executor.execute(_action_of(action), params)
            attempt += 1

        # Watchdog: dead-loop detection on identical action signatures.
        signature = action + "::" + json.dumps(params, sort_keys=True, default=str, ensure_ascii=False)
        try:
            runtime.watchdog.observe(signature)
            repeats = 0
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

        obs = format_observation(result)
        history_entry = f"[{iteration}] {action}: {obs[:300]}"
        masked = runtime.guard.mask_params(params) if runtime.guard else params
        trace_entry = {
            "iteration": iteration,
            "action": action,
            "params": masked,
            "success": result.success,
            "message": result.message[:200],
            "page_state_hash": result.page_state_hash,
            "subtask": decision.get("subtask_goal", ""),
        }

        # Refresh page state for the verify node.
        new_summary = await build_state_summary(runtime.driver)

        return {
            "action_history": [history_entry],
            "action_trace": [trace_entry],
            "last_result": result.to_dict(),
            "state_summary": new_summary,
            "tool_calls_made": state.get("tool_calls_made", 0) + 1,
            "iteration": iteration,
            "page_state_hash": result.page_state_hash,
            "consecutive_repeats": repeats,
            "status": "acting",
            "status_message": f"{action}: {'OK' if result.success else 'FAIL'}",
            "logs": warn_logs + [f"[execute] {iteration}: {action} -> {'OK' if result.success else 'FAIL'}"],
        }

    # -- verify (subtask advancement + drift + termination) -----------

    async def verify_node(state: BrowserState) -> dict:
        # 1. Explicit completion.
        if state.get("final_answer") or state.get("evidence_sufficient"):
            return {"verify_result": "done", "status": "done"}

        # 2. Budget exhausted.
        if state.get("tool_calls_made", 0) >= runtime.watchdog.max_steps:
            return {
                "verify_result": "done",
                "status": "done",
                "status_message": "Budget exhausted",
                "logs": ["[verify] budget exhausted"],
            }

        subtasks = state.get("subtasks", [])
        idx = state.get("current_subtask", 0)
        statuses = list(state.get("subtask_status", []))
        drift = state.get("drift_count", 0)

        # 3. Subtask advancement.
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
                drift = 0
                if idx >= len(subtasks):
                    return {
                        "verify_result": "continue",
                        "current_subtask": idx,
                        "subtask_status": statuses,
                        "task_done_pending": True,
                        "drift_count": 0,
                        "status_message": "All subtasks completed",
                        "logs": ["[verify] all subtasks done → final answer"],
                    }
                return {
                    "verify_result": "continue",
                    "current_subtask": idx,
                    "subtask_status": statuses,
                    "drift_count": 0,
                    "status_message": f"Advanced to subtask {idx}",
                    "logs": [f"[verify] subtask {idx-1} done → advance to {idx}"],
                }

            elif verdict == "drift":
                drift += 1
                if drift >= 3:
                    return {
                        "verify_result": "replan",
                        "current_subtask": idx,
                        "subtask_status": statuses,
                        "drift_count": 0,
                        "status_message": "Drift detected → re-planning",
                        "logs": [f"[verify] drift x{drift} → replan"],
                    }
                return {
                    "verify_result": "continue",
                    "current_subtask": idx,
                    "subtask_status": statuses,
                    "drift_count": drift,
                    "status_message": f"Drift warning ({drift})",
                    "logs": [f"[verify] drift warning {drift}"],
                }
            else:
                drift = 0

        return {
            "verify_result": "continue",
            "current_subtask": idx,
            "subtask_status": statuses,
            "drift_count": drift,
        }

    def route_after_verify(state: BrowserState) -> str:
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


def _make_budget_final_answer(state: BrowserState) -> str:
    steps = state.get("tool_calls_made", 0)
    return f"(预算耗尽：已完成 {steps} 步操作，未显式完成。最后页面状态请见 action_history。)"


async def run_browser_task(
    task: str,
    config: AppConfig = None,
    runtime: BrowserRuntime = None,
) -> dict:
    """Run a browser task in batch mode. Returns the final state dict."""
    if config is None:
        config = AppConfig.from_env()

    owns_runtime = runtime is None
    if runtime is None:
        runtime = BrowserRuntime(config)

    try:
        await runtime.start()
        graph = build_browser_graph(runtime)
        initial = _initial_state(task)
        result = await graph.ainvoke(initial)

        final = dict(result)
        if not final.get("final_answer"):
            final["final_answer"] = _make_budget_final_answer(final)
        return final
    finally:
        if owns_runtime:
            await runtime.stop()
