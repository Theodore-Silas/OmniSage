"""
统一 Agent 入口 (v5.0) — batch + streaming。
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from src.agent.graph import AgentRuntime, build_unified_graph, _initial_state, _make_budget_final_answer
from src.config import AppConfig


async def run_agent_stream(
    task: str,
    config: AppConfig = None,
    runtime: AgentRuntime = None,
) -> AsyncGenerator[dict, None]:
    """Run the unified agent in streaming mode.

    Yields events: {type: "status"|"log"|"action"|"done", content, meta}.
    """
    if config is None:
        config = AppConfig.from_env()

    owns_runtime = runtime is None
    if runtime is None:
        runtime = AgentRuntime(config)

    try:
        graph = build_unified_graph(runtime)
        initial = _initial_state(task)
        final_state = dict(initial)

        async for event in graph.astream(initial, stream_mode="updates"):
            for node_name, update in event.items():
                if not update:
                    continue

                for k, v in update.items():
                    if isinstance(v, list) and isinstance(final_state.get(k), list):
                        final_state[k] = final_state[k] + v
                    else:
                        final_state[k] = v

                msg = update.get("status_message", "")
                if msg:
                    yield {"type": "status", "content": msg, "meta": {"node": node_name}}

                for log in update.get("logs", []):
                    yield {"type": "log", "content": log, "meta": {"node": node_name}}

                for t in update.get("action_trace", []):
                    kind = t.get("kind", "")
                    yield {
                        "type": "action",
                        "content": f"{t['action']}: {'OK' if t['success'] else 'FAIL'}",
                        "meta": {"node": node_name, "trace": t, "kind": kind},
                    }

        answer = final_state.get("final_answer", "")
        if not answer:
            answer = _make_budget_final_answer(final_state)

        yield {
            "type": "done",
            "content": answer,
            "meta": {
                "tool_calls_made": final_state.get("tool_calls_made", 0),
                "iterations": final_state.get("iteration", 0),
                "errors": final_state.get("errors", []),
            },
        }
    finally:
        if owns_runtime:
            await runtime.stop()
