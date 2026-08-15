"""
Browser agent entry points (v4.0) — batch + streaming.

``run_browser_task`` runs a task and returns the final state; ``stream``
yields progress events for a live UI (status / log / action / done).
"""

from __future__ import annotations

from typing import AsyncGenerator, Optional

from src.browser.graph import BrowserRuntime, build_browser_graph, _initial_state, _make_budget_final_answer
from src.config import AppConfig


async def run_browser_task_stream(
    task: str,
    config: AppConfig = None,
    runtime: BrowserRuntime = None,
) -> AsyncGenerator[dict, None]:
    """Run a browser task in streaming mode.

    Yields events: {type: "status"|"log"|"action"|"done", content, meta}.
    """
    if config is None:
        config = AppConfig.from_env()

    owns_runtime = runtime is None
    if runtime is None:
        runtime = BrowserRuntime(config)

    try:
        await runtime.start()
        graph = build_browser_graph(runtime)
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
                    yield {
                        "type": "action",
                        "content": f"{t['action']}: {'OK' if t['success'] else 'FAIL'}",
                        "meta": {"node": node_name, "trace": t},
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
