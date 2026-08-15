"""
Browser agent state (v4.0).

The driver and other non-serializable runtime objects are held OUTSIDE the
state (in a BrowserRuntime closure); the state carries only serializable
fields that flow through the LangGraph nodes.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, TypedDict
from operator import add


class BrowserState(TypedDict, total=False):
    """State for the browser agent loop."""

    task: str
    """Original natural-language task."""

    plan_text: str
    """Structured plan injected into the decision context."""

    subtasks: List[dict]
    """Global planner output (guidance for the LLM)."""

    plan_version: int

    current_subtask: int
    """Index of the currently executing subtask (0-based)."""

    subtask_status: List[str]
    """Per-subtask status: pending | running | done | blocked."""

    verify_result: str
    """Latest verify-node routing signal: continue | replan | done."""

    task_done_pending: bool
    """All subtasks finished; the decide node should emit final_answer."""

    drift_count: int
    """Consecutive drift signals (used to trigger re-planning)."""

    state_summary: str
    """Current page state (URL + title + numbered element list)."""

    action_history: Annotated[List[str], add]
    """Compact per-step action/observation history for the LLM."""

    action_trace: Annotated[List[dict], add]
    """Full action trace (with page_state_hash) for audit + watchdog."""

    last_result: dict
    """Serialized ActionResult of the last action."""

    pending_decision: dict
    """The LLM's latest decision {action, params}."""

    final_answer: str
    evidence_sufficient: bool

    status: str
    status_message: str
    logs: Annotated[List[str], add]
    errors: Annotated[List[str], add]

    tool_calls_made: int
    iteration: int
    page_state_hash: str
    consecutive_repeats: int
