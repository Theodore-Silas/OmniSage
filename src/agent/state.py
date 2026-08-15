"""
统一 Agent 状态 (v5.0)。

一个状态覆盖知识检索与浏览器操作两类任务；非序列化资源（driver/搜索适配器等）
统一由 AgentRuntime 闭包持有，state 只流转可序列化字段。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Dict, List, TypedDict


class UnifiedAgentState(TypedDict, total=False):
    """State for the unified agent loop."""

    task: str
    """Original natural-language task."""

    plan_text: str
    subtasks: List[dict]
    plan_version: int
    current_subtask: int
    subtask_status: List[str]
    verify_result: str
    task_done_pending: bool
    drift_count: int

    state_summary: str
    """Current state: browser DOM/a11y (browser tasks) or last tool observation."""

    last_action_kind: str
    """none | knowledge | browser — drives the perceive node."""

    action_history: Annotated[List[str], add]
    action_trace: Annotated[List[dict], add]
    last_result: dict
    pending_decision: dict

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
