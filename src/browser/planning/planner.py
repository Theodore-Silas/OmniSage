"""
Global planner (v4.0) — decomposes a natural-language task into an ordered
list of subtasks (a coarse-grained plan). Each subtask carries a goal, an
expected end-state (for the verify node), and optional dependencies.

The planner uses the text LLM (DeepSeek) via JSON output; a parse failure
gracefully falls back to a single subtask spanning the whole task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.config import LLMConfig


@dataclass
class Subtask:
    id: int
    goal: str
    expected_state: str = ""
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "expected_state": self.expected_state,
            "depends_on": self.depends_on,
            "status": self.status,
        }


@dataclass
class Plan:
    subtasks: List[Subtask]
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "subtasks": [s.to_dict() for s in self.subtasks]}


_PLAN_PROMPT = """You are a task planner for a web automation agent.

Break the following task into a sequence of small, concrete subtasks. Each subtask
should be a single browser goal (e.g. "open the site", "search for X", "sort by price",
"extract the first 10 results").

Task: {task}

Context (optional): {context}

Return ONLY a JSON array of objects with these fields:
- "goal": string, a concrete browser goal
- "expected_state": string, what the page should show after this subtask succeeds
- "depends_on": array of 1-based indices of subtasks that must finish first (empty if none)

Keep it to at most 6 subtasks. Example:
[{{"goal": "Open the search homepage", "expected_state": "homepage with a search box", "depends_on": []}},
 {{"goal": "Type the query and submit", "expected_state": "results list visible", "depends_on": [1]}}]"""


class GlobalPlanner:
    """LLM-driven task decomposition."""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )
        return self._client

    async def plan(self, task: str, context: str = "") -> Plan:
        """Decompose ``task`` into subtasks."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{
                    "role": "user",
                    "content": _PLAN_PROMPT.format(task=task, context=context),
                }],
                temperature=0.2,
                max_tokens=1500,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            raw = ""

        subtasks = self._parse(raw)
        if not subtasks:
            subtasks = [Subtask(id=1, goal=task, expected_state="task completed")]
        return Plan(subtasks=subtasks)

    @staticmethod
    def _parse(raw: str) -> List[Subtask]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []

        subtasks: List[Subtask] = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            deps = item.get("depends_on", []) or []
            try:
                deps = [int(d) for d in deps]
            except (TypeError, ValueError):
                deps = []
            subtasks.append(Subtask(
                id=i,
                goal=str(item.get("goal", "")),
                expected_state=str(item.get("expected_state", "")),
                depends_on=deps,
            ))
        return subtasks
