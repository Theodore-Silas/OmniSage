"""
Subtask verifier (v4.0) — the verify node's LLM judge.

Decides, per subtask, whether its expected state has been reached ("done"),
whether the agent has drifted off-target ("drift"), or whether to keep going
("continue"). Also summarizes the action history into the final answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from src.config import LLMConfig


_VERIFY_PROMPT = """You are the verifier of a web automation agent.

Subtask goal: {goal}
Expected result after success: {expected_state}

Current page state:
{state_summary}

Action history so far:
{history}

Judge the agent's progress and return ONLY a JSON object with:
- "verdict": one of "done" (subtask goal achieved), "drift" (clearly off-target, e.g. navigated to an unrelated page or repeated failed actions), "continue" (in progress, neither achieved nor drifted)
- "reason": one short sentence

Example: {{"verdict": "done", "reason": "The search results are visible and extracted"}}"""


_SUMMARIZE_PROMPT = """You are a web automation agent finishing a task.

Task: {task}

Action history (steps taken and observations):
{history}

Write a concise final answer reporting what was accomplished, including any
extracted data or key findings. Respond in the user's language (Chinese by
default)."""


class SubtaskVerifier:
    """LLM judge for subtask completion / drift, plus final-answer summary."""

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

    async def evaluate(
        self,
        goal: str,
        expected_state: str,
        state_summary: str,
        history: str,
    ) -> str:
        """Return "done" | "drift" | "continue"."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{
                    "role": "user",
                    "content": _VERIFY_PROMPT.format(
                        goal=goal,
                        expected_state=expected_state,
                        state_summary=state_summary,
                        history=history,
                    ),
                }],
                temperature=0.1,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
        except Exception:
            return "continue"

        verdict = self._parse(raw)
        return verdict if verdict in ("done", "drift", "continue") else "continue"

    async def summarize(self, task: str, history: str) -> str:
        """Generate the final answer from the action history."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{
                    "role": "user",
                    "content": _SUMMARIZE_PROMPT.format(task=task, history=history),
                }],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"(无法生成最终答案：{str(e)[:200]})"

    @staticmethod
    def _parse(raw: str) -> str:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return "continue"
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "continue"
        return str(data.get("verdict", "continue")).strip().lower()
