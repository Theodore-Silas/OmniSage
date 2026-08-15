"""
Local executor (v4.0) — the ReAct "decide" step.

Given the current page state and history, the LLM picks the next atomic
browser action (or final_answer) via function calling over the browser tool
schemas. This is the fine-grained counterpart to the global planner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.browser.action.tools import BROWSER_TOOL_SCHEMAS
from src.config import LLMConfig


@dataclass
class Decision:
    """A single atomic action decision by the LLM."""
    action: str                      # tool name (browser_*) or "final_answer"
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def is_final(self) -> bool:
        return self.action == "final_answer"


UNIFIED_SYSTEM_PROMPT = """You are a general-purpose task agent with TWO kinds of tools:

## Knowledge-retrieval tools (no browser needed)
- search_web / search_papers / search_news / search_blogs — search the internet
- read_page — fetch and read a URL's content via HTTP
- search_wiki / read_wiki_page / follow_link — query the local knowledge base
Use these for research, fact-finding, and summarization tasks.

## Browser-action tools (drive a real browser)
- browser_navigate / browser_click / browser_type / browser_scroll / browser_press /
  browser_hover / browser_select / browser_drag / browser_switch_tab / browser_go_back
- browser_read_a11y — read the page's numbered interactive-element list
- browser_screenshot — visually inspect the page (vision model when available)
- browser_extract — scrape visible text; browser_wait — wait for an element; browser_exec_js — read-only JS
Use these for interactive web tasks (login, form-filling, clicking, dynamic scraping).

## How to work
1. Choose the right tool category for the task: knowledge tasks → search/read; interactive tasks → browser_*.
2. For browser tasks, on a NEW page call browser_read_a11y first; prefer element "index" over vague text.
3. Perform ONE logical action per step. Re-read the page when state changes.
4. When done, call final_answer with the complete result (include citations/sources where relevant).

## Rules
- NEVER guess a URL or element — observe first.
- If an action fails, adapt: re-read, retry, or try a different approach.
- Be economical: at most a few actions per subtask.
- Respond in the user's language (Chinese by default)."""

# Backwards-compatible alias (v4.0 browser-only prompt).
BROWSER_SYSTEM_PROMPT = UNIFIED_SYSTEM_PROMPT


class LocalExecutor:
    """ReAct decision maker over a (configurable) tool-schema set."""

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

    async def decide(
        self,
        task: str,
        subtask_goal: str,
        state_summary: str,
        history: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Decision:
        """Ask the LLM for the next action. Returns a Decision.

        ``tools`` defaults to the browser tool schemas; the unified agent
        passes the full unified pool (knowledge + browser + final_answer).
        """
        if tools is None:
            tools = BROWSER_TOOL_SCHEMAS
        if messages is None:
            messages = [{"role": "system", "content": UNIFIED_SYSTEM_PROMPT}]

        user_content = (
            f"## 任务 (Task)\n{task}\n\n"
            f"## 当前子任务 (Current subtask)\n{subtask_goal}\n\n"
            f"## 当前状态 (Current state)\n{state_summary or '(unknown)'}\n\n"
            f"## 已完成动作 (Action history)\n{history or '(none)'}\n\n"
            "请决定下一步动作 (Decide the next action)."
        )

        call_messages = messages + [{"role": "user", "content": user_content}]

        try:
            resp = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=call_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as e:
            return Decision(action="final_answer", params={"answer": f"(LLM error: {str(e)[:200]})"})

        msg = resp.choices[0].message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return Decision(action=name, params=args)

        content = (msg.content or "").strip()
        if content:
            return Decision(action="final_answer", params={"answer": content})
        return Decision(action="final_answer", params={"answer": "(no action decided)"})
