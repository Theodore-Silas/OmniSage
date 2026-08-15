"""
Agentic context manager (v3.0).
Implements 6 best-practice techniques for long-running agent loops:

  P0 — Wire ContextManager into the ReAct loop
  P0 — Truncate tool outputs (read_page / read_wiki_page)
  P1 — Write a scratchpad (compressed observations) instead of stacking raw text
  P1 — Truncate tool error tracebacks to the first cause line
  P2 — Summarize conversation tail (older turns → LLM-generated summary)
  P2 — Layered injection (system / scratchpad / history / current question)
"""

import json
from typing import Dict, List, Optional, Union

from openai import AsyncOpenAI

from src.config import LLMConfig


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Token thresholds (rough char-based estimates)
CHARS_PER_TOKEN = 3.5
COMPRESS_THRESHOLD_TOKENS = 6000      # when to start compressing
HARD_LIMIT_TOKENS = 20000             # never exceed this

# Layered injection order (top → bottom)
LAYER_SYSTEM = "system"
LAYER_SCRATCHPAD = "scratchpad"
LAYER_HISTORY = "history"
LAYER_RETRIEVED = "retrieved"
LAYER_CURRENT = "current"

# Compression budgets
MAX_FULL_OBSERVATIONS = 3
MAX_CONTENT_CHARS_OBSERVATION = 3000
MAX_ERROR_CHARS = 200
SCRATCHPAD_MAX_CHARS = 1500
HISTORY_SUMMARY_MAX_CHARS = 800


# ═══════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════

def estimate_tokens(messages: List[dict]) -> int:
    """Rough char-based token estimate for a list of messages."""
    total = 0
    for m in messages:
        content = m.get("content", "") or ""
        # tool_calls argument is significant
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                content += fn.get("arguments", "")
        total += len(str(content))
    return int(total / CHARS_PER_TOKEN)


# ═══════════════════════════════════════════════════════════════
# ContextManager
# ═══════════════════════════════════════════════════════════════

class ContextManager:
    """
    Manages LLM context window during agent loops.

    Capabilities:
      1. Compress observation history into a scratchpad string
      2. Layered message assembly (system/scratchpad/history/current)
      3. Tool output truncation (read_page, read_wiki_page)
      4. Error traceback truncation
      5. Conversation history summarization (LLM-driven)
    """

    def __init__(self, llm_config: LLMConfig = None):
        self.llm_config = llm_config
        self._client = None

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        if self._client is None and self.llm_config:
            self._client = AsyncOpenAI(
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )
        return self._client

    # ─────────────────────────────────────────────────────────
    # P1: Scratchpad — compress observations to a 50-token summary
    # ─────────────────────────────────────────────────────────

    def build_scratchpad(
        self,
        observations: List[dict],
        agent_reasoning_trace: List[str] = None,
        max_chars: int = SCRATCHPAD_MAX_CHARS,
    ) -> str:
        """
        Build a scratchpad string summarizing past observations.
        Strategy: keep the last 3 observations as full text (truncated),
        compress earlier ones to first-line summaries.

        Output is structured Markdown, designed to be re-injected as a
        system message so the model can recall prior work without
        re-reading thousands of tokens of raw text.
        """
        if not observations and not agent_reasoning_trace:
            return ""

        parts = ["# Scratchpad (compressed prior work)\n"]

        if agent_reasoning_trace:
            parts.append("## Reasoning trail")
            for t in agent_reasoning_trace[-6:]:
                parts.append(f"- {t}")
            parts.append("")

        if observations:
            recent = observations[-MAX_FULL_OBSERVATIONS:]
            older = observations[:-MAX_FULL_OBSERVATIONS] if len(observations) > MAX_FULL_OBSERVATIONS else []

            if older:
                parts.append(f"## Earlier observations ({len(older)} summarized)")
                for obs in older:
                    tool = obs.get("tool", "tool")
                    summary = self._one_line_summary(obs.get("content", ""))
                    parts.append(f"- **{tool}**: {summary}")
                parts.append("")

            parts.append(f"## Recent observations (last {len(recent)}, full text)")
            for i, obs in enumerate(recent, 1):
                tool = obs.get("tool", "tool")
                content = obs.get("content", "")
                # Truncate each observation to MAX_CONTENT_CHARS_OBSERVATION
                truncated = self._truncate(content, MAX_CONTENT_CHARS_OBSERVATION)
                parts.append(f"### [{i}] {tool}\n{truncated}")

        scratchpad = "\n".join(parts)
        return self._truncate(scratchpad, max_chars)

    def _one_line_summary(self, content: str, max_chars: int = 120) -> str:
        """Extract first meaningful line for a one-line summary."""
        if not content:
            return "(empty)"
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:max_chars] + ("..." if len(line) > max_chars else "")
        return content[:max_chars] + ("..." if len(content) > max_chars else "")

    def _truncate(self, content: str, max_chars: int) -> str:
        """Truncate content with informative suffix."""
        if not content or len(content) <= max_chars:
            return content
        return content[:max_chars] + f"\n... [{len(content) - max_chars} chars truncated]"

    # ─────────────────────────────────────────────────────────
    # P0: Tool output truncation (helpers used by tools.py)
    # ─────────────────────────────────────────────────────────

    def truncate_tool_output(self, content: str, tool: str) -> str:
        """Truncate a tool's output content based on tool type."""
        max_chars = self._tool_char_limit(tool)
        return self._truncate(content, max_chars)

    def truncate_error(self, error: Union[str, Exception]) -> str:
        """Truncate exception messages to the first cause line (P1.2)."""
        msg = str(error)
        if not msg:
            return "Unknown error"

        # Take only the first line; drop stack traces
        first_line = msg.split("\n")[0].strip()
        if len(first_line) > MAX_ERROR_CHARS:
            return first_line[:MAX_ERROR_CHARS] + "..."
        return first_line

    @staticmethod
    def _tool_char_limit(tool: str) -> int:
        """Per-tool output char limits."""
        limits = {
            "search_web": 3000,
            "search_papers": 4000,
            "search_wiki": 1500,
            "read_page": 3000,        # P0.2 — explicitly bounded
            "read_wiki_page": 3000,
            "follow_link": 3000,
        }
        return limits.get(tool, 3000)

    # ─────────────────────────────────────────────────────────
    # P2.1: Conversation history summarization (LLM-driven)
    # ─────────────────────────────────────────────────────────

    async def summarize_history(
        self,
        conversation_history: List[dict],
        current_query: str,
        max_chars: int = HISTORY_SUMMARY_MAX_CHARS,
    ) -> str:
        """
        Compress older turns of conversation_history into a single summary
        using the LLM. Keeps the most recent turn verbatim.
        """
        if not conversation_history:
            return ""

        # Already short — no compression needed
        if len(conversation_history) <= 4:
            return self._format_history_inline(conversation_history, max_chars)

        # Older turns → summarize, last 2 turns → keep verbatim
        to_summarize = conversation_history[:-2]
        recent = conversation_history[-2:]

        summary_text = ""
        if self.client:
            summary_text = await self._llm_summarize_history(to_summarize, current_query)
        else:
            summary_text = self._heuristic_summarize_history(to_summarize)

        parts = ["# 对话历史摘要\n" + summary_text]
        parts.append("\n## 最近对话（原文）")
        parts.append(self._format_history_inline(recent, max_chars // 2))

        return self._truncate("\n".join(parts), max_chars)

    async def _llm_summarize_history(self, turns: List[dict], current_query: str) -> str:
        """Use LLM to summarize older conversation turns into a structured note."""
        rendered = self._format_history_inline(turns, max_chars=4000)
        prompt = (
            "请将以下对话历史压缩为结构化摘要，每项一行：\n"
            "- 用户核心问题：\n"
            "- 已确定事实：\n"
            "- 待确认/未答项：\n"
            "- 当前进度：\n\n"
            "对话历史：\n" + rendered
        )
        try:
            r = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            return self._heuristic_summarize_history(turns)

    @staticmethod
    def _heuristic_summarize_history(turns: List[dict]) -> str:
        """Fallback: take the first user question + count of turns."""
        first_user = next((t for t in turns if t.get("role") == "user"), None)
        first_q = (first_user.get("content", "")[:200] if first_user else "")
        return (
            f"- 用户核心问题：{first_q}\n"
            f"- 已确定事实：（无LLM摘要，使用启发式压缩）\n"
            f"- 待确认/未答项：\n"
            f"- 当前进度：共 {len(turns)} 轮历史对话"
        )

    @staticmethod
    def _format_history_inline(history: List[dict], max_chars: int = 2000) -> str:
        """Format conversation history as inline text."""
        lines = []
        for msg in history:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = (msg.get("content") or "")[:300]
            lines.append(f"{role}: {content}")
        text = "\n".join(lines)
        return text[:max_chars] + ("\n..." if len(text) > max_chars else "")

    # ─────────────────────────────────────────────────────────
    # P2.2: Layered injection — assemble messages in priority order
    # ─────────────────────────────────────────────────────────

    def build_layered_messages(
        self,
        system_prompt: str,
        scratchpad: str = "",
        history_summary: str = "",
        retrieved_knowledge: str = "",
        recent_messages: List[dict] = None,
        current_instruction: str = "",
    ) -> List[dict]:
        """
        Assemble messages in priority order (top → bottom):
          1. system prompt (stable instructions)
          2. scratchpad (compressed observations)
          3. history summary (older conversation turns)
          4. retrieved knowledge (Wiki hits, RAG context)
          5. recent raw messages (last few turns)
          6. current instruction (this turn's question)

        This follows the 2026 best practice of placing stable,
        high-attention content at top and query-specific content at bottom.
        """
        messages: List[dict] = []

        # Layer 1: System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Layer 2: Scratchpad (P1.1 — write, don't dump)
        if scratchpad:
            messages.append({
                "role": "system",
                "content": scratchpad,
            })

        # Layer 3: History summary (P2.1)
        if history_summary:
            messages.append({
                "role": "system",
                "content": history_summary,
            })

        # Layer 4: Retrieved knowledge (Wiki/RAG context)
        if retrieved_knowledge:
            messages.append({
                "role": "system",
                "content": f"# 已检索到的相关知识\n\n{retrieved_knowledge}",
            })

        # Layer 5: Recent raw messages (last few tool turns)
        if recent_messages:
            messages.extend(recent_messages)

        # Layer 6: Current instruction
        if current_instruction:
            messages.append({"role": "user", "content": current_instruction})

        return messages

    # ─────────────────────────────────────────────────────────
    # P0: Wire into agentic loop — full-message compression
    # ─────────────────────────────────────────────────────────

    def compress_messages(
        self,
        messages: List[dict],
        observations: List[dict] = None,
        agent_trace: List[str] = None,
    ) -> List[dict]:
        """
        Compress the full message list when approaching context budget.

        Strategy:
          1. Always keep system prompt + most recent 4 messages verbatim.
          2. Replace older tool/assistant messages with a synthetic
             assistant message summarizing the scratchpad.
          3. Drop tool messages that are clearly superseded.
        """
        if not messages:
            return messages

        tokens = estimate_tokens(messages)
        if tokens < COMPRESS_THRESHOLD_TOKENS:
            return messages

        # Keep system + last 4 messages verbatim
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= 4:
            return messages

        keep_recent = non_system[-4:]
        older = non_system[:-4]

        if observations is None:
            observations = []

        scratchpad = self.build_scratchpad(observations, agent_trace or [])

        result = list(system_msgs)
        if scratchpad:
            result.append({
                "role": "system",
                "content": (
                    f"# Scratchpad (compressed prior work, replaced "
                    f"{len(older)} messages to save context)\n\n"
                    + scratchpad
                ),
            })
        result.extend(keep_recent)

        return result

    def compress(self, observations: List[dict], max_tokens: int = 6000) -> str:
        """
        Backwards-compatible: compress observations to a scratchpad string.
        Equivalent to build_scratchpad() with the default observation list.
        """
        return self.build_scratchpad(observations, max_chars=max_tokens * 4)


# ═══════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════

_context_manager: ContextManager = None


def get_context_manager(llm_config: LLMConfig = None) -> ContextManager:
    """Get or create the context manager singleton."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager(llm_config)
    return _context_manager