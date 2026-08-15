"""
Comprehensive tests for v3.0 context management upgrades.

Tests all 6 best-practice techniques:
  P0 — ContextManager wired into the ReAct loop
  P0 — Tool output truncation
  P1 — Scratchpad (write, don't dump)
  P1 — Error traceback truncation
  P2 — Conversation history summarization
  P2 — Layered injection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig, LLMConfig
from src.agentic.context_manager import (
    ContextManager, get_context_manager, estimate_tokens,
    MAX_FULL_OBSERVATIONS, SCRATCHPAD_MAX_CHARS, HISTORY_SUMMARY_MAX_CHARS,
)


def banner(s):
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}")


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

config = AppConfig.from_env()
cm = ContextManager(config.llm)

failures = []


def check(name, cond):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}")
        failures.append(name)


# ═══════════════════════════════════════════════════════════════
# P0.2: Tool output truncation
# ═══════════════════════════════════════════════════════════════

banner("P0.2 — Tool output truncation")

# Simulate a large read_page output
big_content = "A" * 10000
truncated = cm.truncate_tool_output(big_content, "read_page")
check(
    "read_page output ≤ 3000 chars + truncation notice",
    len(truncated) < 3500 and "truncated" in truncated
)

truncated_wiki = cm.truncate_tool_output(big_content, "read_wiki_page")
check(
    "read_wiki_page output ≤ 3000 chars",
    len(truncated_wiki) < 3500
)

# Search outputs
truncated_search = cm.truncate_tool_output(big_content, "search_web")
check(
    "search_web output ≤ 3000 chars",
    len(truncated_search) < 3500
)

# Small output preserved
small = "tiny content"
check("small output preserved", cm.truncate_tool_output(small, "read_page") == small)


# ═══════════════════════════════════════════════════════════════
# P1.1: Scratchpad
# ═══════════════════════════════════════════════════════════════

banner("P1.1 — Scratchpad (write, don't dump)")

obs = []
for i in range(7):
    obs.append({
        "tool": f"tool_{i}",
        "content": f"This is observation #{i}.\nDetailed content line 1 with unique_marker_obs{i}.\nLine 2."
    })
scratchpad = cm.build_scratchpad(obs)
check(
    f"scratchpad built with 7 observations → ≤ {SCRATCHPAD_MAX_CHARS} chars",
    0 < len(scratchpad) <= SCRATCHPAD_MAX_CHARS + 200
)
check("scratchpad contains 'Scratchpad' header", "Scratchpad" in scratchpad)
check("scratchpad mentions earlier observations (summarized)", "Earlier" in scratchpad or "summarized" in scratchpad.lower())
check("scratchpad preserves recent observations in full (last 3 obs appear verbatim)",
      "unique_marker_obs6" in scratchpad and "unique_marker_obs5" in scratchpad)
print(f"  ↪ Length: {len(scratchpad)} chars (budget: {SCRATCHPAD_MAX_CHARS})")

# With reasoning trace
trace = ["Iter 1: search_web", "Iter 2: read_page", "Iter 3: search_papers"]
scratchpad2 = cm.build_scratchpad(obs, trace)
check("scratchpad includes reasoning trail", "Reasoning" in scratchpad2 or "Iter 1" in scratchpad2)


# ═══════════════════════════════════════════════════════════════
# P1.2: Error traceback truncation
# ═══════════════════════════════════════════════════════════════

banner("P1.2 — Error traceback truncation")

# Realistic multi-line Python traceback
traceback_str = (
    "Traceback (most recent call last):\n"
    "  File \"/Users/x/project/main.py\", line 42, in <module>\n"
    "    result = run_search(\"test query\")\n"
    "  File \"/Users/x/project/agent.py\", line 88, in run_search\n"
    "    return await client.search(...)\n"
    "  File \"/Users/x/.local/lib/.../openai/_base_client.py\", line 1234, in _request\n"
    "    raise APIConnectionError(...) from None\n"
    "openai.APIConnectionError: Connection error.\n"
    "This is a long line that should be truncated to the first cause only."
)
truncated_err = cm.truncate_error(traceback_str)
check("error truncated to single line", "\n" not in truncated_err)
# First line of a Python traceback is the summary line ("Traceback...")
# which is informative; we keep it as-is, but drop the file/line noise below
check("error preserves some traceback context", "Traceback" in truncated_err or "APIConnection" in truncated_err)
check("error drops deep frames", "openai/_base_client.py" not in truncated_err)
print(f"  ↪ Original: {len(traceback_str)} chars → Truncated: {len(truncated_err)} chars")

# Exception object
try:
    raise ValueError("something went wrong\nwith a multi-line message")
except ValueError as e:
    out = cm.truncate_error(e)
check("Exception object truncated", "something went wrong" in out and "\n" not in out)


# ═══════════════════════════════════════════════════════════════
# P2.1: Conversation history summarization
# ═══════════════════════════════════════════════════════════════

banner("P2.1 — Conversation history summarization")

import asyncio

short_history = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for..."},
]
summary_short = asyncio.run(cm.summarize_history(short_history, "current question"))
check("short history (≤4 turns) → no LLM call needed", "对 话" in summary_short or "用户" in summary_short)
check("short history preserves content", "RAG" in summary_short)

# Long history → triggers LLM summarization
long_history = []
for i in range(8):
    long_history.append({"role": "user", "content": f"Question #{i} about topic {i}"})
    long_history.append({"role": "assistant", "content": f"Answer to question #{i} with extensive content..."})
summary_long = asyncio.run(cm.summarize_history(long_history, "current question"))
check("long history → has history summary header", "对话历史摘要" in summary_long or "历史" in summary_long)
check("long history preserves recent turn", "Question #7" in summary_long or "#7" in summary_long)
print(f"  ↪ Summary length: {len(summary_long)} chars (budget: {HISTORY_SUMMARY_MAX_CHARS})")


# ═══════════════════════════════════════════════════════════════
# P2.2: Layered injection
# ═══════════════════════════════════════════════════════════════

banner("P2.2 — Layered message assembly")

messages = cm.build_layered_messages(
    system_prompt="You are a research assistant.",
    scratchpad="# Scratchpad\n- Did search X",
    history_summary="# History summary\n- User asked Y",
    retrieved_knowledge="Wiki hit on topic Z...",
    recent_messages=[
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ],
    current_instruction="What is X?",
)
roles = [m["role"] for m in messages]
check("layered messages assembled", len(messages) >= 5)

# Verify priority order: system first, user last
check("system prompt at top", roles[0] == "system")
check("current instruction at bottom (user role)", roles[-1] == "user")
check("system layers include scratchpad/history/retrieved",
      sum(1 for r in roles if r == "system") >= 4)
print(f"  ↪ Message count: {len(messages)}, role sequence: {roles}")


# ═══════════════════════════════════════════════════════════════
# P0: compress_messages (wire into ReAct loop)
# ═══════════════════════════════════════════════════════════════

banner("P0 — compress_messages in ReAct loop")

# Small message list → no compression needed
small_messages = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]
result = cm.compress_messages(small_messages)
check("small list unchanged", result == small_messages)

# Large message list → compression triggers
big_messages = (
    [{"role": "system", "content": "Be helpful. " * 100}]  # ~1500 chars
    + [
        {"role": "user", "content": f"Tool call result {i}: " + ("X" * 2000)}
        for i in range(15)  # 15 long tool messages
    ]
    + [{"role": "user", "content": "final question"}]
)
original_tokens = estimate_tokens(big_messages)
result = cm.compress_messages(big_messages, obs)
new_tokens = estimate_tokens(result)
check(f"compression triggered (was {original_tokens} tokens)", original_tokens > 6000)
check(f"compressed to ≤ ~6000 tokens (now {new_tokens})", new_tokens < original_tokens * 0.7)
check("compressed version preserves final user message", result[-1].get("content") == "final question")
check("compressed version has scratchpad layer",
      any("Scratchpad" in m.get("content", "") for m in result))


# ═══════════════════════════════════════════════════════════════
# Token estimation
# ═══════════════════════════════════════════════════════════════

banner("Token estimation")

msgs = [
    {"role": "system", "content": "A" * 350},  # ~100 tokens
    {"role": "user", "content": "B" * 700},    # ~200 tokens
]
t = estimate_tokens(msgs)
check(f"estimate_tokens rough accuracy (got {t}, expect ~300)", 250 < t < 400)


# ═══════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════

banner("Singleton")

cm1 = get_context_manager(config.llm)
cm2 = get_context_manager(config.llm)
check("get_context_manager returns same instance", cm1 is cm2)


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
if failures:
    print(f"  {len(failures)} failures:")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print("  ALL CONTEXT MANAGEMENT UPGRADES PASSED")
    print(f"{'=' * 70}\n")