"""
Agentic RAG node implementations (v3.0) — with full context management.
ReAct agent loop: Think → Act → Observe → (continue | answer).

Upgrades applied:
  P0 — ContextManager wired into the ReAct loop (compresses messages before LLM call)
  P0 — Tool output truncation handled inside ContextManager
  P1 — Scratchpad generated from observations, injected as system layer
  P1 — Error traceback truncation to first cause line
  P2 — Conversation history summarization (LLM-driven when available)
  P2 — Layered message assembly (system / scratchpad / history / recent / current)
"""

import json
import re
from typing import Any, Dict, List

from openai import AsyncOpenAI

from src.agent.agentic_state import AgenticState
from src.agent.tools import TOOL_SCHEMAS, execute_tool, execute_tool_calls
from src.agentic.context_manager import get_context_manager, estimate_tokens
from src.config import AppConfig
from src.wiki.manager import WikiManager


# ═══════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """You are SearchAgent, an autonomous research assistant with access to multiple search and reading tools.

## Your Mission
Answer the user's query thoroughly by gathering information from the most relevant sources. Think step by step, use tools strategically, and cite your sources.

## Available Tools
- **search_web(query, max_results?)**: Search the web via DuckDuckGo for general info, news, tutorials, docs.
- **search_papers(query, max_results?)**: Search academic papers via Semantic Scholar + ArXiv.
- **search_wiki(query, top_k?)**: Search the local Wiki knowledge base for previously stored knowledge. Always check this FIRST.
- **read_page(url)**: Fetch and read the full content of a web page. Use AFTER finding a relevant URL.
- **read_wiki_page(path)**: Read a specific Wiki page by its relative path (e.g. 'concepts/rag.md').
- **follow_link(from_path, link_target)**: Navigate from one Wiki page to a linked page.
- **final_answer(answer)**: Call this when you have sufficient evidence. Provide complete answer in Markdown with citations.

## Reasoning Protocol
1. **Analyze** the query — what kind of information is needed? What do I already know?
2. **Search Wiki first** — always check local knowledge before external search.
3. **Plan** — which tools should I use, in what order?
4. **Execute** one or more tools, observe results.
5. **Evaluate** — is the evidence sufficient? Do I have diverse sources? Any contradictions?
6. **Refine** — if not sufficient, try different queries, read specific pages, follow Wiki links.
7. **Answer** — call final_answer with a comprehensive, well-structured response.

## Sufficiency Checklist
Before answering, verify:
- [ ] Searched at least 2 different sources (web + paper, or web + wiki, etc.)
- [ ] Read at least 1-2 full pages (not just search snippets) for key information
- [ ] Checked the Wiki for existing knowledge
- [ ] If contradictions were found, they are explicitly noted
- [ ] ALL factual claims have source citations with URLs

## Rules
- ALWAYS search Wiki first — it's faster and already vetted
- When web and Wiki information conflict, flag the contradiction explicitly
- Each source should be cited as [Source N] with URL
- Keep your thinking concise (1-2 sentences per thought)
- Primary output language: Chinese (中文) unless the user asks otherwise
- Budget: You have a limited number of tool calls. Prioritize wisely.
- NEVER make up information — if you can't find something, say so honestly.
- Use the **Scratchpad** (when present in your context) to recall prior work — do NOT re-read raw tool outputs.

## Output Format (when calling final_answer)
Your final answer should be structured as:

## 核心结论
(3-5 sentences summarizing the key findings)

## 分源分析
### 来源A
...
### 来源B
...

## 观点差异与不确定性
(Point out any contradictions or lack of certainty)

## 参考来源
- [1] Title — URL
- [2] Title — URL"""


# ═══════════════════════════════════════════════════════════════
# Agentic Node — ReAct Think/Act (with layered context)
# ═══════════════════════════════════════════════════════════════

async def agentic_node(state: AgenticState, config: AppConfig) -> dict:
    """
    ReAct agent node: calls LLM with tool definitions.
    Returns either tool_calls (to be executed) or final_answer.

    This is the "Think + Decide" step of the ReAct loop.

    Context management (P0/P1/P2):
      - Compresses older messages via ContextManager before LLM call
      - Injects scratchpad as a system layer summarizing past observations
      - Optionally injects conversation history summary
    """
    query = state.get("query", "")
    messages = state.get("messages", [])
    observations = state.get("observations", [])
    trace = state.get("agent_reasoning_trace", [])

    cm = get_context_manager(config.llm)

    # Initialize messages on first call — layered injection
    if not messages:
        history = state.get("conversation_history", [])
        history_summary = ""
        if history and len(history) >= 4:
            history_summary = await cm.summarize_history(history, query)

        scratchpad = cm.build_scratchpad(observations, trace)

        messages = cm.build_layered_messages(
            system_prompt=AGENT_SYSTEM_PROMPT,
            scratchpad=scratchpad,
            history_summary=history_summary,
            current_instruction=f"Please research and answer the following question:\n\n{query}",
        )

    # Compress if approaching budget (P0: wire ContextManager into loop)
    observations = state.get("observations", [])
    trace = state.get("agent_reasoning_trace", [])
    messages = cm.compress_messages(messages, observations, trace)

    client = AsyncOpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
    )

    try:
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
    except Exception as e:
        return {
            "final_answer": f"LLM call failed: {cm.truncate_error(e)}",
            "evidence_sufficient": True,
            "status": "done",
            "status_message": "Agent error",
            "logs": [f"[agentic] LLM error: {cm.truncate_error(e)}"],
        }

    msg = response.choices[0].message
    new_messages = messages + [msg.model_dump(exclude_none=True)]

    # Case 1: LLM called a tool
    if msg.tool_calls:
        tool_names = [tc.function.name for tc in msg.tool_calls]
        iter_count = state.get("agent_iteration", 0) + 1
        trace.append(f"[Iter {iter_count}] Calling: {', '.join(tool_names)}")

        pending = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

        # Check if any call is final_answer
        has_final = any(tc["function"]["name"] == "final_answer" for tc in pending)
        if has_final:
            final_tc = next(tc for tc in pending if tc["function"]["name"] == "final_answer")
            try:
                final_args = json.loads(final_tc["function"]["arguments"])
                answer = final_args.get("answer", "")
            except json.JSONDecodeError:
                answer = final_tc["function"]["arguments"]

            return {
                "messages": new_messages,
                "final_answer": answer,
                "evidence_sufficient": True,
                "agent_reasoning_trace": trace,
                "agent_iteration": iter_count,
                "tool_calls_made": state.get("tool_calls_made", 0),
                "status": "done",
                "status_message": "Agent completed research",
                "logs": [f"[agentic] Agent finished after {iter_count} iterations"],
            }

        return {
            "messages": new_messages,
            "pending_tool_calls": pending,
            "evidence_sufficient": False,
            "agent_reasoning_trace": trace,
            "agent_iteration": iter_count,
            "status": "acting",
            "status_message": f"Agent acting: {', '.join(tool_names)}",
            "logs": [f"[agentic] Iter {iter_count}: {', '.join(tool_names)}"],
        }

    # Case 2: LLM returned text (no tool call) — treat as final answer
    content = msg.content or ""
    if content.strip():
        return {
            "messages": new_messages,
            "final_answer": content,
            "evidence_sufficient": True,
            "agent_reasoning_trace": trace,
            "status": "done",
            "status_message": "Agent completed research",
            "logs": [f"[agentic] Agent answered directly after {state.get('agent_iteration', 0)} iterations"],
        }

    # Case 3: Empty response — force final answer
    return {
        "messages": new_messages,
        "final_answer": "I was unable to gather sufficient information to answer your question. Please try rephrasing or providing more details.",
        "evidence_sufficient": True,
        "status": "done",
        "status_message": "Agent could not answer",
    }


# ═══════════════════════════════════════════════════════════════
# Tool Node — Execute tool calls (with error truncation)
# ═══════════════════════════════════════════════════════════════

async def tool_execution_node(state: AgenticState, config: AppConfig) -> dict:
    """
    Execute pending tool calls and format observations.

    This is the "Act + Observe" step of the ReAct loop.

    Upgrades:
      - P0.2: Tool outputs truncated via ContextManager per-tool limits
      - P1.2: Errors truncated to first cause line (no raw tracebacks)
    """
    pending = state.get("pending_tool_calls", [])
    messages = state.get("messages", [])

    if not pending:
        return {"pending_tool_calls": [], "logs": ["[tool] No pending tool calls"]}

    cm = get_context_manager(config.llm)

    # Execute tool calls (the executors themselves apply per-tool truncation;
    # we apply error truncation in the wrapper below)
    observations = await _safe_execute_tool_calls(pending, config, cm)

    tool_count = state.get("tool_calls_made", 0) + len(observations)

    # Format observations as tool messages
    new_messages = list(messages)
    for obs in observations:
        tool_msg = {
            "role": "tool",
            "tool_call_id": obs["tool_call_id"],
            "content": obs["content"],
        }
        new_messages.append(tool_msg)

    tool_names = [o["tool"] for o in observations]

    return {
        "messages": new_messages,
        "pending_tool_calls": [],
        "observations": observations,
        "tool_calls_made": tool_count,
        "status_message": f"Observed: {', '.join(tool_names)}",
        "logs": [f"[tool] Executed {len(observations)} tools: {', '.join(tool_names)} (total: {tool_count})"],
    }


async def _safe_execute_tool_calls(tool_calls: List[dict], config: AppConfig, cm) -> List[dict]:
    """
    Wrap execute_tool_calls to:
      - truncate any exception message to first cause line (P1.2)
      - apply per-tool output truncation (P0.2)
    """
    import asyncio

    async def execute_one(tc: dict) -> dict:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        from src.agent.tools import execute_tool
        try:
            content = await execute_tool(name, args, config)
        except Exception as e:
            # P1.2: truncate error to first cause line
            content = f"Tool '{name}' execution error: {cm.truncate_error(e)}"

        # P0.2: truncate large outputs (some executors don't pre-truncate)
        if name != "final_answer":  # never truncate final_answer
            content = cm.truncate_tool_output(content, name)

        return {
            "tool": name,
            "content": content,
            "tool_call_id": tc.get("id", ""),
            "arguments": args,
        }

    if not tool_calls:
        return []

    tasks = [execute_one(tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    observations = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            tc_name = tool_calls[i].get("function", {}).get("name", "unknown")
            observations.append({
                "tool": tc_name,
                "content": f"Tool execution error: {cm.truncate_error(res)}",
                "tool_call_id": tool_calls[i].get("id", ""),
                "arguments": {},
            })
        else:
            observations.append(res)
    return observations


# ═══════════════════════════════════════════════════════════════
# Wiki Pre-Search (reused from v2.0, adapted for agentic)
# ═══════════════════════════════════════════════════════════════

async def agentic_wiki_persist_node(state: AgenticState, config: AppConfig) -> dict:
    """
    Archive agent research results into the Wiki knowledge base.
    Simplified version for agentic mode — archives observations.
    """
    query = state.get("query", "")
    answer = state.get("final_answer", "")
    observations = state.get("observations", [])

    if not answer:
        return {"status": "done", "status_message": "Complete"}

    try:
        wiki = WikiManager()

        # Extract URLs from observations
        search_results = []
        for obs in observations:
            urls = re.findall(r'https?://[^\s\)\]]+', obs.get("content", ""))
            for url in urls[:2]:  # limit per observation
                search_results.append({
                    "source": obs.get("tool", "web").replace("search_", ""),
                    "title": url.split("/")[-1][:80] or "Untitled",
                    "url": url,
                    "snippet": obs.get("content", "")[:200],
                })

        if search_results:
            wiki.archive_search(query, answer, search_results[:10], [])
            wiki.log_operation(
                operation="AGENTIC_SEARCH",
                details=f"Query: {query[:80]}\nObservations: {len(observations)} | Tool calls: {state.get('tool_calls_made', 0)}",
                pages_affected=len(search_results) + 1,
            )

        return {
            "status": "done",
            "status_message": "Complete",
            "logs": [f"[agentic-persist] Archived {len(search_results)} sources to Wiki"],
        }
    except Exception as e:
        return {"status": "done", "status_message": "Complete", "logs": [f"[agentic-persist] {e}"]}