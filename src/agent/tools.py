"""
统一工具集 (v5.0) — 知识检索 + 浏览器动作 + 终止，单一 Agent 自主选择。

三类工具：
  knowledge (8): search_web / search_papers / search_news / search_blogs /
                 read_page / search_wiki / read_wiki_page / follow_link
  browser  (15): browser_navigate / click / type / scroll / press / hover /
                 select / drag / switch_tab / go_back / read_a11y /
                 screenshot / extract / wait / exec_js
  final     (1): final_answer

统一入口：``execute_unified_tool(name, args, runtime)``，由统一运行时装配。
浏览器动作触发 runtime.ensure_browser() 实现 lazy 启动。

保留 ``TOOL_SCHEMAS`` / ``execute_tool`` / ``execute_tool_calls`` 作为 v3.0
向后兼容层（渐进迁移期间旧 agentic_graph 仍可用）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from src.browser.action.tools import (
    BROWSER_TOOL_SCHEMAS,
    _action_of,
    format_observation,
)
from src.config import AppConfig
from src.fetch.fetcher import ContentFetcher, FetchedContent
from src.search.base import SearchResult
from src.search.registry import SearchRegistry
from src.wiki.manager import WikiManager


# ═══════════════════════════════════════════════════════════════
# 1. 知识检索工具 schema（8 个）
# ═══════════════════════════════════════════════════════════════

_QUERY_PARAM = {
    "type": "string",
    "description": "Search keywords. Use specific, targeted terms for best results.",
}
_MAX_RESULTS_PARAM = {
    "type": "integer",
    "description": "Maximum number of results to return (default 5, max 10).",
    "default": 5,
}

KNOWLEDGE_TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for general information, news, tutorials, and documentation. Use for broad queries, current events, how-to questions, and technology topics.",
            "parameters": {
                "type": "object",
                "properties": {"query": _QUERY_PARAM, "max_results": _MAX_RESULTS_PARAM},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search academic papers via Semantic Scholar and ArXiv. Use for research questions, scientific topics, and scholarly references.",
            "parameters": {
                "type": "object",
                "properties": {"query": _QUERY_PARAM, "max_results": _MAX_RESULTS_PARAM},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search recent news articles (via NewsAPI). Use for time-sensitive events, announcements, and current affairs.",
            "parameters": {
                "type": "object",
                "properties": {"query": _QUERY_PARAM, "max_results": _MAX_RESULTS_PARAM},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_blogs",
            "description": "Search technical blog posts via RSS feeds. Use for engineering write-ups, tutorials, and hands-on guides.",
            "parameters": {
                "type": "object",
                "properties": {"query": _QUERY_PARAM, "max_results": _MAX_RESULTS_PARAM},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch and read the full content of a web page by its URL (HTTP fetch + Markdown extraction). Use AFTER finding a relevant URL to get details.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The full URL of the page to read."}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": "Search the local Wiki knowledge base for previously stored knowledge. Always check this first for faster results if the topic may have been researched before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords to find relevant Wiki pages."},
                    "top_k": {"type": "integer", "description": "Max Wiki pages to return (default 3, max 5).", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_wiki_page",
            "description": "Read a specific Wiki page from the local knowledge base by its relative path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative wiki page path, e.g. 'concepts/rag-methods.md'."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_link",
            "description": "Follow a [[wikilink]] from one Wiki page to a related page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_path": {"type": "string", "description": "Current wiki page path."},
                    "link_target": {"type": "string", "description": "The [[wikilink]] target path to follow."},
                },
                "required": ["from_path", "link_target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fused_search",
            "description": "Fused multi-source search: query web+papers+news+blogs in PARALLEL, deduplicate by URL, and rank (BM25 + semantic hybrid). Use as the PRIMARY search for broad research — covers all sources in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _QUERY_PARAM,
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["web", "paper", "news", "blog"]},
                        "description": "Sources to search (default: all four).",
                    },
                    "max_per_source": {"type": "integer", "description": "Max results per source (default 5)."},
                    "top_k": {"type": "integer", "description": "Total results to return after ranking (default 10)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize",
            "description": "Produce a structured research report from all gathered evidence via Map-Reduce (per-source summary → cross-source synthesis with citations). Call AFTER gathering evidence with fused_search/search_*/read_page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The research question (defaults to the current task)."},
                },
            },
        },
    },
]

# 浏览器动作 schema（去掉 final_answer，统一池里只保留一个终止工具）
BROWSER_ACTION_SCHEMAS: List[dict] = [
    s for s in BROWSER_TOOL_SCHEMAS if s["function"]["name"] != "final_answer"
]

FINAL_ANSWER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Call when the task is complete. Provide the final result, synthesized from all gathered information (search results, page reads, and/or browser operations). No further tools will be used after this.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The complete result of the task, with sources/citations where relevant. Format in Markdown."},
            },
            "required": ["answer"],
        },
    },
}

# 统一工具池（24 = 8 + 15 + 1）
UNIFIED_TOOL_SCHEMAS: List[dict] = (
    KNOWLEDGE_TOOL_SCHEMAS + BROWSER_ACTION_SCHEMAS + [FINAL_ANSWER_SCHEMA]
)

# 向后兼容：v3.0 的 TOOL_SCHEMAS（7 知识检索 + final_answer，不含 news/blogs）
TOOL_SCHEMAS: List[dict] = [
    s for s in KNOWLEDGE_TOOL_SCHEMAS
    if s["function"]["name"] not in ("search_news", "search_blogs")
] + [FINAL_ANSWER_SCHEMA]


# ═══════════════════════════════════════════════════════════════
# 2. 知识检索工具 executor
# ═══════════════════════════════════════════════════════════════

def _format_search_results(results: List[SearchResult], label: str, query: str) -> str:
    if not results:
        return f"No {label} results found for '{query}'."
    lines = [f"{label} search results for '{query}' ({len(results)} found):\n"]
    for i, r in enumerate(results, 1):
        meta = r.metadata or {}
        extra = ""
        if meta.get("year") or meta.get("citations"):
            extra = f" ({meta.get('year', 'N/A')}, citations: {meta.get('citations', 'N/A')})"
        authors = ", ".join(meta.get("authors", [])[:3])
        lines.append(f"[{i}] **{r.title}**{extra}")
        if authors:
            lines.append(f"    Authors: {authors}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    {r.snippet[:300]}")
        lines.append("")
    return "\n".join(lines)


async def _search_adapter(source: str, args: dict) -> str:
    query = args.get("query", "")
    max_results = min(int(args.get("max_results", 5) or 5), 10)
    adapter = SearchRegistry.get(source)
    if not adapter:
        return f"Error: {source} search adapter not available."
    try:
        results = await adapter.search(query, max_results=max_results)
    except Exception as e:
        return f"{source} search failed: {str(e)[:200]}"
    return _format_search_results(results, source, query)


async def _execute_search_web(args: dict, config: AppConfig) -> str:
    return await _search_adapter("web", args)


async def _execute_search_papers(args: dict, config: AppConfig) -> str:
    return await _search_adapter("paper", args)


async def _execute_search_news(args: dict, config: AppConfig) -> str:
    return await _search_adapter("news", args)


async def _execute_search_blogs(args: dict, config: AppConfig) -> str:
    return await _search_adapter("blog", args)


async def _execute_search_wiki(args: dict, config: AppConfig) -> str:
    query = args.get("query", "")
    top_k = min(int(args.get("top_k", 3) or 3), 5)
    wiki = WikiManager()
    results = wiki.search_pages(query, top_k=top_k)
    if not results:
        return f"No Wiki pages found for '{query}'."
    lines = [f"Wiki search results for '{query}' ({len(results)} found):\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] **{r['title']}** (type: {r['type']}, score: {r['score']})")
        lines.append(f"    Path: {r['path']}")
        lines.append("")
    return "\n".join(lines)


async def _execute_read_page(args: dict, config: AppConfig) -> str:
    url = args.get("url", "")
    if not url:
        return "Error: No URL provided."
    fetcher = ContentFetcher(
        timeout=config.search.request_timeout,
        max_concurrent=1,
        user_agent=config.search.user_agent,
    )
    try:
        fetched_list = await fetcher.fetch_results(
            [SearchResult(source="web", title="", url=url, snippet="", score=1.0)]
        )
        if not fetched_list:
            return f"Failed to fetch content from {url}."
        fetched = fetched_list[0]
        if not fetched.success:
            return f"Failed to fetch {url}: {fetched.error or 'Unknown error'}. Snippet: {fetched.content[:300]}"
        content = fetched.content[:3000]
        suffix = f"\n... [{len(fetched.content) - 3000} chars truncated]" if len(fetched.content) > 3000 else ""
        return f"Content from {url}\nTitle: {fetched.title}\n---\n{content}{suffix}\n---"
    except Exception as e:
        return f"Error reading page {url}: {str(e)[:200]}"
    finally:
        await fetcher.close()


async def _execute_read_wiki_page(args: dict, config: AppConfig) -> str:
    path = args.get("path", "")
    if not path:
        return "Error: No wiki page path provided."
    wiki = WikiManager()
    page = wiki.load_page(path)
    if not page:
        return f"Wiki page not found: '{path}'. Find available pages via search_wiki."
    md = page.to_markdown()
    content = md[:3000]
    suffix = f"\n... [{len(md) - 3000} chars truncated]" if len(md) > 3000 else ""
    return f"Wiki page: {path}\nTitle: {page.title}\nType: {page.page_type}\n---\n{content}{suffix}\n---"


async def _execute_follow_link(args: dict, config: AppConfig) -> str:
    from_path = args.get("from_path", "")
    link_target = args.get("link_target", "")
    if not link_target:
        return "Error: No link target provided."
    wiki = WikiManager()
    target_page = wiki.load_page(link_target)
    if not target_page and not link_target.endswith(".md"):
        target_page = wiki.load_page(f"{link_target}.md")
        if target_page:
            link_target = f"{link_target}.md"
    if not target_page:
        return f"Link target not found: [[{link_target}]]. Try search_wiki to find related pages."
    md = target_page.to_markdown()
    return (
        f"Followed link to: {link_target}\nTitle: {target_page.title}\n"
        f"Type: {target_page.page_type}\n---\n{md[:5000]}\n---"
    )


_KNOWLEDGE_EXECUTORS = {
    "search_web": _execute_search_web,
    "search_papers": _execute_search_papers,
    "search_news": _execute_search_news,
    "search_blogs": _execute_search_blogs,
    "search_wiki": _execute_search_wiki,
    "read_page": _execute_read_page,
    "read_wiki_page": _execute_read_wiki_page,
    "follow_link": _execute_follow_link,
}


async def execute_knowledge_tool(tool_name: str, args: dict, config: AppConfig) -> str:
    """Execute a knowledge-retrieval tool and return its observation text."""
    executor = _KNOWLEDGE_EXECUTORS.get(tool_name)
    if not executor:
        return f"Error: Unknown knowledge tool '{tool_name}'."
    try:
        return await executor(args, config)
    except Exception as e:
        return f"Tool '{tool_name}' execution error: {str(e)[:300]}"


# ═══════════════════════════════════════════════════════════════
# 2.5 融合检索 + Map-Reduce 总结（v6.0 新增，需要 runtime）
# ═══════════════════════════════════════════════════════════════

async def _execute_fused_search(args: dict, runtime) -> str:
    """并行四源检索 + URL 去重 + 混合排序，并累积证据到 runtime."""
    query = args.get("query", "")
    sources = args.get("sources") or ["web", "paper", "news", "blog"]
    if isinstance(sources, str):
        sources = [s.strip() for s in sources.split(",")]
    max_per_source = min(int(args.get("max_per_source", 5) or 5), 10)
    top_k = min(int(args.get("top_k", 10) or 10), 20)

    adapters = [SearchRegistry.get(s) for s in sources]
    adapters = [a for a in adapters if a is not None]
    if not adapters:
        return "Error: no search adapters available."

    # ① 并行检索
    results_lists = await asyncio.gather(
        *(a.search(query, max_results=max_per_source) for a in adapters),
        return_exceptions=True,
    )

    # ② 扁平 + URL 去重
    seen_urls = set()
    all_results: List[SearchResult] = []
    for rl in results_lists:
        if isinstance(rl, Exception):
            continue
        for r in rl:
            if r.url and r.url in seen_urls:
                continue
            if r.url:
                seen_urls.add(r.url)
            all_results.append(r)

    # ③ 混合排序（BM25 + 语义 + 原始分）
    ranked = runtime.ranker.rank(query, all_results)[:top_k]

    # ④ 累积证据（供 synthesize 使用）
    runtime.evidence.extend(ranked)

    return _format_search_results(ranked, "fused", query)


async def _execute_synthesize(args: dict, runtime) -> str:
    """Map-Reduce 总结：分源总结 + 跨源合成 + 引用附录."""
    query = args.get("query", "") or runtime._last_task
    evidence = runtime.evidence
    if not evidence:
        return "No evidence gathered yet. Call fused_search/search_* first."

    # 按 source 分组
    by_source: Dict[str, List[SearchResult]] = {}
    for r in evidence:
        by_source.setdefault(r.source, []).append(r)

    # Map：每个 source 独立总结（用 snippet 作为内容）
    per_source_summaries: Dict[str, str] = {}
    references = []
    for source, results in by_source.items():
        contents = [
            FetchedContent(url=r.url, title=r.title, content=r.snippet, source=source, success=True, error="")
            for r in results
        ]
        summary = await runtime.summarizer.summarize_per_source(query, source, contents)
        per_source_summaries[source] = summary
        for r in results:
            references.append({"title": r.title, "url": r.url, "source": source})

    # Reduce：跨源合成（含引用附录）
    return await runtime.summarizer.synthesize(query, per_source_summaries, references)


# ═══════════════════════════════════════════════════════════════
# 3. 统一分发入口
# ═══════════════════════════════════════════════════════════════

async def execute_unified_tool(tool_name: str, args: dict, runtime) -> str:
    """Execute any unified tool. ``runtime`` is the AgentRuntime (duck-typed).

    - knowledge tools → HTTP search / fetch / wiki
    - browser tools   → runtime.ensure_browser() (lazy launch) + ActionExecutor
    - final_answer    → passthrough
    """
    if tool_name == "final_answer":
        return str(args.get("answer", ""))

    if tool_name.startswith("browser_"):
        executor = await runtime.ensure_browser()
        result = await executor.execute(_action_of(tool_name), args)
        return format_observation(result)

    if tool_name == "fused_search":
        return await _execute_fused_search(args, runtime)

    if tool_name == "synthesize":
        return await _execute_synthesize(args, runtime)

    return await execute_knowledge_tool(tool_name, args, runtime.config)


# ═══════════════════════════════════════════════════════════════
# 4. 向后兼容层（v3.0）
# ═══════════════════════════════════════════════════════════════

async def execute_tool(tool_name: str, tool_args: dict, config: AppConfig) -> str:
    """DEPRECATED (v3.0): delegate to the knowledge executor."""
    if tool_name == "final_answer":
        return str(tool_args.get("answer", ""))
    return await execute_knowledge_tool(tool_name, tool_args, config)


async def execute_tool_calls(tool_calls: List[dict], config: AppConfig) -> List[dict]:
    """DEPRECATED (v3.0): execute multiple knowledge tool calls in parallel."""
    async def execute_one(tc: dict) -> dict:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content = await execute_tool(name, args, config)
        return {"tool": name, "content": content, "tool_call_id": tc.get("id", ""), "arguments": args}

    if not tool_calls:
        return []
    results = await asyncio.gather(*[execute_one(tc) for tc in tool_calls], return_exceptions=True)
    observations = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            observations.append({
                "tool": tool_calls[i].get("function", {}).get("name", "unknown"),
                "content": f"Tool execution error: {str(res)[:300]}",
                "tool_call_id": tool_calls[i].get("id", ""),
                "arguments": {},
            })
        else:
            observations.append(res)
    return observations
