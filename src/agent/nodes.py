"""
LangGraph node implementations for the search agent pipeline.
"""

import asyncio
from typing import Dict, List

from src.agent.state import AgentState
from src.config import AppConfig
from src.search.base import SearchResult
from src.search.registry import SearchRegistry
from src.search.ranker import ResultRanker
from src.fetch.fetcher import ContentFetcher, FetchedContent
from src.summarize.summarizer import LLMSummarizer
from src.wiki.manager import WikiManager
from src.schema.validator import SchemaValidator
from src.lint.structural import StructuralLinter
from src.lint.error_book import ErrorBook
from src.lint.content import LintScheduler


# ── Node: Query Understanding ───────────────────────────────
QUERY_REWRITE_PROMPT = """You are a search query optimizer. For the user's question,
generate 2-3 alternative search queries (English + Chinese if applicable).

User question: {query}
Respond ONLY with JSON: {{"queries":["q1","q2","q3"],"intent":"brief"}}"""


async def query_understanding_node(state: AgentState, config: AppConfig) -> dict:
    query = state.get("query", "")
    state["logs"].append(f"[query] Processing: {query[:80]}...")

    rewritten = [query]
    try:
        s = LLMSummarizer(config.llm)
        r = await s.client.chat.completions.create(
            model=config.llm.model,
            messages=[{"role": "user", "content": QUERY_REWRITE_PROMPT.format(query=query)}],
            temperature=0.1, max_tokens=300,
        )
        import json
        content = r.choices[0].message.content or "{}"
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        data = json.loads(content.strip())
        rewritten.extend(data.get("queries", []))
        state["logs"].append(f"[query] +{len(data.get('queries',[]))} variants")
    except Exception as e:
        state["logs"].append(f"[query] Rewrite skipped: {e}")

    seen = set()
    unique = [q for q in rewritten if not (q in seen or seen.add(q))]
    state["logs"].append(f"[query] Queries: {unique}")
    return {"rewritten_queries": unique, "status": "searching", "status_message": "Searching..."}


# ── Node: Multi-Source Search ───────────────────────────────
async def multi_source_search_node(state: AgentState, config: AppConfig) -> dict:
    sources = state.get("search_sources", config.enable_sources)
    queries = state.get("rewritten_queries", [state["query"]])
    max_r = state.get("max_results_per_source", config.search.max_results_per_source)
    adapters = SearchRegistry.get_enabled(sources)

    if not adapters:
        state["logs"].append("[search] No adapters enabled!")
        return {"search_results": [], "status": "fetching", "status_message": "No sources enabled"}

    state["logs"].append(f"[search] {len(adapters)} sources, {len(queries)} queries")
    state["status_message"] = f"Searching {len(adapters)} sources..."

    async def search_source(adapter):
        src_results = []
        for q in queries:
            try:
                results = await adapter.search(q, max_results=max_r)
                src_results.extend(results)
            except Exception as e:
                state["logs"].append(f"[search] {adapter.source_name}/{q[:30]}: {e}")
        seen = set()
        unique = []
        for r in sorted(src_results, key=lambda x: x.score, reverse=True):
            norm = r.url.strip("/").lower()
            if norm and norm not in seen:
                seen.add(norm); unique.append(r)
        state["logs"].append(f"[search] {adapter.source_name}: {len(src_results)} raw -> {len(unique)} unique")
        return unique[:max_r]

    # Parallel search across all adapters
    source_result_lists = await asyncio.gather(
        *[search_source(a) for a in adapters], return_exceptions=True
    )

    all_raw = []
    for i, res in enumerate(source_result_lists):
        if isinstance(res, Exception):
            state["logs"].append(f"[search] {adapters[i].source_name} failed: {res}")
        else:
            all_raw.extend(res)

    if not all_raw:
        return {"search_results": [], "status": "fetching", "status_message": "No results found"}

    # Rank results
    try:
        ranker = ResultRanker()
        ranked = ranker.rank(state["query"], all_raw)
        state["logs"].append(f"[search] Ranked {len(ranked)} results")
    except Exception as e:
        state["logs"].append(f"[search] Ranking failed: {e}, using raw order")
        ranked = all_raw

    top = ranked[: config.search.max_fetch_results]
    return {
        "search_results": [
            {"source": r.source, "title": r.title, "url": r.url,
             "snippet": r.snippet, "score": r.score} for r in top
        ],
        "status": "fetching",
        "status_message": f"Found {len(ranked)} results, fetching content...",
    }


# ── Node: Content Fetch ─────────────────────────────────────
async def content_fetch_node(state: AgentState, config: AppConfig) -> dict:
    results_raw = state.get("search_results", [])
    if not results_raw:
        return {"fetched_contents": [], "status": "summarizing", "status_message": "No content to fetch"}

    results = [SearchResult(**r) for r in results_raw[: config.search.max_fetch_results]]
    state["logs"].append(f"[fetch] Fetching {len(results)} URLs")

    fetcher = ContentFetcher(timeout=config.search.request_timeout, max_concurrent=config.search.max_concurrent_fetch, user_agent=config.search.user_agent)
    try:
        fetched = await fetcher.fetch_results(results)
        ok = sum(1 for f in fetched if f.success)
        state["logs"].append(f"[fetch] {ok}/{len(fetched)} success")
        return {
            "fetched_contents": [
                {"url": f.url, "title": f.title, "content": f.content, "source": f.source, "success": f.success, "error": f.error}
                for f in fetched
            ],
            "status": "summarizing",
            "status_message": f"Fetched {ok}/{len(fetched)}, summarizing...",
        }
    finally:
        await fetcher.close()


# ── Node: Summarize ─────────────────────────────────────────
async def summarize_node(state: AgentState, config: AppConfig) -> dict:
    fetched_raw = state.get("fetched_contents", [])
    query = state["query"]
    if not fetched_raw:
        return {"per_source_summaries": {}, "status": "done", "status_message": "No content to summarize"}

    by_source: Dict[str, List[FetchedContent]] = {}
    for f in fetched_raw:
        fc = FetchedContent(**{k: v for k, v in f.items() if k != "error"})
        by_source.setdefault(f["source"], []).append(fc)

    summarizer = LLMSummarizer(config.llm, config.summarize)
    per_source = {}
    for src, contents in by_source.items():
        state["status_message"] = f"Summarizing {src}..."
        try:
            per_source[src] = await summarizer.summarize_per_source(query, src, contents)
            state["logs"].append(f"[summarize] {src}: done ({len(contents)} articles)")
        except Exception as e:
            per_source[src] = f"*{src} 总结失败: {str(e)[:200]}*"
            state["logs"].append(f"[summarize] {src}: FAILED")

    return {"per_source_summaries": per_source, "status": "synthesizing", "status_message": "Synthesizing final answer..."}


# ── Node: Context Injection (before synthesize, for multi-turn) ──
async def context_injection_node(state: AgentState, config: AppConfig) -> dict:
    history = state.get("conversation_history", [])
    if not history:
        return {}

    # Compress recent turns into a short context
    parts = []
    for msg in history[-4:]:  # last 2 turns (user+assistant * 2)
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")[:300]
        parts.append(f"{role}: {content}")

    ctx = "以上是之前的对话历史。请基于历史上下文回答当前问题。\n\n" + "\n".join(parts)
    state["logs"].append(f"[context] Injected {len(history)} history messages")
    return {"conversation_context": ctx}


# ── Node: Synthesize ────────────────────────────────────────
async def synthesize_node(state: AgentState, config: AppConfig) -> dict:
    query = state["query"]
    per_source = state.get("per_source_summaries", {})
    fetched_raw = state.get("fetched_contents", [])
    context = state.get("conversation_context", "")

    if not per_source:
        return {"final_answer": "No content to synthesize.", "status": "done", "status_message": "Done"}

    references = [{"title": f.get("title",""), "url": f.get("url",""), "source": f.get("source","")} for f in fetched_raw]
    summarizer = LLMSummarizer(config.llm, config.summarize)

    try:
        final = await summarizer.synthesize(query, per_source, references, context)
        state["logs"].append(f"[synthesize] Done ({len(final)} chars)")
    except Exception as e:
        state["logs"].append(f"[synthesize] FAILED: {e}")
        final = f"# Synthesis failed\n\n{str(e)[:200]}"

    return {"final_answer": final, "status": "done", "status_message": "Complete"}


# ── Node: Wiki Search (pre-search lookup + composite retrieval) ──
async def wiki_pre_search_node(state: AgentState, config: AppConfig) -> dict:
    """
    Phase D: Wiki-first composite retrieval.
    - High confidence (>=30): answer directly from Wiki, skip external search
    - Medium confidence (>=20): inject Wiki context, still do external search
    - Low confidence (<20): external search only
    """
    query = state["query"]
    wiki = WikiManager()

    wiki_hits = wiki.search_pages(query, top_k=5)
    top_score = wiki_hits[0]["score"] if wiki_hits else 0

    result = {}

    if wiki_hits and top_score >= 30:
        # High confidence — answer from Wiki directly
        best = wiki.load_page(wiki_hits[0]["path"])
        if best:
            answer = (
                f"## 来自知识库的答案\n\n"
                f"此问题在知识库中已有相关记录：\n\n"
                f"> {best.summary}\n\n"
                f"### 关键事实\n"
                + "\n".join(f"- {f}" for f in best.key_facts[:5])
                + "\n\n"
                f"### 详细分析\n{best.analysis[:1000]}\n\n"
                f"---\n"
                f"*来源：Wiki 页面 [[{best.path}]] | "
                f"置信度：{best.confidence:.0%} | "
                f"状态：{best.status}*"
            )
            state["logs"].append(f"[wiki] Direct answer from Wiki: {best.path} (score: {top_score})")
            return {
                "final_answer": answer,
                "status": "done",
                "status_message": "Answered from Wiki knowledge",
            }

    if wiki_hits and top_score >= 20:
        # Medium confidence — inject Wiki context + search externally
        context_parts = []
        for hit in wiki_hits[:3]:
            page = wiki.load_page(hit["path"])
            if page:
                context_parts.append(
                    f"### {page.title} (type: {page.page_type}, confidence: {page.confidence:.0%})\n"
                    f"> {page.summary[:200]}\n"
                    f"Key facts: {'; '.join(page.key_facts[:3])}"
                )
        ctx = "\n\n---\n\n".join(context_parts)
        state["logs"].append(f"[wiki] Injected {len(wiki_hits)} Wiki pages as context (top: {top_score})")
        result["conversation_context"] = (
            f"已有Wiki知识库中的相关内容：\n{ctx}\n\n"
            f"请基于以上Wiki知识和本次搜索结果的综合来回答。"
            f"如果新搜索发现与Wiki矛盾的信息，请明确指出。"
        )

    else:
        state["logs"].append(f"[wiki] No strong Wiki hits (top score: {top_score})")

    result["status"] = "searching"
    return result


# ── Node: Wiki Persist (archive results) ───────────────────
async def wiki_persist_node(state: AgentState, config: AppConfig) -> dict:
    """Archive search results into the Wiki knowledge base."""
    query = state["query"]
    answer = state.get("final_answer", "")
    search_results = state.get("search_results", [])
    fetched_contents = state.get("fetched_contents", [])

    if not answer or not search_results:
        state["logs"].append("[wiki-persist] Nothing to archive")
        return {"status": "done", "status_message": "Complete"}

    state["status_message"] = "Archiving to Wiki..."
    try:
        wiki = WikiManager()
        archived = wiki.archive_search(query, answer, search_results, fetched_contents)

        # Validate archived pages against schema
        validator = SchemaValidator()
        issues_found = 0
        for page_type, page_path in archived.items():
            page = wiki.load_page(page_path)
            if page:
                result = validator.validate_page(page)
                if not result.passed or result.warnings:
                    page = validator.auto_fix_page(page)
                    wiki.save_page(page)
                    issues_found += len(result.issues) + len(result.warnings)
                    state["logs"].append(
                        f"[wiki-lint] {page_path}: {len(result.issues)} issues, "
                        f"{len(result.warnings)} warnings → auto-fixed"
                    )

        # Periodic structural lint (every N searches)
        scheduler = LintScheduler()
        ran_lint = False
        if scheduler.should_lint(increment=True):
            state["logs"].append(f"[lint] Running periodic structural lint (search #{scheduler.get_count()})...")
            linter = StructuralLinter(wiki)
            structural_issues = linter.lint_all()
            if structural_issues:
                eb = ErrorBook()
                added = eb.discover(structural_issues)
                auto = eb.auto_attribute()
                state["logs"].append(
                    f"[lint] Found {len(structural_issues)} issues, "
                    f"{added} new in Error Book, {auto} auto-attributed"
                )
            ran_lint = True

        state["logs"].append(
            f"[wiki-persist] Archived: query + {len(search_results)} sources. "
            f"Wiki: {wiki.get_page_count()} pages. "
            f"Schema fixes: {issues_found}. {'Lint: ran.' if ran_lint else ''}"
        )
    except Exception as e:
        state["logs"].append(f"[wiki-persist] Archive failed: {e}")

    return {"status": "done", "status_message": "Complete"}
