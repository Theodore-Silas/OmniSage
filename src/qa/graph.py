"""
搜索问答主链路 (SearchAgent) — 本地命中优先 + 外部搜索兜底。

状态机：
  START → retrieve（命中：FAISS + Wiki 双通道 + 充分性判断）
    ├─ hit_high → answer（直接回答，零搜索）
    └─ miss     → search（四源并行 + 融合排序 + 抓取）→ summarize（Map-Reduce）
                  → persist（Wiki 归档 + FAISS 记忆）→ END
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional

from langgraph.graph import StateGraph, END

from src.config import AppConfig
from src.fetch.fetcher import ContentFetcher, FetchedContent
from src.knowledge import (
    KnowledgeRegistry,
    ObsidianConnector,
    TencentDocsConnector,
    VectorConnector,
    WikiConnector,
)
from src.qa.context import DEFAULT_BUDGET, clip
from src.qa.state import QAState
from src.search.base import SearchResult
from src.search.ranker import ResultRanker
from src.search.registry import SearchRegistry
from src.storage.vector_store import VectorMemory
from src.summarize.summarizer import LLMSummarizer
from src.wiki.manager import WikiManager

# 充分性阈值
FAISS_HIT_THRESHOLD = 0.80     # 向量相似度阈值
FAISS_MIN_ANSWER = 200         # FAISS 命中答案最短字符
KB_MIN_CONTENT = 400           # 知识库命中内容最短字符（充分性判断）


class QARuntime:
    """搜索问答运行时：装配命中/搜索/总结/沉淀所需资源。"""

    def __init__(self, config: AppConfig, knowledge=None):
        self.config = config
        self.memory = VectorMemory()
        self.wiki = WikiManager()
        self.ranker = ResultRanker()
        self.summarizer = LLMSummarizer(config.llm, config.summarize)

        # 知识库连接器（多源命中：wiki / obsidian / 腾讯文档 / 向量记忆）
        self.knowledge = knowledge or KnowledgeRegistry()
        self.knowledge.register_many([
            WikiConnector(self.wiki),
            VectorConnector(self.memory),
            ObsidianConnector(),          # 读 OBSIDIAN_VAULT_PATH，未配置自动禁用
            TencentDocsConnector(),       # 未注入 MCP 回调时自动禁用
        ])


def _initial_state(query: str) -> QAState:
    return QAState(
        query=query, verdict="", hit_answer="", hit_kind="none", hit_context="",
        search_results=[], fetched=[], final_answer="", status="starting",
        status_message="检索本地知识库...", logs=[],
    )


def _to_search_dicts(results: List[SearchResult]) -> List[dict]:
    return [
        {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
        for r in results
    ]


def _to_fetched_dicts(fetched: List[FetchedContent]) -> List[dict]:
    return [
        {"url": f.url, "title": f.title, "content": f.content, "source": f.source, "success": f.success}
        for f in fetched
    ]


def build_qa_graph(runtime: QARuntime):
    graph = StateGraph(QAState)

    # ── 命中 ──────────────────────────────────────────────────

    async def retrieve_node(state: QAState) -> dict:
        query = state["query"]

        # ① 向量记忆命中（历史相似问答）
        for hit in runtime.knowledge.search_source("vector", query, top_k=3):
            answer = hit.snippet
            if hit.score >= FAISS_HIT_THRESHOLD and len(answer) >= FAISS_MIN_ANSWER:
                return {
                    "verdict": "hit_high", "hit_kind": hit.source, "hit_answer": answer,
                    "status": "cache_hit", "status_message": "命中本地知识（历史问答）",
                    "logs": [f"[retrieve] 向量记忆命中 (sim={hit.score:.2f}, {len(answer)} 字)"],
                }

        # ② 知识库命中（Wiki / Obsidian / 腾讯文档）
        for hit in runtime.knowledge.search_knowledge(query, top_k=5):
            content = runtime.knowledge.get_content(hit)
            if len(content) >= KB_MIN_CONTENT:
                # 检索相关笔记（排除命中的），用于回答末尾附 wikilink 供跳转
                related = []
                try:
                    for rh in runtime.knowledge.search_knowledge(query, top_k=6):
                        if rh.path != hit.path and rh.score >= 2.0:
                            related.append({"source": rh.source, "path": rh.path, "title": rh.title})
                except Exception:
                    related = []
                return {
                    "verdict": "hit_high", "hit_kind": hit.source, "hit_answer": content,
                    "hit_context": f"来源：{hit.source} · {hit.path}",
                    "related": related[:5],
                    "status": "cache_hit", "status_message": f"命中知识库（{hit.source}）",
                    "logs": [f"[retrieve] {hit.source} 命中 ({hit.path}, score={hit.score:.1f})"],
                }

        # 未命中 / 相关内容太少
        return {
            "verdict": "miss", "hit_kind": "none", "hit_answer": "",
            "status": "miss", "status_message": "本地未命中 → 启动搜索",
            "logs": ["[retrieve] 未命中或内容太少 → 启动外部搜索"],
        }

    # ── 高命中直答 ────────────────────────────────────────────

    async def answer_node(state: QAState) -> dict:
        answer = state.get("hit_answer", "")
        kind = state.get("hit_kind", "none")
        context = state.get("hit_context", "")
        if len(answer) > DEFAULT_BUDGET.hit_answer_max:
            answer = clip(answer, DEFAULT_BUDGET.hit_answer_max) + "\n\n...(内容过长，已截断)"
        if kind == "wiki" and context:
            answer = f"{answer}\n\n---\n{context}"
        # 附上相关笔记 wikilink，供用户跳转（知识网络）
        # 若命中笔记本身已含「相关笔记」章节，则跳过，避免重复
        related = state.get("related", [])
        if related and "## 相关笔记" not in answer:
            links = []
            for r in related:
                p = r.get("path", "")
                link = p[:-3] if p.endswith(".md") else p
                if link:
                    links.append(f"- [[{link}]]")
            if links:
                answer = answer.rstrip() + "\n\n---\n\n## 相关笔记\n\n" + "\n".join(links)
        return {
            "final_answer": answer,
            "status": "done",
            "status_message": "已从本地知识库回答",
            "logs": ["[answer] 本地命中，直接回答（零搜索成本）"],
        }

    # ── 搜索兜底 ──────────────────────────────────────────────

    async def search_node(state: QAState) -> dict:
        query = state["query"]
        adapters = list(SearchRegistry.get_all().values())

        # ① 四源并行检索
        results_lists = await asyncio.gather(
            *(a.search(query, max_results=DEFAULT_BUDGET.search_per_source) for a in adapters),
            return_exceptions=True,
        )

        # ② 扁平 + URL 去重
        seen = set()
        all_results: List[SearchResult] = []
        for rl in results_lists:
            if isinstance(rl, Exception):
                continue
            for r in rl:
                if r.url and r.url in seen:
                    continue
                if r.url:
                    seen.add(r.url)
                all_results.append(r)

        # ③ 混合排序
        ranked = runtime.ranker.rank(query, all_results)[:DEFAULT_BUDGET.search_top_k]

        # ④ 抓取 Top-K 正文
        fetcher = ContentFetcher(
            timeout=runtime.config.search.request_timeout,
            max_concurrent=runtime.config.search.max_concurrent_fetch,
            user_agent=runtime.config.search.user_agent,
        )
        try:
            fetched_list = await fetcher.fetch_results(ranked[:DEFAULT_BUDGET.fetch_top_k])
        except Exception:
            fetched_list = []
        finally:
            await fetcher.close()

        return {
            "search_results": _to_search_dicts(ranked),
            "fetched": _to_fetched_dicts(fetched_list),
            "status": "searched",
            "status_message": f"搜索完成（{len(ranked)} 条去重排序结果）",
            "logs": [f"[search] 四源并行 → 去重排序 {len(ranked)} 条 → 抓取 {len(fetched_list)} 篇正文"],
        }

    # ── Map-Reduce 总结 ───────────────────────────────────────

    async def summarize_node(state: QAState) -> dict:
        query = state["query"]
        results = state.get("search_results", [])
        fetched = state.get("fetched", [])

        # 按 source 分组正文（成功抓取的优先，snippet 兜底）
        by_source = {}
        for f in fetched:
            if f.get("success") and f.get("content"):
                by_source.setdefault(f.get("source", "web"), []).append(
                    FetchedContent(f["url"], f["title"], clip(f["content"], DEFAULT_BUDGET.page_content_max), f.get("source", "web"), True, "")
                )

        per_source = {}
        references = []
        for source, contents in by_source.items():
            per_source[source] = await runtime.summarizer.summarize_per_source(query, source, contents)

        # 兜底：若有源无正文，用 snippet 构造
        if not per_source:
            by_src = {}
            for r in results:
                by_src.setdefault(r["source"], []).append(
                    FetchedContent(r["url"], r["title"], r["snippet"][:2000], r["source"], True, "")
                )
            for source, contents in by_src.items():
                per_source[source] = await runtime.summarizer.summarize_per_source(query, source, contents)

        for r in results:
            references.append({"title": r["title"], "url": r["url"], "source": r["source"]})

        answer = await runtime.summarizer.synthesize(query, per_source, references)
        return {
            "final_answer": answer,
            "status": "summarized",
            "status_message": "Map-Reduce 总结完成",
            "logs": ["[summarize] Map-Reduce：分源总结 → 跨源合成（含引用）"],
        }

    # ── 知识沉淀 ──────────────────────────────────────────────

    async def persist_node(state: QAState) -> dict:
        query = state["query"]
        answer = state.get("final_answer", "")
        results = state.get("search_results", [])
        fetched = state.get("fetched", [])

        logs = []
        try:
            runtime.wiki.archive_search(query, answer, results, fetched)
            logs.append("[persist] Wiki 归档（sources + query 页面）")
        except Exception as e:
            logs.append(f"[persist] Wiki 归档失败: {str(e)[:120]}")
        try:
            runtime.memory.store(query, answer, len(results))
            logs.append("[persist] FAISS 记忆已写入")
        except Exception as e:
            logs.append(f"[persist] FAISS 写入失败: {str(e)[:120]}")

        # Obsidian 沉淀：搜索结果也写入本地 vault（若已配置）
        obsidian = runtime.knowledge.get("obsidian")
        if obsidian is not None and obsidian.enabled():
            try:
                path = obsidian.save_note(query, answer, results)
                if path:
                    logs.append(f"[persist] Obsidian 笔记已写入 ({path})")
            except Exception as e:
                logs.append(f"[persist] Obsidian 写入失败: {str(e)[:120]}")

        return {"status": "done", "status_message": "已沉淀到知识库", "logs": logs}

    # ── 路由 ──────────────────────────────────────────────────

    def route(state: QAState) -> str:
        return "answer" if state.get("verdict") == "hit_high" else "search"

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("persist", persist_node)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route, {"answer": "answer", "search": "search"})
    graph.add_edge("answer", END)
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


async def run_qa(query: str, config: AppConfig = None, runtime: QARuntime = None) -> dict:
    """运行搜索问答主链路，返回最终状态。"""
    if config is None:
        config = AppConfig.from_env()
    if runtime is None:
        runtime = QARuntime(config)

    graph = build_qa_graph(runtime)
    result = await graph.ainvoke(_initial_state(query))
    final = dict(result)
    if not final.get("final_answer"):
        final["final_answer"] = "(未生成答案，请检查配置与网络)"
    return final


async def run_qa_stream(query: str, config: AppConfig = None, runtime: QARuntime = None) -> AsyncGenerator[dict, None]:
    """流式运行搜索问答主链路。"""
    if config is None:
        config = AppConfig.from_env()
    if runtime is None:
        runtime = QARuntime(config)

    graph = build_qa_graph(runtime)
    final_state = dict(_initial_state(query))

    async for event in graph.astream(_initial_state(query), stream_mode="updates"):
        for node_name, update in event.items():
            if not update:
                continue
            for k, v in update.items():
                if isinstance(v, list) and isinstance(final_state.get(k), list):
                    final_state[k] = final_state[k] + v
                else:
                    final_state[k] = v
            if update.get("status_message"):
                yield {"type": "status", "content": update["status_message"], "meta": {"node": node_name}}
            for log in update.get("logs", []):
                yield {"type": "log", "content": log, "meta": {"node": node_name}}

    answer = final_state.get("final_answer", "")
    yield {
        "type": "done",
        "content": answer,
        "meta": {"status": final_state.get("status"), "verdict": final_state.get("verdict")},
    }
