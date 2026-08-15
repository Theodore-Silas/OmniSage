"""
项目 Wiki 连接器（封装现有 WikiManager）。
"""

from __future__ import annotations

from typing import List, Optional

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit
from src.wiki.manager import WikiManager


class WikiConnector(KnowledgeBaseConnector):
    """接入项目内置 Wiki 知识库。"""

    name = "wiki"

    def __init__(self, wiki: Optional[WikiManager] = None):
        self.wiki = wiki or WikiManager()

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeHit]:
        results = self.wiki.search_pages(query, top_k=top_k)
        return [
            KnowledgeHit(
                source=self.name,
                title=r.get("title", ""),
                path=r.get("path", ""),
                snippet="",
                score=float(r.get("score", 0)),
                meta={"type": r.get("type", "")},
            )
            for r in results
        ]

    def get_content(self, hit: KnowledgeHit) -> str:
        page = self.wiki.load_page(hit.path)
        if page is None:
            return hit.snippet
        parts = []
        if page.summary:
            parts.append(f"> {page.summary}")
        if page.key_facts:
            parts.append("## 要点\n" + "\n".join(f"- {f}" for f in page.key_facts))
        if page.analysis:
            parts.append(page.analysis)
        return "\n\n".join(parts).strip() or hit.snippet
