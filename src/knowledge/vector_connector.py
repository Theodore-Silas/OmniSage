"""
FAISS 向量记忆连接器（封装现有 VectorMemory，历史相似问答召回）。
"""

from __future__ import annotations

from typing import List, Optional

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit
from src.storage.vector_store import VectorMemory


class VectorConnector(KnowledgeBaseConnector):
    """接入 FAISS 语义记忆（历史问答）。"""

    name = "vector"

    def __init__(self, memory: Optional[VectorMemory] = None):
        self.memory = memory or VectorMemory()

    def search(self, query: str, top_k: int = 3) -> List[KnowledgeHit]:
        records = self.memory.search_similar(query, threshold=0.6, top_k=top_k)
        return [
            KnowledgeHit(
                source=self.name,
                title=r.get("query", ""),
                path="",
                snippet=(r.get("answer", "") or "")[:400],
                score=float(r.get("similarity", 0)),
                meta={"sources_count": r.get("sources_count", 0)},
            )
            for r in records
        ]

    def get_content(self, hit: KnowledgeHit) -> str:
        # 向量命中的 answer 已完整存于 snippet
        return hit.snippet
