"""
知识库连接器注册中心。

统一管理多个 KnowledgeBaseConnector，供命中层做单源/多源查询。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit


class KnowledgeRegistry:
    """管理并查询所有已注册的知识库连接器。"""

    def __init__(self):
        self._connectors: Dict[str, KnowledgeBaseConnector] = {}

    def register(self, connector: KnowledgeBaseConnector) -> None:
        self._connectors[connector.name] = connector

    def register_many(self, connectors: List[KnowledgeBaseConnector]) -> None:
        for c in connectors:
            self.register(c)

    def names(self) -> List[str]:
        return [c.name for c in self._connectors.values() if c.enabled()]

    def get(self, name: str) -> Optional[KnowledgeBaseConnector]:
        """按名字取连接器（可能为 None）。"""
        return self._connectors.get(name)

    def search_source(self, name: str, query: str, top_k: int = 5) -> List[KnowledgeHit]:
        """查询单个源。"""
        conn = self._connectors.get(name)
        if conn is None or not conn.enabled():
            return []
        try:
            return conn.search(query, top_k=top_k)
        except Exception:
            return []

    def search_knowledge(self, query: str, top_k: int = 5, exclude: Optional[set] = None) -> List[KnowledgeHit]:
        """聚合查询所有知识库源（默认排除向量记忆，向量单独处理）。"""
        exclude = exclude or {"vector"}
        hits: List[KnowledgeHit] = []
        for name, conn in self._connectors.items():
            if name in exclude or not conn.enabled():
                continue
            try:
                hits.extend(conn.search(query, top_k=top_k))
            except Exception:
                continue
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def get_content(self, hit: KnowledgeHit) -> str:
        """读取命中的完整内容。"""
        conn = self._connectors.get(hit.source)
        if conn is None:
            return hit.snippet
        try:
            return conn.get_content(hit)
        except Exception:
            return hit.snippet
