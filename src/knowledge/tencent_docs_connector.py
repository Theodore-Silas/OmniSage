"""
腾讯文档连接器（通过 MCP 接入）。

腾讯文档的检索/读取依赖宿主注入的 MCP 调用函数（WorkBuddy 的
`mcp__tencent-docs__*` 工具），本连接器接收可注入的 search/get_content 回调：

  search_fn(query, top_k) -> List[KnowledgeHit]    （同步或 async 均可）
  get_content_fn(hit) -> str

未注入回调时 connector 自动禁用（enabled()==False），命中层跳过该源。
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Callable, List, Optional

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit


class TencentDocsConnector(KnowledgeBaseConnector):
    """接入腾讯文档（依赖宿主注入的 MCP 调用函数）。"""

    name = "tencent_docs"

    def __init__(
        self,
        search_fn: Optional[Callable] = None,
        get_content_fn: Optional[Callable] = None,
    ):
        self._search_fn = search_fn
        self._get_content_fn = get_content_fn

    def enabled(self) -> bool:
        return self._search_fn is not None

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeHit]:
        if self._search_fn is None:
            return []
        result = self._search_fn(query, top_k)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return list(result or [])

    def get_content(self, hit: KnowledgeHit) -> str:
        if self._get_content_fn is None:
            return hit.snippet
        result = self._get_content_fn(hit)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return str(result or hit.snippet)
