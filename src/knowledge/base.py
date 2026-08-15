"""
知识库连接器抽象层 (SearchAgent)。

定义统一的 KnowledgeBaseConnector 接口，让命中层可接入任意本地/云端知识库
（项目 Wiki / Obsidian vault / 腾讯文档 / FAISS 向量记忆等），统一返回
KnowledgeHit。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class KnowledgeHit:
    """一次知识库命中的结果。"""
    source: str                      # 连接器名：wiki / obsidian / tencent_docs / vector
    title: str
    path: str                        # 位置标识（相对路径 / 文档 ID / URL）
    snippet: str                     # 摘要（标题 + 正文片段）
    score: float = 0.0               # 相关度评分（跨源需可比，尽量归一化）
    meta: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBaseConnector(ABC):
    """知识库连接器统一接口。"""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[KnowledgeHit]:
        """全文/语义检索，返回按相关度降序的命中列表。"""

    @abstractmethod
    def get_content(self, hit: KnowledgeHit) -> str:
        """读取命中的完整正文内容（用于命中后回答）。"""

    def enabled(self) -> bool:
        """是否可用（未配置的连接器返回 False，命中层跳过）。"""
        return True


def strip_frontmatter(content: str) -> str:
    """剥离 Obsidian/Markdown 的 YAML frontmatter（--- ... ---）。"""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4:].lstrip("\n")
    return content
