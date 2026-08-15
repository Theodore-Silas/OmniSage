"""
知识库连接器层 (SearchAgent) — 多知识库接入（Wiki / Obsidian / 腾讯文档 / FAISS）。
"""

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit, strip_frontmatter
from src.knowledge.registry import KnowledgeRegistry
from src.knowledge.wiki_connector import WikiConnector
from src.knowledge.vector_connector import VectorConnector
from src.knowledge.obsidian_connector import ObsidianConnector
from src.knowledge.tencent_docs_connector import TencentDocsConnector

__all__ = [
    "KnowledgeBaseConnector",
    "KnowledgeHit",
    "KnowledgeRegistry",
    "WikiConnector",
    "VectorConnector",
    "ObsidianConnector",
    "TencentDocsConnector",
    "strip_frontmatter",
]
