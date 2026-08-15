"""
搜索问答状态 (SearchAgent 问答主链路)。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, List, TypedDict


class QAState(TypedDict, total=False):
    """搜索问答系统状态。"""

    query: str
    """用户输入的问题 / 关键词。"""

    verdict: str
    """命中判断结果：hit_high（高命中直答）| miss（未命中/内容太少）。"""

    hit_answer: str
    """命中本地知识库得到的答案。"""

    hit_kind: str
    """命中来源：faiss（向量记忆）| wiki（知识库页面）| none。"""

    hit_context: str
    """命中的本地知识内容（供回答 / 补充搜索作上下文）。"""

    related: List[dict]
    """命中后检索到的相关笔记（含 source/path/title），用于回答末尾附 wikilink。"""

    search_results: List[dict]
    """搜索 + 融合排序后的结果。"""

    fetched: List[dict]
    """抓取的正文内容。"""

    final_answer: str
    """最终答案。"""

    status: str
    status_message: str
    logs: Annotated[List[str], add]
