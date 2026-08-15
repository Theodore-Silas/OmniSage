"""
搜索问答 (SearchAgent) — 本地命中优先 + 外部搜索兜底。
"""

from src.qa.graph import QARuntime, run_qa, run_qa_stream

__all__ = ["QARuntime", "run_qa", "run_qa_stream"]
