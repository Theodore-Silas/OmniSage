"""
上下文管理 —— 分层 Map-Reduce + 分级截断 + Top-K 限制 + Token 预算。

搜索问答主链路可能一次性检索到多源大量内容（N 篇全文），直接塞给 LLM 会
撑爆上下文窗口、推高成本、稀释关键信息。本模块集中管理上下文预算：

1. 分层 Map-Reduce（核心，避免一次性喂全文）
   - Map：分源独立总结（每源内容压缩为 ≤800 token 摘要）
   - Reduce：跨源合成（各源摘要 + query → 最终答案 ≤2000 token）
   效果：N 篇全文（可达数万字）→ 压缩为 4 源摘要（约 3200 token）→ 合成答案

2. 分级截断（按信息层级设字符上限）
   L0 搜索结果摘要 snippet ≤300 字
   L1 单篇正文 ≤4000 字 / Wiki 页面 ≤3000 字
   L2 每源总结输入 ≤8000 字 / 合成输入 ≤8000 字
   L3 命中答案 ≤3000 字 / 引用附录 ≤3000 字

3. Top-K 限制（控制数量）
   每源搜索 5 条 → 融合排序 Top-10 → 抓取正文 Top-5

4. Token 预算（控制 LLM 输出）
   分源总结 ≤800 token / 跨源合成 ≤2000 token
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    """统一的上下文预算配置。"""

    # 分级截断（字符数）
    snippet_max: int = 300          # L0 搜索结果摘要
    page_content_max: int = 4000    # L1 单篇正文
    wiki_page_max: int = 3000       # L1 Wiki 页面
    per_source_input: int = 8000    # L2 每源总结输入
    synthesis_input: int = 8000     # L2 合成输入
    hit_answer_max: int = 3000      # L3 命中答案
    refs_max: int = 3000            # L3 引用附录

    # Top-K 限制
    search_per_source: int = 5      # 每源搜索条数
    search_top_k: int = 10          # 融合排序后保留
    fetch_top_k: int = 5            # 抓取正文篇数

    # Token 预算（LLM 输出）
    summary_output_tokens: int = 800     # 分源总结输出
    synthesis_output_tokens: int = 2000  # 跨源合成输出


DEFAULT_BUDGET = ContextBudget()


def clip(text: str, max_chars: int, suffix: str = "...") -> str:
    """按字符截断，超长追加省略标记。"""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文约 1 字 ≈ 1 token；英文约 4 字符 ≈ 1 token。本系统以中文场景为主，
    采用「字符数 ≈ token 数」的保守上界估算，保证不超预算。
    """
    return len(text or "")
