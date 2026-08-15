"""
LLM-based summarization layer with streaming support.
"""

from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from src.config import LLMConfig, SummarizeConfig
from src.fetch.fetcher import FetchedContent
from src.schema.loader import get_system_prompt_extension

SUMMARIZE_PER_SOURCE_PROMPT = """你是一个专业的研究助理。请阅读以下来自同一数据源的多篇文章内容，用中文总结核心观点和关键发现。

数据源类型：{source_type}
查询主题：{query}
文章内容：
{contents}

请按以下格式输出总结：
## 核心发现
（2-4句话概括核心主题）
## 要点列表
- **要点1标题**：具体说明（标注来源序号如 [1]）
- **要点2标题**：具体说明 [2]
...
## 关键引用
列出1-3条最重要的原文引用或数据，标注来源。

注意：如果没有抓到有效内容，如实说明；保持客观；观点矛盾时明确指出。"""

SYNTHESIZE_PROMPT = """你是一个资深的研究分析师。请综合以下总结，生成完整报告。

用户查询：{query}
{context}

各数据源总结：
{summaries}

可引用的来源：
{refs}

输出格式：
## 核心结论
（3-5句话提炼核心发现）

## 分源分析
### 网络资讯
### 学术论文
### 新闻
### 博客

## 观点差异与不确定性

## 参考文献
（格式：序号. [标题](URL)）"""


class LLMSummarizer:
    """Map-Reduce summarizer with streaming synthesis support."""

    def __init__(self, llm_config: LLMConfig, summarize_config: SummarizeConfig = None):
        self.llm_config = llm_config
        self.summarize_config = summarize_config or SummarizeConfig()
        self.client = AsyncOpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

    async def summarize_per_source(
        self, query: str, source_name: str, contents: List[FetchedContent]
    ) -> str:
        if not contents:
            return f"*{source_name} 源未获取到有效内容。*"

        blocks = []
        for i, fc in enumerate(contents):
            status = "" if fc.success else " [抓取失败，使用摘要]"
            blocks.append(f"### 文章{i+1}: {fc.title}{status}\nURL: {fc.url}\n\n{fc.content[:4000]}\n")

        source_type_map = {"web": "网页搜索", "paper": "学术论文", "news": "新闻报道", "blog": "技术博客"}
        prompt = SUMMARIZE_PER_SOURCE_PROMPT.format(
            source_type=source_type_map.get(source_name, source_name),
            query=query,
            contents="\n---\n".join(blocks)[:8000],
        )
        try:
            r = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_config.temperature,
                max_tokens=self.summarize_config.summary_max_tokens,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            return f"*{source_name} 源总结失败: {str(e)[:200]}*"

    # ── Batch synthesis ──────────────────────────────────────

    async def synthesize(
        self, query: str, per_source_summaries: Dict[str, str],
        references: Optional[List[dict]] = None,
        conversation_context: str = "",
    ) -> str:
        """Full synthesis, returns complete answer string."""
        prompt = self._build_synth_prompt(query, per_source_summaries, references, conversation_context)
        url_index = self._build_refs(references)[1]
        try:
            r = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=self._build_messages(prompt),
                temperature=self.llm_config.temperature,
                max_tokens=self.summarize_config.synthesize_max_tokens,
            )
            final = r.choices[0].message.content or ""
            return final + self._make_ref_appendix(url_index, final)
        except Exception as e:
            return f"# 合成失败\n\n错误: {str(e)[:200]}"

    # ── Streaming synthesis ──────────────────────────────────

    async def synthesize_stream(
        self, query: str, per_source_summaries: Dict[str, str],
        references: Optional[List[dict]] = None,
        conversation_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """Streaming synthesis, yields token chunks then reference appendix."""
        prompt = self._build_synth_prompt(query, per_source_summaries, references, conversation_context)
        url_index = self._build_refs(references)[1]
        full = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.llm_config.model,
                messages=self._build_messages(prompt),
                temperature=self.llm_config.temperature,
                max_tokens=self.summarize_config.synthesize_max_tokens,
                stream=True,
            )
            async for chunk_data in response:
                delta = chunk_data.choices[0].delta
                token = getattr(delta, "content", "") or ""
                if token:
                    full += token
                    yield token
        except Exception as e:
            full = f"# 合成失败\n\n错误: {str(e)[:200]}"
            yield full

        appendix = self._make_ref_appendix(url_index, full)
        if appendix:
            yield appendix

    def _build_messages(self, prompt: str) -> list:
        """Build message list with schema rules as system message."""
        schema_rules = get_system_prompt_extension()
        if schema_rules:
            return [
                {"role": "system", "content": schema_rules},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    # ── Prompt builder ───────────────────────────────────────

    def _build_synth_prompt(
        self, query: str, per_source_summaries: Dict[str, str],
        references: Optional[List[dict]], conversation_context: str,
    ) -> str:
        source_labels = {"web": "网络资讯", "paper": "学术论文", "news": "新闻", "blog": "博客"}
        blocks = []
        for source, summary in per_source_summaries.items():
            blocks.append(f"### {source_labels.get(source, source)} 总结\n\n{summary}")
        refs_block = self._build_refs(references)[0]
        ctx = f"\n对话历史：\n{conversation_context}" if conversation_context else ""
        return SYNTHESIZE_PROMPT.format(
            query=query, context=ctx,
            summaries="\n\n---\n\n".join(blocks)[:8000],
            refs=refs_block[:3000],
        )

    # ── Reference helpers ────────────────────────────────────

    @staticmethod
    def _build_refs(references):
        url_index = []
        if not references:
            return "（无可用来源）", url_index
        lines = []
        for i, ref in enumerate(references, 1):
            title = ref.get("title", "Untitled")
            url = ref.get("url", "")
            src = ref.get("source", "")
            lines.append(f"[{i}] ({src}) {title[:80]}\n    {url}")
            url_index.append({"index": i, "title": title, "url": url, "source": src})
        return "\n".join(lines), url_index

    @staticmethod
    def _make_ref_appendix(url_index, existing_text):
        if not url_index:
            return ""
        if "参考来源" in existing_text or "## 参考文献" in existing_text:
            return ""
        appendix = "\n\n---\n\n## 参考来源\n\n"
        for ref in url_index:
            appendix += f"{ref['index']}. **{ref['title'][:100]}**  `{ref['source']}`  \n   {ref['url']}\n\n"
        return appendix
