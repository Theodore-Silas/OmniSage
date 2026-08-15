"""
Content validator: cross-page contradiction detection using LLM.
"""

import os
from typing import Dict, List, Tuple

from src.wiki.manager import WikiManager, WikiPage
from src.config import LLMConfig

CONTRADICTION_PROMPT = """You are a fact-checker. Compare these two Wiki page summaries for contradictions.

Page A ({title_a}):
{content_a}

Page B ({title_b}):
{content_b}

Do these pages contain contradictory claims about the same entity/fact?
Answer ONLY in JSON: {{"contradiction": true/false, "description": "brief description if true"}}"""


class ContentValidator:
    """Detects contradictions between Wiki pages using LLM."""

    def __init__(self, llm_config: LLMConfig = None):
        self.llm_config = llm_config or LLMConfig()
        self.wiki = WikiManager()

    async def check_contradictions(self, page_paths: List[str]) -> List[dict]:
        """
        Check for contradictions among a set of Wiki pages.
        Returns list of contradiction findings.
        """
        if len(page_paths) < 2:
            return []

        pages = []
        for p in page_paths:
            page = self.wiki.load_page(p)
            if page:
                pages.append(page)

        findings = []
        # Compare pairs
        for i in range(len(pages)):
            for j in range(i + 1, len(pages)):
                # Only check same-type pages for contradictions
                if pages[i].page_type != pages[j].page_type:
                    continue
                finding = await self._check_pair(pages[i], pages[j])
                if finding:
                    findings.append(finding)

        return findings

    async def _check_pair(self, page_a: WikiPage, page_b: WikiPage) -> dict:
        """Check if two pages contradict each other."""
        content_a = f"Summary: {page_a.summary}\nFacts: {'; '.join(page_a.key_facts[:5])}"
        content_b = f"Summary: {page_b.summary}\nFacts: {'; '.join(page_b.key_facts[:5])}"

        if not self.llm_config.api_key or "test" in self.llm_config.api_key:
            return {}  # skip with no valid API key

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.llm_config.api_key, base_url=self.llm_config.base_url)

            prompt = CONTRADICTION_PROMPT.format(
                title_a=page_a.title, content_a=content_a[:1000],
                title_b=page_b.title, content_b=content_b[:1000],
            )
            response = await client.chat.completions.create(
                model=self.llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=200,
            )

            import json
            content = response.choices[0].message.content or "{}"
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"): content = content[4:]
            data = json.loads(content.strip())

            if data.get("contradiction"):
                return {
                    "page_a": page_a.path,
                    "page_b": page_b.path,
                    "description": data.get("description", "Unspecified contradiction"),
                }
        except Exception:
            pass

        return {}


class LintScheduler:
    """Manages periodic lint runs (every N searches)."""

    def __init__(self, wiki_root: str = None, interval: int = 10):
        if wiki_root is None:
            wiki_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "wiki",
            )
        self.interval = interval
        self.counter_path = os.path.join(wiki_root, ".lint_counter")
        self._ensure_counter()

    def _ensure_counter(self):
        import os
        if not os.path.exists(self.counter_path):
            with open(self.counter_path, "w") as f:
                f.write("0")

    def should_lint(self, increment: bool = True) -> bool:
        """Check if it's time for a lint run. Optionally increment counter."""
        try:
            with open(self.counter_path, "r") as f:
                count = int(f.read().strip())
        except Exception:
            count = 0

        should = count > 0 and count % self.interval == 0

        if increment:
            count += 1
            with open(self.counter_path, "w") as f:
                f.write(str(count))

        return should

    def get_count(self) -> int:
        try:
            with open(self.counter_path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
