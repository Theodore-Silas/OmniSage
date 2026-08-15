"""
RSS / Blog search adapter. Pre-configured tech blog feeds.
No API key needed, uses feedparser.
"""

import asyncio
from typing import List

from .base import BaseSearchAdapter, SearchResult

# Curated list of tech blog RSS feeds
DEFAULT_FEEDS = [
    "https://blog.langchain.dev/rss/",
    "https://openai.com/blog/rss.xml",
    "https://www.anyscale.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://pytorch.org/blog/feed.xml",
    "https://machinelearningmastery.com/feed/",
    "https://lilianweng.github.io/feed.xml",
    "https://blog.google/technology/ai/rss/",
    "https://www.deeplearning.ai/the-batch/feed/",
    "https://news.ycombinator.com/rss",
]


class BlogSearchAdapter(BaseSearchAdapter):
    """Search tech blogs via RSS feed aggregation."""

    def __init__(self, feeds: List[str] = None):
        self.feeds = feeds or DEFAULT_FEEDS

    @property
    def source_name(self) -> str:
        return "blog"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._sync_search, query, max_results)
        except Exception:
            return []

    def _sync_search(self, query: str, max_results: int) -> List[SearchResult]:
        try:
            import feedparser
        except ImportError:
            return []

        query_lower = query.lower()
        query_terms = set(query_lower.split())
        results = []

        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    combined = f"{title} {summary}".lower()

                    # Simple keyword match
                    match_score = sum(1 for t in query_terms if t in combined)
                    if match_score >= 2 or any(t in combined for t in query_terms if len(t) > 3):
                        results.append(SearchResult(
                            source=self.source_name,
                            title=title,
                            url=entry.get("link", ""),
                            snippet=self._clean(summary)[:300],
                            score=match_score / max(len(query_terms), 1),
                            metadata={
                                "published": entry.get("published", ""),
                                "feed": feed_url,
                            },
                        ))
            except Exception:
                continue

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    @staticmethod
    def _clean(text: str) -> str:
        """Strip HTML tags from text."""
        import re
        return re.sub(r"<[^>]+>", "", text).strip()
