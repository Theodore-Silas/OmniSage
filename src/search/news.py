"""
NewsAPI search adapter. Requires NEWSAPI_KEY env var.
Free tier: 100 req/day, articles from past 30 days.
"""

import asyncio
import os
from typing import List

import aiohttp

from .base import BaseSearchAdapter, SearchResult


class NewsSearchAdapter(BaseSearchAdapter):
    """Search news via NewsAPI (newsapi.org)."""

    API_URL = "https://newsapi.org/v2/everything"

    @property
    def source_name(self) -> str:
        return "news"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        api_key = os.getenv("NEWSAPI_KEY", "")
        if not api_key:
            return []

        params = {
            "q": query,
            "apiKey": api_key,
            "pageSize": min(max_results, 20),
            "language": "en",
            "sortBy": "relevancy",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.API_URL, params=params, timeout=10) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            results = []
            for i, article in enumerate(data.get("articles", [])):
                results.append(SearchResult(
                    source=self.source_name,
                    title=article.get("title", "Untitled"),
                    url=article.get("url", ""),
                    snippet=article.get("description", "") or "",
                    score=1.0 - (i * 0.08),
                    metadata={
                        "published_at": article.get("publishedAt", ""),
                        "source_name": article.get("source", {}).get("name", ""),
                    },
                ))
            return results[:max_results]

        except Exception:
            return []
