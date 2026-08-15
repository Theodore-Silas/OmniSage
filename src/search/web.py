"""
DuckDuckGo web search adapter.
"""

import asyncio
from typing import List

from .base import BaseSearchAdapter, SearchResult


class WebSearchAdapter(BaseSearchAdapter):
    """Search the web via DuckDuckGo (free, no API key needed)."""

    @property
    def source_name(self) -> str:
        return "web"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Execute DuckDuckGo text search."""
        loop = asyncio.get_event_loop()
        try:
            raw_results = await loop.run_in_executor(
                None,
                self._sync_search,
                query,
                max_results,
            )
        except Exception as e:
            print(f"[web] Search error: {e}")
            return []

        return [
            SearchResult(
                source=self.source_name,
                title=r.get("title", "Untitled"),
                url=r.get("href", r.get("link", "")),
                snippet=r.get("body", r.get("snippet", "")),
                score=1.0 - (i / max(max_results, 1)),  # simple rank decay
            )
            for i, r in enumerate(raw_results)
        ]

    @staticmethod
    def _sync_search(query: str, max_results: int) -> list:
        """Synchronous DuckDuckGo search call. Tries ddgs first, then duckduckgo_search."""
        # Try new ddgs package first
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return results
        except ImportError:
            pass

        # Fall back to deprecated duckduckgo_search
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return results
        except ImportError:
            raise ImportError(
                "Search backend required. Run: pip install ddgs"
            )
