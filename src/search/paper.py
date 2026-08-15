"""
Academic paper search adapter.
Combines Semantic Scholar API + ArXiv API.
"""

import asyncio
from typing import List

from .base import BaseSearchAdapter, SearchResult


class PaperSearchAdapter(BaseSearchAdapter):
    """
    Search academic papers via Semantic Scholar and ArXiv.

    Semantic Scholar: free, 2B+ papers, returns abstracts + citation counts
    ArXiv: free, preprints mainly for CS/AI/physics
    """

    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    @property
    def source_name(self) -> str:
        return "paper"

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search papers: fetch from Semantic Scholar, then enrich with ArXiv fallback.
        """
        results: List[SearchResult] = []

        # Primary: Semantic Scholar
        try:
            s2_results = await self._search_semantic_scholar(query, max_results)
            results.extend(s2_results)
        except Exception as e:
            print(f"[paper] Semantic Scholar error: {e}")

        # Fallback: ArXiv (if Semantic Scholar returned few results)
        if len(results) < 3:
            try:
                arxiv_results = await self._search_arxiv(query, max(3, max_results - len(results)))
                existing_urls = {r.url for r in results}
                for r in arxiv_results:
                    if r.url not in existing_urls:
                        results.append(r)
            except Exception as e:
                print(f"[paper] ArXiv error: {e}")

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]

    async def _search_semantic_scholar(
        self, query: str, max_results: int
    ) -> List[SearchResult]:
        """Search Semantic Scholar API."""
        import aiohttp

        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,url,abstract,year,citationCount,authors",
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.SEMANTIC_SCHOLAR_URL, params=params, timeout=15
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

        results = []
        for i, paper in enumerate(data.get("data", [])):
            title = paper.get("title", "Untitled")
            abstract = paper.get("abstract") or ""
            year = paper.get("year", "")
            citations = paper.get("citationCount", 0)
            authors = ", ".join(
                a["name"] for a in (paper.get("authors") or [])[:3]
            )
            paper_url = paper.get("url", "")

            snippet = abstract[:300] if abstract else ""
            if year:
                snippet = f"({year}) {snippet}"
            if citations:
                snippet += f" [Citations: {citations}]"

            results.append(
                SearchResult(
                    source=self.source_name,
                    title=title,
                    url=paper_url,
                    snippet=snippet,
                    score=1.0 - (i * 0.08),
                    metadata={
                        "year": year,
                        "citations": citations,
                        "authors": authors,
                    },
                )
            )

        return results

    async def _search_arxiv(
        self, query: str, max_results: int
    ) -> List[SearchResult]:
        """Search ArXiv API as fallback."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_arxiv_search, query, max_results)

    @staticmethod
    def _sync_arxiv_search(query: str, max_results: int) -> List[SearchResult]:
        """Synchronous ArXiv search."""
        try:
            import arxiv
        except ImportError:
            return []

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for i, paper in enumerate(client.results(search)):
            snippet = (paper.summary or "")[:300]
            if paper.published:
                snippet = f"({paper.published.year}) {snippet}"

            authors = ", ".join(
                str(a) for a in (paper.authors or [])[:3]
            )

            results.append(
                SearchResult(
                    source="paper",
                    title=paper.title or "Untitled",
                    url=paper.entry_id or paper.pdf_url or "",
                    snippet=snippet,
                    score=1.0 - (i * 0.08),
                    metadata={
                        "year": paper.published.year if paper.published else "",
                        "authors": authors,
                        "pdf_url": paper.pdf_url or "",
                    },
                )
            )

        return results
