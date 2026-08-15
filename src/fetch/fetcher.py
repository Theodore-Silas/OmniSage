"""
Asynchronous content fetcher: crawl URLs, extract clean text from HTML.
"""

import asyncio
from typing import List, Optional

import aiohttp

from src.search.base import SearchResult


class FetchedContent:
    """Fetched and cleaned content from a URL."""

    def __init__(
        self,
        url: str,
        title: str,
        content: str,
        source: str,
        success: bool = True,
        error: str = "",
    ):
        self.url = url
        self.title = title
        self.content = content
        self.source = source
        self.success = success
        self.error = error

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERR: {self.error}"
        return f"FetchedContent({self.source}: {self.title[:40]}... [{status}])"


class ContentFetcher:
    """
    Fetch and extract clean text content from web pages.
    Uses trafilatura for content extraction (handles HTML→Markdown).
    """

    def __init__(
        self,
        timeout: int = 10,
        max_concurrent: int = 5,
        user_agent: str = "SearchAgent/1.0",
    ):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.user_agent = user_agent
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"User-Agent": self.user_agent}
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    async def fetch_results(
        self, results: List[SearchResult]
    ) -> List[FetchedContent]:
        """Fetch content for multiple search results concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def fetch_one(r: SearchResult) -> FetchedContent:
            async with semaphore:
                return await self._fetch_single(r)

        tasks = [fetch_one(r) for r in results]
        return list(await asyncio.gather(*tasks))

    async def _fetch_single(self, result: SearchResult) -> FetchedContent:
        """Fetch and extract content from a single URL."""
        url = result.url
        if not url:
            return FetchedContent(
                url="",
                title=result.title,
                content=result.snippet,
                source=result.source,
                success=True,  # use snippet as fallback
            )

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return FetchedContent(
                        url=url,
                        title=result.title,
                        content=result.snippet,
                        source=result.source,
                        success=False,
                        error=f"HTTP {resp.status}",
                    )
                html = await resp.text()

            # Extract main content using trafilatura
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, self._extract_content, html, url
            )

            if not content or len(content.strip()) < 50:
                # Fallback: use snippet
                return FetchedContent(
                    url=url,
                    title=result.title,
                    content=result.snippet,
                    source=result.source,
                    success=True,
                    error="Insufficient content extracted, using snippet",
                )

            # Truncate to reasonable size (~3000 tokens ≈ 12000 chars)
            max_chars = 12000
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[...content truncated...]"

            return FetchedContent(
                url=url,
                title=result.title,
                content=content,
                source=result.source,
                success=True,
            )

        except asyncio.TimeoutError:
            return FetchedContent(
                url=url,
                title=result.title,
                content=result.snippet,
                source=result.source,
                success=False,
                error="Timeout",
            )
        except Exception as e:
            return FetchedContent(
                url=url,
                title=result.title,
                content=result.snippet,
                source=result.source,
                success=False,
                error=str(e)[:100],
            )

    @staticmethod
    def _extract_content(html: str, url: str = "") -> str:
        """Extract main content from HTML using trafilatura."""
        try:
            import trafilatura

            # Try trafilatura first (best for articles/news)
            content = trafilatura.extract(
                html,
                output_format="markdown",
                include_links=False,
                include_images=False,
                include_tables=False,
                favor_precision=True,
            )
            if content and len(content.strip()) > 100:
                return content.strip()

            # Fallback: readability
            from readability import Document
            doc = Document(html)
            title = doc.title()
            summary = doc.summary()

            # Use BeautifulSoup to strip HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(summary, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            result = f"# {title}\n\n{text}" if title else text
            return result[:10000]

        except ImportError:
            # Last resort: BeautifulSoup plain text
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                # Remove scripts and styles
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                return text[:8000]
            except Exception:
                return ""

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
