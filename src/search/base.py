"""
Base search adapter interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class SearchResult:
    """Unified search result across all sources."""
    source: str            # "web" | "paper" | "news" | "blog"
    title: str
    url: str
    snippet: str           # brief description / abstract
    score: float = 0.0     # relevance or rank score
    metadata: dict = field(default_factory=dict)  # extra info (year, citations, etc.)


class BaseSearchAdapter(ABC):
    """Abstract base for all search adapters."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Unique source identifier."""
        ...

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Execute search and return results."""
        ...

    async def close(self) -> None:
        """Cleanup resources if needed."""
        pass
