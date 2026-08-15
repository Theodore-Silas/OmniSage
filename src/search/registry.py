"""
Search adapter registry.
"""

from typing import Dict, List, Optional

from .base import BaseSearchAdapter
from .web import WebSearchAdapter
from .paper import PaperSearchAdapter
from .news import NewsSearchAdapter
from .blog import BlogSearchAdapter


class SearchRegistry:
    _adapters: Dict[str, BaseSearchAdapter] = {}

    @classmethod
    def get_all(cls) -> Dict[str, BaseSearchAdapter]:
        if not cls._adapters:
            cls._adapters = {
                "web": WebSearchAdapter(),
                "paper": PaperSearchAdapter(),
                "news": NewsSearchAdapter(),
                "blog": BlogSearchAdapter(),
            }
        return cls._adapters

    @classmethod
    def get(cls, source_name: str) -> Optional[BaseSearchAdapter]:
        return cls.get_all().get(source_name)

    @classmethod
    def get_enabled(cls, enabled_sources: List[str]) -> List[BaseSearchAdapter]:
        all_adapters = cls.get_all()
        return [all_adapters[name] for name in enabled_sources if name in all_adapters]
