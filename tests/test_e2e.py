"""End-to-end functional tests for SearchAgent."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.search.web import WebSearchAdapter
from src.search.paper import PaperSearchAdapter
from src.search.registry import SearchRegistry
from src.fetch.fetcher import ContentFetcher
from src.agent.graph import build_search_agent


async def test_web_search():
    print("\n--- Test: Web Search (DuckDuckGo) ---")
    adapter = WebSearchAdapter()
    try:
        results = await adapter.search("Python LangGraph tutorial", max_results=3)
        print(f"  Results: {len(results)}")
        for r in results:
            print(f"  - {r.title[:60]}... | {r.url[:50]}...")
        return len(results) > 0
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


async def test_paper_search():
    print("\n--- Test: Paper Search (Semantic Scholar) ---")
    adapter = PaperSearchAdapter()
    try:
        results = await adapter.search("retrieval augmented generation", max_results=3)
        print(f"  Results: {len(results)}")
        for r in results:
            print(f"  - [{r.metadata.get('year','?')}] {r.title[:60]}...")
        return len(results) > 0
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


async def test_registry():
    print("\n--- Test: Search Registry ---")
    reg = SearchRegistry.get_enabled(["web", "paper"])
    print(f"  Enabled adapters: {[a.source_name for a in reg]}")
    return len(reg) == 2


async def test_fetcher():
    print("\n--- Test: Content Fetcher ---")
    fetcher = ContentFetcher(timeout=10, max_concurrent=3)
    try:
        from src.search.base import SearchResult
        test_urls = [
            SearchResult(source="web", title="Test", url="https://httpbin.org/html", snippet="Test snippet", score=1.0),
        ]
        results = await fetcher.fetch_results(test_urls)
        for r in results:
            print(f"  URL: {r.url}, Success: {r.success}, Content length: {len(r.content)}")
        await fetcher.close()
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        await fetcher.close()
        return False


async def test_graph_build():
    print("\n--- Test: LangGraph Build ---")
    try:
        config = AppConfig()
        config.llm.api_key = "sk-test-placeholder"
        graph = build_search_agent(config)
        compiled = graph.compile()
        print(f"  Graph compiled OK | Nodes: {list(compiled.get_graph().nodes.keys())}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


async def main():
    print("=" * 60)
    print("SearchAgent End-to-End Tests")
    print("=" * 60)

    results = {}

    results["web_search"] = await test_web_search()
    results["paper_search"] = await test_paper_search()
    results["registry"] = await test_registry()
    results["fetcher"] = await test_fetcher()
    results["graph_build"] = await test_graph_build()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  ALL TESTS PASSED!")
    else:
        print(f"\n  {total - passed} test(s) FAILED!")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
