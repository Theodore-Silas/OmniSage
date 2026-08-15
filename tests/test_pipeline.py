"""Minimal integration test for the agent pipeline."""
import asyncio
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig
from src.agent.graph import run_search_agent


async def main():
    config = AppConfig.from_env()
    config.llm.api_key = "sk-test-no-real-key"
    config.search.max_results_per_source = 1
    config.search.max_fetch_results = 2

    print("Starting pipeline...", flush=True)

    try:
        result = await run_search_agent(
            query="Python LangGraph tutorial",
            sources=["web", "paper"],
            max_results=1,
            config=config,
        )
        print(f"Status: {result.get('status')}", flush=True)
        print(f"Search results: {len(result.get('search_results', []))}", flush=True)
        print(f"Final answer: {len(result.get('final_answer', ''))} chars", flush=True)

        # Check logs
        for log in result.get("logs", []):
            print(f"  LOG: {log}", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
    print("DONE", flush=True)
