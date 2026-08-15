"""Verify all dependencies are installed correctly."""
import sys

errors = []

def check(name, import_stmt):
    try:
        exec(import_stmt)
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, str(e)))
        return False

print("=== Verifying Dependencies ===\n")

# Core
check("torch", "import torch; print(f'    version={torch.__version__}, CUDA={torch.cuda.is_available()}')")
check("langgraph", "import langgraph")
check("langchain_core", "import langchain_core")
check("langchain_openai", "import langchain_openai")
check("openai", "import openai")
check("streamlit", "import streamlit")
check("faiss", "import faiss")

# Search
check("duckduckgo_search", "from duckduckgo_search import DDGS")
check("arxiv", "import arxiv")

# HTTP & Parse
check("aiohttp", "import aiohttp")
check("httpx", "import httpx")
check("trafilatura", "import trafilatura")
check("bs4", "import bs4")
check("readability", "import readability")
check("sentence_transformers", "from sentence_transformers import SentenceTransformer")

# Utils
check("dotenv", "import dotenv")
check("pydantic", "import pydantic")
check("pydantic_settings", "import pydantic_settings")
check("tenacity", "import tenacity")
check("tiktoken", "import tiktoken")

print(f"\n=== Result: {len(errors)}/{19 + 0} failed ===")

if errors:
    print("\nErrors:")
    for name, msg in errors:
        print(f"  - {name}: {msg}")

print("\n=== Project Module Imports ===")

# Project modules
sys.path.insert(0, ".")
check("src.config", "from src.config import AppConfig, LLMConfig")
check("src.search.base", "from src.search.base import BaseSearchAdapter, SearchResult")
check("src.search.web", "from src.search.web import WebSearchAdapter")
check("src.search.paper", "from src.search.paper import PaperSearchAdapter")
check("src.search.registry", "from src.search.registry import SearchRegistry")
check("src.fetch.fetcher", "from src.fetch.fetcher import ContentFetcher, FetchedContent")
check("src.summarize.summarizer", "from src.summarize.summarizer import LLMSummarizer")
check("src.agent.state", "from src.agent.state import AgentState")
check("src.agent.graph", "from src.agent.graph import build_search_agent, run_search_agent")

print(f"\n=== Project Result: {len(errors)} failed ===")

if errors:
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
