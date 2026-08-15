"""
CLI entry point for SearchAgent — 搜索问答（本地命中优先 + 外部搜索兜底）。

Usage:
  python -m src.main "你的问题"          # 先命中本地知识，未命中才搜索
  python -m src.main "问题" -v           # verbose: 显示命中/搜索/沉淀日志
  python -m src.main --streamlit         # 启动 Streamlit UI
"""

import argparse
import asyncio
import time

from src.config import AppConfig
from src.qa import run_qa


def format_output(result: dict, verbose: bool = False) -> str:
    """Format the Q&A result for CLI display."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  Query: {result.get('query', '')}")
    lines.append(f"  Status: {result.get('status', 'unknown')}")
    lines.append(f"  Verdict: {result.get('verdict', '') or '-'}")
    lines.append("=" * 70)

    answer = result.get("final_answer", "")
    lines.append("\n" + (answer or "(No answer generated)"))

    if verbose:
        lines.append("\n" + "-" * 70)
        lines.append("  Logs:")
        lines.append("-" * 70)
        for log in result.get("logs", []):
            lines.append(f"  {log}")

    return "\n".join(lines)


async def main_async(args):
    config = AppConfig.from_env()
    config.verbose = args.verbose

    start_time = time.time()
    result = await run_qa(query=args.query, config=config)
    elapsed = time.time() - start_time

    output = format_output(result, verbose=args.verbose)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output saved to {args.output}")
    else:
        print(output)

    print(f"\n--- Completed in {elapsed:.1f}s ---")


def main():
    parser = argparse.ArgumentParser(
        description="SearchAgent — 搜索问答（本地命中优先 + 外部搜索兜底）"
    )
    parser.add_argument("query", nargs="?", default=None, help="Question / keywords to ask")
    parser.add_argument("--output", "-o", help="Save output to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show retrieve/search/summarize/persist logs")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit web UI instead of CLI")

    args = parser.parse_args()

    if args.streamlit:
        import subprocess
        import os
        ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
        subprocess.run([__import__("sys").executable, "-m", "streamlit", "run", ui_path])
    else:
        if not args.query:
            parser.error("the following arguments are required: query")
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
