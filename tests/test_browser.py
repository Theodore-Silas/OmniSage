"""
End-to-end test for the v4.0 browser agent.

Loads a local demo page, types into the search box, clicks the button, and
extracts the displayed result — exercising navigate → read_a11y → type →
click → extract in a single headless run.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig
from src.browser import run_browser_task


async def main() -> None:
    config = AppConfig.from_env()
    config.browser.headless = True
    config.browser.browser_type = "chromium"
    config.browser.channel = "chrome"  # use system Chrome (no bundled chromium download)

    page = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_page.html"))
    url = "file:///" + page.replace("\\", "/")

    task = (
        f"打开页面 {url} ，在搜索输入框输入 'hello' ，点击搜索按钮，"
        f"然后读取并报告页面显示的结果文本（#result 里的内容）。"
    )

    print("Running browser task...")
    result = await run_browser_task(task, config)

    print("\n=== FINAL ANSWER ===")
    print(result.get("final_answer", "(none)"))
    print("\n=== ACTION TRACE ===")
    for t in result.get("action_trace", []):
        mark = "OK" if t.get("success") else "FAIL"
        print(f"  [{t.get('iteration')}] {t.get('action'):20s} -> {mark}  {t.get('message', '')[:70]}")
    print(f"\nsteps={result.get('tool_calls_made')} iterations={result.get('iteration')}")
    for err in result.get("errors", []):
        print(f"ERROR: {err}")


if __name__ == "__main__":
    asyncio.run(main())
