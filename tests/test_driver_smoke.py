"""
Smoke test for the PlaywrightDriver in isolation (no LLM).

Verifies: launch → navigate → enumerate interactive elements → type → click →
extract text. Run this first to isolate browser-layer issues from LLM issues.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.browser.driver.base import ElementTarget
from src.browser.driver.playwright_driver import PlaywrightDriver


async def main() -> None:
    d = PlaywrightDriver(browser_type="chromium", headless=True, channel="chrome")
    await d.launch()

    page = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_page.html"))
    url = "file:///" + page.replace("\\", "/")
    await d.navigate(url)
    print("title:", await d.get_title())
    print("url:", await d.get_url())

    elems = await d.query_interactive_elements()
    print(f"\n{len(elems)} interactive elements:")
    for e in elems:
        print(f"  [{e.index}] <{e.role}> '{e.short_label()}' css={e.css}")

    # type into input (index 1) then click button (index 2)
    r = await d.type_text(ElementTarget(index=1), "hello")
    print("\ntype ->", r.success, r.message)

    r = await d.click(ElementTarget(index=2))
    print("click ->", r.success, r.message)

    print("\n--- DOM text after interaction ---")
    print(await d.get_dom_snapshot())

    await d.close()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    asyncio.run(main())
