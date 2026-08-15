"""
Perception channel A — DOM / accessibility snapshot (v4.0).

Produces a compact, numbered list of interactable elements that the text LLM
can reason over directly (zero VLM cost). This is the primary perception
channel; channel B (VLM + Set-of-Mark) is only used when this is insufficient.
"""

from __future__ import annotations

from typing import List

from src.browser.driver.base import BrowserDriver, InteractiveElement

MAX_INTERACTIVE = 60


async def get_interactive_elements(driver: BrowserDriver) -> List[InteractiveElement]:
    """Return the current page's interactable elements (with stable ids)."""
    return await driver.query_interactive_elements()


def format_interactive_list(elements: List[InteractiveElement], max_items: int = MAX_INTERACTIVE) -> str:
    """Render the numbered element list for LLM decision-making."""
    lines: List[str] = []
    for e in elements[:max_items]:
        label = e.short_label()
        box = e.bounding_box
        loc = f" @({box['x']},{box['y']})" if box else ""
        lines.append(f'[{e.index}] <{e.role}> "{label}"{loc}')
    if not lines:
        return "(no interactive elements found)"
    return "\n".join(lines)


async def build_state_summary(driver: BrowserDriver) -> str:
    """Compose a full state summary: URL + title + numbered element list."""
    url = await driver.get_url()
    title = await driver.get_title()
    elements = await get_interactive_elements(driver)
    body = format_interactive_list(elements)
    return (
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Interactive elements ({len(elements)}):\n{body}"
    )
