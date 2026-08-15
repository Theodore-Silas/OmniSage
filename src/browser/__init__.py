"""
Browser Agent (v4.0) package — cross-platform automated web agent.

Highlights: Playwright/Selenium driver abstraction, DOM+VLM dual-channel
perception, Set-of-Mark grounding, self-healing locators, four-layer
exception taxonomy with escalation ladder, and hierarchical plan-execute-verify
planning.

Public entry points:
    run_browser_task(task, config=None) -> dict
    run_browser_task_stream(task, config=None) -> AsyncGenerator[dict]
"""

from src.browser.graph import BrowserRuntime, run_browser_task
from src.browser.run import run_browser_task_stream

__all__ = [
    "BrowserRuntime",
    "run_browser_task",
    "run_browser_task_stream",
]
