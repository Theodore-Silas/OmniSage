"""
Browser action tool schemas + executors (v4.0).

Defines the OpenAI function-calling schemas for all browser action primitives
and the dispatch function that maps a tool name to an :class:`ActionExecutor`
call. The LLM reasons over these tools inside the plan/decide nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.browser.action.primitives import ActionExecutor
from src.browser.driver.base import ActionResult


_TARGET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "description": "ARIA role of the element: button, link, textbox, combobox, checkbox, radio, heading, img...",
        },
        "name": {
            "type": "string",
            "description": "Accessible name / visible text of the element (e.g. '登录', 'Search').",
        },
        "text": {
            "type": "string",
            "description": "Fallback visible text to match by substring.",
        },
        "css": {"type": "string", "description": "CSS selector (use only when certain)."},
        "xpath": {"type": "string", "description": "XPath expression (use only when certain)."},
        "index": {
            "type": "integer",
            "description": "Element id from the numbered interactive-elements list (see browser_read_a11y). Strongest locator.",
        },
    },
}


def _target(**extra) -> Dict[str, Any]:
    return _TARGET_SCHEMA


BROWSER_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate the browser to a URL. Use first to open a website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including scheme, e.g. https://example.com"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_read_a11y",
            "description": "Read the current page state: URL, title, and the numbered list of interactive elements. Use this FIRST on any new page to learn element ids before clicking/typing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element. Use the element id from browser_read_a11y, or a role/name/text description.",
            "parameters": {
                "type": "object",
                "properties": {"target": _TARGET_SCHEMA},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text into an input/textarea (clears first by default).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": _TARGET_SCHEMA,
                    "text": {"type": "string", "description": "Text to type."},
                    "clear": {"type": "boolean", "description": "Clear existing content first (default true)."},
                },
                "required": ["target", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press",
            "description": "Press a key on an element (Enter, Tab, Escape, ArrowDown...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": _TARGET_SCHEMA,
                    "key": {"type": "string", "description": "Key name, e.g. Enter, Tab, Escape, ArrowDown."},
                },
                "required": ["target", "key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_hover",
            "description": "Hover over an element (to reveal dropdowns/tooltips).",
            "parameters": {
                "type": "object",
                "properties": {"target": _TARGET_SCHEMA},
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_select",
            "description": "Select an option in a <select> dropdown by visible text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": _TARGET_SCHEMA,
                    "value": {"type": "string", "description": "Option text to select."},
                },
                "required": ["target", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page down or up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up"], "description": "Scroll direction."},
                    "amount": {"type": "integer", "description": "Pixels to scroll (default: one viewport)."},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_go_back",
            "description": "Navigate back one page in browser history.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_drag",
            "description": "Drag an element onto another (for sliders, sortable lists, file uploads).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": _TARGET_SCHEMA,
                    "target": _TARGET_SCHEMA,
                },
                "required": ["source", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_switch_tab",
            "description": "Switch to another open browser tab by index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "Tab index (0-based)."},
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_exec_js",
            "description": "Run a READ-ONLY JavaScript expression in the page and return its value (network/storage/identity APIs are blocked).",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {"type": "string", "description": "JavaScript expression to evaluate, e.g. document.title or document.querySelector('h1').innerText."},
                },
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Capture and visually inspect the current page (uses the vision model when available).",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_page": {"type": "boolean", "description": "Capture the full page (default: viewport)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_extract",
            "description": "Extract the page's visible text content. Use to scrape data after navigating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_nodes": {"type": "integer", "description": "Max text lines to return (default 300)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_wait",
            "description": "Wait for a fixed delay or for a CSS selector to appear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ms": {"type": "integer", "description": "Milliseconds to wait (default 1000)."},
                    "selector": {"type": "string", "description": "CSS selector to wait for (optional)."},
                    "timeout": {"type": "integer", "description": "Selector wait timeout in ms (default 15000)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call when the task is complete. Provide the final result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "description": "The complete result of the task."},
                },
                "required": ["answer"],
            },
        },
    },
]


def _action_of(tool_name: str) -> str:
    """Map a tool name to an ActionExecutor action name."""
    if tool_name.startswith("browser_"):
        return tool_name[len("browser_"):]
    return tool_name


def format_observation(result: ActionResult) -> str:
    """Render an ActionResult as a concise observation string for the LLM."""
    if result.success:
        return f"OK {result.action}: {result.message}"
    err = result.error
    code = f" [{err.code}]" if err else ""
    return f"FAIL {result.action}{code}: {result.message}"


async def execute_browser_tool(
    tool_name: str,
    args: Dict[str, Any],
    executor: ActionExecutor,
) -> str:
    """Execute one browser tool and return its observation text."""
    if tool_name == "final_answer":
        return str(args.get("answer", ""))

    action = _action_of(tool_name)
    result = await executor.execute(action, args)
    return format_observation(result)
