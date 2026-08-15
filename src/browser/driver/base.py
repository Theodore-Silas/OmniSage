"""
Browser driver abstraction (v4.0).

Defines the cross-platform contract between the agent and the underlying
browser automation engine. Two implementations exist:

  PlaywrightDriver  — primary (auto-wait, accessibility tree, native async)
  SeleniumDriver    — fallback (WebDriver protocol, remote grids, legacy)

All higher layers (perception / action / locator / planning) depend ONLY on
this interface, so switching engines never touches agent logic.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.browser.exception.taxonomy import BrowserAgentError


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class ElementTarget:
    """A locator intent produced by the LLM / resolver.

    Exactly one locating field should be populated (or ``strategy`` forces
    the resolution order). ``index`` refers to the 1-based id of an element
    in the interactive-elements list (also used as the Set-of-Mark box id).
    """
    role: str = ""
    name: str = ""
    text: str = ""
    css: str = ""
    xpath: str = ""
    index: int = -1
    strategy: str = ""  # force: role | text | css | xpath | index

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ElementTarget":
        d = d or {}
        return cls(
            role=str(d.get("role", "") or ""),
            name=str(d.get("name", "") or ""),
            text=str(d.get("text", "") or ""),
            css=str(d.get("css", "") or ""),
            xpath=str(d.get("xpath", "") or ""),
            index=int(d.get("index", -1) or -1),
            strategy=str(d.get("strategy", "") or ""),
        )

    def describe(self) -> str:
        if self.index >= 0:
            return f"element #{self.index}"
        if self.role:
            return f"{self.role} '{self.name or self.text}'"
        if self.text:
            return f"text '{self.text}'"
        if self.css:
            return f"css '{self.css}'"
        if self.xpath:
            return f"xpath '{self.xpath}'"
        return "(unspecified target)"

    def is_empty(self) -> bool:
        return not (
            self.role or self.name or self.text or self.css
            or self.xpath or self.index >= 0
        )


@dataclass
class InteractiveElement:
    """One interactable element discovered on the current page."""
    index: int
    tag: str
    role: str
    name: str
    text: str
    css: str
    visible: bool = True
    bounding_box: Dict[str, float] = field(default_factory=dict)

    def short_label(self) -> str:
        label = self.name or self.text or self.role or self.tag
        return label.strip()[:60] or self.tag


@dataclass
class ActionResult:
    """Result of a single browser action, with page-state metadata for
    the verify node and the dead-loop watchdog."""
    success: bool
    action: str
    message: str = ""
    error: Optional[BrowserAgentError] = None
    url: str = ""
    title: str = ""
    page_state_hash: str = ""
    screenshot_b64: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "message": self.message,
            "error": self.error.to_dict() if self.error else None,
            "url": self.url,
            "title": self.title,
            "page_state_hash": self.page_state_hash,
            "screenshot_b64": self.screenshot_b64[:200],  # only a stub in logs
        }


# ─────────────────────────────────────────────────────────────
# Driver interface
# ─────────────────────────────────────────────────────────────

class BrowserDriver(ABC):
    """Cross-platform browser automation contract (async)."""

    # -- lifecycle -------------------------------------------------

    @abstractmethod
    async def launch(self) -> None:
        """Start the browser. Raises LaunchError on failure."""

    @abstractmethod
    async def close(self) -> None:
        """Tear the browser down. Safe to call repeatedly."""

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the browser process is still usable."""

    # -- navigation ------------------------------------------------

    @abstractmethod
    async def navigate(self, url: str) -> None:
        """Navigate to a URL, waiting for the load event."""

    @abstractmethod
    async def go_back(self) -> None:
        """Navigate back one page in history."""

    # -- element lookup --------------------------------------------

    @abstractmethod
    async def query_interactive_elements(self) -> List[InteractiveElement]:
        """Enumerate interactable elements, assigning stable data-index ids."""

    @abstractmethod
    async def find(self, target: ElementTarget) -> Any:
        """Resolve an ElementTarget to an engine-specific element handle.

        Raises ElementNotFoundError (or subclass) when no match is found.
        """

    # -- actions (each returns ActionResult) -------------------------

    @abstractmethod
    async def click(self, target: ElementTarget) -> ActionResult: ...

    @abstractmethod
    async def type_text(self, target: ElementTarget, text: str, clear: bool = True) -> ActionResult: ...

    @abstractmethod
    async def press(self, target: ElementTarget, key: str) -> ActionResult: ...

    @abstractmethod
    async def hover(self, target: ElementTarget) -> ActionResult: ...

    @abstractmethod
    async def select(self, target: ElementTarget, value: str) -> ActionResult: ...

    @abstractmethod
    async def scroll(self, direction: str, amount: int = 0) -> ActionResult: ...

    @abstractmethod
    async def drag(self, source: ElementTarget, target: ElementTarget) -> ActionResult: ...

    @abstractmethod
    async def switch_tab(self, index: int = 0) -> ActionResult: ...

    # -- perception -------------------------------------------------

    @abstractmethod
    async def screenshot(self, full_page: bool = False) -> bytes:
        """Capture a PNG screenshot of the viewport (or full page)."""

    @abstractmethod
    async def get_a11y_tree(self) -> Dict[str, Any]:
        """Return the accessibility tree snapshot."""

    @abstractmethod
    async def get_dom_snapshot(self, max_nodes: int = 300) -> str:
        """Return a compact text serialization of the DOM."""

    @abstractmethod
    async def get_url(self) -> str: ...

    @abstractmethod
    async def get_title(self) -> str: ...

    # -- util -------------------------------------------------------

    @abstractmethod
    async def execute_js(self, script: str) -> Any:
        """Evaluate JavaScript in the page context (sandboxed by caller)."""

    @abstractmethod
    async def wait_for_selector(self, selector: str, timeout: int = 15000) -> bool:
        """Wait until a selector appears (True) or timeout (False)."""

    async def get_page_state_hash(self) -> str:
        """A stable hash of the current page state, used for dead-loop detection."""
        try:
            url = await self.get_url()
            title = await self.get_title()
            elems = await self.query_interactive_elements()
            n = len(elems)
        except Exception:
            url, title, n = "?", "?", 0
        key = f"{url}|{title}|{n}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
