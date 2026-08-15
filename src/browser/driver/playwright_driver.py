"""
Playwright driver implementation (v4.0) — primary browser engine.

Key features:
  - native async (no thread wrapper)
  - auto-waiting locators (no brittle sleeps)
  - accessibility tree + interactive-element enumeration via JS
  - stable ``data-agent-index`` id injection for index-based grounding
  - Playwright errors translated into the four-layer taxonomy
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from src.browser.driver.base import (
    ActionResult,
    BrowserDriver,
    ElementTarget,
    InteractiveElement,
)
from src.browser.exception.taxonomy import (
    BrowserAgentError,
    ElementNotFoundError,
    ElementObscuredError,
    LaunchError,
    PageTimeoutError,
    classify_exception,
)


# JS that enumerates interactable elements and stamps a stable data-agent-index.
_ENUMERATE_JS = """
() => {
  const SELECTOR = 'a, button, input, textarea, select, option, [role], [onclick], [tabindex], summary, [contenteditable="true"]';
  const els = Array.from(document.querySelectorAll(SELECTOR));
  const out = [];
  let idx = 1;
  const cssFor = (el) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    if (el.name) return `[name="${CSS.escape(el.name)}"]`;
    return `[data-agent-index="${el.getAttribute('data-agent-index')}"]`;
  };
  const roleFor = (el) => {
    const r = el.getAttribute('role');
    if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link';
    if (t === 'button') return 'button';
    if (t === 'input') {
      const tp = (el.getAttribute('type') || 'text').toLowerCase();
      if (tp === 'submit') return 'button';
      if (tp === 'checkbox') return 'checkbox';
      if (tp === 'radio') return 'radio';
      return 'textbox';
    }
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textbox';
    if (t === 'img') return 'img';
    return t;
  };
  for (const el of els) {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    const visible = style.display !== 'none' && style.visibility !== 'hidden'
      && rect.width > 1 && rect.height > 1;
    if (!visible) continue;
    el.setAttribute('data-agent-index', String(idx));
    const name = el.getAttribute('aria-label')
      || ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && el.value ? el.value : '')
      || el.getAttribute('placeholder')
      || el.getAttribute('title')
      || el.getAttribute('alt')
      || (el.innerText || '').trim().slice(0, 60)
      || el.getAttribute('value')
      || '';
    out.push({
      index: idx,
      tag: el.tagName.toLowerCase(),
      role: roleFor(el),
      name: name,
      text: (el.innerText || '').trim().slice(0, 120),
      css: cssFor(el),
      visible: true,
      bounding_box: { x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height) },
    });
    idx++;
  }
  return out;
}
"""

_DOM_TEXT_JS = """
(maxNodes) => {
  const clone = document.body ? document.body.cloneNode(true) : document.createElement('body');
  clone.querySelectorAll('script, style, noscript, svg, canvas, iframe').forEach(n => n.remove());
  const text = clone.innerText || '';
  const lines = text.split('\\n').map(l => l.trim()).filter(Boolean);
  return lines.slice(0, maxNodes).join('\\n');
}
"""


class PlaywrightDriver(BrowserDriver):
    """Playwright-based browser driver."""

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        default_timeout: int = 15000,
        navigation_timeout: int = 30000,
        user_agent: str = "",
        channel: str = "",
    ):
        self._browser_type = browser_type
        self._channel = channel  # "" | "chrome" | "msedge"
        self._headless = headless
        self._viewport = {"width": viewport_width, "height": viewport_height}
        self._default_timeout = default_timeout
        self._navigation_timeout = navigation_timeout
        self._user_agent = user_agent

        self._pw = None
        self._browser = None
        self._page = None

    # -- lifecycle ---------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._browser is not None and self._page is not None

    async def launch(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:  # pragma: no cover
            raise LaunchError(
                "Playwright is not installed. Run: pip install playwright && playwright install chromium"
            ) from e

        try:
            self._pw = await async_playwright().start()
            launcher = getattr(self._pw, self._browser_type, None)
            if launcher is None:
                raise LaunchError(f"Unknown browser type: {self._browser_type}")

            launch_kwargs: Dict[str, Any] = {"headless": self._headless}
            if self._channel:
                launch_kwargs["channel"] = self._channel
            if self._user_agent:
                launch_kwargs["user_agent"] = self._user_agent

            self._browser = await launcher.launch(**launch_kwargs)
            context_kwargs: Dict[str, Any] = {"viewport": self._viewport}
            self._context = await self._browser.new_context(**context_kwargs)
            self._context.set_default_timeout(self._default_timeout)
            self._context.set_default_navigation_timeout(self._navigation_timeout)
            self._page = await self._context.new_page()
        except LaunchError:
            raise
        except Exception as e:
            await self._safe_cleanup()
            raise LaunchError(f"Failed to launch browser: {str(e)[:300]}") from e

    async def close(self) -> None:
        await self._safe_cleanup()

    async def _safe_cleanup(self) -> None:
        for closer in (self._page, self._context, self._browser, self._pw):
            if closer is None:
                continue
            try:
                if closer is self._pw:
                    await closer.stop()
                else:
                    await closer.close()
            except Exception:
                pass
        self._page = self._context = self._browser = self._pw = None

    @property
    def _active_page(self):
        if self._page is None:
            raise LaunchError("Browser is not launched.")
        return self._page

    # -- navigation ---------------------------------------------------

    async def navigate(self, url: str) -> None:
        await self._active_page.goto(url, wait_until="load", timeout=self._navigation_timeout)

    async def go_back(self) -> None:
        await self._active_page.go_back(wait_until="load", timeout=self._navigation_timeout)

    # -- element lookup -----------------------------------------------

    async def query_interactive_elements(self) -> List[InteractiveElement]:
        try:
            raw = await self._active_page.evaluate(_ENUMERATE_JS)
        except Exception as e:
            raise classify_exception(e)
        return [
            InteractiveElement(
                index=int(it["index"]),
                tag=it.get("tag", ""),
                role=it.get("role", ""),
                name=it.get("name", ""),
                text=it.get("text", ""),
                css=it.get("css", ""),
                visible=bool(it.get("visible", True)),
                bounding_box=it.get("bounding_box", {}),
            )
            for it in (raw or [])
        ]

    async def find(self, target: ElementTarget):
        page = self._active_page
        strategy = target.strategy or self._default_strategy(target)

        try:
            if target.index >= 0 or strategy == "index":
                loc = page.locator(f'[data-agent-index="{target.index}"]')
            elif strategy == "css" and target.css:
                loc = page.locator(target.css)
            elif strategy == "xpath" and target.xpath:
                loc = page.locator(f"xpath={target.xpath}")
            elif strategy == "role" and target.role:
                loc = page.get_by_role(target.role, name=target.name or None)
            elif strategy == "text" and target.text:
                loc = page.get_by_text(target.text, exact=False)
            else:
                raise ElementNotFoundError(f"Cannot resolve target: {target.describe()}")

            # Auto-wait for the element to be attached.
            await loc.first.wait_for(state="attached", timeout=self._default_timeout)
            return loc.first
        except BrowserAgentError:
            raise
        except Exception as e:
            raise ElementNotFoundError(f"Element not found: {target.describe()} ({str(e)[:200]})")

    @staticmethod
    def _default_strategy(target: ElementTarget) -> str:
        if target.index >= 0:
            return "index"
        if target.css:
            return "css"
        if target.xpath:
            return "xpath"
        if target.role:
            return "role"
        if target.text:
            return "text"
        return ""

    # -- actions -------------------------------------------------------

    async def click(self, target: ElementTarget) -> ActionResult:
        try:
            loc = await self.find(target)
            await loc.click(timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("click", target, e)
        except Exception as e:
            err = classify_exception(e)
            if isinstance(err, PageTimeoutError):
                err = ElementObscuredError(f"Not clickable (obscured/hidden): {target.describe()}")
            return await self._fail("click", target, err)
        return await self._ok("click", target)

    async def type_text(self, target: ElementTarget, text: str, clear: bool = True) -> ActionResult:
        try:
            loc = await self.find(target)
            if clear:
                await loc.fill(text, timeout=self._default_timeout)
            else:
                await loc.type(text, timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("type", target, e)
        except Exception as e:
            return await self._fail("type", target, classify_exception(e))
        return await self._ok("type", target, f"typed {len(text)} chars")

    async def press(self, target: ElementTarget, key: str) -> ActionResult:
        try:
            loc = await self.find(target)
            await loc.press(key, timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("press", target, e)
        except Exception as e:
            return await self._fail("press", target, classify_exception(e))
        return await self._ok("press", target, f"pressed {key}")

    async def hover(self, target: ElementTarget) -> ActionResult:
        try:
            loc = await self.find(target)
            await loc.hover(timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("hover", target, e)
        except Exception as e:
            return await self._fail("hover", target, classify_exception(e))
        return await self._ok("hover", target)

    async def select(self, target: ElementTarget, value: str) -> ActionResult:
        try:
            loc = await self.find(target)
            await loc.select_option(value, timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("select", target, e)
        except Exception as e:
            return await self._fail("select", target, classify_exception(e))
        return await self._ok("select", target, f"selected '{value}'")

    async def scroll(self, direction: str, amount: int = 0) -> ActionResult:
        try:
            page = self._active_page
            if amount <= 0:
                amount = self._viewport["height"]
            delta = amount if direction in ("down", "up") else amount
            if direction == "up":
                delta = -delta
            elif direction in ("down",):
                delta = delta
            else:  # arbitrary vertical
                delta = delta
            await page.mouse.wheel(0, delta)
            await page.wait_for_timeout(250)
        except Exception as e:
            return await self._fail("scroll", ElementTarget(), classify_exception(e))
        return await self._ok("scroll", ElementTarget(), f"scrolled {direction} {amount}px")

    async def drag(self, source: ElementTarget, target: ElementTarget) -> ActionResult:
        try:
            src = await self.find(source)
            dst = await self.find(target)
            await src.drag_to(dst, timeout=self._default_timeout)
        except BrowserAgentError as e:
            return await self._fail("drag", source, e)
        except Exception as e:
            return await self._fail("drag", source, classify_exception(e))
        return await self._ok("drag", source, f"dragged to {target.describe()}")

    async def switch_tab(self, index: int = 0) -> ActionResult:
        try:
            pages = self._context.pages if self._context else []
            if index < 0 or index >= len(pages):
                return await self._fail("switch_tab", ElementTarget(),
                                        ElementNotFoundError(f"Tab index {index} out of range ({len(pages)} tabs)"))
            await pages[index].bring_to_front()
            self._page = pages[index]
        except BrowserAgentError as e:
            return await self._fail("switch_tab", ElementTarget(), e)
        except Exception as e:
            return await self._fail("switch_tab", ElementTarget(), classify_exception(e))
        return await self._ok("switch_tab", ElementTarget(), f"switched to tab {index}")

    # -- perception ----------------------------------------------------

    async def screenshot(self, full_page: bool = False) -> bytes:
        return await self._active_page.screenshot(full_page=full_page, type="png")

    async def get_a11y_tree(self) -> Dict[str, Any]:
        try:
            return await self._active_page.accessibility.snapshot() or {}
        except Exception:
            return {}

    async def get_dom_snapshot(self, max_nodes: int = 300) -> str:
        try:
            return await self._active_page.evaluate(_DOM_TEXT_JS, max_nodes)
        except Exception:
            return ""

    async def get_url(self) -> str:
        try:
            return self._active_page.url
        except Exception:
            return ""

    async def get_title(self) -> str:
        try:
            return await self._active_page.title()
        except Exception:
            return ""

    # -- util ----------------------------------------------------------

    async def execute_js(self, script: str) -> Any:
        return await self._active_page.evaluate(script)

    async def wait_for_selector(self, selector: str, timeout: int = 15000) -> bool:
        try:
            await self._active_page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    # -- result helpers ------------------------------------------------

    async def _ok(self, action: str, target: ElementTarget, message: str = "") -> ActionResult:
        return ActionResult(
            success=True,
            action=action,
            message=message or f"{action} {target.describe()}",
            url=await self.get_url(),
            title=await self.get_title(),
            page_state_hash=await self.get_page_state_hash(),
        )

    async def _fail(self, action: str, target: ElementTarget, error: BrowserAgentError) -> ActionResult:
        shot = ""
        try:
            shot = base64.b64encode(await self.screenshot()).decode("ascii")
        except Exception:
            pass
        return ActionResult(
            success=False,
            action=action,
            message=error.message,
            error=error,
            url=await self.get_url(),
            title=await self.get_title(),
            page_state_hash=await self.get_page_state_hash(),
            screenshot_b64=shot,
        )
