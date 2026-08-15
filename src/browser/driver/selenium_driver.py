"""
Selenium driver implementation (v4.0) — fallback browser engine.

Wraps Selenium's synchronous WebDriver API with ``asyncio.to_thread`` so it
presents the same async :class:`BrowserDriver` contract as Playwright.

Use cases: environments that mandate the WebDriver protocol, remote Selenium
Grids / BrowserStack, and legacy browsers. Selenium is imported lazily so the
package remains importable without it installed.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, Dict, List

from src.browser.driver.base import (
    ActionResult,
    BrowserDriver,
    ElementTarget,
    InteractiveElement,
)
from src.browser.exception.taxonomy import (
    BrowserAgentError,
    ElementNotFoundError,
    LaunchError,
    classify_exception,
)


_ENUMERATE_JS = """
() => {
  const SELECTOR = 'a, button, input, textarea, select, option, [role], [onclick], [tabindex], summary, [contenteditable="true"]';
  const els = Array.from(document.querySelectorAll(SELECTOR));
  const out = [];
  let idx = 1;
  const roleFor = (el) => {
    const r = el.getAttribute('role'); if (r) return r;
    const t = el.tagName.toLowerCase();
    if (t === 'a') return 'link';
    if (t === 'button') return 'button';
    if (t === 'input') { const tp = (el.getAttribute('type')||'text').toLowerCase();
      if (tp === 'submit') return 'button'; if (tp === 'checkbox') return 'checkbox'; return 'textbox'; }
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textbox';
    return t;
  };
  for (const el of els) {
    const s = window.getComputedStyle(el); const r = el.getBoundingClientRect();
    const visible = s.display !== 'none' && s.visibility !== 'hidden' && r.width > 1 && r.height > 1;
    if (!visible) continue;
    el.setAttribute('data-agent-index', String(idx));
    out.push({ index: idx, tag: el.tagName.toLowerCase(), role: roleFor(el),
      name: (el.getAttribute('aria-label') || ((el.tagName==='INPUT'||el.tagName==='TEXTAREA')&&el.value?el.value:'') || el.getAttribute('placeholder')||el.getAttribute('title')||(el.innerText||'').trim().slice(0,60)||''),
      text: (el.innerText||'').trim().slice(0,120),
      css: el.id ? ('#' + CSS.escape(el.id)) : '[data-agent-index="' + idx + '"]',
      visible: true,
      bounding_box: { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height) } });
    idx++;
  }
  return out;
}
"""


class SeleniumDriver(BrowserDriver):
    """Selenium WebDriver-based browser driver."""

    def __init__(
        self,
        browser_type: str = "chrome",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        default_timeout: int = 15000,
        navigation_timeout: int = 30000,
    ):
        self._browser_type = browser_type if browser_type in ("chrome", "firefox") else "chrome"
        self._headless = headless
        self._viewport_w = viewport_width
        self._viewport_h = viewport_height
        self._default_timeout = default_timeout / 1000.0  # seconds for Selenium
        self._navigation_timeout = navigation_timeout / 1000.0
        self._driver = None

    @property
    def is_alive(self) -> bool:
        return self._driver is not None

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # -- lifecycle ---------------------------------------------------

    async def launch(self) -> None:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
        except ImportError as e:  # pragma: no cover
            raise LaunchError("Selenium is not installed. Run: pip install selenium") from e

        try:
            if self._browser_type == "firefox":
                opts = FirefoxOptions()
                if self._headless:
                    opts.add_argument("--headless")
                self._driver = await self._run(webdriver.Firefox, options=opts)
            else:
                opts = ChromeOptions()
                if self._headless:
                    opts.add_argument("--headless=new")
                opts.add_argument(f"--window-size={self._viewport_w},{self._viewport_h}")
                self._driver = await self._run(webdriver.Chrome, options=opts)
            await self._run(self._driver.set_page_load_timeout, self._navigation_timeout)
        except LaunchError:
            raise
        except Exception as e:
            raise LaunchError(f"Failed to launch Selenium browser: {str(e)[:300]}") from e

    async def close(self) -> None:
        if self._driver is not None:
            try:
                await self._run(self._driver.quit)
            except Exception:
                pass
            self._driver = None

    # -- navigation ---------------------------------------------------

    async def navigate(self, url: str) -> None:
        await self._run(self._driver.get, url)

    async def go_back(self) -> None:
        await self._run(self._driver.back)

    # -- element lookup -----------------------------------------------

    async def query_interactive_elements(self) -> List[InteractiveElement]:
        try:
            raw = await self._run(self._driver.execute_script, _ENUMERATE_JS)
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
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        strategy = target.strategy or self._default_strategy(target)
        by, value = By.CSS_SELECTOR, ""

        if target.index >= 0 or strategy == "index":
            by, value = By.CSS_SELECTOR, f'[data-agent-index="{target.index}"]'
        elif strategy == "css" and target.css:
            by, value = By.CSS_SELECTOR, target.css
        elif strategy == "xpath" and target.xpath:
            by, value = By.XPATH, target.xpath
        elif strategy == "role" and target.role:
            by, value = By.CSS_SELECTOR, self._role_to_selector(target.role)
        elif strategy == "text" and target.text:
            by, value = By.XPATH, f'//*[contains(normalize-space(text()), "{target.text}")]'
        else:
            raise ElementNotFoundError(f"Cannot resolve target: {target.describe()}")

        try:
            wait = WebDriverWait(self._driver, self._default_timeout)
            return await self._run(wait.until, EC.presence_of_element_located((by, value)))
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

    @staticmethod
    def _role_to_selector(role: str) -> str:
        mapping = {
            "button": 'button, [role="button"], input[type="submit"], input[type="button"]',
            "link": 'a, [role="link"]',
            "textbox": 'input[type="text"], input:not([type]), textarea, [role="textbox"]',
            "checkbox": 'input[type="checkbox"], [role="checkbox"]',
            "combobox": 'select, [role="combobox"]',
            "radio": 'input[type="radio"]',
            "heading": 'h1,h2,h3,h4,h5,h6,[role="heading"]',
        }
        return mapping.get(role, f'[role="{role}"]')

    # -- actions -------------------------------------------------------

    async def click(self, target: ElementTarget) -> ActionResult:
        try:
            el = await self.find(target)
            await self._run(el.click)
        except BrowserAgentError as e:
            return await self._fail("click", target, e)
        except Exception as e:
            return await self._fail("click", target, classify_exception(e))
        return await self._ok("click", target)

    async def type_text(self, target: ElementTarget, text: str, clear: bool = True) -> ActionResult:
        try:
            el = await self.find(target)
            if clear:
                await self._run(el.clear)
            await self._run(el.send_keys, text)
        except BrowserAgentError as e:
            return await self._fail("type", target, e)
        except Exception as e:
            return await self._fail("type", target, classify_exception(e))
        return await self._ok("type", target, f"typed {len(text)} chars")

    async def press(self, target: ElementTarget, key: str) -> ActionResult:
        try:
            el = await self.find(target)
            from selenium.webdriver.common.keys import Keys
            key_map = {"Enter": Keys.ENTER, "Tab": Keys.TAB, "Escape": Keys.ESCAPE, "ArrowDown": Keys.ARROW_DOWN, "ArrowUp": Keys.ARROW_UP}
            await self._run(el.send_keys, key_map.get(key, key))
        except BrowserAgentError as e:
            return await self._fail("press", target, e)
        except Exception as e:
            return await self._fail("press", target, classify_exception(e))
        return await self._ok("press", target, f"pressed {key}")

    async def hover(self, target: ElementTarget) -> ActionResult:
        try:
            from selenium.webdriver import ActionChains
            el = await self.find(target)
            await self._run(ActionChains(self._driver).move_to_element(el).perform)
        except BrowserAgentError as e:
            return await self._fail("hover", target, e)
        except Exception as e:
            return await self._fail("hover", target, classify_exception(e))
        return await self._ok("hover", target)

    async def select(self, target: ElementTarget, value: str) -> ActionResult:
        try:
            from selenium.webdriver.support.ui import Select as Sel
            el = await self.find(target)
            await self._run(Sel(el).select_by_visible_text, value)
        except BrowserAgentError as e:
            return await self._fail("select", target, e)
        except Exception as e:
            return await self._fail("select", target, classify_exception(e))
        return await self._ok("select", target, f"selected '{value}'")

    async def scroll(self, direction: str, amount: int = 0) -> ActionResult:
        try:
            if amount <= 0:
                amount = self._viewport_h
            delta = -amount if direction == "up" else amount
            await self._run(self._driver.execute_script, f"window.scrollBy(0, {delta});")
        except Exception as e:
            return await self._fail("scroll", ElementTarget(), classify_exception(e))
        return await self._ok("scroll", ElementTarget(), f"scrolled {direction} {amount}px")

    async def drag(self, source: ElementTarget, target: ElementTarget) -> ActionResult:
        try:
            from selenium.webdriver import ActionChains
            src = await self.find(source)
            dst = await self.find(target)
            await self._run(ActionChains(self._driver).drag_and_drop(src, dst).perform)
        except BrowserAgentError as e:
            return await self._fail("drag", source, e)
        except Exception as e:
            return await self._fail("drag", source, classify_exception(e))
        return await self._ok("drag", source, f"dragged to {target.describe()}")

    async def switch_tab(self, index: int = 0) -> ActionResult:
        try:
            handles = await self._run(lambda: self._driver.window_handles)
            if index < 0 or index >= len(handles):
                return await self._fail("switch_tab", ElementTarget(),
                                        ElementNotFoundError(f"Tab index {index} out of range ({len(handles)} tabs)"))
            await self._run(self._driver.switch_to.window, handles[index])
        except BrowserAgentError as e:
            return await self._fail("switch_tab", ElementTarget(), e)
        except Exception as e:
            return await self._fail("switch_tab", ElementTarget(), classify_exception(e))
        return await self._ok("switch_tab", ElementTarget(), f"switched to tab {index}")

    # -- perception ----------------------------------------------------

    async def screenshot(self, full_page: bool = False) -> bytes:
        if full_page:
            return await self._run(self._driver.get_screenshot_as_png)
        return await self._run(self._driver.get_screenshot_as_png)

    async def get_a11y_tree(self) -> Dict[str, Any]:
        # Selenium has no native accessibility API — return an empty tree.
        return {}

    async def get_dom_snapshot(self, max_nodes: int = 300) -> str:
        try:
            text = await self._run(
                self._driver.execute_script,
                "const c=document.body.cloneNode(true); c.querySelectorAll('script,style,svg,canvas,iframe').forEach(n=>n.remove()); return (c.innerText||'');",
            )
            lines = [l.strip() for l in (text or "").split("\n") if l.strip()]
            return "\n".join(lines[:max_nodes])
        except Exception:
            return ""

    async def get_url(self) -> str:
        try:
            return await self._run(lambda: self._driver.current_url)
        except Exception:
            return ""

    async def get_title(self) -> str:
        try:
            return await self._run(lambda: self._driver.title)
        except Exception:
            return ""

    # -- util ----------------------------------------------------------

    async def execute_js(self, script: str) -> Any:
        return await self._run(self._driver.execute_script, script)

    async def wait_for_selector(self, selector: str, timeout: int = 15000) -> bool:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        try:
            wait = WebDriverWait(self._driver, timeout / 1000.0)
            await self._run(wait.until, EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
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
