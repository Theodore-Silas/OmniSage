"""
Action primitives (v4.0) — the single execution point for browser actions.

Wires together driver + resolver + safety guard, exposing one method
``execute(action, params)`` used by the tool executors and the planning layer.
Perception-style actions (read_a11y / screenshot / extract) return their
observation text in the ActionResult.message field.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Optional

from src.browser.driver.base import ActionResult, BrowserDriver, ElementTarget
from src.browser.exception.taxonomy import BrowserAgentError, classify_exception
from src.browser.locator.resolver import LocatorResolver
from src.browser.perception.dom_snapshot import build_state_summary
from src.browser.perception.som import annotate_and_encode
from src.browser.perception.vlm import VLMClient


class ActionExecutor:
    """Executes browser action primitives with safety + self-healing."""

    def __init__(
        self,
        driver: BrowserDriver,
        resolver: LocatorResolver,
        guard: Optional[Any] = None,
        vlm_client: Optional[VLMClient] = None,
        task: str = "",
    ):
        self.driver = driver
        self.resolver = resolver
        self.guard = guard
        self.vlm = vlm_client
        self.task = task

    async def execute(self, action: str, params: Dict[str, Any]) -> ActionResult:
        """Dispatch a single action by name. Unknown actions return a failure."""
        if self.guard is not None and not self.guard.check(action, params):
            return await self._fail(action, "Blocked by safety guard")

        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            return await self._fail(action, f"Unknown action '{action}'")
        try:
            return await handler(params)
        except BrowserAgentError as e:
            return await self._fail(action, e.message, e)
        except Exception as e:
            return await self._fail(action, str(e)[:300], classify_exception(e))

    # -- navigation / utility -------------------------------------------

    async def _do_navigate(self, p: Dict[str, Any]) -> ActionResult:
        url = str(p.get("url", "")).strip()
        if not url:
            return await self._fail("navigate", "No URL provided")
        try:
            await self.driver.navigate(url)
            return await self._ok("navigate", f"Navigated to {url}")
        except Exception as e:
            return await self._fail("navigate", str(e)[:300], classify_exception(e))

    async def _do_go_back(self, p: Dict[str, Any]) -> ActionResult:
        try:
            await self.driver.go_back()
            return await self._ok("go_back", "Navigated back")
        except Exception as e:
            return await self._fail("go_back", str(e)[:300], classify_exception(e))

    async def _do_wait(self, p: Dict[str, Any]) -> ActionResult:
        selector = str(p.get("selector", "")).strip()
        if selector:
            ok = await self.driver.wait_for_selector(selector, int(p.get("timeout", 15000) or 15000))
            return await self._ok("wait", f"Selector {'appeared' if ok else 'timed out'}: {selector}")
        import asyncio
        await asyncio.sleep(min(int(p.get("ms", 1000) or 1000), 10000))
        return await self._ok("wait", f"Waited {p.get('ms', 1000)}ms")

    # -- element actions --------------------------------------------------

    async def _do_click(self, p: Dict[str, Any]) -> ActionResult:
        t = ElementTarget.from_dict(p.get("target"))
        if t.is_empty():
            return await self._fail("click", "No target specified")
        return await self.resolver.act("click", t)

    async def _do_type(self, p: Dict[str, Any]) -> ActionResult:
        t = ElementTarget.from_dict(p.get("target"))
        text = str(p.get("text", ""))
        clear = bool(p.get("clear", True))
        if t.is_empty() or not text:
            return await self._fail("type", "Missing target or text")
        return await self.resolver.act("type_text", t, text=text, clear=clear)

    async def _do_press(self, p: Dict[str, Any]) -> ActionResult:
        t = ElementTarget.from_dict(p.get("target"))
        key = str(p.get("key", "Enter"))
        if t.is_empty():
            return await self._fail("press", "No target specified")
        return await self.resolver.act("press", t, key=key)

    async def _do_hover(self, p: Dict[str, Any]) -> ActionResult:
        t = ElementTarget.from_dict(p.get("target"))
        if t.is_empty():
            return await self._fail("hover", "No target specified")
        return await self.resolver.act("hover", t)

    async def _do_select(self, p: Dict[str, Any]) -> ActionResult:
        t = ElementTarget.from_dict(p.get("target"))
        value = str(p.get("value", ""))
        if t.is_empty() or not value:
            return await self._fail("select", "Missing target or value")
        return await self.resolver.act("select", t, value=value)

    async def _do_scroll(self, p: Dict[str, Any]) -> ActionResult:
        direction = str(p.get("direction", "down"))
        amount = int(p.get("amount", 0) or 0)
        return await self.driver.scroll(direction, amount)

    async def _do_drag(self, p: Dict[str, Any]) -> ActionResult:
        src = ElementTarget.from_dict(p.get("source"))
        dst = ElementTarget.from_dict(p.get("target"))
        if src.is_empty() or dst.is_empty():
            return await self._fail("drag", "Missing source or target")
        return await self.driver.drag(src, dst)

    async def _do_switch_tab(self, p: Dict[str, Any]) -> ActionResult:
        index = int(p.get("index", 0) or 0)
        return await self.driver.switch_tab(index)

    async def _do_exec_js(self, p: Dict[str, Any]) -> ActionResult:
        script = str(p.get("script", "")).strip()
        if not script:
            return await self._fail("exec_js", "No script provided")
        try:
            result = await self.driver.execute_js(script)
            text = str(result) if result is not None else "(no return value)"
            return await self._ok("exec_js", text[:2000])
        except Exception as e:
            return await self._fail("exec_js", str(e)[:300], classify_exception(e))

    # -- perception actions ----------------------------------------------

    async def _do_read_a11y(self, p: Dict[str, Any]) -> ActionResult:
        summary = await build_state_summary(self.driver)
        return await self._ok("read_a11y", summary)

    async def _do_screenshot(self, p: Dict[str, Any]) -> ActionResult:
        full_page = bool(p.get("full_page", False))
        shot = await self.driver.screenshot(full_page=full_page)
        if self.vlm is not None and self.vlm.enabled:
            elements = await self.driver.query_interactive_elements()
            som_bytes = annotate_and_encode(shot, elements) if not full_page else None
            result = await self.vlm.perceive(shot, som_bytes, self.task, elements)
            if result is not None:
                text = (
                    f"[VLM] page: {result.page_summary}\n"
                    f"[VLM] suggested: {result.action} element #{result.element_id} — {result.reason}"
                )
                return await self._ok("screenshot", text)
        # Fallback: return the DOM element list
        summary = await build_state_summary(self.driver)
        return await self._ok("screenshot", f"(VLM disabled, DOM view)\n{summary}")

    async def _do_extract(self, p: Dict[str, Any]) -> ActionResult:
        max_nodes = int(p.get("max_nodes", 300) or 300)
        text = await self.driver.get_dom_snapshot(max_nodes=max_nodes)
        if not text:
            text = "(empty page or no extractable text)"
        return await self._ok("extract", text)

    # -- result helpers ----------------------------------------------------

    async def _ok(self, action: str, message: str = "") -> ActionResult:
        return ActionResult(
            success=True,
            action=action,
            message=message,
            url=await self.driver.get_url(),
            title=await self.driver.get_title(),
            page_state_hash=await self.driver.get_page_state_hash(),
        )

    async def _fail(self, action: str, message: str, error: Optional[BrowserAgentError] = None) -> ActionResult:
        return ActionResult(
            success=False,
            action=action,
            message=message,
            error=error,
            url=await self.driver.get_url(),
            title=await self.driver.get_title(),
            page_state_hash=await self.driver.get_page_state_hash(),
        )
