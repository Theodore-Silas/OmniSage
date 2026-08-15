"""
Multi-strategy element resolver with self-healing (v4.0).

Resolution order (strongest → weakest):

  1. cached stable css selector (LocatorCache)
  2. explicit strategy sequence: index → role+name → text → css → xpath
  3. fuzzy match against the interactive-elements list (rapidfuzz / difflib)
  4. VLM re-perception (screenshot + Set-of-Mark → element_id → index)

Each successful fuzzy/VLM resolution is written back to the cache so future
runs reuse the stable selector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.browser.driver.base import ActionResult, BrowserDriver, ElementTarget
from src.browser.exception.taxonomy import ElementNotFoundError
from src.browser.locator.cache import LocatorCache
from src.browser.perception.som import annotate_and_encode
from src.browser.perception.vlm import VLMClient

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    import difflib

    class _Fuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    _fuzz = _Fuzz()


FUZZY_THRESHOLD = 60.0


class LocatorResolver:
    """Resolves ElementTargets to element handles with multi-level fallback."""

    def __init__(
        self,
        driver: BrowserDriver,
        vlm_client: Optional[VLMClient] = None,
        cache: Optional[LocatorCache] = None,
        task: str = "",
    ):
        self.driver = driver
        self.vlm = vlm_client
        self.cache = cache or LocatorCache()
        self.task = task

    # -- target expansion ------------------------------------------------

    @staticmethod
    def expand_target(target: ElementTarget) -> List[ElementTarget]:
        """Expand a (possibly vague) target into a strongest-first strategy list."""
        seq: List[ElementTarget] = []
        if target.index >= 0:
            seq.append(ElementTarget(index=target.index, strategy="index"))
        if target.role:
            seq.append(ElementTarget(role=target.role, name=target.name, strategy="role"))
        if target.text or target.name:
            seq.append(ElementTarget(text=target.text or target.name, strategy="text"))
        if target.css:
            seq.append(ElementTarget(css=target.css, strategy="css"))
        if target.xpath:
            seq.append(ElementTarget(xpath=target.xpath, strategy="xpath"))
        if target.strategy and target.strategy not in ("index", "role", "text", "css", "xpath"):
            seq.insert(0, target)
        return seq

    # -- resolution -------------------------------------------------------

    async def resolve(self, target: ElementTarget) -> Tuple[object, Optional[str]]:
        """Resolve a target to (element_handle, stable_css_or_None).

        Raises ElementNotFoundError when every strategy fails.
        """
        url = await self.driver.get_url()

        # 0. cached stable selector
        cached_css = self.cache.get_css(url, target)
        if cached_css:
            try:
                handle = await self.driver.find(ElementTarget(css=cached_css, strategy="css"))
                return handle, cached_css
            except ElementNotFoundError:
                pass

        # 1. explicit strategy sequence
        for t in self.expand_target(target):
            try:
                handle = await self.driver.find(t)
                css = await self._css_for_index(t.index) if t.index >= 0 else None
                return handle, css or t.css or None
            except ElementNotFoundError:
                continue

        # 2. fuzzy match against interactive elements
        ft = await self._fuzzy_match(target)
        if ft is not None:
            try:
                handle = await self.driver.find(ft)
                css = await self._css_for_index(ft.index)
                self.cache.put_css(url, target, css)
                return handle, css
            except ElementNotFoundError:
                pass

        # 3. VLM re-perception
        vt = await self._vlm_relocate(target)
        if vt is not None:
            try:
                handle = await self.driver.find(vt)
                css = await self._css_for_index(vt.index)
                self.cache.put_css(url, target, css)
                return handle, css
            except ElementNotFoundError:
                pass

        raise ElementNotFoundError(f"All strategies failed for {target.describe()}")

    # -- action execution with self-healing --------------------------------

    async def act(self, action: str, target: ElementTarget, **kwargs) -> ActionResult:
        """Resolve + execute a driver action, writing back learned locators."""
        try:
            handle, css = await self.resolve(target)
        except ElementNotFoundError as e:
            return ActionResult(
                success=False, action=action, message=e.message, error=e,
            )

        method = getattr(self.driver, action, None)
        if method is None:
            return ActionResult(
                success=False, action=action, message=f"Unknown action '{action}'",
            )

        # For index-based targets, learn the stable css for the cache.
        try:
            result = await method(target, **kwargs)
        except Exception as e:
            from src.browser.exception.taxonomy import classify_exception
            result = ActionResult(
                success=False, action=action, message=str(e)[:300],
                error=classify_exception(e),
            )

        return result

    # -- fallback helpers ----------------------------------------------------

    async def _fuzzy_match(self, target: ElementTarget) -> Optional[ElementTarget]:
        needle = (target.name or target.text or "").strip()
        if not needle:
            return None
        try:
            elements = await self.driver.query_interactive_elements()
        except Exception:
            return None

        best_idx, best_score = -1, 0.0
        for e in elements:
            cand = (e.name or e.text or "").strip()
            if not cand:
                continue
            score = _fuzz.ratio(needle.lower(), cand.lower())
            if score > best_score:
                best_idx, best_score = e.index, score

        if best_idx > 0 and best_score >= FUZZY_THRESHOLD:
            return ElementTarget(index=best_idx, strategy="index")
        return None

    async def _vlm_relocate(self, target: ElementTarget) -> Optional[ElementTarget]:
        if self.vlm is None or not self.vlm.enabled:
            return None
        try:
            shot = await self.driver.screenshot(full_page=False)
            elements = await self.driver.query_interactive_elements()
            som_bytes = annotate_and_encode(shot, elements)
            goal = f"{self.task} — locate the element '{target.name or target.text or target.describe()}'"
            result = await self.vlm.perceive(shot, som_bytes, goal, elements)
            if result and result.element_id >= 0:
                return ElementTarget(index=result.element_id, strategy="index")
        except Exception:
            return None
        return None

    async def _css_for_index(self, index: int) -> Optional[str]:
        try:
            elements = await self.driver.query_interactive_elements()
            for e in elements:
                if e.index == index:
                    return e.css
        except Exception:
            pass
        return None
