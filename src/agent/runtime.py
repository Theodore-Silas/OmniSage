"""
统一 Agent 运行时 (v5.0)。

装配一个 Agent 所需的全部资源，按需惰性初始化：

  知识侧  — SearchRegistry（4 源）/ ContentFetcher / WikiManager
  浏览器侧 — Playwright/Selenium Driver（LAZY 启动）/ VLM / LocatorResolver /
            ActionExecutor
  决策侧  — GlobalPlanner / LocalExecutor / SubtaskVerifier
  控制侧  — Watchdog / SafetyGuard

浏览器仅在首次调用 browser_* 工具时才 launch（ensure_browser），
纯搜索任务零浏览器开销。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.browser.action.primitives import ActionExecutor
from src.browser.driver.base import BrowserDriver
from src.browser.driver.playwright_driver import PlaywrightDriver
from src.browser.driver.selenium_driver import SeleniumDriver
from src.browser.exception.escalation import Watchdog
from src.browser.locator.cache import LocatorCache
from src.browser.locator.resolver import LocatorResolver
from src.browser.perception.vlm import VLMClient
from src.browser.planning.executor import LocalExecutor
from src.browser.planning.planner import GlobalPlanner
from src.browser.planning.verifier import SubtaskVerifier
from src.browser.safety.guard import SafetyGuard
from src.agentic.context_manager import ContextManager
from src.agentic.sufficiency import SufficiencyChecker
from src.search.ranker import ResultRanker
from src.summarize.summarizer import LLMSummarizer
from src.config import AppConfig


def create_driver(config: AppConfig) -> BrowserDriver:
    """Factory: build the configured driver (playwright by default)."""
    bc = config.browser
    if bc.driver == "selenium":
        return SeleniumDriver(
            browser_type=bc.browser_type,
            headless=bc.headless,
            viewport_width=bc.viewport_width,
            viewport_height=bc.viewport_height,
            default_timeout=bc.default_timeout,
            navigation_timeout=bc.navigation_timeout,
        )
    return PlaywrightDriver(
        browser_type=bc.browser_type,
        headless=bc.headless,
        viewport_width=bc.viewport_width,
        viewport_height=bc.viewport_height,
        default_timeout=bc.default_timeout,
        navigation_timeout=bc.navigation_timeout,
        user_agent=bc.user_agent,
        channel=bc.channel,
    )


class AgentRuntime:
    """Unified runtime holding all resources the agent nodes need.

    ``confirm_callback`` (optional) is an async ``fn(action, params, hint) ->
    bool`` used as the human confirmation gate for destructive actions.
    """

    def __init__(self, config: AppConfig, confirm_callback: Optional[Callable] = None):
        self.config = config
        self.confirm_callback = confirm_callback

        # ── 浏览器资源（lazy）──
        self._driver: Optional[BrowserDriver] = None
        self.vlm = VLMClient(config.vlm)
        self.guard = SafetyGuard(config.browser.allowed_domains)
        self.cache = LocatorCache()
        self._resolver: Optional[LocatorResolver] = None
        self._browser_executor: Optional[ActionExecutor] = None

        # ── 决策侧 ──
        self.planner = GlobalPlanner(config.llm)
        self.local = LocalExecutor(config.llm)
        self.verifier = SubtaskVerifier(config.llm)

        # ── 检索 / 总结 / 上下文（v6.0 接入）──
        self.ranker = ResultRanker()
        self.summarizer = LLMSummarizer(config.llm, config.summarize)
        self.sufficiency = SufficiencyChecker(config.llm)
        self.context = ContextManager(config.llm)
        self.evidence: list = []    # 累积的 SearchResult（融合检索）
        self.fetched: list = []     # 累积的 FetchedContent（read_page）
        self._last_task = ""        # 供 synthesize 回填 query

        # ── 控制侧 ──
        self.watchdog = Watchdog(max_steps=config.browser.max_steps)

    # ── 浏览器 lazy 启动 ──────────────────────────────────────────

    async def ensure_browser(self) -> ActionExecutor:
        """Lazily launch the browser (if needed) and return the action executor."""
        if self._driver is None:
            self._driver = create_driver(self.config)
            await self._driver.launch()
            self._resolver = LocatorResolver(self._driver, self.vlm, self.cache, task="")
            self._browser_executor = ActionExecutor(
                self._driver, self._resolver, self.guard, self.vlm, task=""
            )
        return self._browser_executor

    @property
    def driver(self) -> Optional[BrowserDriver]:
        return self._driver

    @property
    def browser_alive(self) -> bool:
        return self._driver is not None and self._driver.is_alive

    def set_task(self, task: str) -> None:
        """Propagate the current task to resolver/executor (for VLM prompts)."""
        self._last_task = task
        if self._resolver is not None:
            self._resolver.task = task
        if self._browser_executor is not None:
            self._browser_executor.task = task

    # ── 生命周期 ──────────────────────────────────────────────────

    async def stop(self) -> None:
        """Tear down the browser (if launched). Safe to call repeatedly."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._resolver = None
            self._browser_executor = None
