"""
SearchAgent Configuration
Supports DeepSeek, Qwen (OpenAI-compatible APIs), and OpenAI.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "deepseek"  # deepseek | qwen | openai
    model: str = "deepseek-chat"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        configs = {
            "deepseek": {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            },
            "qwen": {
                "model": os.getenv("QWEN_MODEL", "qwen-plus"),
                "api_key": os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
                "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            },
            "openai": {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            },
        }
        cfg = configs.get(provider, configs["deepseek"])
        return cls(
            provider=provider,
            model=cfg["model"],
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        )


@dataclass
class SearchConfig:
    """Search configuration."""
    max_results_per_source: int = 5
    max_fetch_results: int = 10       # top-K after dedup+rank
    request_timeout: int = 10         # seconds per HTTP request
    max_concurrent_fetch: int = 5     # async semaphore
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )


@dataclass
class SummarizeConfig:
    """Summarization configuration."""
    max_chunk_tokens: int = 3000      # per-source summary max input tokens
    summary_max_tokens: int = 800     # per-source summary output
    synthesize_max_tokens: int = 2000  # final synthesis output


@dataclass
class AgenticConfig:
    """Agentic RAG (v3.0) configuration."""
    max_tool_calls: int = 12              # max tool calls per query
    max_iterations: int = 20              # max ReAct loop iterations
    empty_search_limit: int = 3           # consecutive empty searches → stop
    sufficiency_min_confidence: float = 0.7  # threshold for evidence sufficiency
    context_compress_threshold: int = 6   # observations before compression kicks in
    enable_self_correction: bool = True   # agent can re-search on contradictions


@dataclass
class BrowserConfig:
    """Browser Agent (v4.0) configuration."""
    driver: str = "playwright"       # playwright (首选) | selenium (备选)
    browser_type: str = "chromium"   # chromium | firefox | webkit | chrome
    channel: str = ""                # "" (bundled chromium) | "chrome" | "msedge" (system browser)
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    default_timeout: int = 15000     # ms, per-action auto-wait
    navigation_timeout: int = 30000  # ms
    max_steps: int = 30              # max atomic actions per task
    screenshot_quality: int = 70     # JPEG quality for VLM screenshots
    allowed_domains: List[str] = field(default_factory=list)  # empty = no restriction
    persist_session: bool = False    # keep browser open across tasks (reuse cookies/state)
    user_agent: str = ""             # empty = browser default

    @classmethod
    def from_env(cls) -> "BrowserConfig":
        domains = os.getenv("BROWSER_ALLOWED_DOMAINS", "")
        return cls(
            driver=os.getenv("BROWSER_DRIVER", "playwright"),
            browser_type=os.getenv("BROWSER_TYPE", "chromium"),
            channel=os.getenv("BROWSER_CHANNEL", ""),
            headless=os.getenv("BROWSER_HEADLESS", "1").lower() in ("1", "true", "yes"),
            viewport_width=int(os.getenv("BROWSER_VIEWPORT_WIDTH", "1280")),
            viewport_height=int(os.getenv("BROWSER_VIEWPORT_HEIGHT", "800")),
            default_timeout=int(os.getenv("BROWSER_TIMEOUT", "15000")),
            navigation_timeout=int(os.getenv("BROWSER_NAV_TIMEOUT", "30000")),
            max_steps=int(os.getenv("BROWSER_MAX_STEPS", "30")),
            screenshot_quality=int(os.getenv("BROWSER_SCREENSHOT_QUALITY", "70")),
            allowed_domains=[d.strip() for d in domains.split(",") if d.strip()],
            persist_session=os.getenv("BROWSER_PERSIST_SESSION", "0").lower() in ("1", "true", "yes"),
            user_agent=os.getenv("BROWSER_USER_AGENT", ""),
        )


@dataclass
class VLMConfig:
    """Visual language model configuration (v4.0 perception channel B).

    provider: "qwen-vl" | "openai" | "local" | "none"
    When "none" (no vision key configured), the browser agent falls back
    to the DOM/a11y channel only — VLM is an optional enhancement.
    """
    provider: str = "none"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    local_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # for provider == "local"

    @classmethod
    def from_env(cls) -> "VLMConfig":
        # Auto-detect: openai key → openai, qwen/dashscope key → qwen-vl, else none
        openai_key = os.getenv("OPENAI_API_KEY", "")
        qwen_key = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
        provider = os.getenv("VLM_PROVIDER", "").strip().lower()

        if not provider:
            if openai_key:
                provider = "openai"
            elif qwen_key:
                provider = "qwen-vl"
            else:
                provider = "none"

        if provider == "openai":
            model = os.getenv("VLM_MODEL", "gpt-4o")
            api_key = openai_key
            base_url = os.getenv("VLM_BASE_URL", "https://api.openai.com/v1")
        elif provider == "qwen-vl":
            model = os.getenv("VLM_MODEL", "qwen-vl-max")
            api_key = qwen_key
            base_url = os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        elif provider == "local":
            model = os.getenv("VLM_MODEL", "")
            api_key = ""
            base_url = ""
        else:
            model, api_key, base_url = "", "", ""

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=float(os.getenv("VLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("VLM_MAX_TOKENS", "1024")),
            local_model=os.getenv("VLM_LOCAL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        )

    @property
    def enabled(self) -> bool:
        return self.provider not in ("none", "")


@dataclass
class AppConfig:
    """Application-level configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    search: SearchConfig = field(default_factory=SearchConfig)
    summarize: SummarizeConfig = field(default_factory=SummarizeConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    enable_sources: List[str] = field(default_factory=lambda: ["web", "paper"])
    verbose: bool = False
    mode: str = "fast"  # "fast" | "agentic" | "browser"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            llm=LLMConfig.from_env(),
            browser=BrowserConfig.from_env(),
            vlm=VLMConfig.from_env(),
            enable_sources=os.getenv("ENABLE_SOURCES", "web,paper").split(","),
            verbose=os.getenv("VERBOSE", "").lower() in ("1", "true", "yes"),
            mode=os.getenv("SEARCH_MODE", "fast"),
        )
