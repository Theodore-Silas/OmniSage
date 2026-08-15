"""
Browser agent exception taxonomy (v4.0).

Four-layer error classification used by the escalation ladder, the watchdog
and the ErrorBook integration. Layers:

  L0 browser  — launch / crash / navigation / network / timeout
  L1 element  — not found / stale / obscured / not visible
  L2 action   — JS blocked / form validation / unexpected navigation
  L3 task     — dead loop / plan stuck / unrecoverable

Each exception carries a stable `code` and a `recoverable` flag so the
escalation ladder can pick the right recovery strategy without isinstance
cascades.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


LEVEL_BROWSER = "browser"
LEVEL_ELEMENT = "element"
LEVEL_ACTION = "action"
LEVEL_TASK = "task"

# Escalation priority: recover the cheapest error first (lower == earlier).
LEVEL_ORDER = {
    LEVEL_ELEMENT: 0,
    LEVEL_ACTION: 1,
    LEVEL_BROWSER: 2,
    LEVEL_TASK: 3,
}


class BrowserAgentError(Exception):
    """Base class for all browser-agent errors."""

    level: str = LEVEL_BROWSER
    code: str = "browser_agent_error"
    recoverable: bool = True

    def __init__(self, message: str = "", detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message or self.code
        self.detail = detail or {}
        # Tracks how far the escalation ladder has gone for this error.
        self.escalation_stage: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "detail": self.detail,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(code={self.code}, message={self.message!r})"


# ─────────────────────────────────────────────────────────────
# L0 — browser level
# ─────────────────────────────────────────────────────────────

class LaunchError(BrowserAgentError):
    level = LEVEL_BROWSER
    code = "launch_error"
    recoverable = False


class BrowserCrashError(BrowserAgentError):
    level = LEVEL_BROWSER
    code = "browser_crash"
    recoverable = True


class NavigationError(BrowserAgentError):
    level = LEVEL_BROWSER
    code = "navigation_error"
    recoverable = True


class NetworkError(BrowserAgentError):
    level = LEVEL_BROWSER
    code = "network_error"
    recoverable = True


class PageTimeoutError(BrowserAgentError):
    level = LEVEL_BROWSER
    code = "page_timeout"
    recoverable = True


# ─────────────────────────────────────────────────────────────
# L1 — element level
# ─────────────────────────────────────────────────────────────

class ElementNotFoundError(BrowserAgentError):
    level = LEVEL_ELEMENT
    code = "element_not_found"
    recoverable = True


class ElementStaleError(BrowserAgentError):
    level = LEVEL_ELEMENT
    code = "element_stale"
    recoverable = True


class ElementObscuredError(BrowserAgentError):
    level = LEVEL_ELEMENT
    code = "element_obscured"
    recoverable = True


class ElementNotVisibleError(BrowserAgentError):
    level = LEVEL_ELEMENT
    code = "element_not_visible"
    recoverable = True


# ─────────────────────────────────────────────────────────────
# L2 — action level
# ─────────────────────────────────────────────────────────────

class ActionBlockedError(BrowserAgentError):
    level = LEVEL_ACTION
    code = "action_blocked"
    recoverable = True


class FormValidationError(BrowserAgentError):
    level = LEVEL_ACTION
    code = "form_validation_failed"
    recoverable = True


class UnexpectedNavigationError(BrowserAgentError):
    level = LEVEL_ACTION
    code = "unexpected_navigation"
    recoverable = True


# ─────────────────────────────────────────────────────────────
# L3 — task level
# ─────────────────────────────────────────────────────────────

class DeadLoopError(BrowserAgentError):
    level = LEVEL_TASK
    code = "dead_loop"
    recoverable = True


class PlanStuckError(BrowserAgentError):
    level = LEVEL_TASK
    code = "plan_stuck"
    recoverable = True


class TaskUnrecoverableError(BrowserAgentError):
    level = LEVEL_TASK
    code = "task_unrecoverable"
    recoverable = False


# ─────────────────────────────────────────────────────────────
# Classification helpers
# ─────────────────────────────────────────────────────────────

# Heuristic mapping from raw exception message substrings → error class.
_MESSAGE_HINTS = (
    (("element", "not found", "no such element", "cannot find", "找不到", "not attached"), ElementNotFoundError),
    (("stale", "detached", "no longer in the dom"), ElementStaleError),
    (("obscured", "not clickable", "intercepts pointer", "hidden", "not visible", "invisible"), ElementObscuredError),
    (("timeout", "timed out", "wait until"), PageTimeoutError),
    (("navigat", "url", "redirect"), NavigationError),
    (("net", "dns", "connection", "refused", "err_"), NetworkError),
    (("crash", "crashed", "closed", "disconnected", "session deleted"), BrowserCrashError),
    (("validat", "invalid", "required"), FormValidationError),
    (("blocked", "forbidden", "denied", "captcha", "verification"), ActionBlockedError),
)


def classify_exception(exc: Exception) -> BrowserAgentError:
    """Wrap an arbitrary exception into a typed :class:`BrowserAgentError`.

    Already-typed errors pass through unchanged. Unknown exceptions are
    classified heuristically by message content; unmatched ones default to
    a recoverable L0 ``BrowserAgentError``.
    """
    if isinstance(exc, BrowserAgentError):
        return exc

    msg = str(exc).lower()
    for hints, cls in _MESSAGE_HINTS:
        if any(h in msg for h in hints):
            return cls(str(exc)[:300])

    return BrowserAgentError(str(exc)[:300] or "unknown browser error")


def level_of(exc: Exception) -> str:
    """Return the layer tag (browser/element/action/task) for an exception."""
    return classify_exception(exc).level


def is_recoverable(exc: Exception) -> bool:
    return classify_exception(exc).recoverable
