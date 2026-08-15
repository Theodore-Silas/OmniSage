"""
Escalation ladder + watchdog (v4.0).

The seven-stage escalation ladder is applied by the control layer when an
action fails. The watchdog detects dead loops (repeated identical action on
the same page state) and raises :class:`DeadLoopError` to force re-planning.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, List, Optional, Tuple

from src.browser.exception.taxonomy import (
    BrowserAgentError,
    DeadLoopError,
    classify_exception,
    level_of,
)

# The seven escalation stages, in order.
STAGE_RETRY = "retry"
STAGE_RELOCATE = "relocate"
STAGE_REPERCEIVE = "reperceive"
STAGE_WAIT = "wait"
STAGE_ALTERNATIVE = "alternative"
STAGE_HUMAN = "human"
STAGE_GIVE_UP = "give_up"

ESCALATION_LADDER = [
    STAGE_RETRY,
    STAGE_RELOCATE,
    STAGE_REPERCEIVE,
    STAGE_WAIT,
    STAGE_ALTERNATIVE,
    STAGE_HUMAN,
    STAGE_GIVE_UP,
]

# Per-layer default entry point into the ladder.
_LAYER_ENTRY = {
    "element": STAGE_RELOCATE,   # element errors skip plain retry → try re-locating
    "action": STAGE_RETRY,
    "browser": STAGE_RETRY,
    "task": STAGE_ALTERNATIVE,
}


def suggest_next_stage(error: BrowserAgentError, current_stage: Optional[str]) -> str:
    """Return the next escalation stage after ``current_stage``.

    Errors start at their layer's natural entry point; each call advances one
    step down the ladder until ``give_up``.
    """
    if current_stage is None:
        return _LAYER_ENTRY.get(error.level, STAGE_RETRY)

    if current_stage == STAGE_GIVE_UP:
        return STAGE_GIVE_UP

    idx = ESCALATION_LADDER.index(current_stage)
    return ESCALATION_LADDER[min(idx + 1, len(ESCALATION_LADDER) - 1)]


async def retry_with_backoff(
    coro_factory: Callable[[], Awaitable],
    retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    retryable: Optional[Callable[[Exception], bool]] = None,
):
    """Retry an async operation with exponential backoff.

    ``retryable`` filters which exceptions are worth retrying; by default
    anything recoverable is retried.
    """
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last = e
            err = classify_exception(e)
            if attempt >= retries or not err.recoverable:
                raise
            if retryable is not None and not retryable(e):
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]


class Watchdog:
    """Detects dead loops by tracking repeated identical action signatures.

    A "signature" is a stable string identifying one concrete action
    (e.g. ``browser_type::{"text":"hello"}``). Repeating the *exact same*
    action N times in a row is treated as a loop; distinct params (e.g.
    typing different text) do NOT count as a loop.
    """

    def __init__(self, max_repeats: int = 3, max_steps: int = 30):
        self.max_repeats = max_repeats
        self.max_steps = max_steps
        self._history: List[str] = []
        self._step_count = 0

    def observe(self, signature: str) -> None:
        """Record one step; raise DeadLoopError if a loop is detected."""
        self._step_count += 1
        self._history.append(signature or "")

        if self._step_count > self.max_steps:
            raise DeadLoopError(f"Step budget exceeded ({self.max_steps})")

        if len(self._history) >= self.max_repeats:
            recent = self._history[-self.max_repeats:]
            if all(s == recent[0] and s != "" for s in recent):
                raise DeadLoopError(
                    f"Dead loop detected: identical action '{recent[0][:90]}' repeated {self.max_repeats}x"
                )

    def reset(self) -> None:
        self._history.clear()
        self._step_count = 0

    @property
    def steps(self) -> int:
        return self._step_count
