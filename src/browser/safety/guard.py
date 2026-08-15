"""
Safety guard (v4.0) — action allowlist, domain boundary, sensitive-data masking.

Browser agents can perform real, irreversible actions, so this guard enforces:

  1. action allowlist (unknown/risky actions like raw JS are rejected)
  2. domain boundary for navigation (optional allowlist)
  3. sensitive-field masking in logs/traces

Destructive semantic actions (pay/delete/submit) are surfaced through the
``destructive_hint`` so the control layer can open a human confirmation gate.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse

ALLOWED_ACTIONS = {
    "navigate", "go_back", "click", "type", "press", "hover", "select",
    "scroll", "drag", "switch_tab", "read_a11y", "screenshot", "extract",
    "wait", "exec_js",
}

SENSITIVE_FIELDS = {
    "password", "passwd", "pwd", "card", "card_number", "credit", "cvv",
    "secret", "token", "api_key", "apikey", "access_token",
}

# Forbidden APIs in the sandboxed exec_js tool.
FORBIDDEN_JS = (
    "fetch(", "XMLHttpRequest", "localStorage", "sessionStorage",
    "document.cookie", "sendBeacon", "form.submit", "window.open",
    "WebSocket", "eval(",
)

# Substrings that hint at a destructive/high-stakes action.
DESTRUCTIVE_HINTS = (
    "submit", "pay", "payment", "checkout", "confirm", "delete", "remove",
    "logout", "sign_out", "sign out", "buy", "purchase", "send", "post",
)


class SafetyGuard:
    """Enforces browser-agent safety boundaries."""

    def __init__(self, allowed_domains: List[str] = None):
        self.allowed_domains = [d.lower().strip() for d in (allowed_domains or []) if d.strip()]

    def check(self, action: str, params: Dict[str, Any]) -> bool:
        """Return True if the action is permitted."""
        if action not in ALLOWED_ACTIONS:
            return False
        if action == "navigate":
            url = str(params.get("url", ""))
            if not self._url_allowed(url):
                return False
        if action == "exec_js":
            script = str(params.get("script", ""))
            if not self.check_exec_js(script):
                return False
        return True

    def check_exec_js(self, script: str) -> bool:
        """Reject scripts that touch network/storage/identity-sensitive APIs."""
        lowered = script.lower()
        return not any(frag.lower() in lowered for frag in FORBIDDEN_JS)

    def destructive_hint(self, action: str, params: Dict[str, Any]) -> str:
        """Return a human-readable reason if the action looks destructive, else ''."""
        text = " ".join(str(v) for v in params.values()).lower()
        for hint in DESTRUCTIVE_HINTS:
            if hint in text:
                return f"action '{action}' may be destructive (matched '{hint}')"
        return ""

    def mask_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of params with sensitive values masked."""
        return {
            k: ("***" if k.lower() in SENSITIVE_FIELDS else v)
            for k, v in (params or {}).items()
        }

    def _url_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        host = (urlparse(url).netloc or "").lower()
        if not host:
            return True
        return any(host == d or host.endswith("." + d) for d in self.allowed_domains)
