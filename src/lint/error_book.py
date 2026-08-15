"""
Error Book: persistent error tracking with 5-stage lifecycle.
Based on LLM-Wiki paper (Ming et al., 2026).
"""

import os
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional


class ErrorBook:
    """
    Persistent error tracking for the Wiki knowledge base.

    5-stage lifecycle:
    1. Discover  — detection validates pages, finds errors
    2. Attribute — trace each error to root cause
    3. Constrain — convert root cause into natural language constraint
    4. Inject    — append active constraints to agent prompts
    5. Verify    — re-check periodically, close if fixed

    Stored as error_book.yaml in the wiki root.
    """

    def __init__(self, wiki_root: str = None):
        if wiki_root is None:
            wiki_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "wiki",
            )
        self.wiki_root = wiki_root
        self.path = os.path.join(wiki_root, "error_book.yaml")
        self.entries: List[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.entries = data.get("entries", [])
        else:
            self.entries = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            yaml.dump({"entries": self.entries}, f, allow_unicode=True, default_flow_style=False)

    # ── Lifecycle Stage 1: Discover ──────────────────────────

    def discover(self, lint_issues: list) -> int:
        """
        Register newly discovered errors.
        Returns count of new unique errors added.
        """
        added = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for issue in lint_issues:
            # Deduplicate by page_path + issue_type combination
            key = f"{issue.page_path}|{getattr(issue, 'issue_type', 'lint')}"
            if any(e.get("key") == key for e in self.entries):
                continue

            entry = {
                "id": f"err-{len(self.entries) + added + 1:03d}",
                "key": key,
                "type": getattr(issue, "issue_type", "lint"),
                "severity": getattr(issue, "severity", "warning"),
                "page": issue.page_path,
                "description": getattr(issue, "description", str(issue)),
                "discovered": now,
                "status": "open",
                "root_cause": "",
                "constraint": "",
                "verified_at": "",
                "fix_count": 0,
            }
            self.entries.append(entry)
            added += 1

        if added:
            self._save()
        return added

    # ── Lifecycle Stage 2-3: Attribute + Constrain ──────────

    def attribute_and_constrain(self, entry_id: str, root_cause: str, constraint: str):
        """Attribute root cause and generate a constraint rule."""
        for e in self.entries:
            if e["id"] == entry_id:
                e["root_cause"] = root_cause
                e["constraint"] = constraint
                e["status"] = "constrained"
                self._save()
                return True
        return False

    def auto_attribute(self):
        """Auto-generate constraints for common error types."""
        patterns = {
            "dangling-link": {
                "root_cause": "LLM generated a wikilink to a non-existent page during summarization",
                "constraint": "NEVER create a wikilink [[...]] unless the target page is explicitly listed in index.md",
            },
            "incomplete-page": {
                "root_cause": "Summary generation did not include all required sections",
                "constraint": "Every page MUST include: summary blockquote, Key Facts section, and Sources section",
            },
            "orphan-page": {
                "root_cause": "New page was created but no existing pages were updated to link to it",
                "constraint": "When creating a new page, also update at least one existing related page to add a wikilink to the new page",
            },
            "index-gap": {
                "root_cause": "Page file was saved but the index.md was not updated",
                "constraint": "Every time a new page is saved, index.md MUST be updated with an [[link]] to the new page",
            },
        }

        updated = 0
        for e in self.entries:
            if e["status"] == "open" and e["type"] in patterns:
                pat = patterns[e["type"]]
                e["root_cause"] = pat["root_cause"]
                e["constraint"] = pat["constraint"]
                e["status"] = "constrained"
                updated += 1
        if updated:
            self._save()
        return updated

    # ── Lifecycle Stage 4: Inject ────────────────────────────

    def get_active_constraints(self) -> str:
        """Get all active constraints as a prompt snippet."""
        active = [e for e in self.entries if e["status"] in ("constrained", "open") and e.get("constraint")]
        if not active:
            return ""

        lines = ["## Error Book Constraints (MUST follow to avoid past errors)"]
        for e in sorted(active, key=lambda x: x["severity"]):
            lines.append(f"- [{e['severity'].upper()}] {e['constraint']}")
        return "\n".join(lines)

    # ── Lifecycle Stage 5: Verify ────────────────────────────

    def verify_and_close(self, entry_id: str, fixed: bool = True):
        """Verify if an error has been fixed, and close it."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for e in self.entries:
            if e["id"] == entry_id:
                if fixed:
                    e["status"] = "closed"
                    e["verified_at"] = now
                else:
                    e["fix_count"] = e.get("fix_count", 0) + 1
                self._save()
                return True
        return False

    def periodic_verify(self, current_pages: List[str]) -> int:
        """Re-check all open errors against current Wiki state. Close fixed ones."""
        closed = 0
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        for e in self.entries:
            if e["status"] == "closed":
                continue
            # Simple check: if the page no longer exists, the error is resolved
            if e["page"] not in current_pages and e.get("page") != "index.md":
                e["status"] = "closed"
                e["verified_at"] = now
                closed += 1

        if closed:
            self._save()
        return closed

    # ── Stats ────────────────────────────────────────────────

    @property
    def open_count(self) -> int:
        return sum(1 for e in self.entries if e["status"] != "closed")

    @property
    def total_count(self) -> int:
        return len(self.entries)
