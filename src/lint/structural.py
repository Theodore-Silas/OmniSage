"""
Structural linter: detects dangling links, inconsistent indexes, orphan pages.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Set

from src.wiki.manager import WikiManager, WikiPage


@dataclass
class LintIssue:
    """A single lint finding."""
    page_path: str
    issue_type: str       # dangling-link | incomplete-page | orphan-page | index-gap | format-error
    severity: str         # warning | error
    description: str
    fix_suggestion: str = ""


class StructuralLinter:
    """Validates Wiki structural integrity."""

    def __init__(self, wiki: WikiManager = None):
        self.wiki = wiki or WikiManager()

    def lint_all(self) -> List[LintIssue]:
        issues = []
        all_pages = self.wiki.list_pages()
        all_paths = set(all_pages)

        for page_path in all_pages:
            page = self.wiki.load_page(page_path)
            if page is None:
                issues.append(LintIssue(
                    page_path=page_path, issue_type="format-error", severity="error",
                    description="Page exists but cannot be parsed",
                ))
                continue

            # Check: dangling wikilinks (skip URLs)
            for link in page.related_pages + page.sources:
                link_path = link.strip()
                if not link_path:
                    continue
                if link_path.startswith("http://") or link_path.startswith("https://"):
                    continue  # skip external URLs
                if link_path not in all_paths:
                    issues.append(LintIssue(
                        page_path=page_path, issue_type="dangling-link", severity="warning",
                        description=f"Link [[{link_path}]] points to non-existent page",
                        fix_suggestion=f"Create {link_path} or remove this link",
                    ))

            # Check: incomplete page (missing required sections)
            missing = []
            if not page.summary:
                missing.append("summary (blockquote)")
            if not page.key_facts and page.page_type != "source":
                missing.append("Key Facts")
            if not page.sources:
                missing.append("Sources")
            if missing:
                issues.append(LintIssue(
                    page_path=page_path, issue_type="incomplete-page", severity="warning",
                    description=f"Missing sections: {', '.join(missing)}",
                    fix_suggestion="Use wiki_periodic_fix to complete this page",
                ))

        # Check: orphan pages (no incoming links)
        linked_to: Set[str] = set()
        for page_path in all_pages:
            page = self.wiki.load_page(page_path)
            if page:
                for link in page.related_pages + page.sources:
                    linked_to.add(link.strip())

        for page_path in all_pages:
            if page_path not in linked_to and not page_path.startswith("sources/"):
                # Exclude index.md, log.md, and source snapshots
                if page_path not in ("index.md", "log.md"):
                    issues.append(LintIssue(
                        page_path=page_path, issue_type="orphan-page", severity="warning",
                        description="No other pages link to this page",
                        fix_suggestion="Add wikilinks from related concept/entity pages",
                    ))

        # Check: index.md consistency
        idx_path = os.path.join(self.wiki.root, "index.md")
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                idx_content = f.read()
            indexed_pages = set(re.findall(r"\[\[(.+?)\]\]", idx_content))
            non_source_pages = {p for p in all_paths if not p.startswith("sources/")}
            missing_from_idx = non_source_pages - indexed_pages - {"index.md", "log.md", "AGENTS.md", "error_book.yaml"}
            for p in missing_from_idx:
                issues.append(LintIssue(
                    page_path="index.md", issue_type="index-gap", severity="warning",
                    description=f"Page {p} exists but is not listed in index.md",
                ))

        return issues
