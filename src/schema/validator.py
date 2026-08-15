"""
Schema validator: checks wiki pages against AGENTS.md quality standards.
"""

from dataclasses import dataclass, field
from typing import List

from src.wiki.manager import WikiPage


@dataclass
class ValidationResult:
    """Result of a single page validation."""
    page_path: str
    passed: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SchemaValidator:
    """Validates Wiki pages against schema-defined quality standards."""

    def validate_page(self, page: WikiPage) -> ValidationResult:
        """Validate a single WikiPage against quality standards."""
        result = ValidationResult(page_path=page.path, passed=True)

        # Rule 1: Every claim must be traceable to a source
        if not page.sources and page.page_type not in ("source",):
            result.issues.append("No sources listed — claims are not traceable")

        # Rule 2: Contradictions must be explicitly marked
        # (We can't auto-detect this, but we flag if the section is missing and page is high-confidence)
        if page.confidence < 0.5 and page.status != "draft":
            result.issues.append(
                f"Confidence {page.confidence} should be marked status: draft"
            )

        # Rule 3: Use wikilinks for cross-page references
        if page.related_pages:
            for link in page.related_pages:
                if not link.endswith(".md"):
                    result.warnings.append(
                        f"Related page link '{link}' should use .md extension"
                    )

        # Rule 4: All required sections should be present
        required_sections = ["Key Facts", "Sources"]
        missing = [
            s for s in required_sections
            if not self._has_section(page, s)
        ]
        if missing:
            result.warnings.append(f"Missing sections: {', '.join(missing)}")

        # Rule 5: Tags should be lowercase, no spaces
        if page.tags:
            bad_tags = [t for t in page.tags if " " in t or t != t.lower()]
            if bad_tags:
                result.warnings.append(f"Tags should be lowercase, no spaces: {bad_tags}")

        result.passed = len(result.issues) == 0
        return result

    def validate_all(self, pages: List[WikiPage]) -> List[ValidationResult]:
        """Validate multiple pages."""
        return [self.validate_page(p) for p in pages]

    @staticmethod
    def _has_section(page: WikiPage, section_name: str) -> bool:
        """Check if a page has a non-empty section."""
        if section_name == "Key Facts":
            return len(page.key_facts) > 0
        if section_name == "Sources":
            return len(page.sources) > 0
        if section_name == "Analysis":
            return len(page.analysis) > 50
        if section_name == "Related Pages":
            return len(page.related_pages) > 0
        return True

    def auto_fix_page(self, page: WikiPage) -> WikiPage:
        """Auto-fix common issues in a page to meet schema standards."""
        result = self.validate_page(page)

        # If confidence < 0.5, set status to draft
        if page.confidence < 0.5:
            page.status = "draft"

        # Normalize tags
        page.tags = [t.lower().replace(" ", "-") for t in page.tags]

        # If no sources, lower confidence
        if not page.sources and page.page_type != "source":
            page.confidence = min(page.confidence, 0.4)

        return page
