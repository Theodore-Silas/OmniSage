"""
Schema loader: reads AGENTS.md and injects governance rules into agent prompts.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class SchemaRules:
    """Parsed rules from AGENTS.md for injection into LLM prompts."""
    page_template: str = ""
    quality_standards: List[str] = field(default_factory=list)
    workflow_rules: List[str] = field(default_factory=list)
    directory_structure: str = ""

    def to_system_prompt(self) -> str:
        """Convert rules into a condensed system prompt for LLM context."""
        parts = []

        if self.quality_standards:
            parts.append("## Quality Standards (MUST follow)")
            for i, rule in enumerate(self.quality_standards, 1):
                clean_rule = re.sub(r"^\d+\.\s*", "", rule)
                parts.append(f"{i}. {clean_rule}")

        if self.workflow_rules:
            parts.append("\n## Workflow Rules")
            for rule in self.workflow_rules:
                parts.append(f"- {rule}")

        if self.page_template:
            parts.append(f"\n## Output Format\n{self.page_template}")

        return "\n".join(parts)


class SchemaLoader:
    """Loads and parses AGENTS.md for agent governance."""

    def __init__(self, wiki_root: str = None):
        if wiki_root is None:
            wiki_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "wiki",
            )
        self.wiki_root = wiki_root
        self.schema_path = os.path.join(wiki_root, "AGENTS.md")
        self.rules: SchemaRules = SchemaRules()

    def load(self) -> SchemaRules:
        """Load and parse AGENTS.md."""
        if not os.path.exists(self.schema_path):
            return self.rules  # return empty rules

        with open(self.schema_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._parse(content)
        return self.rules

    def _parse(self, content: str):
        """Parse AGENTS.md into structured rules."""
        # Extract page template (between ```markdown ... ```)
        import re
        tmpl_match = re.search(r"```markdown\n(.*?)```", content, re.DOTALL)
        if tmpl_match:
            self.rules.page_template = tmpl_match.group(1).strip()

        # Extract quality standards (numbered list under "### Quality Standards")
        qs_match = re.search(
            r"### Quality Standards\n((?:\d+\..*\n?)+)", content
        )
        if qs_match:
            self.rules.quality_standards = [
                line.strip()
                for line in qs_match.group(1).strip().split("\n")
                if line.strip()
            ]

        # Extract workflow rules (under "### Search" and "### Persist")
        for section in ["### Search", "### Persist"]:
            # Find the section and get its numbered list
            pattern = rf"{section}\n((?:\d+\..*\n?)+)"
            m = re.search(pattern, content)
            if m:
                self.rules.workflow_rules.extend([
                    line.strip() for line in m.group(1).strip().split("\n") if line.strip()
                ])

        # Extract directory structure
        ds_match = re.search(r"## Directory Structure\n((?:- .*\n?)+)", content)
        if ds_match:
            self.rules.directory_structure = ds_match.group(1).strip()


# Singleton
_schema_loader: SchemaLoader = None


def get_schema_rules() -> SchemaRules:
    """Get parsed schema rules (lazy singleton)."""
    global _schema_loader
    if _schema_loader is None:
        _schema_loader = SchemaLoader()
        _schema_loader.load()
    return _schema_loader.rules


def get_system_prompt_extension() -> str:
    """Get schema rules + Error Book constraints as a system prompt for LLM calls."""
    parts = [get_schema_rules().to_system_prompt()]

    # Inject Error Book active constraints
    try:
        from src.lint.error_book import ErrorBook
        eb = ErrorBook()
        eb_constraints = eb.get_active_constraints()
        if eb_constraints:
            parts.append("\n" + eb_constraints)
    except Exception:
        pass

    return "\n".join(parts)
