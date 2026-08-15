"""
Wiki page data model and WikiManager for persistent knowledge storage.
"""

import os
import re
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class WikiPage:
    """A single Wiki page in the knowledge base."""

    path: str                          # relative path from wiki/ root, e.g. "concepts/rag-methods.md"
    title: str
    page_type: str                     # concept | entity | query | synthesis | source
    summary: str = ""                  # one-sentence summary (blockquote)
    key_facts: List[str] = field(default_factory=list)
    analysis: str = ""
    related_pages: List[str] = field(default_factory=list)   # [[page]] links
    sources: List[str] = field(default_factory=list)          # [[source-page]] links
    contradictions: str = ""
    tags: List[str] = field(default_factory=list)
    status: str = "draft"
    confidence: float = 0.5
    created: str = ""
    updated: str = ""

    def to_markdown(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not self.created:
            self.created = now
        self.updated = now

        frontmatter = {
            "type": self.page_type,
            "created": self.created,
            "updated": self.updated,
            "tags": self.tags,
            "status": self.status,
            "confidence": self.confidence,
        }
        yaml_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()

        md = f"---\n{yaml_str}\n---\n\n# {self.title}\n\n"
        if self.summary:
            md += f"> {self.summary}\n\n"

        if self.key_facts:
            md += "## Key Facts\n"
            for i, fact in enumerate(self.key_facts, 1):
                md += f"- {fact}\n"
            md += "\n"

        if self.analysis:
            md += "## Analysis\n"
            md += f"{self.analysis}\n\n"

        if self.related_pages:
            md += "## Related Pages\n"
            for p in self.related_pages:
                md += f"- [[{p}]]\n"
            md += "\n"

        if self.sources:
            md += "## Sources\n"
            for s in self.sources:
                md += f"- [[{s}]]\n"
            md += "\n"

        if self.contradictions:
            md += "## Contradictions\n"
            md += f"{self.contradictions}\n\n"

        return md

    @classmethod
    def from_markdown(cls, path: str, content: str) -> "WikiPage":
        page = cls(path=path, title="", page_type="concept")
        page.created = ""
        page.updated = ""

        # Parse frontmatter
        fm_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        body = content
        if fm_match:
            try:
                fm = yaml.safe_load(fm_match.group(1)) or {}
                page.page_type = fm.get("type", "concept")
                page.tags = fm.get("tags", [])
                page.status = fm.get("status", "draft")
                page.confidence = fm.get("confidence", 0.5)
                page.created = str(fm.get("created", ""))
                page.updated = str(fm.get("updated", ""))
            except Exception:
                pass
            body = content[fm_match.end():]

        # Parse sections
        page.title = cls._extract_section(body, r"^# (.+)", "")
        page.summary = cls._extract_section(body, r"^> (.+)", "")
        page.key_facts = cls._extract_list(body, "Key Facts")
        page.analysis = cls._extract_section(body, r"## Analysis\n(.+?)(?=\n## |\Z)", "")
        page.related_pages = cls._extract_wikilinks(body, "Related Pages")
        page.sources = cls._extract_wikilinks(body, "Sources")
        page.contradictions = cls._extract_section(body, r"## Contradictions\n(.+?)(?=\n## |\Z)", "")

        return page

    @staticmethod
    def _extract_section(text: str, pattern: str, default: str) -> str:
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else default

    @staticmethod
    def _extract_list(text: str, section_name: str) -> List[str]:
        pattern = rf"## {section_name}\n((?:- .+\n?)+)"
        m = re.search(pattern, text)
        if not m:
            return []
        return [line.strip("- ").strip() for line in m.group(1).strip().split("\n") if line.startswith("- ")]

    @staticmethod
    def _extract_wikilinks(text: str, section_name: str) -> List[str]:
        pattern = rf"## {section_name}\n((?:- \[\[.+?\]\].*\n?)+)"
        m = re.search(pattern, text)
        if not m:
            return []
        return re.findall(r"\[\[(.+?)\]\]", m.group(1))


class WikiManager:
    """
    Manages the persistent Wiki knowledge base.
    Handles page CRUD, index maintenance, and operation logging.
    """

    def __init__(self, wiki_root: str = None):
        if wiki_root is None:
            wiki_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "wiki",
            )
        self.root = wiki_root
        self._ensure_dirs()

    def _ensure_dirs(self):
        for d in ["concepts", "entities", "queries", "synthesis", "sources"]:
            os.makedirs(os.path.join(self.root, d), exist_ok=True)

    # ── Page CRUD ────────────────────────────────────────────

    def save_page(self, page: WikiPage) -> str:
        """Save a WikiPage to disk. Returns the full file path."""
        filepath = os.path.join(self.root, page.path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page.to_markdown())
        self._update_index(page)
        return filepath

    def load_page(self, rel_path: str) -> Optional[WikiPage]:
        """Load a WikiPage from disk."""
        filepath = os.path.join(self.root, rel_path)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return WikiPage.from_markdown(rel_path, content)

    def list_pages(self, page_type: str = None) -> List[str]:
        """List all Wiki pages, optionally filtered by type."""
        pages = []
        subdirs = [page_type] if page_type else ["concepts", "entities", "queries", "synthesis", "sources"]
        for sub in subdirs:
            d = os.path.join(self.root, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".md"):
                        pages.append(f"{sub}/{f}")
        return sorted(pages)

    def search_pages(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple full-text search across all Wiki pages."""
        results = []
        query_lower = query.lower()
        for page_path in self.list_pages():
            page = self.load_page(page_path)
            if page is None:
                continue
            content = page.to_markdown().lower()
            score = 0
            if query_lower in page.title.lower():
                score += 10
            if query_lower in page.summary.lower():
                score += 5
            # Count keyword matches
            for term in query_lower.split():
                score += content.count(term)
            if score > 0:
                results.append({"path": page_path, "title": page.title, "score": score, "type": page.page_type})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ── Index management ─────────────────────────────────────

    def _update_index(self, page: WikiPage):
        """Update index.md with the new/updated page."""
        idx_path = os.path.join(self.root, "index.md")
        if not os.path.exists(idx_path):
            return

        with open(idx_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Determine which section to update
        section_map = {
            "concept": "## Concepts",
            "entity": "## Entities",
            "query": "## Query Archives",
            "synthesis": "## Syntheses",
            "source": "## Sources",
        }
        section = section_map.get(page.page_type)
        if not section:
            return

        # Build entry line
        line = f"- [[{page.path}]]"
        if page.summary:
            line += f" — {page.summary[:80]}"

        # Check if entry already exists
        if f"[[{page.path}]]" in content:
            return  # already indexed

        # Insert after section header
        pattern = rf"({re.escape(section)}.*?\n)"
        m = re.search(pattern, content)
        if m:
            insert_pos = m.end()
            content = content[:insert_pos] + line + "\n" + content[insert_pos:]

        # Update timestamp
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = re.sub(r"_Last updated:.*_", f"_Last updated: {now}_", content)

        with open(idx_path, "w", encoding="utf-8") as f:
            f.write(content)

    # ── Logging ──────────────────────────────────────────────

    def log_operation(self, operation: str, details: str = "", pages_affected: int = 0):
        """Append an operation entry to log.md."""
        log_path = os.path.join(self.root, "log.md")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        entry = f"## {timestamp} — {operation}\n"
        if details:
            entry += f"{details}\n"
        entry += f"Pages affected: {pages_affected}\n\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # ── Search result archival ───────────────────────────────

    def archive_search(
        self,
        query: str,
        answer: str,
        search_results: List[dict],
        fetched_contents: List[dict],
    ) -> Dict[str, str]:
        """
        Archive a complete search session into the Wiki.
        Returns dict mapping page types to their paths.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower())[:50]
        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archived = {}

        # 1. Save source snapshots
        source_pages = []
        for i, s in enumerate(search_results):
            src_path = f"sources/{date_prefix}-{slug}-src-{i+1}.md"
            src_page = WikiPage(
                path=src_path,
                title=s.get("title", "Untitled")[:100],
                page_type="source",
                summary=s.get("snippet", "")[:200],
                tags=[s.get("source", "web")],
                sources=[s.get("url", "")],
                status="established",
                confidence=0.9,
            )
            self.save_page(src_page)
            source_pages.append(src_path)

        # 2. Save query answer page
        query_path = f"queries/{date_prefix}-{slug}.md"
        query_page = WikiPage(
            path=query_path,
            title=query[:100],
            page_type="query",
            summary=answer.split("\n")[0][:120] if answer else "",
            analysis=answer,
            sources=source_pages,
            tags=["auto-archived"],
            status="established",
            confidence=0.7,
        )
        self.save_page(query_page)
        archived["query"] = query_path

        # 3. Log the operation
        self.log_operation(
            operation="ARCHIVE_SEARCH",
            details=f"Query: {query[:80]}\nSources: {len(search_results)} | Fetched: {len(fetched_contents)}",
            pages_affected=len(source_pages) + 1,
        )

        return archived

    def get_page_count(self) -> int:
        """Get total number of Wiki pages."""
        return len(self.list_pages())
