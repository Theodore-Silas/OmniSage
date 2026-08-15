"""
Obsidian vault 连接器。

把本地 Obsidian 知识库（一个 Markdown 文件夹）接入命中层：
  - 遍历 vault 下所有 .md（自动排除 .obsidian / .trash 等隐藏目录）
  - 全文匹配评分（文件名 / 标题 / 正文关键词 / tag / wikilink）
  - 命中后剥离 YAML frontmatter 返回正文

通过环境变量 OBSIDIAN_VAULT_PATH 指定 vault 路径；未配置时 connector 自动禁用。
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import List, Optional

import yaml

from src.knowledge.base import KnowledgeBaseConnector, KnowledgeHit, strip_frontmatter

# 排除的隐藏目录（Obsidian 内部/回收站等）
_EXCLUDED_DIRS = {".obsidian", ".trash", ".git", "node_modules", ".smart-env"}


class ObsidianConnector(KnowledgeBaseConnector):
    """接入本地 Obsidian vault。"""

    name = "obsidian"

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = (vault_path or "").strip() or os.getenv("OBSIDIAN_VAULT_PATH", "").strip()
        self._notes: List[dict] = []  # [{path, title, tags, content}]

    def enabled(self) -> bool:
        return bool(self.vault_path) and os.path.isdir(self.vault_path)

    # ── 扫描 vault ─────────────────────────────────────────────

    def _scan(self) -> List[dict]:
        """扫描 vault 下所有 .md，返回笔记元数据（惰性缓存）。"""
        if self._notes:
            return self._notes
        notes = []
        for root, dirs, files in os.walk(self.vault_path):
            dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS and not d.startswith(".")]
            for f in files:
                if not f.endswith(".md"):
                    continue
                full = os.path.join(root, f)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                except Exception:
                    continue
                title = f[:-3]  # 文件名去 .md 作为标题
                m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                if m:
                    title = m.group(1).strip()
                tags = re.findall(r"#([\w\u4e00-\u9fff/-]+)", content)
                rel = os.path.relpath(full, self.vault_path).replace("\\", "/")
                notes.append({"path": rel, "title": title, "tags": tags, "content": content})
        self._notes = notes
        return notes

    # ── 检索 ───────────────────────────────────────────────────

    @staticmethod
    def _char_ngrams(text: str, n: int = 2) -> set:
        """字符级 n-gram（中文无空格分词，用 2-gram 做模糊匹配）。"""
        text = re.sub(r"\s+", "", text)
        if len(text) < n:
            return {text} if text else set()
        return {text[i:i + n] for i in range(len(text) - n + 1)}

    def search(self, query: str, top_k: int = 5) -> List[KnowledgeHit]:
        if not self.enabled():
            return []

        q = query.lower().strip()
        grams = self._char_ngrams(q, 2)
        english_terms = [t for t in re.split(r"\s+", q) if len(t) > 1]
        hits: List[KnowledgeHit] = []

        for note in self._scan():
            title_l = note["title"].lower()
            content_l = note["content"].lower()
            score = 0.0

            # 短 query 完全命中标题（强信号）
            if q and q in title_l:
                score += 20
            # 字符 2-gram 与标题/正文的重叠度
            title_overlap = len(grams & self._char_ngrams(title_l, 2))
            score += title_overlap * 2
            for g in grams:
                score += content_l.count(g) * 0.3
            # 英文/数字词匹配
            for term in english_terms:
                score += content_l.count(term)
                if term in title_l:
                    score += 3
            # tag 匹配
            if any(term in " ".join(note["tags"]).lower() for term in english_terms if term):
                score += 2

            if score <= 0:
                continue

            snippet = strip_frontmatter(note["content"])
            snippet = re.sub(r"\n{3,}", "\n\n", snippet).strip()[:300]
            hits.append(KnowledgeHit(
                source=self.name,
                title=note["title"],
                path=note["path"],
                snippet=snippet,
                score=score,
                meta={"tags": note["tags"]},
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    # ── 读取正文 ───────────────────────────────────────────────

    def get_content(self, hit: KnowledgeHit) -> str:
        if not self.enabled():
            return hit.snippet
        full = os.path.join(self.vault_path, hit.path)
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                return strip_frontmatter(fh.read())
        except Exception:
            return hit.snippet

    # ── 写入笔记（搜索沉淀）────────────────────────────────────

    def save_note(
        self,
        title: str,
        content: str,
        sources: Optional[List[dict]] = None,
        tags: Optional[List[str]] = None,
        subdir: str = "SearchAgent",
        link_related: bool = True,
    ) -> str:
        """把搜索结果写成一篇 Obsidian 笔记，返回相对路径（失败返回空串）。

        笔记结构（Obsidian 友好）：
          ---frontmatter---
          # 标题
          ## 答案
          ## 来源（可点击链接）
          ## 相关笔记（[[wikilink]] 双向链接）

        ``link_related=True`` 时自动检索 vault 内相关笔记并加 [[wikilink]]，
        配合 Obsidian 的反向链接（backlinks）形成知识网络。
        """
        if not self.enabled():
            return ""

        slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", title.lower()).strip("-")[:50] or "note"
        date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date}-{slug}.md"
        rel_path = f"{subdir}/{filename}".replace("\\", "/")

        # 检索相关笔记（写入前 _notes 不含新笔记，无需排除自己）
        related = []
        if link_related:
            try:
                hits = self.search(title, top_k=6)
                related = [h for h in hits if h.path != rel_path and h.score >= 2.0][:5]
            except Exception:
                related = []

        d = os.path.join(self.vault_path, subdir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)

        frontmatter = {
            "type": "query",
            "tags": tags or ["search-agent", "auto-archived"],
            "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        }
        fm = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()

        body = f"# {title}\n\n## 答案\n\n{content.strip()}\n"
        if sources:
            body += "\n## 来源\n\n"
            for s in sources:
                t = (s.get("title") or "").strip() or s.get("url", "")
                u = s.get("url", "")
                body += f"- [{t}]({u})\n" if u else f"- {t}\n"
        if related:
            body += "\n## 相关笔记\n\n"
            for h in related:
                link = h.path[:-3] if h.path.endswith(".md") else h.path
                body += f"- [[{link}]]\n"

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"---\n{fm}\n---\n\n{body}")
        except Exception:
            return ""

        # 反向链接：在已有关联笔记里补上指向新笔记的 wikilink（双向互链）
        if related:
            for h in related:
                try:
                    self._add_backlink(h.path, rel_path)
                except Exception:
                    pass

        # 使缓存失效，下次 search 能检索到新笔记
        self._notes = []
        return rel_path

    def _add_backlink(self, target_path: str, new_path: str) -> bool:
        """在 target_path 笔记的「相关笔记」章节补上指向 new_path 的 wikilink。

        实现双向互链：新笔记正向链接已有笔记的同时，已有笔记也反向链接新笔记。
        返回 True 表示有改动（新增了链接）。
        """
        if not self.enabled():
            return False
        full = os.path.join(self.vault_path, target_path)
        try:
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return False

        link = new_path[:-3] if new_path.endswith(".md") else new_path
        if f"[[{link}]]" in content:
            return False  # 已存在，跳过

        line = f"- [[{link}]]\n"
        if "## 相关笔记" in content:
            idx = content.find("## 相关笔记")
            next_section = content.find("\n## ", idx + 1)
            if next_section == -1:
                content = content.rstrip() + "\n" + line
            else:
                content = content[:next_section] + line + content[next_section:]
        else:
            content = content.rstrip() + "\n\n## 相关笔记\n\n" + line

        try:
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception:
            return False
