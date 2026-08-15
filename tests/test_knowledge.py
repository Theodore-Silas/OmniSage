"""
知识库连接器测试 — Obsidian vault 命中 + 集成验证。

验证：
  1. ObsidianConnector 能检索本地 vault 中的 .md 笔记
  2. get_content 剥离 YAML frontmatter 返回正文
  3. run_qa 集成：指向测试 vault 后，问答能命中 Obsidian（不启动搜索）
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowledge import ObsidianConnector
from src.config import AppConfig
from src.qa import QARuntime, run_qa


NOTE_VECTOR = """---
tags: [ai, ml, vector]
---

# 向量数据库

向量数据库是专门用于存储、索引和检索高维向量（embedding）的数据库，支持相似性搜索（向量检索）。

## 核心概念
向量数据库通过将文本、图像等数据编码为高维向量，然后基于向量间的距离（余弦相似度、欧氏距离等）进行最近邻检索。

## 常见产品
- FAISS：Facebook 开源的高效相似性搜索库，常用于内存级向量检索。
- Milvus：云原生分布式向量数据库，支持十亿级向量规模。
- Pinecone：托管式向量数据库服务，开箱即用。

## 典型应用场景
- RAG（检索增强生成）：为大语言模型提供外部知识检索。
- 推荐系统：基于用户/物品向量的相似性推荐。
- 语义搜索：按语义而非关键词匹配检索文档。

向量数据库的核心价值在于将「暴力全量比对」优化为「近似最近邻搜索（ANN）」，从而在海量高维向量中实现毫秒级检索。

## 选型建议
选择向量数据库时需要综合考虑数据规模、性能要求、部署方式与成本：小规模实验可优先使用 FAISS 或 Chroma，生产级大规模场景可考虑 Milvus 或 Qdrant，追求开箱即用的托管服务则可选 Pinecone。向量数据库与大语言模型结合（RAG）已成为当前构建知识问答、智能客服等应用的主流范式。
"""

NOTE_ML = """---
tags: [ml]
---

# 机器学习基础

机器学习是人工智能的一个分支，研究如何让计算机从数据中自动学习规律并做出预测或决策。

## 三大范式
- 监督学习：从带标签数据学习映射关系。
- 无监督学习：从无标签数据发现内在结构。
- 强化学习：通过与环境交互、试错来学习策略。

机器学习已经广泛应用于计算机视觉、自然语言处理、推荐系统等领域。
"""

NOTE_DEEP = """---
tags: [dl]
---

# 深度学习

深度学习是机器学习的一个子领域，使用多层神经网络从数据中学习层次化的特征表示，在图像识别、语音识别、自然语言处理等任务上取得了突破性进展。
"""


def create_test_vault() -> str:
    vault = tempfile.mkdtemp(prefix="searchagent_vault_")
    os.makedirs(os.path.join(vault, "AI"), exist_ok=True)
    with open(os.path.join(vault, "向量数据库.md"), "w", encoding="utf-8") as f:
        f.write(NOTE_VECTOR)
    with open(os.path.join(vault, "机器学习基础.md"), "w", encoding="utf-8") as f:
        f.write(NOTE_ML)
    with open(os.path.join(vault, "AI", "深度学习.md"), "w", encoding="utf-8") as f:
        f.write(NOTE_DEEP)
    return vault


def test_obsidian_connector() -> None:
    vault = create_test_vault()
    conn = ObsidianConnector(vault)
    assert conn.enabled(), "应启用（vault 存在）"

    hits = conn.search("向量数据库", top_k=3)
    assert hits, "应有命中"
    print(f"\n[Obsidian] 命中: {[(h.title, round(h.score, 1)) for h in hits]}")

    content = conn.get_content(hits[0])
    assert "向量数据库" in content, "正文应包含关键词"
    assert not content.startswith("---"), "应剥离 frontmatter"
    assert "FAISS" in content, "正文应保留"
    print(f"[Obsidian] 正文前 60 字: {content[:60]}...")
    print("✓ Obsidian 连接器通过（检索 + frontmatter 剥离）")


def test_save_note() -> None:
    """验证搜索结果可写入 Obsidian 并再次命中（知识沉淀 + 双向链接）。"""
    vault = create_test_vault()
    conn = ObsidianConnector(vault)

    # 用相关主题写入，验证自动关联已有「向量数据库」笔记（wikilink）
    path = conn.save_note(
        "Milvus 向量数据库",
        "Milvus 是云原生分布式向量数据库，支持十亿级向量规模，常用于生产级 RAG 应用。",
        [{"title": "Milvus 官网", "url": "https://milvus.io"}],
    )
    assert path, "save_note 应返回相对路径"
    print(f"\n[save_note] 保存路径: {path}")

    with open(os.path.join(vault, path), "r", encoding="utf-8") as f:
        note_content = f.read()
    assert "[[" in note_content, "应包含 wikilink（双向链接）"
    assert "向量数据库" in note_content, "应关联到已有「向量数据库」笔记"
    print("[save_note] 已自动添加相关笔记 wikilink")

    hits = conn.search("Milvus", top_k=3)
    assert hits, "写入后应能命中新笔记"
    print(f"[save_note] 命中: {[(h.title, round(h.score, 1)) for h in hits]}")
    print("✓ save_note 通过（写入 + 双向链接 + 缓存失效 + 再次命中）")


async def test_run_qa_obsidian_hit() -> None:
    vault = create_test_vault()
    os.environ["OBSIDIAN_VAULT_PATH"] = vault

    config = AppConfig.from_env()
    runtime = QARuntime(config)  # 自动注册 ObsidianConnector(vault)

    result = await run_qa("向量数据库有哪些常见产品？", config, runtime)
    print(f"\n[run_qa] verdict={result.get('verdict')}  hit_kind={result.get('hit_kind')}")
    print(f"[run_qa] 日志: {result.get('logs', [])}")
    print(f"[run_qa] 答案: {result.get('final_answer', '')[:120]}...")

    assert result.get("verdict") == "hit_high", "应命中 Obsidian 知识库"
    assert result.get("hit_kind") == "obsidian", f"hit_kind 应为 obsidian，实际 {result.get('hit_kind')}"
    print("✓ run_qa 集成通过（命中 Obsidian，未启动搜索）")


def main() -> None:
    test_obsidian_connector()
    test_save_note()
    asyncio.run(test_run_qa_obsidian_hit())
    print("\n=== 知识库连接器测试全部通过 ===")


if __name__ == "__main__":
    main()
