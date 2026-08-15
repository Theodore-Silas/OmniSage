# SearchAgent

一个「**本地命中优先、外部搜索兜底**」的多知识库搜索问答智能体。

> 基于 LangGraph 编排：输入问题后先在本地知识库（Wiki / Obsidian / 腾讯文档 / FAISS）命中已有答案，命中充分直接秒回；未命中或内容太少才启动多源搜索，经 Map-Reduce 总结后输出带引用的答案，并把结果沉淀为可复用知识、自动双向链接形成知识网络 —— **越用越聪明**。

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-4B8BBE)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFE)
![FAISS](https://img.shields.io/badge/Vector-FAISS-0866FF)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

---

## ✨ 特性

- **命中优先**：输入先查本地多知识库，命中充分秒回（零搜索、零 LLM 成本），未命中才搜索
- **多知识库接入**：统一连接器抽象，Wiki / Obsidian / 腾讯文档 / FAISS 四类知识库可插拔扩展
- **多源检索**：DuckDuckGo / Semantic Scholar / ArXiv / NewsAPI / RSS 四源并行 + BM25 与语义混合排序
- **智能总结**：Map-Reduce 分层总结（分源独立总结 → 跨源合成），输出带引用来源的结构化答案
- **上下文管理**：分层压缩 + 分级截断 + Top-K 限制 + Token 预算，检索全文压缩比约 16:1
- **知识沉淀**：搜索后三重沉淀（Wiki + FAISS + Obsidian），重复问题永久命中
- **双向链接**：沉淀笔记自动关联相关笔记，形成 Obsidian 知识网络

## 🏗️ 架构

```
输入问题
   │
   ▼
本地命中（多知识库：FAISS / Wiki / Obsidian / 腾讯文档）
   │
   ├─ 命中充分 ──► 直接回答（秒回，零搜索）
   │
   └─ 未命中/内容太少 ──► 四源并行搜索 → 融合排序 → Map-Reduce 总结
                                            │
                                            ▼
                              回答 + 知识沉淀（Wiki + FAISS + Obsidian）
```

## 🚀 快速开始

### 环境要求

- Python 3.10+（开发环境为 3.12）
- 一个 LLM API Key（DeepSeek 推荐，或 Qwen / OpenAI 兼容接口）

### 安装

```bash
# 1. 创建并激活环境（可选）
conda create -n searchagent python=3.12 -y
conda activate searchagent

# 2. 安装依赖
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`（或直接新建 `.env`），填写你的 API Key：

```bash
# .env
LLM_PROVIDER=deepseek            # deepseek | qwen | openai
DEEPSEEK_API_KEY=sk-your-key     # 你的 API Key
DEEPSEEK_MODEL=deepseek-chat

# 可选：接入 Obsidian 本地知识库
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault
```

> ⚠️ `.env` 含密钥，已被 `.gitignore` 忽略，请勿提交到仓库。

### 运行

```bash
# 命令行提问（先命中本地，未命中才搜索）
python -m src.main "FAISS 向量数据库是什么？"

# 查看执行日志（命中/搜索/总结/沉淀）
python -m src.main "FAISS 向量数据库是什么？" -v

# Web UI
python -m src.main --streamlit
```

## 📖 使用示例

```
第 1 次问「FAISS 是什么」
  → 本地未命中 → 四源搜索 → Map-Reduce 总结（带引用）→ 沉淀到知识库     ≈10s

第 2 次问「FAISS 是什么」
  → 命中本地知识库 → 直接回答（零搜索、零 LLM 成本）                    <1s
```

## 🗂️ 项目结构

```
src/
├── qa/           # 搜索问答主链路（命中→回答 / 搜索→总结→沉淀）
│   ├── context.py    # 上下文管理（分级截断 + Token 预算）
│   └── graph.py      # LangGraph 状态图
├── knowledge/    # 知识库连接器层（Wiki / Obsidian / 腾讯文档 / FAISS）
├── search/       # 四源搜索适配器 + 融合排序
├── summarize/    # Map-Reduce 总结
├── storage/      # FAISS 向量记忆
├── wiki/         # Wiki 知识持久化（运行时自动生成）
├── agentic/      # 充分性判断 + 上下文压缩
├── browser/      # 浏览器自动化（可选扩展）
├── config.py     # 配置
├── main.py       # CLI 入口
└── ui/           # Streamlit UI

benchmarks/       # 指标模拟测试
tests/            # 测试
prompts/          # prompt 模板
```

## 🧩 核心机制

| 机制 | 说明 |
|------|------|
| **多知识库命中** | `KnowledgeBaseConnector` 统一接口，Wiki / Obsidian / 腾讯文档 / FAISS 双通道命中 + 置信度评分 |
| **充分性判断** | 命中内容充分（向量 ≥200 字 / 知识库 ≥400 字）才直答，否则启动搜索 |
| **融合检索** | 四源并行 + URL 去重 + BM25/语义混合排序（6:3:1） |
| **Map-Reduce 总结** | 分源独立总结 → 跨源合成 → 核心结论 + 分源分析 + 观点差异 + 参考文献 |
| **上下文管理** | 分层压缩（压缩比 ≈16:1）+ 分级截断 + Top-K + Token 预算 |
| **知识沉淀** | 搜索后三重沉淀（Wiki 归档 + FAISS 记忆 + Obsidian 笔记）+ 双向链接 |

## ⚙️ 配置说明

| 配置项 | 说明 | 默认 |
|--------|------|------|
| `LLM_PROVIDER` | LLM 提供商 | `deepseek` |
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填） | — |
| `OBSIDIAN_VAULT_PATH` | Obsidian vault 路径（可选） | 空（禁用） |
| `BROWSER_CHANNEL` | 浏览器内核（可选扩展） | `chrome` |

完整配置见 `src/config.py`。

## 🛠️ 技术栈

Python · LangGraph · DeepSeek API · FAISS · Streamlit · DuckDuckGo · Semantic Scholar · ArXiv · NewsAPI · RSS

## 📄 说明

- 本项目聚焦「搜索问答」主链路；`src/browser/` 为早期浏览器自动化扩展，保留但不参与问答主链路。
- `wiki/`、`.searchagent_memory/` 为运行时数据（含个人搜索沉淀），已加入 `.gitignore`，不随代码分发。
- 详细设计文档见 [`DESIGN.md`](DESIGN.md)。
