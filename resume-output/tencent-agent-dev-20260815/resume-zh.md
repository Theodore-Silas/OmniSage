<!-- preflight: template=star · length=1page · language=zh · auto-selected: false -->

# [姓名]　·　Agent 开发工程师（校招 · 技术方向）

电话：[电话待填写]　｜　邮箱：[邮箱待填写]　｜　意向城市：[城市待填写]

---

## 项目经历

### SearchAgent —— 本地命中优先的多知识库搜索问答 Agent

**S（情境）**：知识问答场景中，多源搜索每次都要多次网络请求 + 多次 LLM 调用，重复问题反复搜索、成本难控；多源全文直接喂大模型会撑爆上下文窗口、稀释关键信息。

**T（任务）**：设计并实现一个「本地命中优先、外部搜索兜底」的搜索问答 Agent，在保证答案质量的前提下控制 token 成本与响应时延，并让系统越用越省。

**技术栈**：Python · LangGraph · DeepSeek（LLM）· RAG · FAISS · asyncio · Streamlit

**A（行动）· R（结果）** —— 三大技术点，均按 STAR 展开：

**① 长上下文管理**
- S：四源检索最多返回约 3.2 万字，直接喂 LLM 撑爆窗口、推高 token 成本。
- A：设计四层机制——Map-Reduce 分层压缩（Map 分源独立总结 ≤800 token → Reduce 跨源合成 ≤2000 token）、分级截断（摘要 300 / 正文 4000 / 每源 8000 字）、Top-K 数量控制（每源 5 → 融合排序 Top-10 → 抓取 Top-5）、Token 预算。
- R：压缩比约 16:1（3.2 万字 → 约 3200 token 摘要 → ≤2000 token 答案），单次 LLM 输入严格控在预算内。

**② RAG 多知识库检索增强**
- S：本地知识分散在 Wiki / Obsidian / 腾讯文档 / 历史问答多个源，缺少统一检索与语义增强。
- A：抽象知识库连接器（KnowledgeBaseConnector）统一接入四类知识源、命中源可插拔；FAISS（all-MiniLM-L6-v2，384 维）做语义检索，中文用字符 2-gram（免分词）提升召回；未命中时四源并行搜索 + URL 去重 + BM25/语义混合排序（6:3:1）。
- R：答案带完整引用来源；每次搜索结果归档 Wiki、写入 FAISS 记忆，形成「越用越聪明」的正循环。

**③ Agent 编排与两级路由**
- S：重复问题每次都走完整搜索链路，既慢又贵，缺少统一的编排与路由控制。
- A：用 LangGraph 状态机编排全链路，`conditional_edges` 驱动命中优先的两级路由——第一级先查本地多知识库（FAISS 向量记忆 → Wiki/Obsidian/腾讯文档，按相似度 + 充分性分级判定），命中充分直接回答；第二级未命中/内容不足才切到外部四源搜索（网页/论文/新闻/博客）→ Map-Reduce 总结 → 写回知识库。
- R：命中路径 0 次 LLM 调用、首答 98ms（缓存 1ms）；搜索路径 5 次 LLM 调用、秒级——成本降低 100%，重复问题命中率 0% → 100%。

---

## 技能

- 编程语言：Python
- Agent 框架：LangGraph
- 大模型：DeepSeek（OpenAI 兼容）、Prompt Engineering
- 检索增强：RAG、FAISS 向量检索、BM25/语义混合排序
- 其他：asyncio 并发、Streamlit、单元测试

---

## 指标说明（可复现）

本地命中耗时、LLM 调用次数、命中率、压缩比由 `benchmarks/benchmark.py` 模拟测试得出，方法见项目仓库。
