# JD 匹配说明 —— 腾讯校招「Agent 开发工程师」× SearchAgent

> 目标 JD：Agent开发工程师（技术 · 应届毕业生，join.qq.com post_id=1282707395466077184）
> 简历版本：`tencent-agent-dev-20260815`（派生自项目根 `RESUME_FINAL.md`，原文件未改动）

## 一、JD 关键词对齐

| JD 要求 | 简历对应点 | 覆盖 |
|---|---|---|
| 主流 Agent 框架（LangGraph 等）+ 实际搭建经验 | LangGraph 状态机 + `conditional_edges` 两级路由 | ✅ |
| Planning（任务规划） | Map-Reduce 分源任务分解 + 跨源合成 | ✅ |
| Memory（记忆系统） | FAISS 向量记忆 + Wiki 知识归档 | ✅ |
| Tool Use（工具调用） | 四源搜索工具 + 知识库连接器 | ✅ |
| 大模型 + Prompt Engineering | DeepSeek + 分级截断/系统提示词设计 | ✅ |
| 成本/性能管控（岗位职责③） | 命中优先零搜索成本 + Token 预算 | ✅ |
| 加分项：RAG + 向量检索 | FAISS 向量检索 + 多知识库 RAG | ✅ |
| 加分项：复杂工作流编排 | LangGraph 状态机全链路编排 | ✅ |
| Reflection（反思机制） | 充分性自检（SufficiencyChecker 5 维评分）| ⚠️ 弱覆盖 |

## 二、需你确认/补强的点

1. **Reflection 弱覆盖**：JD 要求「深入理解 Reflection 机制」。你的项目里充分性判断（`src/agentic/sufficiency.py`）是一个轻量自检，但并非完整的 Reflection 循环。若想强化，可考虑补一句「检索结果充分性自检，不足时触发补充搜索」——这已是事实，建议在面试中主动展开。
2. **「多 Agent 协作」未写**：项目是单 Agent（LangGraph 单图），v4-v6 的 multiagent 已降级为可选扩展。简历未虚标「多 Agent 协作」，符合正直原则；若后续实现多 Agent 框架可再补。
3. **联系信息为占位符**：姓名/电话/邮箱/城市投递前必须替换真实信息。

## 三、量化指标来源（可追溯）

| 指标 | 数值 | 来源 |
|---|---|---|
| 本地命中耗时 | 首次 98ms / 缓存 1ms | `RESUME_FINAL.md` 指标说明，`benchmarks/benchmark.py` |
| LLM 调用 | 命中 0 次 vs 搜索 5 次 | 静态分析两条路径 |
| 重复问题命中率 | 0% → 100% | 3 组无关问题沉淀前后对比 |
| 上下文压缩比 | ≈16:1 | 4 源 × 8000 字 ≈ 3.2 万字 → ≤2000 token |

> 以上数据均来自你项目仓库内已有的 benchmark 结论，简历未新增任何未经证实的数据。
