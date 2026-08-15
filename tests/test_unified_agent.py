"""
Unified agent (v5.0) E2E tests — knowledge path + browser path (lazy launch).

Verifies:
  1. A knowledge task does NOT launch the browser (lazy browser).
  2. A browser task DOES launch the browser and completes the full loop.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig
from src.agent.graph import AgentRuntime, run_agent


async def test_knowledge_task() -> None:
    config = AppConfig.from_env()
    runtime = AgentRuntime(config)

    result = await run_agent("用一句话介绍 FAISS 向量数据库是什么", config, runtime)

    assert runtime.browser_alive is False, "知识检索任务不应启动浏览器"
    answer = result.get("final_answer", "")
    assert answer, "应有最终答案"
    print(f"\n[知识任务] browser_alive={runtime.browser_alive} steps={result.get('tool_calls_made')}")
    print(f"[知识任务] 答案: {answer[:120]}...")
    await runtime.stop()
    print("✓ 知识任务通过（lazy 浏览器未启动）")


async def test_browser_task() -> None:
    config = AppConfig.from_env()
    config.browser.channel = "chrome"
    config.browser.headless = True
    runtime = AgentRuntime(config)

    page = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_page.html"))
    url = "file:///" + page.replace("\\", "/")

    result = await run_agent(
        f"打开页面 {url} ，在输入框输入 'hello' ，点击搜索按钮，报告页面结果文本",
        config,
        runtime,
    )

    assert runtime.browser_alive is True, "浏览器任务应启动浏览器"
    answer = result.get("final_answer", "")
    assert "hello" in answer or "3 条" in answer, f"答案应包含结果，实际: {answer[:200]}"
    print(f"\n[浏览器任务] browser_alive={runtime.browser_alive} steps={result.get('tool_calls_made')}")
    print(f"[浏览器任务] 动作: " + " → ".join(t["action"] for t in result.get("action_trace", [])))
    print(f"[浏览器任务] 答案: {answer[:160]}...")
    await runtime.stop()
    print("✓ 浏览器任务通过（lazy 启动 + 完整闭环）")


async def main() -> None:
    await test_knowledge_task()
    await test_browser_task()
    print("\n=== 统一 Agent E2E 全部通过 ===")


if __name__ == "__main__":
    asyncio.run(main())
