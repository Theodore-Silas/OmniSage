"""
搜索问答主链路 E2E 测试。

验证两条路径：
  1. 首次提问 → 未命中 → 四源搜索 → Map-Reduce 总结 → 知识沉淀
  2. 再次提问（同类）→ 命中本地知识库 → 直接回答（零搜索）
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import AppConfig
from src.qa import QARuntime, run_qa


async def main() -> None:
    config = AppConfig.from_env()
    runtime = QARuntime(config)

    query = "FAISS 向量数据库是什么？"

    print("=" * 60)
    print("[第一次提问] 预期：未命中 → 搜索 → 总结 → 沉淀")
    r1 = await run_qa(query, config, runtime)
    print(f"  verdict={r1.get('verdict')}  status={r1.get('status')}")
    print(f"  日志: {r1.get('logs', [])}")
    print(f"  答案: {r1.get('final_answer', '')[:200]}...")

    print("\n" + "=" * 60)
    print("[第二次提问] 预期：命中本地知识库 → 直接回答")
    r2 = await run_qa(query, config, runtime)
    print(f"  verdict={r2.get('verdict')}  status={r2.get('status')}  hit_kind={r2.get('hit_kind')}")
    print(f"  日志: {r2.get('logs', [])}")
    print(f"  答案: {r2.get('final_answer', '')[:200]}...")

    print("\n" + "=" * 60)
    ok = (r1.get("verdict") == "miss" and r2.get("verdict") == "hit_high")
    print("✓ 两条路径均符合预期" if ok else "✗ 与预期不符")
    if not ok:
        print(f"  (r1 verdict={r1.get('verdict')}, r2 verdict={r2.get('verdict')})")


if __name__ == "__main__":
    asyncio.run(main())
