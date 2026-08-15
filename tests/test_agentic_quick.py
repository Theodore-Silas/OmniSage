"""Quick agentic graph structure test — no heavy deps needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Agentic Graph Structure Test ===\n")

# 1. Graph compilation
from src.config import AppConfig
from src.agent.agentic_graph import build_agentic_graph, _make_agentic_initial_state
config = AppConfig.from_env()
g = build_agentic_graph(config)
c = g.compile()
print("[OK] Agentic graph compiled")

# 2. Initial state
state = _make_agentic_initial_state("test", ["web"], 5)
assert state["max_tool_calls"] == 12
assert state["evidence_sufficient"] == False
print("[OK] Initial state: max_tool_calls=12, evidence=False")

# 3. Tool schemas
from src.agent.tools import TOOL_SCHEMAS, TOOL_EXECUTORS
names = [t["function"]["name"] for t in TOOL_SCHEMAS]
assert len(names) == 7, f"Expected 7 tools, got {len(names)}"
assert "final_answer" in names
print(f"[OK] {len(TOOL_SCHEMAS)} tool schemas: {names}")

# 4. Tool executors
assert len(TOOL_EXECUTORS) == 6  # final_answer is handled specially
print(f"[OK] {len(TOOL_EXECUTORS)} tool executors: {list(TOOL_EXECUTORS.keys())}")

# 5. Context manager
from src.agentic.context_manager import ContextManager
cm = ContextManager()
obs = [{"tool": f"tool_{i}", "content": f"Content {i} " * 50} for i in range(8)]
compressed = cm.compress(obs)
assert len(compressed) > 0
assert "Earlier observations" in compressed
print(f"[OK] Context compression: 8 obs → {len(compressed)} chars")

# 6. Fast mode still works
from src.agent.graph import build_search_agent
# Test without importing ranker (numpy-dependent)
print("[OK] Graph module loads successfully")

# 7. Config has agentic fields
assert hasattr(config, "agentic")
assert hasattr(config, "mode")
assert hasattr(config.agentic, "max_tool_calls")
print(f"[OK] Config: agentic.max_tool_calls={config.agentic.max_tool_calls}, mode={config.mode}")

# 8. Agent system prompt
from src.agent.agentic_nodes import AGENT_SYSTEM_PROMPT
assert "search_web" in AGENT_SYSTEM_PROMPT
assert "search_wiki" in AGENT_SYSTEM_PROMPT
assert "final_answer" in AGENT_SYSTEM_PROMPT
print("[OK] Agent system prompt contains all tool references")

# 9. Mode switching in graph builder
from src.agent.graph import build_search_agent as bsa
import importlib.util
has_numpy = importlib.util.find_spec("numpy") is not None
if has_numpy:
    fg = bsa(config, mode="fast")
    fg.compile()
    print("[OK] Fast graph compiles (numpy available)")
else:
    print("[SKIP] Fast graph (numpy not installed)")

ag = bsa(config, mode="agentic")
ag.compile()
print("[OK] Agentic graph compiles via unified builder")

print("\n=== ALL TESTS PASSED ===")
