"""Verify agentic RAG (v3.0) imports and graph construction."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

errors = []

def check(name, import_stmt):
    try:
        exec(import_stmt)
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, str(e)))
        return False

print("=== Agentic RAG (v3.0) Import Verification ===\n")

# v3.0 modules
check("agentic_state", "from src.agent.agentic_state import AgenticState")
check("tools", "from src.agent.tools import TOOL_SCHEMAS, TOOL_EXECUTORS, execute_tool, execute_tool_calls")
check("agentic_nodes", "from src.agent.agentic_nodes import agentic_node, tool_execution_node, agentic_wiki_persist_node, AGENT_SYSTEM_PROMPT")
check("agentic_graph", "from src.agent.agentic_graph import build_agentic_graph, run_agentic_search, run_agentic_search_stream, _make_agentic_initial_state")
check("context_manager", "from src.agentic.context_manager import ContextManager, get_context_manager")
check("sufficiency", "from src.agentic.sufficiency import SufficiencyChecker, get_sufficiency_checker")

# Verify tool schemas have correct format
print("\n=== Tool Schema Verification ===")
try:
    from src.agent.tools import TOOL_SCHEMAS
    tool_names = [t["function"]["name"] for t in TOOL_SCHEMAS]
    print(f"  Tools: {tool_names}")
    assert len(TOOL_SCHEMAS) == 7, f"Expected 7 tools, got {len(TOOL_SCHEMAS)}"
    assert "search_web" in tool_names
    assert "search_papers" in tool_names
    assert "search_wiki" in tool_names
    assert "read_page" in tool_names
    assert "read_wiki_page" in tool_names
    assert "follow_link" in tool_names
    assert "final_answer" in tool_names
    print("  [OK] All 7 tools defined with correct names")
except Exception as e:
    print(f"  [FAIL] Tool schema check: {e}")
    errors.append(("tool_schemas", str(e)))

# Verify tool executors exist for each tool (except final_answer)
print("\n=== Tool Executor Verification ===")
try:
    from src.agent.tools import TOOL_EXECUTORS
    executor_names = list(TOOL_EXECUTORS.keys())
    expected = ["search_web", "search_papers", "search_wiki", "read_page", "read_wiki_page", "follow_link"]
    for name in expected:
        assert name in executor_names, f"Missing executor for {name}"
    print(f"  [OK] All 6 executors defined: {executor_names}")
except Exception as e:
    print(f"  [FAIL] Executor check: {e}")
    errors.append(("tool_executors", str(e)))

# Verify agentic graph builds
print("\n=== Agentic Graph Build Verification ===")
try:
    from src.config import AppConfig
    from src.agent.agentic_graph import build_agentic_graph
    config = AppConfig.from_env()
    graph = build_agentic_graph(config)
    compiled = graph.compile()
    print(f"  [OK] Agentic graph compiled successfully")
except Exception as e:
    print(f"  [FAIL] Agentic graph build: {e}")
    errors.append(("agentic_graph", str(e)))

# Verify fast graph with mode parameter
print("\n=== Fast Graph with Mode Parameter ===")
try:
    from src.config import AppConfig
    from src.agent.graph import build_search_agent
    config = AppConfig.from_env()
    fast_graph = build_search_agent(config, mode="fast")
    fast_compiled = fast_graph.compile()
    print(f"  [OK] Fast graph compiled with mode='fast'")
except Exception as e:
    print(f"  [FAIL] Fast graph: {e}")
    errors.append(("fast_graph", str(e)))

# Verify initial state
print("\n=== Initial State Verification ===")
try:
    from src.agent.agentic_graph import _make_agentic_initial_state
    state = _make_agentic_initial_state("test query", ["web"], 5)
    assert state["query"] == "test query"
    assert state["max_tool_calls"] == 12
    assert state["tool_calls_made"] == 0
    assert state["evidence_sufficient"] == False
    assert state["observations"] == []
    print(f"  [OK] Agentic initial state correct")
except Exception as e:
    print(f"  [FAIL] Initial state: {e}")
    errors.append(("initial_state", str(e)))

# Verify Config has agentic settings
print("\n=== Config Verification ===")
try:
    from src.config import AppConfig, AgenticConfig
    cfg = AppConfig.from_env()
    assert hasattr(cfg, "agentic"), "Missing agentic config"
    assert hasattr(cfg, "mode"), "Missing mode field"
    assert cfg.agentic.max_tool_calls == 12
    assert cfg.mode in ("fast", "agentic")
    print(f"  [OK] AppConfig has agentic settings (mode={cfg.mode})")
except Exception as e:
    print(f"  [FAIL] Config check: {e}")
    errors.append(("config", str(e)))

# Verify context manager
print("\n=== Context Manager Verification ===")
try:
    from src.agentic.context_manager import ContextManager
    cm = ContextManager()
    obs = [
        {"tool": "search_web", "content": "Result 1: some content here about AI."},
        {"tool": "search_web", "content": "Result 2: more content."},
        {"tool": "read_page", "content": "Detailed page content... " * 100},
        {"tool": "search_papers", "content": "Paper results found."},
        {"tool": "read_page", "content": "Another detailed read."},
    ]
    compressed = cm.compress(obs)
    assert len(compressed) > 0
    # Should compress when > 3 observations
    assert "Earlier observations" in compressed or len(obs) <= 3
    print(f"  [OK] Context manager compresses {len(obs)} observations into {len(compressed)} chars")
except Exception as e:
    print(f"  [FAIL] Context manager: {e}")
    errors.append(("context_manager", str(e)))

print(f"\n=== Summary: {len(errors)} errors ===")
if errors:
    for name, msg in errors:
        print(f"  - {name}: {msg}")
    sys.exit(1)
else:
    print("ALL AGENTIC v3.0 CHECKS PASSED")
