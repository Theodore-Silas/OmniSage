"""
Streamlit web UI -- DeepSeek-style chat with streaming + multi-turn + memory.
"""

import asyncio
import sys
import os
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import AppConfig
from src.qa import run_qa_stream
from src.storage.vector_store import VectorMemory

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="SearchAgent", page_icon="\U0001f50d", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#fff;--bg-soft:#f7f8fa;--text:#1a1a1a;--text-muted:#6b7280;--text-faint:#9ca3af;--border:#e5e7eb;--accent:#4D6BFE;--accent-soft:#eef1ff;--accent-hover:#3d5be6}
html,body,[class*="css"]{font-family:'Inter',-apple-system,BlinkMacSystemFont,'PingFang SC','HarmonyOS Sans SC','Microsoft YaHei',sans-serif;color:var(--text)}
.main .block-container{padding-top:0;padding-bottom:6rem;max-width:760px}
[data-testid="stSidebar"]{background:var(--bg-soft);border-right:1px solid var(--border)}
[data-testid="stSidebar"] .block-container{padding:1rem .75rem}
[data-testid="stSidebarNav"]{display:none}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
.brand{display:flex;align-items:center;gap:8px;padding:.5rem .75rem 1.25rem;font-size:1rem;font-weight:600;color:var(--text);letter-spacing:-.01em}
.brand-logo{width:24px;height:24px;border-radius:6px;background:var(--accent);display:inline-flex;align-items:center;justify-content:center;color:#fff;font-size:.85rem;font-weight:700}
.new-chat-btn button{width:100%;background:var(--accent)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:500!important;padding:.55rem 1rem!important}
.new-chat-btn button:hover{background:var(--accent-hover)!important}
.history-title{font-size:.72rem;color:var(--text-faint);text-transform:uppercase;letter-spacing:.06em;padding:.5rem .75rem;font-weight:500;margin-top:1.5rem}
.history-item{padding:.55rem .75rem;border-radius:6px;cursor:pointer;color:var(--text);font-size:.83rem;margin-bottom:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:background .12s}
.history-item:hover{background:rgba(0,0,0,.04)}
.settings-section{margin-top:1rem;padding-top:1rem;border-top:1px solid var(--border)}
.settings-title{font-size:.72rem;color:var(--text-faint);text-transform:uppercase;letter-spacing:.06em;padding:.25rem .75rem .5rem;font-weight:500}
.recommend-title{font-size:.72rem;color:var(--text-faint);text-transform:uppercase;letter-spacing:.06em;padding:.5rem .75rem .25rem;font-weight:500}
.recommend-item{padding:.35rem .75rem;font-size:.8rem;color:var(--text-muted);cursor:pointer;border-radius:4px;transition:all .12s}
.recommend-item:hover{background:var(--accent-soft);color:var(--accent)}
.welcome{text-align:center;padding:5rem 1rem 3rem}
.welcome-greeting{font-size:1.5rem;font-weight:500;color:var(--text);margin-bottom:1.5rem}
.welcome-greeting .dot{display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:8px;background:var(--accent);color:#fff;font-weight:600;margin-right:10px}
.modes{display:flex;justify-content:center;gap:10px;margin-bottom:1.5rem;flex-wrap:wrap}
.mode-pill{padding:.5rem 1.1rem;border-radius:999px;background:var(--bg-soft);border:1px solid var(--border);color:var(--text);font-size:.85rem;cursor:pointer;transition:all .15s}
.mode-pill.active{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.suggestions{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:2rem;padding:0 1rem}
.suggestion-chip{padding:.45rem .95rem;background:var(--bg-soft);border:1px solid var(--border);border-radius:999px;color:var(--text-muted);font-size:.83rem;cursor:pointer;transition:all .15s}
.suggestion-chip:hover{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.stTextArea textarea{border-radius:16px!important;border:1px solid var(--border)!important;font-size:.95rem!important;padding:16px 60px 16px 18px!important;line-height:1.6!important;min-height:64px!important;transition:border-color .2s,box-shadow .2s;resize:none!important}
.stTextArea textarea:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(77,107,254,.1)!important}
.stTextArea label{display:none}
.stButton>button{border-radius:999px!important;font-size:.85rem!important;font-weight:500!important;padding:.45rem 1rem!important;height:auto!important;transition:all .15s}
.msg-user{background:var(--bg-soft);padding:.85rem 1.1rem;border-radius:14px;margin:.75rem 0;font-size:.95rem;line-height:1.6}
.msg-ai{padding:.85rem .25rem;margin:.75rem 0;font-size:.95rem;line-height:1.7;color:var(--text)}
.msg-meta{font-size:.72rem;color:var(--text-faint);margin-top:.35rem}
.report h2{font-size:1.05rem!important;font-weight:600!important;margin-top:1.2rem!important;margin-bottom:.5rem!important;color:var(--text)}
.report h3{font-size:.92rem!important;font-weight:600!important;margin-top:.9rem!important;color:var(--text)}
.report p,.report li{font-size:.9rem!important;line-height:1.7!important;color:#2a2a2a}
.report a{color:var(--accent);text-decoration:none;word-break:break-all}
.report a:hover{text-decoration:underline}
.report code{background:var(--bg-soft);padding:.1rem .4rem;border-radius:4px;font-size:.85em;color:#d63384}
.status-line{font-size:.8rem;color:var(--text-faint);padding:.3rem .5rem}
hr{margin:1rem 0!important;border-color:var(--border)!important}
.streamlit-expanderHeader{font-size:.83rem!important;color:var(--text-muted)!important}
.agent-trace{margin:.5rem 0;border-left:2px solid var(--accent);padding-left:1rem}
.agent-thought{font-size:.82rem;color:var(--text-muted);padding:.25rem 0;display:flex;align-items:flex-start;gap:6px}
.agent-thought .icon{font-size:.9rem;flex-shrink:0}
.agent-action{font-size:.8rem;color:var(--accent);padding:.2rem 0 .2rem 1.5rem;font-weight:500}
.agent-observation{font-size:.78rem;color:var(--text-faint);padding:.15rem 0 .15rem 1.5rem;line-height:1.5;max-height:120px;overflow-y:auto;background:var(--bg-soft);border-radius:6px;padding:.4rem .6rem;margin:.2rem 0 .2rem 1.5rem}
.agent-answer{margin-top:.75rem}
.mode-toggle{display:flex;gap:4px;background:var(--bg-soft);border-radius:8px;padding:3px}
.mode-btn{padding:.35rem .8rem;border:none;border-radius:6px;background:transparent;color:var(--text-muted);font-size:.78rem;cursor:pointer;font-weight:500;transition:all .15s}
.mode-btn.active{background:#fff;color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.08)}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = VectorMemory()
if "_last_input" not in st.session_state:
    st.session_state._last_input = ""


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-logo">S</span><span>SearchAgent</span></div>', unsafe_allow_html=True)
    st.caption("Unified agent · knowledge + browser")

    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("\u2728  New Chat", key="new_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # History
    st.markdown('<div class="history-title">History</div>', unsafe_allow_html=True)
    if st.session_state.messages:
        seen = []
        for msg in st.session_state.messages:
            if msg["role"] == "user" and msg["content"] not in seen:
                seen.append(msg["content"])
        for q in reversed(seen[-15:]):
            label = q[:30] + ("..." if len(q) > 30 else "")
            if st.button(label, key=f"h_{q[:20]}", use_container_width=True):
                # Find the index of this query's user message and truncate
                target_idx = None
                for i, m in enumerate(st.session_state.messages):
                    if m["role"] == "user" and m["content"] == q:
                        target_idx = i
                        break
                if target_idx is not None:
                    # Find next assistant message
                    next_ai = None
                    for j in range(target_idx + 1, len(st.session_state.messages)):
                        if st.session_state.messages[j]["role"] == "assistant":
                            next_ai = j; break
                    if next_ai:
                        st.session_state.messages = st.session_state.messages[:next_ai + 1]
                        st.rerun()
    else:
        st.markdown('<div style="padding:.5rem .75rem;color:var(--text-faint);font-size:.82rem">No history</div>', unsafe_allow_html=True)

    # Wiki Pages
    st.markdown('<div class="history-title" style="margin-top:1.5rem">Wiki Knowledge</div>', unsafe_allow_html=True)
    try:
        from src.wiki.manager import WikiManager
        wiki_mgr = WikiManager()
        wiki_pages = wiki_mgr.list_pages()
        if wiki_pages:
            page_count = len(wiki_pages)
            st.caption(f"{page_count} pages indexed")
            recent = sorted(wiki_pages, reverse=True)[:10]
            for wp in recent:
                display = wp.replace(".md", "").replace("/", " › ")
                if st.button(display[:35], key=f"wiki_{wp[:30]}", use_container_width=True):
                    page = wiki_mgr.load_page(wp)
                    if page:
                        st.session_state["_show_wiki_page"] = page.to_markdown()
                        st.rerun()
        else:
            st.markdown('<div style="padding:.5rem .75rem;color:var(--text-faint);font-size:.82rem">Empty — search to populate</div>', unsafe_allow_html=True)
    except Exception:
        pass

    # Recommendations from memory
    mem = st.session_state.memory
    if mem.size() > 0 and st.session_state.messages:
        last_q = st.session_state.messages[-1].get("content","") if st.session_state.messages else ""
        recs = mem.get_recommendations(last_q, top_k=3) if last_q else []
        if recs:
            st.markdown('<div class="recommend-title">Related</div>', unsafe_allow_html=True)
            for rec in recs:
                label = rec[:28] + "..." if len(rec) > 28 else rec
                if st.button(label, key=f"rec_{rec[:15]}", use_container_width=True):
                    st.session_state["_prefill"] = rec
                    st.rerun()

    # Settings
    st.markdown('<div class="settings-section"><div class="settings-title">Settings</div>', unsafe_allow_html=True)
    llm_provider = st.selectbox("LLM", ["deepseek","qwen","openai"], index=0, label_visibility="collapsed")
    api_key_env = {"deepseek":"DEEPSEEK_API_KEY","qwen":"DASHSCOPE_API_KEY","openai":"OPENAI_API_KEY"}[llm_provider]
    default_model = {"deepseek":"deepseek-chat","qwen":"qwen-plus","openai":"gpt-4o"}[llm_provider]
    api_key = st.text_input("API Key", type="password", value=os.getenv(api_key_env,""), label_visibility="collapsed", placeholder=f"{llm_provider} API Key")
    model = st.text_input("Model", value=default_model, label_visibility="collapsed")
    sources = st.multiselect("Sources", ["web","paper","news","blog"], default=["web","paper"], format_func=lambda x:{"web":"Web","paper":"Papers","news":"News","blog":"Blogs"}.get(x,x), label_visibility="collapsed")
    max_results = st.slider("Results/source", 1,10,5, label_visibility="collapsed")
    verbose = st.checkbox("Debug", value=False)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Main content ────────────────────────────────────────────
# Show Wiki page if requested
if "_show_wiki_page" in st.session_state and st.session_state["_show_wiki_page"]:
    with st.expander("Wiki Page", expanded=True):
        st.markdown(st.session_state["_show_wiki_page"])
        if st.button("Close", key="close_wiki"):
            del st.session_state["_show_wiki_page"]
            st.rerun()

# Show welcome if empty
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><div class="welcome-greeting">'
        '<span class="dot">S</span>Ask anything — search or operate the web</div></div>',
        unsafe_allow_html=True)

    st.markdown(
        '<div class="modes"><span class="mode-pill active">Knowledge</span>'
        '<span class="mode-pill">Browser</span><span class="mode-pill">Unified</span></div>',
        unsafe_allow_html=True)

    # Suggestion chips — real buttons that fill the query input
    suggestions = [
        "Compare FAISS, Milvus and Pinecone for production",
        "打开 https://news.ycombinator.com 抓取前 5 条新闻标题",
        "Latest advances in autonomous driving 2025",
        "打开 https://www.python.org 并总结首页内容",
    ]

    # Render suggestions as a wrapped row of real buttons
    cols = st.columns(len(suggestions))
    for col, sug in zip(cols, suggestions):
        with col:
            if st.button(sug[:24] + ("…" if len(sug) > 24 else ""), key=f"sug_{sug[:15]}", use_container_width=True):
                st.session_state["_prefill"] = sug
                st.rerun()

# ── Helper: render agent trace ──────────────────────────────
def _render_trace(placeholder, entries: list):
    """Render the agent reasoning trace inline."""
    if not entries:
        return
    html = '<div class="agent-trace">'
    for entry in entries[-8:]:  # show last 8 entries only
        t = entry.get("type", "")
        content = entry.get("content", "")
        if t == "thought":
            html += f'<div class="agent-thought"><span class="icon">\U0001f4ad</span> {content}</div>'
        elif t == "action":
            html += f'<div class="agent-action">\u25b6 {content}</div>'
        elif t == "observation":
            tool = entry.get("tool", "")
            short = content[:200] + ("..." if len(content) > 200 else "")
            html += f'<div class="agent-observation"><strong>{tool}</strong>: {short}</div>'
    html += '</div>'
    placeholder.markdown(html, unsafe_allow_html=True)


# Show messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        meta = []
        if msg.get("elapsed"): meta.append(f"{msg['elapsed']}s")
        sc = msg.get("sources_count", 0)
        if sc: meta.append(f"{sc} steps")
        # Show trace for agentic mode
        trace = msg.get("trace", [])
        if trace:
            _render_trace(st.empty(), trace)
        st.markdown(f'<div class="msg-ai"><div class="report">{msg["content"]}</div><div class="msg-meta">{" · ".join(meta)}</div></div>', unsafe_allow_html=True)

# ── Input ───────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
query = st.text_area(
    "Query", placeholder="Send a message...", height=68,
    label_visibility="collapsed", key="user_input",
    value=prefill if prefill else "",
)

col1, col2 = st.columns([10, 1])
with col2:
    send = st.button("\u27a4", type="primary", use_container_width=True, key="send_btn")

# ── Send handler ─────────────────────────────────────────────
def execute_search(user_query: str):
    config = AppConfig.from_env()
    config.enable_sources = sources
    config.verbose = verbose
    config.llm.provider = llm_provider
    if api_key: config.llm.api_key = api_key
    if model: config.llm.model = model
    config.llm.base_url = {
        "deepseek":"https://api.deepseek.com/v1",
        "qwen":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai":"https://api.openai.com/v1",
    }[llm_provider]
    config.search.max_results_per_source = max_results

    # Build conversation history from messages
    conv_history = []
    for m in st.session_state.messages[-6:]:  # last 3 turns
        conv_history.append({"role": m["role"], "content": m.get("content","")})

    st.session_state.messages.append({"role": "user", "content": user_query})

    # Placeholder for AI response
    ai_idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": "", "elapsed": "", "sources_count": 0, "trace": []})
    st.rerun()


def finish_streaming(user_query: str, ai_idx: int):
    """Actually run the streaming search and update the message in-place."""
    config = AppConfig.from_env()
    config.enable_sources = sources
    config.verbose = verbose
    config.llm.provider = llm_provider
    if api_key: config.llm.api_key = api_key
    if model: config.llm.model = model
    config.llm.base_url = {
        "deepseek":"https://api.deepseek.com/v1",
        "qwen":"https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai":"https://api.openai.com/v1",
    }[llm_provider]
    config.search.max_results_per_source = max_results

    conv_history = []
    for m in st.session_state.messages[:-1]:  # exclude the placeholder
        conv_history.append({"role": m["role"], "content": m.get("content","")})

    status_placeholder = st.empty()
    trace_placeholder = st.empty()
    answer_placeholder = st.empty()
    full_text = ""
    search_meta = {}
    trace_entries = []
    start = time.time()

    # Initial visible feedback so the user knows something is happening
    status_placeholder.caption("\u25cf \u6b63\u5728\u542f\u52a8\u4ee3\u7406...")

    try:
        async def stream():
            agen = run_qa_stream(query=user_query, config=config)
            async for event in agen:
                yield event

        gen = stream()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while True:
                try:
                    event = loop.run_until_complete(gen.__anext__())
                except StopAsyncIteration:
                    break

                etype = event.get("type", "")

                if etype == "status":
                    status_placeholder.caption(f"\u25cf {event['content']}")
                elif etype == "log":
                    trace_entries.append({"type": "thought", "content": event["content"]})
                    _render_trace(trace_placeholder, trace_entries)
                elif etype == "thought":
                    trace_entries.append({"type": "thought", "content": event["content"]})
                    _render_trace(trace_placeholder, trace_entries)
                    status_placeholder.caption(f"\u25cf {event['content']}")
                elif etype == "action":
                    trace_entries.append({"type": "action", "content": event["content"]})
                    _render_trace(trace_placeholder, trace_entries)
                    status_placeholder.caption(f"\u25cf {event['content']}")
                elif etype == "observation":
                    trace_entries.append({"type": "observation", "content": event["content"], "tool": event.get("meta", {}).get("tool", "")})
                    _render_trace(trace_placeholder, trace_entries)
                elif etype == "chunk":
                    full_text += event["content"]
                    answer_placeholder.markdown(f'<div class="report">{full_text}</div>', unsafe_allow_html=True)
                elif etype == "done":
                    full_text = event["content"]
                    search_meta = event.get("meta", {})
                    status_placeholder.empty()
                    trace_placeholder.empty()
                    answer_placeholder.markdown(f'<div class="report">{full_text}</div>', unsafe_allow_html=True)
                    break
        finally:
            loop.close()

    except Exception as e:
        full_text = f"\u26a0\ufe0f \u51fa\u9519\uff1a{str(e)}\n\n\u8bf7\u68c0\u67e5\u4ee3\u7406\u914d\u7f6e\uff08API Key\u3001\u6a21\u578b\u540d\u3001\u7f51\u7edc\u8bbf\u95ee\uff09\u540e\u91cd\u8bd5\u3002"
        status_placeholder.empty()
        trace_placeholder.empty()
        answer_placeholder.markdown(f'<div class="report">{full_text}</div>', unsafe_allow_html=True)

    elapsed = time.time() - start
    sr_count = search_meta.get("search_results", len(search_meta.get("logs", [])))
    if isinstance(sr_count, list):
        sr_count = len(sr_count)

    st.session_state.messages[ai_idx] = {
        "role": "assistant",
        "content": full_text,
        "elapsed": f"{elapsed:.1f}",
        "sources_count": search_meta.get("tool_calls_made", sr_count),
        "trace": trace_entries,
    }

    # Store in memory
    if full_text:
        st.session_state.memory.store(user_query, full_text, search_meta.get("tool_calls_made", 0))


# Check if we need to stream (there's a placeholder message)
placeholder_idx = None
for i, m in enumerate(st.session_state.messages):
    if m["role"] == "assistant" and m["content"] == "" and m.get("elapsed") == "":
        placeholder_idx = i
        break

if placeholder_idx is not None:
    user_msg = st.session_state.messages[placeholder_idx - 1]["content"]
    finish_streaming(user_msg, placeholder_idx)
    st.rerun()

if send and query.strip():
    execute_search(query.strip())
