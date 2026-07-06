"""
RAG Chatbot — Streamlit UI

Run with:  streamlit run app.py
Ingest data first:  python ingest.py
"""
import logging
import threading
import time
import traceback
import uuid
import streamlit as st
from rag import RagConfig, RagPipeline, Confidence, CONF_META, OrchestratorResult
from rag.session_store import SessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Background warm-up ────────────────────────────────────────────────────────
_warmup_status: dict = {}          # module-level: persists across Streamlit reruns
_warmup_thread: threading.Thread | None = None


def _run_warmup(pipeline: RagPipeline) -> None:
    try:
        status = pipeline.warmup()
        _warmup_status.update(status)
        logger.info("Warm-up done: %s", status)
    except Exception:
        logger.warning("Warm-up failed", exc_info=True)
        _warmup_status.update({"bm25": "failed", "reranker": "failed"})


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    .session-item {
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        color: #e6edf3;
        border: 1px solid transparent;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .session-item.active {
        background: #1f3a5f;
        border-color: #388bfd;
        color: #58a6ff;
        font-weight: 600;
    }
    .session-item:hover { background: #21262d; }
    .source-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 6px;
    }
    .source-title { font-weight: 600; font-size: 13px; color: #e6edf3; }
    .source-meta { font-size: 11px; color: #7d8590; margin-top: 3px; }
    .conf-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 12px; font-size: 12px;
        font-weight: 600; margin-bottom: 8px;
    }
    .stat-box {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 8px; padding: 14px; text-align: center;
    }
    .stat-num { font-size: 26px; font-weight: 700; color: #58a6ff; }
    .stat-label { font-size: 11px; color: #7d8590; margin-top: 2px; }
    .empty-state { text-align: center; padding: 60px 20px; color: #7d8590; }
    .empty-state h3 { color: #e6edf3; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _new_session_id() -> str:
    return uuid.uuid4().hex[:8]


def init_state():
    if "pipeline" not in st.session_state:
        logger.info("Initialising RagPipeline...")
        try:
            st.session_state.config = RagConfig.from_env()
            st.session_state.pipeline = RagPipeline(st.session_state.config)
            logger.info("RagPipeline ready. Chunks in DB: %d", st.session_state.pipeline.total_chunks())
        except Exception:
            logger.exception("RagPipeline init failed")
            st.error(f"Pipeline init failed:\n```\n{traceback.format_exc()}\n```")
            st.stop()

    global _warmup_thread
    if _warmup_thread is None:
        _warmup_status.update({"bm25": "loading", "reranker": "loading"})
        _warmup_thread = threading.Thread(
            target=_run_warmup, args=(st.session_state.pipeline,), daemon=True)
        _warmup_thread.start()
        logger.info("Background warm-up started (BM25 + reranker)")

    if "store" not in st.session_state:
        logger.info("Initialising SessionStore...")
        try:
            st.session_state.store = SessionStore(st.session_state.config)
            logger.info("SessionStore ready")
        except Exception:
            logger.exception("SessionStore init failed")
            st.warning(f"MongoDB unavailable — sessions won't persist:\n```\n{traceback.format_exc()}\n```")
            st.session_state.store = None

    # Load sessions from MongoDB on first run
    if "sessions" not in st.session_state:
        loaded = st.session_state.store.load_all() if st.session_state.store else {}
        if loaded:
            st.session_state.sessions = loaded
            st.session_state.current_session = next(iter(loaded))
        else:
            first_id = _new_session_id()
            first_session = {"name": "Chat 1", "messages": []}
            st.session_state.sessions = {first_id: first_session}
            st.session_state.current_session = first_id
            if st.session_state.store:
                st.session_state.store.save(first_id, first_session)

    if "current_session" not in st.session_state:
        st.session_state.current_session = next(iter(st.session_state.sessions))


init_state()

pipeline: RagPipeline = st.session_state.pipeline
store: SessionStore = st.session_state.store
sessions: dict = st.session_state.sessions
current_id: str = st.session_state.current_session

# Guard: current session might have been deleted
if current_id not in sessions:
    current_id = next(iter(sessions))
    st.session_state.current_session = current_id

current_session = sessions[current_id]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Chatbot")
    st.markdown("---")

    # ── Chat sessions ─────────────────────────────────────────────────────────
    col_sess, col_new = st.columns([3, 1])
    with col_sess:
        st.markdown("### 💬 Sessions")
    with col_new:
        if st.button("＋", help="New chat session", use_container_width=True):
            new_id = _new_session_id()
            new_sess = {"name": f"Chat {len(sessions) + 1}", "messages": []}
            sessions[new_id] = new_sess
            if store:
                store.save(new_id, new_sess)
            st.session_state.current_session = new_id
            st.rerun()

    for sid, sess in list(sessions.items()):
        is_active = sid == current_id
        col_a, col_b = st.columns([5, 1])
        with col_a:
            label = ("▶ " if is_active else "") + sess["name"]
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.current_session = sid
                st.rerun()
        with col_b:
            if len(sessions) > 1:
                if st.button("✕", key=f"del_sess_{sid}", help="Delete session"):
                    del sessions[sid]
                    if store:
                        store.delete(sid)
                    if is_active:
                        st.session_state.current_session = next(iter(sessions))
                    st.rerun()

    st.markdown("---")

    # ── Rename current session ────────────────────────────────────────────────
    new_name = st.text_input(
        "Rename session",
        value=current_session["name"],
        key=f"rename_{current_id}",
        label_visibility="collapsed",
        placeholder="Session name...",
    )
    if new_name and new_name != current_session["name"]:
        current_session["name"] = new_name
        if store:
            store.save(current_id, current_session)

    if st.button("🧹 Clear Chat", use_container_width=True, disabled=not current_session["messages"]):
        current_session["messages"] = []
        if store:
            store.save(current_id, current_session)
        st.rerun()

    st.markdown("---")

    # ── Knowledge base stats ──────────────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    sources = pipeline.list_sources()
    total_chunks = pipeline.total_chunks()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{len(sources)}</div>
            <div class="stat-label">Sources</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-num">{total_chunks}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    st.caption("Run `python ingest.py` to add data.")

    if sources:
        st.markdown("")
        for src in sources:
            type_icon = {"web": "🌐", "pdf": "📄", "docx": "📝", "text": "📋", "csv": "📊"}.get(src["type"], "📁")
            st.markdown(f"""<div class="source-card">
                <div class="source-title">{type_icon} {src['title']}</div>
                <div class="source-meta">{src['chunks']} chunks · {src['type']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ System")
    _si = {"loading": ("🟡", "Loading…"), "ready": ("🟢", "Ready"), "failed": ("🔴", "Failed")}
    for component, label in [("bm25", "BM25 Index"), ("reranker", "Reranker")]:
        icon, text = _si.get(_warmup_status.get(component, "loading"), ("🟡", "Loading…"))
        st.markdown(f"{icon} **{label}** — {text}")


# ── Main: Chat Interface ──────────────────────────────────────────────────────
st.markdown(f"# {current_session['name']}")

if not sources:
    st.markdown("""<div class="empty-state">
        <h3>No knowledge base yet</h3>
        <p>Run <code>python ingest.py</code> after dropping files in <code>data/</code>
        or adding URLs to <code>data/web_urls.txt</code>.</p>
    </div>""", unsafe_allow_html=True)

# Display chat history for current session
messages = current_session["messages"]

for msg in messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta: OrchestratorResult = msg["meta"]
            conf_info = CONF_META.get(meta.confidence, CONF_META[Confidence.NONE])
            st.markdown(
                f'<div class="conf-badge" style="background:{conf_info["color"]}22;'
                f'color:{conf_info["color"]};border:1px solid {conf_info["color"]}44">'
                f'{conf_info["icon"]} {conf_info["label"]}'
                f' · {meta.latency_ms:.0f}ms</div>',
                unsafe_allow_html=True,
            )
            if getattr(meta, "staleness_note", ""):
                st.caption(f"⏳ {meta.staleness_note}")
            if meta.sources:
                with st.expander(f"📎 {len(meta.sources)} source(s) used", expanded=False):
                    for s in meta.sources:
                        st.markdown(f"- **{s['title']}** — {s.get('section', '')} *(score: {s['score']})*")

# Chat input
if prompt := st.chat_input(f"Ask a question… ({current_session['name']})"):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]

    with st.chat_message("assistant", avatar="🧠"):
        full_response = ""
        meta_result = None
        placeholder = st.empty()

        try:
            for chunk in pipeline.stream(prompt, history):
                if isinstance(chunk, str):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, OrchestratorResult):
                    meta_result = chunk

            placeholder.markdown(full_response)

            if meta_result:
                conf_info = CONF_META.get(meta_result.confidence, CONF_META[Confidence.NONE])
                st.markdown(
                    f'<div class="conf-badge" style="background:{conf_info["color"]}22;'
                    f'color:{conf_info["color"]};border:1px solid {conf_info["color"]}44">'
                    f'{conf_info["icon"]} {conf_info["label"]}'
                    f' · {meta_result.latency_ms:.0f}ms</div>',
                    unsafe_allow_html=True,
                )
                if getattr(meta_result, "staleness_note", ""):
                    st.caption(f"⏳ {meta_result.staleness_note}")
                if meta_result.sources:
                    with st.expander(f"📎 {len(meta_result.sources)} source(s) used", expanded=False):
                        for s in meta_result.sources:
                            st.markdown(f"- **{s['title']}** — {s.get('section', '')} *(score: {s['score']})*")

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Chat processing failed for prompt: %r", prompt)
            full_response = f"Error: {e}"
            placeholder.error(f"**Error:** {e}\n\n```\n{tb}\n```")
            meta_result = OrchestratorResult(response=full_response, confidence=Confidence.NONE)

    messages.append({
        "role": "assistant",
        "content": full_response,
        "meta": meta_result,
    })
    if store:
        store.save(current_id, current_session)

# Auto-refresh until warm-up complete (polls at 0.75s intervals)
if not all(v in ("ready", "failed") for v in _warmup_status.values()):
    time.sleep(0.75)
    st.rerun()
