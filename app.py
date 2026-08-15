"""
app.py
------
Main Streamlit entry point for the AI Study Assistant.

Tabs:
    📥 Ingest       — Upload PDFs, add YouTube/web URLs
    📋 Summary      — Generate structured topic summaries
    🃏 Flashcards   — Generate and review flashcard decks
    📝 Quiz         — Generate and take interactive quizzes
    📊 Progress     — View quiz history and performance profile

Run with:
    streamlit run app.py

Requires .env file with:
    OPENAI_API_KEY=sk-...         (for quiz chain + embeddings)
    GOOGLE_API_KEY=your_key_here  (for summary + flashcard chains)
"""

import streamlit as st

# ── Page config — must be the FIRST Streamlit call ────────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after set_page_config) ───────────────────────────────────────
import os
import json
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from vector_store.store_manager import get_store
from chains.quiz_chain import generate_quiz, score_quiz
from chains.flashcard_chain import generate_flashcards, deck_to_csv, deck_to_anki_format
from chains.summary_chain import generate_summary, get_suggested_quiz_topics
from models.schemas import DifficultyLevel, PerformanceProfile

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Fonts ───────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ──────────────────────────────────────────────── */
:root {
    --bg:           #0f1117;
    --surface:      #1a1d27;
    --surface-2:    #222535;
    --border:       #2e3148;
    --accent:       #7c9bff;
    --accent-2:     #b79dfd;
    --success:      #34d399;
    --warning:      #fbbf24;
    --danger:       #f87171;
    --text-primary: #f4f5fb;
    --text-secondary: #c3c7e0;
    --text-muted:   #9297b8;
    --font-body:    'Sora', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --radius:       12px;
    --radius-sm:    8px;
}

/* ── Base — force every text-bearing element to a readable color ─── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}

/* Streamlit ships its own inline color rules (near-black) on markdown
   text, labels, and widget chrome that otherwise beat the rule above.
   Reassert color on every generic text container so nothing renders
   dark-on-dark. */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp div[data-testid="stMarkdownContainer"],
.stApp div[data-testid="stMarkdownContainer"] p,
.stApp div[data-testid="stText"],
.stApp .stMarkdown,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stCaptionContainer"] {
    color: var(--text-primary) !important;
}

/* ── App background ─────────────────────────────────────────────── */
.stApp {
    background: var(--bg) !important;
}

/* ── Top header bar (the white strip above the title) ─────────────── */
[data-testid="stHeader"] {
    background: var(--bg) !important;
    background-color: var(--bg) !important;
}
[data-testid="stDecoration"] {
    background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}
[data-testid="stToolbar"] {
    background: var(--bg) !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Headings ───────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--text-muted) !important;
}
[data-testid="stTabs"] button p {
    color: inherit !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: var(--border) !important;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
    color: var(--text-primary) !important;
    transition: all 0.15s ease !important;
}
.stButton > button p {
    color: inherit !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(124,155,255,0.1) !important;
}
.stButton > button:disabled {
    color: var(--text-muted) !important;
    opacity: 0.5 !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #0f1117 !important;
    border-color: var(--accent) !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"] p {
    color: #0f1117 !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.88 !important;
    color: #0f1117 !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--surface-2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Inputs ─────────────────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] {
    font-family: var(--font-body) !important;
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 1 !important;
}
[data-baseweb="select"] span {
    color: var(--text-primary) !important;
}
/* Dropdown menu (selectbox options list) */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-baseweb="popover"] [role="option"]:hover {
    background: var(--surface) !important;
}

/* Radio / checkbox labels */
.stRadio label, .stCheckbox label, .stSlider label,
.stRadio [data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}
.stRadio [role="radiogroup"] label span {
    color: var(--text-primary) !important;
}

/* Slider track/value text */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
.stSlider [data-baseweb="slider"] div {
    color: var(--text-muted) !important;
}
[data-testid="stThumbValue"] {
    color: var(--text-primary) !important;
    background: var(--surface-2) !important;
}

/* Toggle */
.stToggle label p {
    color: var(--text-primary) !important;
}

/* ── Cards ──────────────────────────────────────────────────────── */
.study-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    color: var(--text-primary);
}
.study-card * {
    color: inherit;
}
.study-card.accent-left {
    border-left: 3px solid var(--accent);
}
.study-card.success {
    border-left: 3px solid var(--success);
    background: rgba(52,211,153,0.08);
}
.study-card.danger {
    border-left: 3px solid var(--danger);
    background: rgba(248,113,113,0.08);
}

/* ── Source badge ───────────────────────────────────────────────── */
.source-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    background: rgba(124,155,255,0.15);
    color: var(--accent) !important;
    border: 1px solid rgba(124,155,255,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 3px 2px 0;
}

/* ── Tag badge ──────────────────────────────────────────────────── */
.tag-badge {
    display: inline-block;
    font-size: 0.72rem;
    background: rgba(183,157,253,0.15);
    color: var(--accent-2) !important;
    border: 1px solid rgba(183,157,253,0.3);
    border-radius: 4px;
    padding: 2px 8px;
    margin: 2px 3px 2px 0;
}

/* ── Section header ─────────────────────────────────────────────── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted) !important;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Progress bar ───────────────────────────────────────────────── */
.score-bar-wrap {
    background: var(--surface-2);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 0.4rem 0;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Flashcard flip ─────────────────────────────────────────────── */
.flip-card {
    perspective: 1000px;
    cursor: pointer;
    height: 200px;
    margin-bottom: 1rem;
}
.flip-card-inner {
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.5s ease;
    transform-style: preserve-3d;
}
.flip-card.flipped .flip-card-inner {
    transform: rotateY(180deg);
}
.flip-card-front, .flip-card-back {
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
    text-align: center;
    color: var(--text-primary);
}
.flip-card-front {
    background: var(--surface);
    border: 1px solid var(--border);
}
.flip-card-back {
    background: var(--surface-2);
    border: 1px solid var(--accent);
    transform: rotateY(180deg);
}

/* ── Metric box ─────────────────────────────────────────────────── */
.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-box .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent) !important;
    line-height: 1;
}
.metric-box .label {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Streamlit overrides ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}
[data-testid="stFileUploader"] section {
    background: var(--surface) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploader"] button {
    background: var(--surface-2) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--surface) !important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
    color: var(--text-primary) !important;
}

/* Alert boxes (info / success / warning / error) — keep tinted
   backgrounds but force legible text on top of them */
.stAlert, .stAlert p, .stAlert div, .stAlert span {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stAlertContentInfo"] {
    background: rgba(124,155,255,0.1) !important;
}
div[data-testid="stAlertContentSuccess"] {
    background: rgba(52,211,153,0.1) !important;
}
div[data-testid="stAlertContentWarning"] {
    background: rgba(251,191,36,0.1) !important;
}
div[data-testid="stAlertContentError"] {
    background: rgba(248,113,113,0.1) !important;
}

/* Spinner text */
.stSpinner p {
    color: var(--text-primary) !important;
}

/* Progress bar widget (st.progress) */
[data-testid="stProgress"] > div > div {
    background: var(--accent) !important;
}
[data-testid="stProgress"] > div {
    background: var(--surface-2) !important;
}

/* Columns / containers shouldn't leak white backgrounds */
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialise all session state keys exactly once per session."""
    defaults = {
        # Ingestion
        "ingested_sources"    : [],          # list of source dicts from store

        # Quiz state
        "active_quiz"         : None,        # current Quiz object
        "quiz_answers"        : {},          # {question_index: selected_option_index}
        "quiz_submitted"      : False,
        "quiz_result"         : None,        # result dict from score_quiz()
        "quiz_topic"          : "",

        # Flashcard state
        "active_deck"         : None,        # current FlashcardDeck object
        "card_index"          : 0,           # which card is shown
        "card_flipped"        : False,       # is current card flipped?

        # Summary state
        "active_summary"      : None,        # current TopicSummary object
        "summary_quiz_topics" : [],          # topics extracted from summary

        # Performance tracking
        "performance_profile" : PerformanceProfile(),
        "quiz_history"        : [],          # list of score result dicts
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the persistent sidebar with ingestion status."""
    with st.sidebar:
        st.markdown("""
        <div style='padding: 0.5rem 0 1.5rem 0;'>
            <div style='font-size:1.6rem; font-weight:700; letter-spacing:-0.03em; color:var(--text-primary);'>
                🎓 Study Assistant
            </div>
            <div style='font-size:0.8rem; color:var(--text-muted); margin-top:0.25rem;'>
                Powered by RAG + Gemini
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Knowledge Base</div>',
                    unsafe_allow_html=True)

        # Refresh source list from store
        try:
            store   = get_store()
            summary = store.get_summary()
            sources = store.list_sources()
            st.session_state.ingested_sources = sources
        except Exception:
            summary = {"total_chunks": 0, "total_sources": 0, "by_type": {}}
            sources = []

        # Stats row
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{summary['total_sources']}</div>
                <div class="label">Sources</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value">{summary['total_chunks']}</div>
                <div class="label">Chunks</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Source list
        if sources:
            for s in sources:
                icon = {"pdf": "📄", "youtube": "▶️", "website": "🌐"}.get(
                    s["doc_type"], "📁"
                )
                st.markdown(f"""
                <div class="study-card" style="padding:0.6rem 0.9rem; margin-bottom:0.4rem;">
                    <div style="font-size:0.8rem; font-weight:500;">
                        {icon} {s['source'][:35]}{'…' if len(s['source'])>35 else ''}
                    </div>
                    <div style="font-size:0.7rem; color:var(--text-muted);">
                        {s['chunks']} chunks · {s['doc_type']}
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Clear all button
            if st.button("🗑️ Clear All Sources", use_container_width=True):
                try:
                    get_store().clear_all()
                    st.session_state.ingested_sources = []
                    st.success("Knowledge base cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Clear failed: {e}")
        else:
            st.markdown("""
            <div style='color:var(--text-muted); font-size:0.82rem;
                        text-align:center; padding:1rem 0;'>
                No documents yet.<br>Go to the Ingest tab to add study material.
            </div>""", unsafe_allow_html=True)

        # Quiz performance snapshot
        profile = st.session_state.performance_profile
        if profile.history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Quiz Performance</div>',
                        unsafe_allow_html=True)
            last = profile.history[-1]
            pct  = last.percentage
            bar_color = (
                "var(--success)"  if pct >= 70 else
                "var(--warning)"  if pct >= 40 else
                "var(--danger)"
            )
            st.markdown(f"""
            <div style='font-size:0.8rem; color:var(--text-muted);
                        margin-bottom:0.3rem;'>
                Last quiz: {last.topic}
            </div>
            <div class='score-bar-wrap'>
                <div class='score-bar-fill'
                     style='width:{pct}%; background:{bar_color};'></div>
            </div>
            <div style='font-size:0.75rem; color:var(--text-muted);
                        text-align:right;'>{last.score}/{last.total} · {pct}%</div>
            """, unsafe_allow_html=True)

render_sidebar()


# ══════════════════════════════════════════════════════════════════════════
# MAIN TITLE
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style='padding: 1.5rem 0 1rem 0;'>
    <h1 style='margin:0; font-size:2rem; letter-spacing:-0.04em; color:var(--text-primary);'>
        AI Study Assistant
    </h1>
    <p style='color:var(--text-muted); margin:0.3rem 0 0 0; font-size:0.9rem;'>
        Feed your notes, lectures, and textbooks — get quizzes, flashcards, and summaries.
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════

tab_ingest, tab_summary, tab_flash, tab_quiz, tab_progress = st.tabs([
    "📥 Ingest",
    "📋 Summary",
    "🃏 Flashcards",
    "📝 Quiz",
    "📊 Progress",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════

with tab_ingest:
    st.markdown("### Add Study Material")
    st.markdown(
        "<p style='color:var(--text-muted); font-size:0.88rem;'>"
        "Upload PDFs, add YouTube lecture links, or paste website URLs. "
        "All content is chunked, embedded, and stored locally."
        "</p>", unsafe_allow_html=True
    )

    col_pdf, col_links = st.columns([1, 1], gap="large")

    # ── PDF Upload ───────────────────────────────────────────────────────
    with col_pdf:
        st.markdown('<div class="section-header">📄 PDF Documents</div>',
                    unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("⚡ Ingest PDFs", type="primary",
                         use_container_width=True):
                from ingestion.pdf_loader import ingest_pdf

                progress = st.progress(0)
                total    = len(uploaded_files)
                success  = 0

                for i, uf in enumerate(uploaded_files):
                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            # Save uploaded bytes to a temp file
                            with tempfile.NamedTemporaryFile(
                                suffix=".pdf", delete=False
                            ) as tmp:
                                tmp.write(uf.read())
                                tmp_path = tmp.name

                            chunks = ingest_pdf(tmp_path)

                            # Rename source metadata to the original filename
                            for c in chunks:
                                c.metadata["source"] = uf.name

                            get_store().add_documents(chunks)
                            success += 1
                            os.unlink(tmp_path)

                        except Exception as e:
                            st.error(f"Failed: {uf.name} — {e}")

                    progress.progress((i + 1) / total)

                progress.empty()
                if success:
                    st.success(f"✅ Ingested {success}/{total} PDF(s) successfully.")
                    st.rerun()

    # ── YouTube + Web URLs ───────────────────────────────────────────────
    with col_links:
        st.markdown('<div class="section-header">🔗 YouTube & Web URLs</div>',
                    unsafe_allow_html=True)

        url_input = st.text_area(
            "URLs (one per line)",
            placeholder=(
                "https://youtu.be/VIDEO_ID\n"
                "https://en.wikipedia.org/wiki/Transformer\n"
                "https://arxiv.org/abs/1706.03762"
            ),
            height=130,
            label_visibility="collapsed",
        )

        if url_input.strip():
            if st.button("⚡ Ingest URLs", type="primary",
                         use_container_width=True):
                from ingestion.youtube_loader import ingest_youtube
                from ingestion.web_loader import ingest_url

                raw_urls = [u.strip() for u in url_input.strip().splitlines()
                            if u.strip()]
                progress = st.progress(0)
                success  = 0

                for i, url in enumerate(raw_urls):
                    with st.spinner(f"Processing {url[:55]}…"):
                        try:
                            # Detect YouTube vs general web
                            if any(x in url for x in
                                   ["youtube.com", "youtu.be"]):
                                chunks = ingest_youtube(url)
                            else:
                                chunks = ingest_url(url)

                            get_store().add_documents(chunks)
                            success += 1
                        except Exception as e:
                            st.error(f"Failed: {url[:55]} — {e}")

                    progress.progress((i + 1) / len(raw_urls))

                progress.empty()
                if success:
                    st.success(
                        f"✅ Ingested {success}/{len(raw_urls)} URL(s)."
                    )
                    st.rerun()

    # ── Ingestion status table ───────────────────────────────────────────
    sources = st.session_state.ingested_sources
    if sources:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Ingested Sources</div>',
                    unsafe_allow_html=True)

        for s in sources:
            icon = {"pdf": "📄", "youtube": "▶️", "website": "🌐"}.get(
                s["doc_type"], "📁"
            )
            col_name, col_type, col_chunks, col_del = st.columns(
                [4, 1.5, 1, 0.8]
            )
            with col_name:
                st.markdown(f"""
                <div style='font-size:0.85rem; padding-top:0.45rem; color:var(--text-primary);'>
                    {icon} {s['source']}
                </div>""", unsafe_allow_html=True)
            with col_type:
                st.markdown(f"""
                <div class='source-badge' style='margin-top:0.35rem;
                     display:inline-block;'>{s['doc_type']}</div>
                """, unsafe_allow_html=True)
            with col_chunks:
                st.markdown(f"""
                <div style='font-size:0.82rem; color:var(--text-muted);
                            padding-top:0.45rem;'>{s['chunks']} chunks</div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("✕", key=f"del_{s['source']}"):
                    get_store().delete_source(s["source"])
                    st.rerun()
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "No documents ingested yet. Upload PDFs or add URLs above to get started."
        )
# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════

with tab_summary:
    st.markdown("### Topic Summary")
    st.markdown(
        "<p style='color:var(--text-muted); font-size:0.88rem;'>"
        "Generate a structured study guide for any topic in your knowledge base."
        "</p>", unsafe_allow_html=True
    )

    # ── Controls ─────────────────────────────────────────────────────────
    col_input, col_opts = st.columns([3, 1], gap="large")

    with col_input:
        summary_topic = st.text_input(
            "Topic",
            placeholder="e.g. backpropagation, attention mechanism, gradient descent…",
            key="summary_topic_input",
        )

    with col_opts:
        summary_doc_type = st.selectbox(
            "Filter by source type",
            options=["All", "pdf", "youtube", "website"],
            key="summary_doc_type",
        )

    generate_summary_btn = st.button(
        "✨ Generate Summary", type="primary", key="gen_summary_btn"
    )

    if generate_summary_btn:
        if not summary_topic.strip():
            st.warning("Please enter a topic.")
        elif not st.session_state.ingested_sources:
            st.warning("No documents ingested yet. Go to the Ingest tab first.")
        else:
            doc_type_filter = (
                None if summary_doc_type == "All" else summary_doc_type
            )
            with st.spinner(f"Summarising '{summary_topic}'…"):
                try:
                    result = generate_summary(
                        topic    = summary_topic,
                        doc_type = doc_type_filter,
                    )
                    st.session_state.active_summary      = result["summary"]
                    st.session_state.summary_quiz_topics = (
                        get_suggested_quiz_topics(result["summary"])
                    )
                    strategy_label = (
                        "map-reduce (large context)"
                        if result["strategy"] == "map_reduce"
                        else "single pass"
                    )
                    st.success(
                        f"Summary generated using {strategy_label} · "
                        f"{result['chunks_used']} chunks retrieved"
                    )
                except ValueError as e:
                    st.error(str(e))
                except RuntimeError as e:
                    st.error(f"Generation failed: {e}")

    # ── Render summary ────────────────────────────────────────────────────
    summary = st.session_state.active_summary
    if summary:
        st.markdown("<br>", unsafe_allow_html=True)

        # Header
        st.markdown(f"""
        <div class='study-card accent-left'>
            <div style='font-size:1.25rem; font-weight:600;
                        letter-spacing:-0.02em;'>{summary.topic}</div>
            <div style='margin-top:0.4rem;'>
                <span class='source-badge'>{summary.source_hint[:60]}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        col_kp, col_terms = st.columns([1, 1], gap="large")

        # Key points
        with col_kp:
            st.markdown('<div class="section-header">Key Points</div>',
                        unsafe_allow_html=True)
            for point in summary.key_points:
                st.markdown(f"""
                <div class='study-card' style='padding:0.7rem 1rem;
                     margin-bottom:0.5rem;'>
                    <span style='color:var(--accent); margin-right:0.5rem;'>▸</span>
                    {point}
                </div>""", unsafe_allow_html=True)

        # Key terms
        with col_terms:
            st.markdown('<div class="section-header">Key Terms</div>',
                        unsafe_allow_html=True)
            for kt in summary.key_terms:
                st.markdown(f"""
                <div class='study-card' style='padding:0.7rem 1rem;
                     margin-bottom:0.5rem;'>
                    <span style='font-weight:600; color:var(--accent-2);'>
                        {kt.term}
                    </span><br>
                    <span style='font-size:0.85rem; color:var(--text-muted);'>
                        {kt.definition}
                    </span>
                </div>""", unsafe_allow_html=True)

        # Detailed summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Detailed Summary</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='study-card' style='line-height:1.75; font-size:0.9rem;'>
            {summary.detailed_summary}
        </div>""", unsafe_allow_html=True)

        # Suggested questions + prerequisites side by side
        col_sq, col_pre = st.columns([3, 2], gap="large")

        with col_sq:
            st.markdown('<div class="section-header">Suggested Questions</div>',
                        unsafe_allow_html=True)
            for i, q in enumerate(summary.suggested_questions, 1):
                st.markdown(f"""
                <div class='study-card' style='padding:0.65rem 1rem;
                     margin-bottom:0.4rem; font-size:0.875rem;'>
                    <span style='color:var(--text-muted);
                                 font-family:var(--font-mono);
                                 font-size:0.75rem;'>Q{i}</span>
                    &nbsp;{q}
                </div>""", unsafe_allow_html=True)

        with col_pre:
            if summary.prerequisites:
                st.markdown('<div class="section-header">Prerequisites</div>',
                            unsafe_allow_html=True)
                for p in summary.prerequisites:
                    st.markdown(f"""
                    <div class='study-card' style='padding:0.65rem 1rem;
                         margin-bottom:0.4rem; font-size:0.875rem;'>
                        <span style='color:var(--warning);
                                     margin-right:0.4rem;'>⚑</span>{p}
                    </div>""", unsafe_allow_html=True)

        # Quick-jump buttons to generate quiz/flashcards on a sub-topic
        if st.session_state.summary_quiz_topics:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Drill Down — Generate Quiz or '
                'Flashcards on a Sub-topic</div>',
                unsafe_allow_html=True
            )
            topics_display = st.session_state.summary_quiz_topics[:8]
            cols = st.columns(min(4, len(topics_display)))
            for i, t in enumerate(topics_display):
                with cols[i % 4]:
                    if st.button(
                        t[:30], key=f"drill_{i}",
                        use_container_width=True,
                        help=f"Pre-fill quiz/flashcard topic with '{t}'"
                    ):
                        st.session_state.quiz_topic = t
                        st.info(
                            f"Topic set to **{t}**. "
                            f"Switch to the Quiz or Flashcards tab."
                        )


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — FLASHCARDS
# ══════════════════════════════════════════════════════════════════════════

with tab_flash:
    st.markdown("### Flashcard Deck")
    st.markdown(
        "<p style='color:var(--text-muted); font-size:0.88rem;'>"
        "Generate flashcards from your study material and review them "
        "with an interactive card flip interface."
        "</p>", unsafe_allow_html=True
    )

    # ── Controls ─────────────────────────────────────────────────────────
    col_fi, col_fn, col_fdt = st.columns([3, 1, 1], gap="medium")

    with col_fi:
        flash_topic = st.text_input(
            "Topic",
            value=st.session_state.get("quiz_topic", ""),
            placeholder="e.g. transformer architecture, linked lists…",
            key="flash_topic_input",
        )
    with col_fn:
        num_cards = st.slider("Cards", min_value=5, max_value=20,
                              value=10, key="num_cards_slider")
    with col_fdt:
        flash_doc_type = st.selectbox(
            "Source type", ["All", "pdf", "youtube", "website"],
            key="flash_doc_type"
        )

    if st.button("🃏 Generate Flashcards", type="primary",
                 key="gen_flash_btn"):
        if not flash_topic.strip():
            st.warning("Please enter a topic.")
        elif not st.session_state.ingested_sources:
            st.warning("No documents ingested yet.")
        else:
            doc_type_filter = (
                None if flash_doc_type == "All" else flash_doc_type
            )
            with st.spinner(f"Generating {num_cards} flashcards on '{flash_topic}'…"):
                try:
                    result = generate_flashcards(
                        topic     = flash_topic,
                        num_cards = num_cards,
                        doc_type  = doc_type_filter,
                    )
                    st.session_state.active_deck  = result["deck"]
                    st.session_state.card_index   = 0
                    st.session_state.card_flipped = False
                    st.success(
                        f"Generated {result['deck'].card_count} cards · "
                        f"{result['chunks_used']} chunks retrieved · "
                        f"Model: {result['model']}"
                    )
                except (ValueError, RuntimeError, EnvironmentError) as e:
                    st.error(str(e))

    # ── Deck viewer ───────────────────────────────────────────────────────
    deck = st.session_state.active_deck
    if deck:
        st.markdown("<br>", unsafe_allow_html=True)

        # Deck header
        st.markdown(f"""
        <div class='study-card accent-left'
             style='display:flex; justify-content:space-between;
                    align-items:center;'>
            <div>
                <span style='font-size:1.1rem; font-weight:600;'>
                    {deck.topic}
                </span>
                <span style='color:var(--text-muted); font-size:0.82rem;
                              margin-left:0.75rem;'>
                    {deck.card_count} cards
                </span>
            </div>
            <span class='source-badge'>{deck.source_hint[:45]}</span>
        </div>""", unsafe_allow_html=True)

        # Navigation controls
        idx = st.session_state.card_index
        col_prev, col_counter, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("← Prev", use_container_width=True,
                         disabled=(idx == 0), key="card_prev"):
                st.session_state.card_index  -= 1
                st.session_state.card_flipped = False
                st.rerun()

        with col_counter:
            st.markdown(f"""
            <div style='text-align:center; padding:0.55rem 0;
                        color:var(--text-muted); font-size:0.85rem;'>
                Card <strong style='color:var(--text-primary);'>
                {idx + 1}</strong> of {deck.card_count}
            </div>""", unsafe_allow_html=True)

        with col_next:
            if st.button("Next →", use_container_width=True,
                         disabled=(idx == deck.card_count - 1),
                         key="card_next"):
                st.session_state.card_index  += 1
                st.session_state.card_flipped = False
                st.rerun()

        # Current card
        card     = deck.cards[idx]
        flipped  = st.session_state.card_flipped
        flip_cls = "flip-card flipped" if flipped else "flip-card"

        st.markdown(f"""
        <div class="{flip_cls}" id="flashcard" style="margin:0.5rem 0;">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <div class='tag-badge' style='margin-bottom:0.75rem;'>
                        {card.topic_tag}
                    </div>
                    <div style='font-size:1rem; font-weight:500;
                                line-height:1.5;'>{card.front}</div>
                    <div style='font-size:0.72rem; color:var(--text-muted);
                                margin-top:1rem;'>Click to reveal</div>
                </div>
                <div class="flip-card-back">
                    <div style='font-size:0.9rem; line-height:1.6;
                                margin-bottom:0.75rem;'>{card.back}</div>
                    {"<div style='font-size:0.8rem; color:var(--accent-2); "
                     "border-top:1px solid var(--border); padding-top:0.6rem; "
                     f"margin-top:0.2rem;'>💡 {card.example}</div>"
                     if card.example else ""}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        col_flip, col_gap = st.columns([1, 3])
        with col_flip:
            flip_label = "🔄 Show Front" if flipped else "🔄 Flip Card"
            if st.button(flip_label, use_container_width=True, key="flip_btn"):
                st.session_state.card_flipped = not st.session_state.card_flipped
                st.rerun()

        # All cards list (collapsible)
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"📋 View all {deck.card_count} cards"):
            for i, c in enumerate(deck.cards):
                st.markdown(f"""
                <div class='study-card' style='padding:0.75rem 1rem;
                     margin-bottom:0.5rem;'>
                    <div style='font-size:0.72rem; color:var(--text-muted);
                                margin-bottom:0.3rem; font-family:var(--font-mono);'>
                        #{i+1} · <span class='tag-badge'>{c.topic_tag}</span>
                    </div>
                    <div style='font-weight:500; margin-bottom:0.3rem;'>
                        {c.front}
                    </div>
                    <div style='font-size:0.85rem; color:var(--text-muted);'>
                        {c.back}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Exports
        st.markdown('<div class="section-header" style="margin-top:1rem;">'
                    'Export</div>', unsafe_allow_html=True)
        col_csv, col_anki, _ = st.columns([1, 1, 2])

        with col_csv:
            csv_data = deck_to_csv(deck)
            st.download_button(
                label="⬇ Download CSV",
                data=csv_data,
                file_name=f"{deck.topic.replace(' ','_')}_flashcards.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_anki:
            anki_data = deck_to_anki_format(deck)
            st.download_button(
                label="⬇ Download for Anki",
                data=anki_data,
                file_name=f"{deck.topic.replace(' ','_')}_anki.txt",
                mime="text/plain",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — QUIZ
# ══════════════════════════════════════════════════════════════════════════

with tab_quiz:
    st.markdown("### Interactive Quiz")
    st.markdown(
        "<p style='color:var(--text-muted); font-size:0.88rem;'>"
        "Test your understanding with AI-generated questions grounded "
        "in your study material."
        "</p>", unsafe_allow_html=True
    )

    # ── Controls ─────────────────────────────────────────────────────────
    col_qt, col_qn, col_qd, col_qdt = st.columns([3, 1, 1.5, 1.5], gap="medium")

    with col_qt:
        quiz_topic_input = st.text_input(
            "Topic",
            value=st.session_state.get("quiz_topic", ""),
            placeholder="e.g. backpropagation, sorting algorithms…",
            key="quiz_topic_text",
        )
    with col_qn:
        num_questions = st.slider("Questions", 3, 10, 5, key="num_q_slider")
    with col_qd:
        difficulty_str = st.selectbox(
            "Difficulty",
            ["intermediate", "beginner", "advanced"],
            key="quiz_diff_select",
        )
    with col_qdt:
        quiz_doc_type = st.selectbox(
            "Source type", ["All", "pdf", "youtube", "website"],
            key="quiz_doc_type"
        )

    col_gen, col_faith = st.columns([2, 1])
    with col_gen:
        gen_quiz_btn = st.button("✨ Generate Quiz", type="primary",
                                 key="gen_quiz_btn", use_container_width=True)
    with col_faith:
        faith_check = st.toggle("Faithfulness check", value=False,
                                key="faith_toggle",
                                help="Verify each question is grounded in context. "
                                     "Adds ~2s per quiz.")

    if gen_quiz_btn:
        if not quiz_topic_input.strip():
            st.warning("Please enter a topic.")
        elif not st.session_state.ingested_sources:
            st.warning("No documents ingested yet.")
        else:
            doc_type_filter = (
                None if quiz_doc_type == "All" else quiz_doc_type
            )
            difficulty = DifficultyLevel(difficulty_str)

            with st.spinner(f"Generating {num_questions}-question quiz on "
                            f"'{quiz_topic_input}'…"):
                try:
                    result = generate_quiz(
                        topic                  = quiz_topic_input,
                        num_questions          = num_questions,
                        difficulty             = difficulty,
                        doc_type               = doc_type_filter,
                        run_faithfulness_check = faith_check,
                    )
                    st.session_state.active_quiz    = result["quiz"]
                    st.session_state.quiz_answers   = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_result    = None
                    st.session_state.quiz_topic     = quiz_topic_input
                    st.session_state["_verdicts"]   = result["verdicts"]
                    st.success(
                        f"Generated {result['quiz'].total_marks} questions · "
                        f"{result['chunks_used']} chunks retrieved"
                    )
                except (ValueError, RuntimeError) as e:
                    st.error(str(e))

    # ── Render quiz ───────────────────────────────────────────────────────
    quiz     = st.session_state.active_quiz
    verdicts = st.session_state.get("_verdicts", [])

    if quiz:
        st.markdown("<br>", unsafe_allow_html=True)

        # Quiz header
        st.markdown(f"""
        <div class='study-card accent-left'>
            <div style='font-size:1.1rem; font-weight:600;'>
                {quiz.topic}
            </div>
            <div style='margin-top:0.3rem;'>
                <span class='source-badge'>{quiz.source_hint[:60]}</span>
                <span class='tag-badge'>{quiz.total_marks} questions</span>
            </div>
        </div>""", unsafe_allow_html=True)

        submitted = st.session_state.quiz_submitted

        # ── Question cards ────────────────────────────────────────────────
        for i, q in enumerate(quiz.questions):
            # Faithfulness badge
            faith_badge = ""
            if verdicts:
                v = next((x for x in verdicts
                          if x.get("question_index") == i), None)
                if v and v.get("verdict") == "unsupported":
                    faith_badge = (
                        "<span style='font-size:0.7rem; color:var(--warning);"
                        " background:rgba(251,191,36,0.1); border:1px solid "
                        "rgba(251,191,36,0.3); border-radius:4px; padding:1px 7px;"
                        " margin-left:0.5rem;'>⚠ unverified</span>"
                    )

            card_cls = "study-card"
            if submitted:
                ans = st.session_state.quiz_answers.get(i)
                if ans is not None:
                    card_cls = (
                        "study-card success"
                        if ans == q.get_correct_index()
                        else "study-card danger"
                    )

            st.markdown(f"""
            <div class='{card_cls}'>
                <div style='font-size:0.72rem; color:var(--text-muted);
                            font-family:var(--font-mono); margin-bottom:0.4rem;'>
                    Q{i+1} ·
                    <span class='tag-badge'>{q.topic_tag}</span>
                    <span class='tag-badge'>{q.difficulty.value}</span>
                    {faith_badge}
                </div>
                <div style='font-size:0.95rem; font-weight:500;
                            margin-bottom:0.75rem;'>{q.question}</div>
            </div>""", unsafe_allow_html=True)

            # Option radio buttons
            option_labels = [
                f"{'✓ ' if submitted and j == q.get_correct_index() else ''}"
                f"{chr(65+j)}. {opt.option}"
                for j, opt in enumerate(q.options)
            ]

            selected = st.radio(
                f"q{i}",
                options=list(range(4)),
                format_func=lambda j, i=i: option_labels[j],
                key=f"quiz_q_{i}",
                disabled=submitted,
                label_visibility="collapsed",
                horizontal=False,
            )
            st.session_state.quiz_answers[i] = selected

            # Explanation (shown after submission)
            if submitted:
                is_correct = (selected == q.get_correct_index())
                icon = "✅" if is_correct else "❌"
                st.markdown(f"""
                <div style='background:rgba(124,155,255,0.08);
                            border:1px solid var(--border);
                            border-radius:var(--radius-sm);
                            padding:0.7rem 1rem;
                            font-size:0.85rem;
                            color:var(--text-primary);
                            margin: -0.5rem 0 1rem 0;'>
                    {icon}
                    <strong>{'Correct!' if is_correct else 'Incorrect.'}</strong>
                    &nbsp;{q.explanation}
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

        # ── Submit / Reset ────────────────────────────────────────────────
        if not submitted:
            answered = len(st.session_state.quiz_answers)
            all_done = answered == quiz.total_marks

            if st.button(
                "📨 Submit Quiz", type="primary",
                disabled=not all_done,
                key="submit_quiz_btn",
                use_container_width=True,
                help="Answer all questions before submitting."
            ):
                answers_list = [
                    st.session_state.quiz_answers.get(i, 0)
                    for i in range(quiz.total_marks)
                ]
                result = score_quiz(quiz, answers_list)
                st.session_state.quiz_submitted = True
                st.session_state.quiz_result    = result

                # Record to performance profile
                profile = st.session_state.performance_profile
                profile.syllabus_topics = list(
                    set(profile.syllabus_topics + [quiz.topic])
                )
                profile.add_quiz_result(result)
                st.session_state.quiz_history.append(result)
                st.rerun()

        else:
            # ── Score panel ───────────────────────────────────────────────
            result = st.session_state.quiz_result
            if result:
                pct   = result["percentage"]
                score = result["score"]
                total = result["total"]
                bar_color = (
                    "var(--success)" if pct >= 70 else
                    "var(--warning)" if pct >= 40 else
                    "var(--danger)"
                )
                grade = (
                    "Excellent! 🎉" if pct >= 80 else
                    "Good work 👍" if pct >= 60 else
                    "Keep studying 📖"
                )

                st.markdown(f"""
                <div class='study-card' style='text-align:center;
                     padding:1.5rem; margin-bottom:1rem;'>
                    <div style='font-size:2.5rem; font-weight:700;
                                color:{bar_color};'>{pct}%</div>
                    <div style='color:var(--text-muted); margin:0.3rem 0;'>
                        {score} / {total} correct · {grade}
                    </div>
                    <div class='score-bar-wrap' style='max-width:320px;
                         margin:0.75rem auto 0;'>
                        <div class='score-bar-fill'
                             style='width:{pct}%; background:{bar_color};'>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Difficulty breakdown
                breakdown = result.get("difficulty_breakdown", {})
                if breakdown:
                    st.markdown('<div class="section-header">Score by Difficulty'
                                '</div>', unsafe_allow_html=True)
                    bcols = st.columns(len(breakdown))
                    for ci, (diff, data) in enumerate(breakdown.items()):
                        bpct = (
                            round(data["correct"] / data["total"] * 100)
                            if data["total"] else 0
                        )
                        with bcols[ci]:
                            st.markdown(f"""
                            <div class='metric-box'>
                                <div class='value'>{bpct}%</div>
                                <div class='label'>{diff}</div>
                                <div style='font-size:0.7rem;
                                            color:var(--text-muted);'>
                                    {data['correct']}/{data['total']}
                                </div>
                            </div>""", unsafe_allow_html=True)

            col_retry, col_new = st.columns(2)
            with col_retry:
                if st.button("🔄 Retake Quiz", use_container_width=True):
                    st.session_state.quiz_answers   = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_result    = None
                    st.rerun()
            with col_new:
                if st.button("✨ New Quiz", type="primary",
                             use_container_width=True):
                    st.session_state.active_quiz    = None
                    st.session_state.quiz_answers   = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_result    = None
                    st.session_state["_verdicts"]   = []
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — PROGRESS
# ══════════════════════════════════════════════════════════════════════════

with tab_progress:
    st.markdown("### Progress & Performance")
    st.markdown(
        "<p style='color:var(--text-muted); font-size:0.88rem;'>"
        "Track your quiz history and see where to focus next."
        "</p>", unsafe_allow_html=True
    )

    profile = st.session_state.performance_profile
    history = st.session_state.quiz_history

    if not history:
        st.info("No quizzes completed yet. Take a quiz to see your progress here.")
    else:
        # ── Headline metrics ──────────────────────────────────────────────
        total_quizzes  = len(history)
        avg_pct        = round(
            sum(r["percentage"] for r in history) / total_quizzes, 1
        )
        total_q        = sum(r["total"]   for r in history)
        total_correct  = sum(r["score"]   for r in history)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)

        for col, val, label in [
            (m1, total_quizzes,   "Quizzes Taken"),
            (m2, f"{avg_pct}%",   "Avg Score"),
            (m3, total_correct,   "Total Correct"),
            (m4, total_q,         "Total Questions"),
        ]:
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='value'>{val}</div>
                    <div class='label'>{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Strong vs Weak topics ─────────────────────────────────────────
        col_strong, col_weak = st.columns(2, gap="large")

        with col_strong:
            st.markdown('<div class="section-header">💪 Strong Topics (≥ 70%)'
                        '</div>', unsafe_allow_html=True)
            if profile.strong_topics:
                for t in profile.strong_topics:
                    st.markdown(f"""
                    <div class='study-card success'
                         style='padding:0.6rem 1rem; margin-bottom:0.4rem;
                                font-size:0.875rem;'>
                        ✓ {t}
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='color:var(--text-muted); font-size:0.85rem;'>"
                    "No strong topics yet.</div>", unsafe_allow_html=True
                )

        with col_weak:
            st.markdown('<div class="section-header">📖 Needs Work (< 70%)'
                        '</div>', unsafe_allow_html=True)
            if profile.weak_topics:
                for t in profile.weak_topics:
                    col_t, col_btn = st.columns([3, 1])
                    with col_t:
                        st.markdown(f"""
                        <div class='study-card danger'
                             style='padding:0.6rem 1rem; margin-bottom:0.4rem;
                                    font-size:0.875rem;'>
                            ✗ {t}
                        </div>""", unsafe_allow_html=True)
                    with col_btn:
                        if st.button("Retry", key=f"retry_{t}",
                                     use_container_width=True):
                            st.session_state.quiz_topic = t
                            st.info(
                                f"Topic set to **{t}**. "
                                "Switch to the Quiz tab."
                            )
            else:
                st.markdown(
                    "<div style='color:var(--text-muted); font-size:0.85rem;'>"
                    "No weak topics yet.</div>", unsafe_allow_html=True
                )

        # ── Quiz history log ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Quiz History</div>',
                    unsafe_allow_html=True)

        for i, r in enumerate(reversed(history)):
            pct    = r["percentage"]
            bar_c  = (
                "var(--success)" if pct >= 70 else
                "var(--warning)" if pct >= 40 else
                "var(--danger)"
            )
            st.markdown(f"""
            <div class='study-card' style='padding:0.75rem 1.25rem;
                 margin-bottom:0.5rem;
                 display:flex; justify-content:space-between;
                 align-items:center;'>
                <div>
                    <span style='font-weight:500;'>{r['topic']}</span>
                    <span style='color:var(--text-muted); font-size:0.8rem;
                                 margin-left:0.5rem;'>
                        {r['score']}/{r['total']} questions
                    </span>
                </div>
                <span style='font-size:1.1rem; font-weight:700;
                             color:{bar_c};'>{pct}%</span>
            </div>""", unsafe_allow_html=True)

        # ── Reset progress ────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Reset Progress", key="reset_progress"):
            st.session_state.performance_profile = PerformanceProfile()
            st.session_state.quiz_history        = []
            st.success("Progress reset.")
            st.rerun()