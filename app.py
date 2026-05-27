"""
app.py — Streamlit demo for the Multimodal RAG Pipeline
Run: streamlit run app.py
"""

import streamlit as st
import tempfile, os
from pathlib import Path
from src.rag_pipeline import MultimodalRAGPipeline

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main { background: #0e1117; }
.source-card {
    background: #1a1a2e;
    border: 1px solid #16213e;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
}
.score-badge {
    background: #0f3460;
    color: #e94560;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────
st.title("🧠 Multimodal RAG System")
st.caption("Upload PDFs, text files, or images — then ask questions in natural language.")
st.divider()

# ── Sidebar: Upload & Settings ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    backend = st.selectbox(
        "LLM Backend",
        ["extractive", "openai", "huggingface"],
        help="extractive = no API key needed | openai = needs OPENAI_API_KEY"
    )
    top_k = st.slider("Retrieved chunks (top-k)", 1, 10, 5)
    chunk_size = st.slider("Chunk size (words)", 128, 1024, 512, step=64)

    st.divider()
    st.header("📁 Upload Documents")
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded and st.button("🚀 Build Index", use_container_width=True):
        with st.spinner("Ingesting and indexing documents..."):
            pipeline = MultimodalRAGPipeline(
                generator_backend=backend,
                top_k=top_k,
                chunk_size=chunk_size,
            )
            # Save uploads to temp dir and ingest
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded:
                    out = os.path.join(tmpdir, f.name)
                    with open(out, "wb") as fp:
                        fp.write(f.read())
                pipeline.ingest(tmpdir)

            st.session_state.pipeline = pipeline
            st.session_state.history  = []
        st.success(f"✅ Indexed {len(pipeline.store.documents)} chunks from {len(uploaded)} file(s)")

    st.divider()
    st.markdown("""
    **How it works:**
    1. Upload your documents
    2. Click **Build Index**
    3. Ask questions below

    **Supported formats:**
    - 📄 PDF (text extraction per page)
    - 📝 TXT / Markdown
    - 🖼️ Images (BLIP captioning)

    **Tech Stack:**
    - Embeddings: `sentence-transformers`
    - Vector DB: `FAISS`
    - Generation: `OpenAI` / `flan-t5` / extractive
    """)

# ── Main: Query Interface ─────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Ask a Question")

    if st.session_state.pipeline is None:
        st.info("👈 Upload documents and build the index first.")
    else:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What is the main conclusion of the paper?",
            label_visibility="collapsed"
        )

        if st.button("🔍 Search & Answer", use_container_width=True) and query:
            with st.spinner("Retrieving and generating answer..."):
                response = st.session_state.pipeline.query(query)
                st.session_state.history.append(response)

        # Show latest answer
        if st.session_state.history:
            latest = st.session_state.history[-1]

            st.markdown("### 🤖 Answer")
            st.markdown(f"""
            <div style='background:#1a1a2e; border-left:4px solid #e94560;
                        padding:1.2rem; border-radius:8px; font-size:1rem; line-height:1.7'>
            {latest.answer}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📚 Retrieved Sources")
            for r in latest.sources:
                modality_icon = {"text": "📄", "image": "🖼️", "table": "📊"}.get(r.document.modality, "📄")
                st.markdown(f"""
                <div class='source-card'>
                    <b>{modality_icon} {Path(r.document.source).name}</b>
                    &nbsp;&nbsp;<span class='score-badge'>score: {r.score:.3f}</span>
                    &nbsp;&nbsp;<span style='color:#888'>rank #{r.rank}</span>
                    {'&nbsp;| page ' + str(r.document.page) if r.document.page else ''}
                    <br><br>
                    <span style='color:#aaa'>{r.document.content[:300]}...</span>
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.subheader("🕑 Query History")
    if not st.session_state.history:
        st.caption("No queries yet.")
    else:
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"Q{len(st.session_state.history)-i}: {h.query[:50]}..."):
                st.write(h.answer)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
