"""
app.py
------
Phase 19: Streamlit Application.

Interactive UI for the Enterprise Document Intelligence & RAG Assistant.
Features:
- Document upload (PDF/DOCX) and one-click (re-)indexing
- Natural-language question box
- AI-generated, grounded answer with source references
- Retrieved-chunk preview (transparency into what the model saw)
- Conversation history with memory across turns
- Clear/reset conversation
- Basic error handling (missing API key, empty knowledge base, etc.)

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py
"""

import os
import sys
import shutil

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from document_loader import load_documents
from chunking import chunk_documents
from retriever import VectorStore
from rag_pipeline import answer_question, ConversationTurn, DEFAULT_SCORE_THRESHOLD

DOCS_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
STORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")

st.set_page_config(page_title="Enterprise AI Document Assistant", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[ConversationTurn]
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list[dict] for rendering (includes sources)
if "store" not in st.session_state:
    st.session_state.store = None
if "index_status" not in st.session_state:
    st.session_state.index_status = "not_built"


def get_store() -> VectorStore | None:
    if st.session_state.store is not None:
        return st.session_state.store
    if os.path.exists(os.path.join(STORE_DIR, "index.faiss")):
        try:
            st.session_state.store = VectorStore.load(STORE_DIR)
            st.session_state.index_status = "loaded"
        except Exception as e:
            st.session_state.index_status = f"error: {e}"
    return st.session_state.store


def build_index():
    os.makedirs(DOCS_DIR, exist_ok=True)
    records = load_documents(DOCS_DIR)
    if not records:
        st.session_state.index_status = "no_documents"
        return
    chunks = chunk_documents(records)
    store = VectorStore()
    with st.spinner(f"Embedding {len(chunks)} chunks and building the vector index..."):
        store.build(chunks)
        store.save(STORE_DIR)
    st.session_state.store = store
    st.session_state.index_status = "loaded"
    st.session_state.num_chunks = len(chunks)
    st.session_state.num_docs = len({r.doc_name for r in records})


# ---------------------------------------------------------------------------
# Sidebar: document upload + indexing status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF / DOCX", type=["pdf", "docx"], accept_multiple_files=True
    )
    if uploaded_files:
        os.makedirs(DOCS_DIR, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DOCS_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to the document collection.")

    st.divider()

    existing_docs = []
    if os.path.exists(DOCS_DIR):
        existing_docs = [f for f in os.listdir(DOCS_DIR) if f.endswith((".pdf", ".docx"))]

    st.subheader("Current Collection")
    if existing_docs:
        for d in existing_docs:
            st.markdown(f"- {d}")
    else:
        st.caption("No documents yet. Upload files above.")

    st.divider()

    if st.button("🔄 Build / Rebuild Index", use_container_width=True, type="primary"):
        build_index()
    if existing_docs and getattr(st.session_state, "index_status", None) not in ["loaded", "building"]:
        build_index()
    
    status = st.session_state.index_status
    if status == "loaded":
        n_chunks = getattr(st.session_state, "num_chunks", None)
        st.success(f"Index ready" + (f" ({n_chunks} chunks)" if n_chunks else ""))
    elif status == "no_documents":
        st.warning("No documents found to index. Upload files first.")
    elif status.startswith("error"):
        st.error(f"Index load error: {status}")
    else:
        st.info("Index not built yet.")

    st.divider()
    top_k = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=4)
    threshold = st.slider("Similarity threshold", min_value=0.0, max_value=0.8,
                           value=DEFAULT_SCORE_THRESHOLD, step=0.05,
                           help="Chunks below this cosine similarity are discarded before "
                                "reaching the LLM — this is what prevents hallucinated answers "
                                "when the knowledge base has no relevant information.")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.display_history = []
        st.rerun()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set. Set it in your environment before asking "
                    "questions, e.g. `export ANTHROPIC_API_KEY=sk-ant-...`")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("📄 Enterprise AI Document Assistant")
st.caption("Ask natural-language questions grounded in your organization's documents — "
           "with cited sources and honest 'I don't know' answers when the docs don't cover it.")

store = get_store()

# Render conversation history
for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("📚 Sources"):
                for s in turn["sources"]:
                    st.markdown(f"- {s}")
        if turn["role"] == "assistant" and turn.get("chunks"):
            with st.expander("🔍 Retrieved context (preview)"):
                for c in turn["chunks"]:
                    st.markdown(f"**{c.doc_name} — Page {c.page_number}** (score: {c.score:.3f})")
                    st.text(c.text[:300] + ("..." if len(c.text) > 300 else ""))
                    st.markdown("---")

question = st.chat_input("Ask a question about your documents, e.g. 'What is the leave policy?'")

if question:
    st.session_state.display_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if store is None or getattr(store, "index", None) is None:
            error_msg = ("No document index found. Please upload documents and click "
                          "'Build / Rebuild Index' in the sidebar first.")
            st.error(error_msg)
            st.session_state.display_history.append({"role": "assistant", "content": error_msg})
        
        else:
            try:
                with st.spinner("Retrieving relevant context and generating answer..."):
                    result = answer_question(
                        store, question, top_k=top_k, score_threshold=threshold,
                        history=st.session_state.history,
                    )
                st.markdown(result.answer)

                if result.sources:
                    with st.expander("📚 Sources", expanded=True):
                        for s in result.sources:
                            st.markdown(f"- {s}")

                if result.retrieved_chunks:
                    with st.expander("🔍 Retrieved context (preview)"):
                        for c in result.retrieved_chunks:
                            st.markdown(f"**{c.doc_name} — Page {c.page_number}** (score: {c.score:.3f})")
                            st.text(c.text[:300] + ("..." if len(c.text) > 300 else ""))
                            st.markdown("---")

                st.session_state.display_history.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "chunks": result.retrieved_chunks,
                })
                st.session_state.history.append(ConversationTurn(role="user", content=question))
                st.session_state.history.append(ConversationTurn(role="assistant", content=result.answer))

            except RuntimeError as e:
                st.error(str(e))
                st.session_state.display_history.append({"role": "assistant", "content": str(e)})
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                st.error(error_msg)
                st.session_state.display_history.append({"role": "assistant", "content": error_msg})
