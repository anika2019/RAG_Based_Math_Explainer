import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from rag import (
    VECTOR_STORE_PATH,
    build_knowledge_base,
    get_math_help,
    initialize_math_rag,
)


load_dotenv()


APP_TITLE = "RAG Based Math Solver (UP Board / NCERT)"

def _hydrate_env_from_streamlit_secrets() -> None:
    """Populate env vars from Streamlit secrets when running in cloud."""
    if "GOOGLE_API_KEY" not in os.environ and "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


def _parse_urls(raw: str) -> list[str]:
    urls: list[str] = []
    for line in raw.replace(",", "\n").splitlines():
        u = line.strip()
        if not u:
            continue
        urls.append(u)
    # de-dup, keep order
    deduped: list[str] = []
    seen: set[str] = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    return deduped


@st.cache_resource(show_spinner=False)
def _init_models():
    return initialize_math_rag()


def _vector_store_exists(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _load_vector_store(path: Path, embeddings):
    # FAISS persistence includes a pickle file; LangChain requires this flag for local loads.
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧮", layout="wide")
    st.title(APP_TITLE)
    st.caption("Enter URLs to build a knowledge base, then ask your math question.")

    _hydrate_env_from_streamlit_secrets()
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    if not google_key or not groq_key:
        st.error(
            "Missing API keys. Set `GOOGLE_API_KEY` and `GROQ_API_KEY` in your `.env` (or environment)."
        )
        st.stop()

    with st.sidebar:
        st.subheader("Knowledge base")
        default_urls = "\n".join(
            [
                "https://openstax.org/details/books/elementary-algebra-2e",
                "https://www.ck12.org/book/ck-12-algebra-i-second-edition/section/10.0/",
            ]
        )
        urls_raw = st.text_area(
            "URLs (comma or newline separated)",
            value=default_urls,
            height=140,
        )
        rebuild = st.checkbox(
            "Rebuild knowledge base (re-scrape URLs)",
            value=not _vector_store_exists(VECTOR_STORE_PATH),
        )
        st.write(f"Vector store path: `{VECTOR_STORE_PATH}`")

    embeddings, llm = _init_models()

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    def ensure_vector_store():
        if not rebuild and st.session_state.vector_store is not None:
            return st.session_state.vector_store

        if not rebuild and _vector_store_exists(VECTOR_STORE_PATH):
            with st.spinner("Loading existing knowledge base..."):
                st.session_state.vector_store = _load_vector_store(VECTOR_STORE_PATH, embeddings)
            return st.session_state.vector_store

        urls = _parse_urls(urls_raw)
        if not urls:
            st.warning("Please provide at least one URL.")
            return None

        with st.spinner("Building knowledge base (scraping + embedding)..."):
            VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            st.session_state.vector_store = build_knowledge_base(urls, embeddings)
        return st.session_state.vector_store

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        user_query = st.text_area(
            "Ask a math question (Hindi or English)",
            placeholder="e.g. द्विघात समीकरण को हल करने का सूत्र क्या है?",
            height=120,
        )
        ask = st.button("Get Answer", type="primary", use_container_width=True)

    with col2:
        st.subheader("Status")
        st.write("Ready when the knowledge base is available.")
        st.write(
            f"Knowledge base: {'✅ found' if _vector_store_exists(VECTOR_STORE_PATH) else '❌ not built yet'}"
        )

    if ask:
        if not user_query.strip():
            st.warning("Please enter a question.")
            st.stop()

        v_store = ensure_vector_store()
        if v_store is None:
            st.stop()

        with st.spinner("Thinking..."):
            answer = get_math_help(v_store, llm, user_query.strip())

        st.subheader("Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
