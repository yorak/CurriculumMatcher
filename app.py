"""
CurriculumMatcher — Tampere ITC/Computing Sciences Course Explorer

Run with:
    streamlit run app.py

Prerequisites:
    python fetch_and_index.py   (once, to fetch data and build the index)
"""

import os
import sys

import streamlit as st

st.set_page_config(
    page_title="CurriculumMatcher",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Check that the index exists before importing the pipeline
# ---------------------------------------------------------------------------
INDEX_PATH = os.path.join("data", "chroma")
if not os.path.isdir(INDEX_PATH):
    st.error(
        "No index found. Please run `python fetch_and_index.py` first to "
        "fetch course data and build the vector index."
    )
    st.stop()

try:
    import chromadb

    client = chromadb.PersistentClient(path=INDEX_PATH)
    col = client.get_collection("curriculum_itc")
    course_count = col.count()
    if course_count == 0:
        st.error("Index exists but is empty. Re-run `python fetch_and_index.py`.")
        st.stop()
except Exception as exc:
    st.error(f"Failed to open index: {exc}")
    st.stop()

from pipeline import query as run_query  # noqa: E402  (import after index check)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("CurriculumMatcher")
    st.caption(f"Indexed courses: {course_count}")
    st.divider()

    st.markdown("**Top-k results**")
    top_k = st.slider("How many courses to retrieve", 4, 16, 8)

    st.divider()
    st.markdown("**Example queries**")
    examples = [
        "What courses cover functional safety and IEC 61508?",
        "Show me the embedded systems core courses and what students learn",
        "Which modules cover software testing for safety-critical systems?",
        "Who at Tampere teaches operations research?",
        "What do students learn in the machine learning courses?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    st.markdown(
        "_Data from [Tampere University Sisu Kori API](https://sisu.tuni.fi/kori/swagger-ui)_"
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------------------------
# Main layout: chat (left) + sources (right)
# ---------------------------------------------------------------------------
st.header("Tampere ITC / Computing Sciences Curriculum Explorer")

chat_col, sources_col = st.columns([3, 2])

with chat_col:
    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Prefill from example button
    prefill = st.session_state.pop("prefill", "")

    user_input = st.chat_input(
        "Ask about courses, topics, or what students would know...",
        key="chat_input",
    )

    # Allow example button click to trigger a query
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with chat_col:
            with st.chat_message("user"):
                st.markdown(user_input)

        # Run query
        with st.spinner("Searching curriculum..."):
            try:
                result = run_query(user_input, k=top_k)
            except Exception as exc:
                result = {
                    "answer": f"Error: {exc}",
                    "sources": [],
                }

        # Show assistant reply
        st.session_state["messages"].append(
            {"role": "assistant", "content": result["answer"]}
        )
        with chat_col:
            with st.chat_message("assistant"):
                st.markdown(result["answer"])

        # Store sources for the right panel
        st.session_state["last_sources"] = result["sources"]

        st.rerun()

with sources_col:
    st.subheader("Retrieved courses")
    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.caption("Sources will appear here after your first query.")
    else:
        for src in sources:
            score_pct = int(src["score"] * 100)
            with st.expander(f"**{src['code']}** — {score_pct}% match"):
                st.markdown(f"**{src['name']}**")
                st.caption(f"Credits: {src['credits']}")
                st.progress(src["score"])
