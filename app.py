"""
Milestone 5 — A simple UI so a non-technical client can try it.

This is the demo you'll screen-record for your portfolio Loom. The cached store
loads once. Each answer shows its sources in expandable sections, which is the
detail that makes it feel trustworthy and "senior."

Run:
    streamlit run app.py
"""
import streamlit as st
from dotenv import load_dotenv

import rag
from store import VectorStore

load_dotenv()

st.set_page_config(page_title="DocsRAG", page_icon="📄")
st.title("📄 DocsRAG — Ask your documents")


@st.cache_resource
def get_store():
    return VectorStore.load("index.pkl")


try:
    store = get_store()
except FileNotFoundError:
    st.error("No index found. Run `python build_index.py` first.")
    st.stop()

question = st.text_input("Ask a question about your documents:")
if question:
    with st.spinner("Thinking..."):
        out = rag.answer(question, store=store)
    st.markdown("### Answer")
    st.write(out["answer"])
    st.markdown("### Sources")
    for s in out["sources"]:
        with st.expander(f"{s['source']} #{s['position']}  (similarity {s['score']})"):
            st.write(s["snippet"] + "...")
