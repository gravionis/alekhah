import streamlit as st
from typing import List
import os

from ingestion import list_documents, ingest_files
from qa import answer_question


ST_KNOWLEDGE_DIR = "data/knowledge"


def ingestion_page():
    st.title("Ingestion")
    st.write("Drop `.pdf` and `.md` files into the `knowledge` folder, then select files and click Ingest.")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh"):
            st.rerun()
    with col2:
        st.write("")

    files = list_documents(ST_KNOWLEDGE_DIR)
    if not files:
        st.info(f"No .pdf or .md files found in `{ST_KNOWLEDGE_DIR}`. Add files and click Refresh.")
        return

    selected = {}
    st.write("Select files to ingest:")
    for fn in files:
        selected[fn] = st.checkbox(fn, value=False)

    ingest_btn = st.button("ingest")
    if ingest_btn:
        to_ingest = [fn for fn, sel in selected.items() if sel]
        if not to_ingest:
            st.warning("No files selected for ingestion.")
        else:
            st.info(f"Starting ingestion for {len(to_ingest)} file(s)")
            progress = st.progress(0)
            results = []
            for i, fn in enumerate(to_ingest, start=1):
                res = ingest_files([fn], knowledge_dir=ST_KNOWLEDGE_DIR)
                results.extend(res)
                progress.progress(int(i / len(to_ingest) * 100))
            st.write("### Results")
            for r in results:
                if r.get("status") == "ok":
                    st.success(f"{r['filename']}: ingested {r.get('chunks', 0)} chunks -> {r.get('out_path')}")
                elif r.get("status") == "empty":
                    st.warning(f"{r['filename']}: file had no extractable text.")
                else:
                    st.error(f"{r['filename']}: {r.get('status')} - {r.get('error')}")


def qa_page():
    st.title("Question Answer")
    st.write("Ask a question against the ingested contents (Phase 1: JSON vector store)")

    question = st.text_area("Question", height=120)
    k = st.number_input("Top k matches", min_value=1, max_value=20, value=3, step=1)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Search"):
            if not question or not question.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching..."):
                    try:
                        result = answer_question(question, k=int(k))
                    except Exception as e:
                        st.error(f"Error during search: {e}")
                        return

                st.subheader("Answer")
                if result.get("answer"):
                    st.write(result.get("answer"))
                else:
                    st.info("No answer could be composed from the ingested content.")

                st.subheader("Sources")
                matches = result.get("matches") or []
                if not matches:
                    st.info("No matching chunks found.")
                else:
                    for m in matches:
                        st.markdown(f"**File:** {m.get('filename')}  ")
                        st.markdown(f"**Checksum:** {m.get('checksum')}  ")
                        st.markdown(f"**Chunk index:** {m.get('index')} (chars {m.get('char_start')} - {m.get('char_end')})  ")
                        st.markdown(f"**Score:** {m.get('score'):.4f}  ")
                        st.write(m.get('snippet'))
                        st.markdown("---")

    with col2:
        st.write("Use the left pane to enter a question and run retrieval.")


def rule_gen_page():
    st.title("Rule Generator")
    st.write("Placeholder for Rule Generator page (Phase 1).")


PAGES = {
    "Ingestion": ingestion_page,
    "Question Answer": qa_page,
    "Rule Generator": rule_gen_page,
}


def main():
    st.sidebar.title("Pages")
    page = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[page]()


if __name__ == "__main__":
    main()
