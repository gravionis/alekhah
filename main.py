import streamlit as st
from typing import List
import os
import pandas as pd  # Add this import for table display
import fitz  # PyMuPDF for PDF handling

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
                    for match in matches:
                        filename = match.get("filename")
                        snippet = match.get("truncated_snippet")
                        char_start = match.get("char_start")
                        char_end = match.get("char_end")

                        # Create a clickable link for each filename
                        link = f"?file={filename}&start={char_start}&end={char_end}"
                        st.markdown(
                            f"[**{filename}**]({link}) - {snippet}",
                            unsafe_allow_html=True,
                        )

    with col2:
        # Display the PDF preview if a file is selected
        query_params = st.query_params
        selected_file = query_params.get("file", [None])[0]
        char_start = int(query_params.get("start", [0])[0])
        char_end = int(query_params.get("end", [0])[0])

        if selected_file:
            st.subheader(f"Preview: {selected_file}")
            try:
                file_path = os.path.join(ST_KNOWLEDGE_DIR, selected_file)
                with fitz.open(file_path) as pdf:
                    # Find the page containing the snippet
                    snippet_page = None
                    for page_num in range(len(pdf)):
                        page = pdf[page_num]
                        text = page.get_text("text")
                        if text and str(char_start) in text:
                            snippet_page = page_num
                            break

                    if snippet_page is not None:
                        page = pdf[snippet_page]
                        # Render the page as an image
                        pix = page.get_pixmap()
                        st.image(
                            pix.tobytes(),
                            caption=f"Page {snippet_page + 1}",
                            use_column_width=True,
                        )
                    else:
                        st.info("Snippet not found in the document.")
            except Exception as e:
                st.error(f"Error loading PDF: {e}")


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
