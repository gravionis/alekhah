import streamlit as st
from typing import List
import os
import pandas as pd
import fitz  # PyMuPDF for PDF handling
from PIL import Image
import io

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

    # Initialize selected file in session state
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = "c628_2007.pdf"
    if "pdf_page" not in st.session_state:
        st.session_state.pdf_page = 0

    with col1:
        # Store search results in session state to persist them
        if st.button("Search"):
            if not question or not question.strip():
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching..."):
                    try:
                        result = answer_question(question, k=int(k))
                        st.session_state.search_result = result
                    except Exception as e:
                        st.error(f"Error during search: {e}")
                        return

        # Display search results if they exist
        if "search_result" in st.session_state:
            result = st.session_state.search_result

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

                    # Create a clickable button for each filename
                    if st.button(f"📄 {filename}", key=f"pdf_{filename}_{char_start}"):
                        st.session_state.selected_pdf = filename
                        st.session_state.pdf_page = 0
                        st.session_state.highlight_text = match.get("snippet", "")
                        st.session_state.char_start = char_start
                        st.session_state.char_end = char_end

                    st.write(f"_{snippet}_")
                    st.divider()

    with col2:
        # Display the selected PDF
        selected_file = st.session_state.selected_pdf
        st.subheader(f"Preview: {selected_file}")
        try:
            file_path = os.path.join(ST_KNOWLEDGE_DIR, selected_file)

            if not os.path.exists(file_path):
                st.warning(f"File not found: {selected_file}")
                return

            pdf = fitz.Document(file_path)
            total_pages = len(pdf)

            # Find the page containing the highlight text (only on first load)
            if "highlight_text" in st.session_state and st.session_state.highlight_text:
                if "last_highlight" not in st.session_state or st.session_state.last_highlight != st.session_state.highlight_text:
                    search_text = st.session_state.highlight_text
                    for page_num in range(total_pages):
                        page = pdf[page_num]
                        text_instances = page.search_for(search_text[:100])  # Search first 100 chars
                        if text_instances:
                            st.session_state.pdf_page = page_num
                            st.session_state.last_highlight = st.session_state.highlight_text
                            break

            # Navigation buttons
            col_prev, col_info, col_next = st.columns([1, 2, 1])

            with col_prev:
                if st.button("←", disabled=(st.session_state.pdf_page == 0)):
                    st.session_state.pdf_page -= 1
                    st.rerun()

            with col_info:
                st.write(f"Page {st.session_state.pdf_page + 1} of {total_pages}")

            with col_next:
                if st.button("→", disabled=(st.session_state.pdf_page >= total_pages - 1)):
                    st.session_state.pdf_page += 1
                    st.rerun()

            # Render current page with highlights
            page = pdf[st.session_state.pdf_page]

            # Highlight the search text if it exists on this page
            if "highlight_text" in st.session_state and st.session_state.highlight_text:
                search_text = st.session_state.highlight_text
                text_instances = page.search_for(search_text[:100])
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=[1, 1, 0])  # Yellow highlight
                    highlight.update()

            pix = page.get_pixmap()

            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            st.image(
                img
            )

            pdf.close()
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