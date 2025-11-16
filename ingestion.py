"""ingestion.py

Utility functions for listing documents, extracting text, chunking, and a simple
ingest pipeline. Now uses sentence-transformers for real embeddings via LangChain's
HuggingFaceEmbeddings wrapper. Stores chunk + metadata + embeddings to JSON files
under `data/vectors/`.

This file uses LangChain's RecursiveCharacterTextSplitter for intelligent text
chunking and PyPDF2 for PDF text extraction.
"""

import os
import hashlib
import json
import time
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_documents(knowledge_dir: str = "data/knowledge") -> List[str]:
    """Return sorted list of filenames (not full path) with .pdf or .md in folder."""
    ensure_dir(knowledge_dir)
    files = [f for f in os.listdir(knowledge_dir) if f.lower().endswith((".pdf", ".md"))]
    return sorted(files)


def compute_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        text = p.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def read_md_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize_text(text: str) -> str:
    # Simple normalization: trim and collapse multiple blank lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    filtered = []
    prev_blank = False
    for ln in lines:
        is_blank = not ln.strip()
        if is_blank and prev_blank:
            continue
        filtered.append(ln)
        prev_blank = is_blank
    return "\n".join(filtered).strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter.

    Returns list of dicts with text, start, end, index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0

    # Use LangChain's intelligent text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Split the text
    split_texts = text_splitter.split_text(text)

    # Build chunks with metadata
    chunks = []
    current_pos = 0
    for index, chunk_text in enumerate(split_texts):
        # Find the actual position of this chunk in the original text
        start = text.find(chunk_text, current_pos)
        if start == -1:
            # Fallback if exact match not found (shouldn't happen normally)
            start = current_pos
        end = start + len(chunk_text)

        chunks.append({
            "index": index,
            "text": chunk_text,
            "char_start": start,
            "char_end": end,
        })
        current_pos = end

    return chunks


def get_embeddings_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Get HuggingFace embeddings model via LangChain wrapper.

    Default model: all-mpnet-base-v2 (most accurate sentence-transformers model)
    Alternative: sentence-transformers/all-MiniLM-L6-v2 (faster, smaller)
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
    )


def ingest_file(
    filename: str,
    knowledge_dir: str = "data/knowledge",
    chunk_size: int = 1000,
    overlap: int = 200,
    out_dir: str = "data/vectors",
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> Dict:
    """Ingest a single file. Returns status dict with counts and output path.

    Behavior: parses .md/.pdf, normalizes, chunks using LangChain's text splitter,
    generates embeddings using HuggingFace models, and writes a JSON file with
    chunks & metadata to `out_dir/{filename}.{checksum}.json`.
    """
    ensure_dir(knowledge_dir)
    ensure_dir(out_dir)
    path = os.path.join(knowledge_dir, filename)
    if not os.path.exists(path):
        return {"filename": filename, "status": "missing", "error": "file not found"}

    try:
        # Initialize embeddings model
        embeddings_model = get_embeddings_model(embeddings_model_name)

        checksum = compute_checksum(path)
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            text = read_pdf_text(path)
        elif ext == "md":
            text = read_md_text(path)
        else:
            return {"filename": filename, "status": "unsupported"}

        if not text or not text.strip():
            return {"filename": filename, "status": "empty"}

        text = normalize_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        # Extract texts for batch embedding
        chunk_texts = [c["text"] for c in chunks]

        # Generate embeddings in batch (more efficient)
        embeddings = embeddings_model.embed_documents(chunk_texts)

        # enrich each chunk with snippet and embedding
        for i, c in enumerate(chunks):
            snippet = c["text"]
            c["snippet"] = snippet
            c["embedding"] = embeddings[i]
            del c["text"]

        out = {
            "filename": filename,
            "checksum": checksum,
            "ingest_timestamp": int(time.time()),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embedding_model": embeddings_model_name,
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "chunks": chunks,
        }
        out_path = os.path.join(out_dir, f"{filename}.{checksum}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return {"filename": filename, "status": "ok", "chunks": len(chunks), "out_path": out_path}
    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}


def ingest_files(
    filenames: List[str],
    knowledge_dir: str = "data/knowledge",
    chunk_size: int = 2000,
    overlap: int = 400,
    out_dir: str = "data/vectors",
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2"
) -> List[Dict]:
    results = []
    for fn in filenames:
        res = ingest_file(
            fn,
            knowledge_dir=knowledge_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            out_dir=out_dir,
            embeddings_model_name=embeddings_model_name
        )
        results.append(res)
    return results


if __name__ == "__main__":
    # quick manual smoke test
    print("Listing knowledge docs:", list_documents())