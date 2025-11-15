"""ingestion.py

Utility functions for listing documents, extracting text, chunking, and a simple
ingest pipeline. Phase 1 implementation stores chunk + metadata + deterministic
pseudo-embeddings to JSON files under `data/vectors/` as a fallback when a
pgvector DB is not configured.

This file intentionally keeps dependencies small: it uses PyPDF2 for PDF text
extraction. Embedding generation is a deterministic stub so the pipeline can be
tested without external model dependencies.
"""

import os
import hashlib
import json
import time
from typing import List, Dict

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


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
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is not installed; cannot read PDFs")
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
    """Chunk by characters. Returns list of dicts with text, start, end, index."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        overlap = 0

    chunks = []
    start = 0
    text_len = len(text)
    index = 0
    while start < text_len:
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "index": index,
            "text": chunk_text,
            "char_start": start,
            "char_end": min(end, text_len),
        })
        index += 1
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def embed_stub(text: str, dim: int = 32) -> List[float]:
    """Deterministic pseudo-embedding: derived from sha256 hash of text.

    - Uses a small dimension by default so storage is compact for testing.
    - Values are floats in range [-1, 1].
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand or repeat digest to reach required bytes
    needed = dim * 4
    data = (h * ((needed // len(h)) + 1))[:needed]
    vals = []
    for i in range(0, needed, 4):
        chunk = data[i : i + 4]
        val = int.from_bytes(chunk, "big", signed=False)
        # normalize to [-1, 1]
        vals.append((val / (2 ** 32 - 1)) * 2 - 1)
    return vals


def ingest_file(filename: str, knowledge_dir: str = "knowledge", chunk_size: int = 1000, overlap: int = 200, out_dir: str = "data/vectors") -> Dict:
    """Ingest a single file. Returns status dict with counts and output path.

    Behavior: parses .md/.pdf, normalizes, chunks, generates pseudo-embeddings, and
    writes a JSON file with chunks & metadata to `out_dir/{filename}.{checksum}.json`.
    """
    ensure_dir(knowledge_dir)
    ensure_dir(out_dir)
    path = os.path.join(knowledge_dir, filename)
    if not os.path.exists(path):
        return {"filename": filename, "status": "missing", "error": "file not found"}

    try:
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
        # enrich each chunk with snippet and embedding
        for c in chunks:
            snippet = c["text"][:400]
            c["snippet"] = snippet
            c["embedding"] = embed_stub(c["text"])  # small deterministic vector
            # remove full text from chunk payload to keep stored files smaller
            # but keep a snippet and char offsets
            del c["text"]

        out = {
            "filename": filename,
            "checksum": checksum,
            "ingest_timestamp": int(time.time()),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunks": chunks,
        }
        out_path = os.path.join(out_dir, f"{filename}.{checksum}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return {"filename": filename, "status": "ok", "chunks": len(chunks), "out_path": out_path}
    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}


def ingest_files(filenames: List[str], knowledge_dir: str = "knowledge", chunk_size: int = 1000, overlap: int = 200, out_dir: str = "data/vectors") -> List[Dict]:
    results = []
    for fn in filenames:
        res = ingest_file(fn, knowledge_dir=knowledge_dir, chunk_size=chunk_size, overlap=overlap, out_dir=out_dir)
        results.append(res)
    return results


if __name__ == "__main__":
    # quick manual smoke test
    print("Listing knowledge docs:", list_documents())

