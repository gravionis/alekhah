"""qa.py

Simple question-answering over the ingested vector JSON files produced by
`ingestion.py` (Phase 1). This module loads JSON files from `data/vectors/`,
computes a deterministic query embedding using `ingestion.embed_stub`, performs
cosine-similarity search against stored chunk embeddings, and returns the top-k
matching chunks along with a short aggregated answer composed from snippets.

API:
- answer_question(question: str, k: int = 3) -> dict

CLI:
- python qa.py "your question" --k 3

This file is intentionally lightweight and has no external ML/embedding
dependencies; it uses the deterministic `embed_stub` from `ingestion.py` so it
works in Phase 1.
"""

import os
import json
import math
import heapq
import argparse
import logging
from typing import List, Dict, Any, Optional

# local import from same package
from ingestion import embed_stub
import llm  # new: use llm.get_llm()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

VECTORS_DIR = os.path.join("data", "vectors")


def load_index(vectors_dir: str = VECTORS_DIR) -> List[Dict[str, Any]]:
    """Load all chunk entries from JSON files under `vectors_dir`.

    Returns a flat list of chunk dicts with the following keys ensured:
      - filename, checksum, index, char_start, char_end, snippet, embedding

    Files or chunks that are malformed will be skipped with a logged warning.
    """
    if not os.path.isdir(vectors_dir):
        raise FileNotFoundError(f"vectors directory not found: {vectors_dir}")

    items: List[Dict[str, Any]] = []
    for fn in sorted(os.listdir(vectors_dir)):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(vectors_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning("failed to read %s: %s", path, e)
            continue

        filename = payload.get("filename")
        checksum = payload.get("checksum")
        chunks = payload.get("chunks") or []
        if not isinstance(chunks, list):
            logger.warning("no chunks array in %s", path)
            continue

        for c in chunks:
            emb = c.get("embedding")
            snippet = c.get("snippet")
            index = c.get("index")
            char_start = c.get("char_start")
            char_end = c.get("char_end")

            if not (isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb)):
                logger.warning("skipping chunk with invalid embedding in %s (file: %s index: %s)", path, filename, index)
                continue

            items.append({
                "filename": filename,
                "checksum": checksum,
                "index": index,
                "char_start": char_start,
                "char_end": char_end,
                "snippet": snippet,
                "embedding": emb,
            })

    return items


def _cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between vectors a and b. Returns value in [-1,1]."""
    if len(a) != len(b):
        raise ValueError("vector dimensionality mismatch")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def answer_question(question: str, k: int = 3, vectors_dir: str = VECTORS_DIR, max_answer_chars: int = 10000) -> Dict[str, Any]:
    """Return an answer and the top-k matching chunks for `question`.

    Output format:
      {
        "question": str,
        "answer": str,
        "matches": [ { filename, checksum, index, char_start, char_end, snippet, score } ]
      }

    The `answer` is a simple concatenation of top snippets (trimmed to
    `max_answer_chars`). This is intentionally simple for Phase 1.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be > 0")

    items = load_index(vectors_dir=vectors_dir)
    if not items:
        return {"question": question, "answer": "", "matches": []}

    # compute query embedding
    q_emb = embed_stub(question)
    emb_dim = len(q_emb)

    # verify dims
    for it in items[:5]:
        if len(it["embedding"]) != emb_dim:
            raise ValueError("embedding dimension mismatch between query and stored vectors")

    # compute top-k via heap
    heap: List[tuple] = []  # min-heap of (score, idx)
    for idx, it in enumerate(items):
        try:
            score = _cosine(q_emb, it["embedding"])
        except Exception:
            # if anything goes wrong, treat as very low score
            score = -1.0
        if len(heap) < k:
            heapq.heappush(heap, (score, idx))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, idx))

    # extract and sort by descending score
    top = sorted(heap, key=lambda x: x[0], reverse=True)
    matches = []
    snippets = []

    # prefer linking to data/originals if present
    originals_dir = os.path.abspath(os.path.join("data", "originals"))

    for score, idx in top:
        it = items[idx]
        filename = it.get("filename")
        char_start = it.get("char_start", "")
        char_end = it.get("char_end", "")

        # build link: prefer file:// to originals if exists, else relative path
        if filename:
            candidate = os.path.join(originals_dir, filename)
            if os.path.exists(candidate):
                link = f"file://{candidate}#chars={char_start}-{char_end}"
            else:
                link = f"./{filename}#chars={char_start}-{char_end}"
        else:
            link = ""

        matches.append({
            "filename": filename,
            "index": it.get("index"),
            "char_start": char_start,
            "char_end": char_end,
            "snippet": it.get("snippet"),
            "score": float(score),
            "link": link,
        })
        if it.get("snippet"):
            snippets.append(it.get("snippet"))

    # Build a short aggregated answer by concatenating snippets, deduplicating by text
    seen = set()
    pieces = []
    for s in snippets:
        if s in seen:
            continue
        seen.add(s)
        pieces.append(s)
        if sum(len(p) for p in pieces) >= max_answer_chars:
            break

    # joined text to feed the LLM
    joined = "\n\n".join(pieces)

    # Use LLM to summarize the joined snippets into a concise answer; fall back to concatenation
    try:
        client = llm.get_llm()
        summarized = client.summarize(joined, question=question, max_chars=max_answer_chars)
    except Exception:
        summarized = ""

    if summarized:
        answer = summarized
    else:
        # fallback deterministic answer (existing behavior)
        answer = joined
        if len(answer) > max_answer_chars:
            answer = answer[:max_answer_chars].rsplit(" ", 1)[0] + "..."

    return {
        "question": question,
        "answer": answer,
        "matches": matches,
    }


def summarize_matches(matches: List[Dict[str, Any]], question: Optional[str] = None, max_chars: int = 1000) -> Dict[str, str]:
    """Produce a summary (via LLM when available) and a Markdown reference table with links.

    Returns:
      {
        "summary": "<concise summary text>",
        "references_table": "<markdown table as string>"
      }
    """
    # build markdown reference table (filename as link when possible)
    headers = ["filename", "checksum", "index", "char_start", "char_end", "score"]
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    snippets: List[str] = []
    # attempt to link to a likely originals directory; fallback to relative filename
    originals_dir = os.path.abspath(os.path.join("data", "originals"))

    for m in matches:
        filename = m.get("filename") or ""
        char_start = m.get("char_start", "")
        char_end = m.get("char_end", "")
        # prefer linking to data/originals if that file exists; otherwise link to relative filename
        if filename:
            candidate = os.path.join(originals_dir, filename)
            if os.path.exists(candidate):
                # include char fragment to help the user locate the snippet
                link = f"file://{candidate}#chars={char_start}-{char_end}"
            else:
                # relative link (may be resolvable in repo viewers)
                link = f"./{filename}"
            md_name = f"[{filename}]({link})"
        else:
            md_name = ""

        table_lines.append(
            "| {} | {} | {} | {} | {} | {:.6f} |".format(
                md_name,
                m.get("checksum", "") or "",
                m.get("index", "") or "",
                char_start,
                char_end,
                float(m.get("score", 0.0)),
            )
        )
        s = m.get("snippet")
        if isinstance(s, str) and s.strip():
            snippets.append(s.strip())

    references_table = "\n".join(table_lines)

    # deduplicate snippets preserving order
    seen = set()
    unique_snips = []
    for s in snippets:
        if s in seen:
            continue
        seen.add(s)
        unique_snips.append(s)

    joined = "\n\n".join(unique_snips)

    # Use LLM client to summarize (falls back internally if no external LLM)
    client = llm.get_llm()
    summary = client.summarize(joined, question=question, max_chars=max_chars)

    return {"summary": summary, "references_table": references_table}


def answer_and_summarize(question: str, k: int = 3, vectors_dir: str = VECTORS_DIR, max_answer_chars: int = 1000, summary_chars: int = 1000) -> Dict[str, Any]:
    """Run the vector search and produce both the simple answer and a summarized view with references."""
    out = answer_question(question, k=k, vectors_dir=vectors_dir, max_answer_chars=max_answer_chars)
    summ = summarize_matches(out.get("matches", []), question=question, max_chars=summary_chars)
    out["summary"] = summ["summary"]
    out["references_table"] = summ["references_table"]
    return out


def _main():
    p = argparse.ArgumentParser(description="Ask a question against ingested vectors (Phase 1: JSON fallback)")
    p.add_argument("question", nargs="+", help="Question to ask")
    p.add_argument("--k", type=int, default=3, help="Number of top matches to return")
    p.add_argument("--vectors-dir", default=VECTORS_DIR, help="Directory where vector JSON files are stored")
    p.add_argument("--summarize", action="store_true", help="Also produce a summary and reference table from top matches")
    args = p.parse_args()
    question = " ".join(args.question)
    try:
        if args.summarize:
            out = answer_and_summarize(question, k=args.k, vectors_dir=args.vectors_dir)
        else:
            out = answer_question(question, k=args.k, vectors_dir=args.vectors_dir)
    except Exception as e:
        logger.error("error answering question: %s", e)
        raise
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
