# Business Requirements Document (BRD)

Last updated: 2025-11-16

Project: Document-backed Q&A & Rule Generator

Overview
--------
This document captures the business and functional requirements for a system that ingests documents (PDF and Markdown in Phase 1) placed into a `knowledge` folder, stores chunked embeddings in a pgvector-enabled PostgreSQL instance, and exposes a Streamlit application with three pages: Ingestion, Question Answer, and Rule Generator.

Purpose
-------
Provide a clear, actionable specification for Phase 1 of a document ingestion and semantic search system so developers and stakeholders have aligned expectations for features, acceptance criteria, and next steps.

Scope (Phase 1)
----------------
In scope:
- Support for reading `.pdf` and `.md` files placed in a `knowledge` folder.
- A Streamlit Ingestion page that lists files, supports refresh, per-file checkboxes, and an "ingest" action.
- Chunking documents with configurable chunk size and overlap and storing embeddings + metadata in pgvector.
- Streamlit application skeleton pages for Question Answer and Rule Generator (placeholders acceptable in Phase 1).

Out of scope (Phase 1):
- Live filesystem watchers or automatic background ingestion.
- Additional file formats (DOCX, PPTX, images/OCR) — reserved for future phases.
- Authentication, multi-tenant isolation, and production-grade role-based access control.

Stakeholders
------------
- Product Owner: defines acceptance criteria and prioritization.
- Data Engineer: implements parsing, chunking, embeddings pipeline, and pgvector schema.
- Backend Engineer: manages DB connectivity, ingestion orchestration, and idempotency.
- Frontend Engineer: Streamlit UI for ingestion and other pages.
- QA: validates acceptance criteria and functionality.
- End Users / Analysts: add documents to `knowledge` and use the UI to query and generate rules.

User Personas
-------------
- Non-technical analyst: drops files into `knowledge`, uses the Streamlit UI to ingest and ask questions.
- Developer/operator: configures DB connection, embedding model, and chunking parameters.

Functional Requirements
-----------------------
System-level
- The system must accept and parse PDF and Markdown files for Phase 1.
- Embeddings must be stored in a PostgreSQL database with the pgvector extension enabled.
- Each stored vector must include metadata linking it to the source file and chunk (filename, chunk index, short text excerpt, char offsets, ingest timestamp, checksum).

Ingestion page (Streamlit) — Phase 1 specifics
1. File listing
   - Display all files with extensions `.pdf` and `.md` that exist in the repository folder named `knowledge` (relative to the app's working directory).
2. Refresh
   - Provide a "Refresh" button which re-reads the `knowledge` folder and updates the displayed list.
3. Selection
   - Provide a checkbox per file so users can select or deselect files for ingestion.
4. Ingest action
   - Provide a global "ingest" button. When clicked for selected files, the app must:
     a. Parse the file to extract text (PDF text extraction for `.pdf`, direct read for `.md`).
     b. Normalize text (strip excessive whitespace, normalize newlines).
     c. Chunk text into contiguous pieces using configured chunk size and overlap.
     d. Produce embeddings for each chunk using the configured embedding model.
     e. Write vectors to the pgvector table with metadata (see Data Model section).
     f. Display a progress indicator and a final status message per file (success/failure and counts).
5. Idempotency
   - Ingestion should be idempotent: repeated ingestion of the same file should either update existing vectors (based on checksum/version) or skip duplicates while reporting the behavior.

Question Answer page (Streamlit) — Phase 1 (placeholder acceptable)
- Provide a question input box and a "Search" button.
- Query the pgvector store for top-k similar chunks.
- Return a composed answer using retrieved chunks and list the source files and chunk excerpts. (In Phase 1 this may be a basic proof-of-concept or placeholder UI backed by a simple retrieval flow.)

Rule Generator page (Streamlit) — Phase 1 (placeholder acceptable)
- Provide a UI to compose or derive rules from retrieved content or free text.
- Allow exporting generated rules as plain text or downloadable file.
- Show provenance linking rules back to source documents/chunks (can be minimal in Phase 1).

Data Model and Storage
----------------------
Primary pgvector table (example logical schema):
- id: UUID or serial primary key
- vector: VECTOR (pgvector column)
- filename: text
- chunk_index: integer
- chunk_text_snippet: text (truncated preview)
- char_start: integer
- char_end: integer
- ingest_timestamp: timestamptz
- checksum: text (e.g., file hash)
- metadata: jsonb (optional free-form metadata)

Configuration
- chunk_size (tokens/characters)
- chunk_overlap
- embedding_model (local or remote model identifier)
- pgvector connection string (from environment variable)

Acceptance Criteria (Phase 1)
-----------------------------
1. The Ingestion page lists all `.pdf` and `.md` files currently present in `knowledge`.
2. Clicking "Refresh" updates the list to reflect additions/removals in `knowledge`.
3. Selecting files via checkboxes and clicking "ingest" stores chunk vectors and metadata in the pgvector table.
4. A clear on-screen status shows success/failure and counts per file after ingestion.
5. The system demonstrates idempotent handling of re-ingestion (skip or update duplicates per chosen strategy).
6. Streamlit app contains navigable pages for Ingestion, Question Answer, and Rule Generator (the latter two may show placeholder content in Phase 1).

Non-functional Requirements
---------------------------
- Performance: reasonable ingestion time for typical documents (e.g., up to 50 PDF pages) — target configurable timeouts.
- Reliability: ingestion should fail gracefully and provide helpful error messages.
- Observability: log ingestion start/end, per-file vector counts, and errors.
- Security: DB credentials and embedding keys must be provided via environment variables; no secrets checked into source control.
- Scalability: design choices should allow future background processing and horizontal scaling.

Idempotency Strategy (recommended)
---------------------------------
- Compute a checksum (e.g., SHA256) of the file contents at ingest time.
- Store checksum and ingest_timestamp in a file-level metadata table.
- On ingest, if checksum matches existing record, either skip ingestion or replace vectors for that file (implementation choice). Document the chosen behavior in code and UI.

Error Handling and Edge Cases
-----------------------------
- Empty or unreadable files: skip with a clear error message in the UI.
- Very large files: surface a warning and support configurable maximum file size.
- Non-text PDFs (scanned images): Phase 1 should surface a parsing failure; OCR support planned for later phases.

Risks and Mitigations
---------------------
- Large documents may cause long ingestion times — mitigation: max file size, chunking controls, and progress reporting.
- Poor PDF text extraction may reduce search quality — mitigation: document restrictions and add parser fallbacks in future.
- Duplicate ingestion may bloat storage — mitigation: use checksum-based idempotency.

Roadmap / Future Phases
-----------------------
Phase 2 and beyond:
- Add support for DOCX, PPTX, images (OCR), and other file formats.
- Introduce a background ingestion worker and file watcher for automatic ingestion.
- Add authentication and role-based access controls.
- Improve UI: file previews, ingestion history, batch selection, and scheduling.
- Add model-selection UI and hybrid search (BM25 + vector).

Quality Gates
-------------
Before merging Phase 1 changes, validate:
- Build/lint (Streamlit Python app should import without runtime errors).
- Manual smoke test for ingestion flow using a small set of sample `.md` and `.pdf` files placed into `knowledge`.
- Verify vectors and metadata are written to pgvector.

Open Questions / Assumptions
---------------------------
- The `knowledge` folder is on the same machine/container as the Streamlit app and accessible from the app's working directory.
- A Postgres instance with pgvector is available and reachable using env vars.
- Embedding model (remote or local) is configured and available via environment variables or a service.

Appendix: Minimal Acceptance Test Plan (manual)
----------------------------------------------
1. Prepare test files: create `knowledge/test.md` and `knowledge/test.pdf` (small files).
2. Run the Streamlit app and open the Ingestion page.
3. Click "Refresh" — verify both files appear.
4. Select both files via checkboxes and click "ingest".
5. Observe progress and final status: expect success and a vector count > 0 per file.
6. Check the pgvector table for vectors and metadata for `test.md` and `test.pdf`.
7. Re-run ingest for the same files and verify idempotency behavior (skip or update) and proper logging.

## Phase 1 QA Implementation (qa.py)

New in the repository for Phase 1 is a lightweight QA module implemented as `qa.py`.
This file implements a simple retrieval-based QA fallback that operates over the
Phase 1 vector storage format (JSON files under `data/vectors/`) produced by the
ingestion pipeline.

Key behaviors (Phase 1):
- Loads JSON vector files from `data/vectors/` (each file contains `filename`,
  `checksum`, and a `chunks` array where each chunk contains `index`,
  `char_start`, `char_end`, `snippet` and a deterministic `embedding`).
- Uses the deterministic `embed_stub` from `ingestion.py` to compute a query
  embedding for the user's question (no external model required in Phase 1).
- Performs cosine-similarity search over stored embeddings and returns the
  top-k matching chunks.
- Returns provenance information for each match: `filename`, `checksum`,
  `index`, `char_start`, `char_end`, `snippet`, and a similarity `score`.
- Produces a short aggregated answer by concatenating top snippets (trimmed to
  a configurable max length). This is intentionally simple for Phase 1.

API and CLI
- Programmatic API: `answer_question(question: str, k: int = 3) -> dict`.
  Output shape:
  {
    "question": str,
    "answer": str,  # short concatenated snippet-based answer
    "matches": [
      {"filename": str, "checksum": str, "index": int, "char_start": int,
       "char_end": int, "snippet": str, "score": float}
    ]
  }
- CLI: `python qa.py "your question" --k 3` prints the JSON output to stdout.

Data shapes and invariants
- Query embedding dimension must match stored embedding dimension (Phase 1 uses
  the same deterministic `embed_stub` so this holds by default).
- Stored JSON file schema is the one produced by the `ingest_*` utilities in
  `ingestion.py` (chunks array with `snippet` and `embedding`).
- The QA flow will skip malformed files or chunks and log warnings.

Phase 1 Acceptance Criteria (update)
- The repository contains a working `qa.py` module that can query the Phase 1
  vector store (`data/vectors/`) and return the top-k matching chunks.
- The QA output includes provenance linking each returned snippet back to its
  source file and chunk offsets (filename, checksum, chunk index, char offsets).
- A simple CLI (`python qa.py "..." --k N`) produces JSON with `answer` and
  `matches` that can be consumed by the Streamlit app or used for manual smoke
  testing.

Developer notes / next steps
- In Phase 2, swap out the deterministic `embed_stub` for a model-backed
  embedding function and replace the JSON file index with pgvector-backed
  storage or a hybrid approach.
- The Streamlit Question Answer page can call `answer_question` (or invoke the
  CLI) to display results and provenance. For Phase 1 the Streamlit page may
  show the JSON output or a formatted view of the `matches` plus the aggregated
  `answer`.
- Consider caching the in-memory index for improved query latency on repeated
  queries; ensure cache invalidation occurs after re-ingestion or refresh.


-- End of BRD --
