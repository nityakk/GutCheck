"""
RAG store: documents -> chunks -> embeddings, with cosine similarity search.
Uses SQLite (single file) and a local sentence-transformers model. No API key required.
First run will download the model (~80MB) to ~/.cache/huggingface/.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

DB_PATH = Path(__file__).parent.parent / "gutcheck.db"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-d, fast, ~130MB
EMBED_DIM = 384

_model = None


def get_model():
    """Lazy-load the embedding model. Uses fastembed (no torch dependency)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=EMBED_MODEL)
    return _model


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_rag_db():
    """Create the documents + chunks tables. Idempotent."""
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                source_type TEXT NOT NULL,    -- pdf | text | journal | article | doctor_note | lab
                uploaded_at TEXT NOT NULL,
                doc_date TEXT,                -- date associated with the content (for filtering)
                title TEXT,
                summary TEXT,                 -- one-line LLM summary
                page_count INTEGER,
                raw_path TEXT                  -- path to original file (in uploads/)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                ord INTEGER NOT NULL,         -- chunk order within doc
                text TEXT NOT NULL,
                page INTEGER,
                embedding TEXT NOT NULL,      -- JSON list of floats
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_docs_date ON documents(doc_date DESC);
        """)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Simple recursive-ish chunker: prefer paragraph boundaries, fall back to char windows.
    Personal-corpus scale, so we don't need a tokenizer-aware splitter."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # First try splitting on paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            if len(p) > max_chars:
                # Long paragraph: char-window with overlap
                for i in range(0, len(p), max_chars - overlap):
                    chunks.append(p[i : i + max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Batch-embed a list of strings using a local sentence-transformers model.
    `input_type` is accepted but ignored (kept for API parity with hosted providers
    like Voyage that prepend different prompts for query vs document)."""
    if not texts:
        return []
    model = get_model()
    # sentence-transformers handles batching internally; we still chunk to bound memory
    out: list[list[float]] = []
    BATCH = 64
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        # fastembed.embed returns a generator of numpy arrays
        embeddings = list(model.embed(batch))
        out.extend(emb.tolist() for emb in embeddings)
    return out


def add_document(
    filename: str,
    text: str,
    source_type: str = "text",
    doc_date: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    page_count: Optional[int] = None,
    raw_path: Optional[str] = None,
    page_map: Optional[list[int]] = None,
) -> int:
    """Ingest a document: chunk, embed, store. Returns document_id.

    page_map: optional list aligning each chunk to its source page number.
              If omitted, chunks won't have page numbers.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("document is empty")
    embeddings = embed_texts(chunks, input_type="document")

    with db() as conn:
        cur = conn.execute(
            """INSERT INTO documents
               (filename, source_type, uploaded_at, doc_date, title, summary, page_count, raw_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, source_type, datetime.now().isoformat(), doc_date, title,
             summary, page_count, raw_path),
        )
        doc_id = cur.lastrowid
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            page = page_map[i] if page_map and i < len(page_map) else None
            conn.execute(
                """INSERT INTO chunks (document_id, ord, text, page, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_id, i, chunk, page, json.dumps(emb)),
            )
    return doc_id


def list_documents() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: int):
    with db() as conn:
        # chunks cascade via FK, but make it explicit in case the DB was opened
        # without foreign_keys=ON
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a is shape (d,) or (n, d), b is (m, d). Returns (n, m) or (m,)."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T


def search(
    query: str,
    *,
    k: int = 6,
    source_types: Optional[list[str]] = None,
    date_from: Optional[str] = None,    # YYYY-MM-DD
    date_to: Optional[str] = None,
    document_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Semantic search over chunks with optional metadata filters.

    Returns list of {chunk_id, document_id, filename, source_type, doc_date,
                     page, text, score} sorted by score desc.
    """
    q_emb_list = embed_texts([query], input_type="query")
    if not q_emb_list:
        return []
    q_emb = np.array(q_emb_list[0], dtype=np.float32)

    # Build WHERE clause for metadata filter at DB layer so we don't pull all rows
    where_parts = []
    params: list = []
    if source_types:
        placeholders = ",".join("?" * len(source_types))
        where_parts.append(f"d.source_type IN ({placeholders})")
        params.extend(source_types)
    if date_from:
        where_parts.append("(d.doc_date IS NULL OR d.doc_date >= ?)")
        params.append(date_from)
    if date_to:
        where_parts.append("(d.doc_date IS NULL OR d.doc_date <= ?)")
        params.append(date_to)
    if document_ids:
        placeholders = ",".join("?" * len(document_ids))
        where_parts.append(f"c.document_id IN ({placeholders})")
        params.extend(document_ids)
    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    with db() as conn:
        rows = conn.execute(
            f"""SELECT c.id AS chunk_id, c.document_id, c.text, c.page, c.embedding,
                       d.filename, d.source_type, d.doc_date, d.title
                  FROM chunks c JOIN documents d ON d.id = c.document_id
                  {where_sql}""",
            params,
        ).fetchall()

    if not rows:
        return []

    # Compute similarity in NumPy — fast even for thousands of chunks
    embs = np.array([json.loads(r["embedding"]) for r in rows], dtype=np.float32)
    scores = cosine_sim(q_emb, embs)
    top_idx = np.argsort(-scores)[:k]

    return [
        {
            "chunk_id": rows[i]["chunk_id"],
            "document_id": rows[i]["document_id"],
            "filename": rows[i]["filename"],
            "source_type": rows[i]["source_type"],
            "doc_date": rows[i]["doc_date"],
            "title": rows[i]["title"],
            "page": rows[i]["page"],
            "text": rows[i]["text"],
            "score": float(scores[i]),
        }
        for i in top_idx
    ]
