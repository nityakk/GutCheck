"""
Ingestion: take an uploaded file, extract text (with optional page tracking),
generate a quick LLM summary, and hand off to the RAG store.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from pypdf import PdfReader

from .store import add_document, chunk_text

UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
SUMMARY_MODEL = "claude-haiku-4-5"

_anthropic: Optional[Anthropic] = None


def get_anthropic() -> Anthropic:
    global _anthropic
    if _anthropic is None:
        _anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic


def extract_pdf(path: Path) -> tuple[str, int, list[str]]:
    """Returns (joined_text, page_count, per_page_texts)."""
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return ("\n\n".join(pages), len(pages), pages)


def text_with_page_map(per_page_texts: list[str]) -> tuple[str, list[int]]:
    """Combine pages and return a flat text + a parallel list mapping each
    chunk produced from this text to its source page. We re-chunk after the
    join so the page map needs to be reconstructed by tracking offsets."""
    full = "\n\n".join(per_page_texts)
    chunks = chunk_text(full)
    # Build offset → page map by walking the original page texts
    page_breaks: list[tuple[int, int]] = []  # (cumulative_offset, page_number)
    cursor = 0
    for i, p in enumerate(per_page_texts):
        page_breaks.append((cursor, i + 1))
        cursor += len(p) + 2  # +2 for the "\n\n" join

    def page_for_offset(offset: int) -> int:
        page = 1
        for cum, pg in page_breaks:
            if offset >= cum:
                page = pg
            else:
                break
        return page

    page_map: list[int] = []
    search_cursor = 0
    for ch in chunks:
        # find this chunk in full, starting from search_cursor
        idx = full.find(ch, search_cursor)
        if idx < 0:
            page_map.append(1)
        else:
            page_map.append(page_for_offset(idx))
            search_cursor = idx + len(ch) - 50  # allow small overlaps
    return full, page_map


def summarize(text: str, filename: str) -> tuple[str, str]:
    """Generate a (title, one-line summary) for the document via Claude.
    Falls back to filename + first 80 chars if the API is unavailable."""
    snippet = text[:6000]
    try:
        client = get_anthropic()
        resp = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=300,
            system=(
                "You generate a concise title and one-sentence summary for a personal "
                "health document. Output ONLY JSON: {\"title\": \"...\", \"summary\": \"...\"}. "
                "Title: 3-8 words. Summary: 1-2 sentences describing what's in this document "
                "(e.g. 'Lipid panel from May 2024 showing elevated LDL')."
            ),
            messages=[
                {"role": "user", "content": f"Filename: {filename}\n\n---\n{snippet}"},
                {"role": "assistant", "content": "{"},
            ],
            temperature=0.2,
        )
        raw = "{" + resp.content[0].text
        end = raw.rfind("}")
        if end != -1:
            raw = raw[: end + 1]
        import json as _json
        data = _json.loads(raw)
        return data.get("title", filename), data.get("summary", "")
    except Exception:
        return filename, text[:120].replace("\n", " ").strip()


def ingest_file(
    file_path: Path,
    *,
    source_type: str = "text",
    doc_date: Optional[str] = None,
) -> int:
    """Ingest a file from disk. Returns document_id."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        _, page_count, page_texts = extract_pdf(file_path)
        full, page_map = text_with_page_map(page_texts)
        text = full
    else:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        page_count = None
        page_map = None

    if not text.strip():
        raise ValueError(f"no extractable text in {file_path.name}")

    title, summary = summarize(text, file_path.name)

    return add_document(
        filename=file_path.name,
        text=text,
        source_type=source_type,
        doc_date=doc_date,
        title=title,
        summary=summary,
        page_count=page_count,
        raw_path=str(file_path),
        page_map=page_map,
    )
