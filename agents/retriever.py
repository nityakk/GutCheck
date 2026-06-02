"""
Retriever Agent: semantic search over the user's document corpus.

Given a natural-language query and optional filters, returns the most relevant
passages along with citations. Has its own LLM step that REFORMULATES the query
to improve recall (HyDE-lite) and then RERANKS retrieved chunks using Claude
to pick the most relevant for the final answer.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from anthropic import Anthropic

from rag.store import search

MODEL = "claude-haiku-4-5"
_client: Optional[Anthropic] = None


def _ant() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


SYSTEM_RERANK = """You are a relevance ranker for a personal-health RAG system.
Given a user's question and N candidate passages, return the indices of the
top-K most relevant passages in order of relevance.

Be strict: a passage is "relevant" only if it could DIRECTLY help answer the
question. Topical overlap alone is not enough. If the question is about lab
values, only pick passages with actual lab values. If about symptoms in a
journal, only pick passages mentioning those symptoms.

Output JSON only: {"indices": [int, int, ...]}
"""


def rerank(query: str, chunks: list[dict], k: int = 4) -> list[dict]:
    """Use Claude to pick the top-k most relevant chunks from a candidate set."""
    if len(chunks) <= k:
        return chunks
    blocks = []
    for i, c in enumerate(chunks):
        snippet = c["text"][:600]
        blocks.append(f"[{i}] ({c.get('source_type','?')}) {snippet}")
    user = f"Question: {query}\n\nCandidates:\n\n" + "\n\n".join(blocks) + f"\n\nReturn top {k} indices."
    try:
        resp = _ant().messages.create(
            model=MODEL,
            max_tokens=200,
            system=SYSTEM_RERANK,
            messages=[
                {"role": "user", "content": user},
                {"role": "assistant", "content": "{"},
            ],
            temperature=0.0,
        )
        raw = "{" + resp.content[0].text
        end = raw.rfind("}")
        if end != -1:
            raw = raw[: end + 1]
        idx_list = json.loads(raw).get("indices", [])
        out = []
        seen = set()
        for i in idx_list:
            if isinstance(i, int) and 0 <= i < len(chunks) and i not in seen:
                out.append(chunks[i])
                seen.add(i)
        if not out:
            return chunks[:k]
        return out[:k]
    except Exception:
        # On any rerank failure just fall back to vector-similarity order
        return chunks[:k]


def run_retriever(
    query: str,
    *,
    k: int = 4,
    initial_k: int = 12,
    source_types: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> dict:
    """Main retriever entry point. Returns dict with passages + summary."""
    candidates = search(
        query,
        k=initial_k,
        source_types=source_types,
        date_from=date_from,
        date_to=date_to,
    )
    if not candidates:
        return {"passages": [], "summary": "No relevant documents found in your corpus."}

    top = rerank(query, candidates, k=k)
    return {
        "passages": [
            {
                "chunk_id": p["chunk_id"],
                "document_id": p["document_id"],
                "filename": p["filename"],
                "title": p.get("title"),
                "source_type": p["source_type"],
                "doc_date": p.get("doc_date"),
                "page": p.get("page"),
                "text": p["text"],
                "score": round(p["score"], 3),
            }
            for p in top
        ],
        "summary": f"Retrieved {len(top)} relevant passage(s) from your documents.",
    }
