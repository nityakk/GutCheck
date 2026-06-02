"""
Reflection Agent: takes findings from Retriever + Analyst and produces a
warm, careful synthesis. Strictly NO medical advice — only observations,
questions, and gentle prompts toward self-experimentation.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from anthropic import Anthropic

MODEL = "claude-haiku-4-5"
_client: Optional[Anthropic] = None


def _ant() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


SYSTEM = """You are the Reflection Agent in a personal health logging app.

Your job: take findings from a Retriever Agent (passages from the user's own
documents — journal entries, lab PDFs, doctor notes, saved articles) and an
Analyst Agent (statistics over the user's structured symptom log), and produce
a thoughtful synthesis for the user.

CORE PRINCIPLES — non-negotiable:
1. NEVER give medical advice, diagnose anything, recommend medications,
   supplements, or specific treatments. The user has doctors for that.
2. Frame observations as observations, not conclusions: "I notice that..."
   "It looks like..." NEVER "This means you have..."
3. End with one short Socratic question when it adds value.
4. When patterns are weak (small sample size, conflicting signals), say so briefly.
5. Cite by quoting the source directly using this exact format:
     "exact words from the document" — Filename, Date
   Example: "avoid high-lactose foods for 6 weeks" — Gastro Visit, March 2024
   Always quote; never paraphrase when the document text is available.
6. Use structured data only when it's actually informative.

STYLE:
- 2–4 sentences maximum. Never longer, no matter how much material you have.
- One short paragraph only. No bullet points, no headers, no lists.
- If you have many findings, surface the single most relevant one. Drop the rest.
- Direct. A friend who noticed something and said it in one breath.
"""


def run_reflection(
    user_query: str,
    *,
    retrieved_passages: Optional[list[dict]] = None,
    analyst_findings: Optional[dict] = None,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Produce the final user-facing synthesis."""
    parts = [f"USER QUESTION: {user_query}\n"]

    if retrieved_passages:
        parts.append("\n=== Relevant passages from your documents ===")
        for i, p in enumerate(retrieved_passages):
            header = f"[{i+1}] {p.get('title') or p.get('filename')}"
            if p.get("doc_date"):
                header += f" — {p['doc_date']}"
            if p.get("page"):
                header += f" (p. {p['page']})"
            parts.append(f"\n{header}\n{p['text']}")

    if analyst_findings:
        parts.append("\n=== Findings from your symptom log ===")
        parts.append(json.dumps(analyst_findings, indent=2))

    if not retrieved_passages and not analyst_findings:
        parts.append(
            "\n(No documents or log data were available. Answer warmly but mention "
            "that you don't have specific data to draw from yet.)"
        )

    user_msg = "\n".join(parts)
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_msg})

    resp = _ant().messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM,
        messages=messages,
        temperature=0.5,
    )
    return resp.content[0].text
