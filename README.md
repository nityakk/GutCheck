# HealthLog

A personal health intelligence app: log symptoms in plain language, upload your medical documents, and ask a multi-agent system about patterns in your own data.

Built because filling out a form is the last thing you want to do when you feel terrible — and because finding patterns across years of journal entries, lab PDFs, and symptom logs is something a human alone can't realistically do.

> **Not medical advice.** This is a personal logging tool. Talk to a doctor.

---

## Features

**Conversational logging**
Type naturally — *"bad cramping after dinner last night, maybe a 7, definitely the pasta"* — and Claude extracts symptoms, severity, category, foods, triggers, and dates. Asks at most one follow-up per turn.

**Weekly food dump**
No time to log daily? Dump the whole week at once: *"this week I had pasta Monday, sushi Wednesday, wine most nights."* Each day gets its own entry with foods populated, no symptoms or severity required.

**Multi-day entries**
*"May 20th–22nd — bad stomach ache, pain 7/10"* creates a single entry spanning three days and renders as a block on the calendar.

**Calendar view**
Month grid with color-coded symptom chips per day. Period bars span their full date range. Click any day to see the full detail panel. Edit any entry directly from the calendar — change dates, symptoms, foods, triggers, severity, notes — without re-logging.

**Trends**
60-day severity chart, food/trigger correlations ranked by average gut severity vs. your personal baseline, category breakdown, top symptoms. Correlations look at same-day and next-day gut response to account for delayed reactions.

**Document corpus**
Upload PDFs of lab results, doctor notes, journals, or articles. Each document is chunked, embedded with a local model, and stored in SQLite alongside your symptom log. No external embedding API needed.

**Multi-agent assistant**
Ask questions like *"what are my worst triggers and what has my doctor said about them?"* An orchestrator routes across two specialist agents:
- **Retriever** — semantic search over your documents, with an LLM rerank pass to pick the most directly relevant passages
- **Analyst** — structured queries over your symptom log (overview, correlations, trend, period overlap)

The orchestrator runs both in sequence for any health question: it gets the data analysis first, then searches your documents using terminology the data surfaced. The **Reflection** agent synthesizes findings with inline citations in `"exact quote" — Source, Date` format, in 2–4 sentences.

---

## Architecture

```
Browser (vanilla JS + HTML)
        │
        ▼
FastAPI (app.py)
  ├── /api/log/*          Conversational logging → claude-haiku-4-5
  ├── /api/chat           Agent chat → Orchestrator
  ├── /api/documents/*    Upload → RAG ingest pipeline
  ├── /api/entries/*      CRUD + full edit (PUT)
  ├── /api/periods/*      CRUD
  └── /api/stats          Trend aggregations (pure Python)

Orchestrator (claude-sonnet-4-6, tool-use loop, max 6 iters)
  ├── analyze_data  →  Analyst Agent  (pure Python + SQL, no LLM)
  │                      overview | correlations | trend | period_overlap
  ├── search_documents →  Retriever Agent (claude-haiku-4-5 rerank)
  │                         fastembed BAAI/bge-small-en-v1.5 → cosine sim → top-12
  │                         → LLM rerank → top-4 passages
  └── synthesize_answer →  Reflection Agent (claude-haiku-4-5)
                             data findings + document passages → 2–4 sentence synthesis

SQLite (gutcheck.db) — single file for everything
  ├── sessions / entries / periods   (symptom log)
  └── documents / chunks             (RAG corpus: text + 384-d embeddings as JSON)
```

**Analyst is sandboxed to a fixed set of named analyses** — no arbitrary SQL — so a poisoned document can't instruct it to delete data.

---

## Models used

| Where | Model | Why |
|---|---|---|
| Orchestrator | `claude-sonnet-4-6` | Tool-use routing, multi-step reasoning |
| Logging extraction | `claude-haiku-4-5` | Structured JSON extraction, fast |
| Document summarization | `claude-haiku-4-5` | Title + one-line summary on ingest |
| Retriever rerank | `claude-haiku-4-5` | Pick top-4 from 12 candidates |
| Reflection / synthesis | `claude-haiku-4-5` | Final answer, inline citations |
| Embeddings | `BAAI/bge-small-en-v1.5` | Local via fastembed, no API key, ~130MB |

---

## Setup

```bash
git clone <repo> healthlog && cd healthlog
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your Anthropic API key: https://console.anthropic.com

uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000).

First document upload will download the embedding model (~130MB) to `~/.cache/huggingface/` and cache it for all future runs.

---

## Project layout

```
healthlog/
├── app.py                  # FastAPI app, logging flow, all REST endpoints
├── agents/
│   ├── orchestrator.py     # Tool-use routing loop (Sonnet)
│   ├── retriever.py        # Semantic search + LLM rerank (Haiku)
│   ├── analyst.py          # Fixed-set SQL analyses, no LLM
│   └── reflection.py       # Synthesis with inline citations (Haiku)
├── rag/
│   ├── ingest.py           # PDF/text extraction, LLM summary, page tracking
│   └── store.py            # Chunking, local embeddings, cosine search
├── static/index.html       # Single-page UI (no build step)
├── requirements.txt
├── .env.example
└── uploads/                # Original uploaded files (gitignored)
```

---

## Roadmap

- [ ] Voice input (Whisper)
- [ ] Photo food logging — vision model extracts foods from meal photos
- [ ] Confidence indicators on correlations (flag when n < 5)
- [ ] Weekly digest — LLM-generated summary of the past 7 days
- [ ] Apple Health export ingestion
- [ ] Drag-to-resize entries on the calendar

---

## License

MIT
