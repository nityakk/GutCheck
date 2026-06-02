"""
HealthLog (née GutCheck) — conversational symptom logger + multi-agent health intelligence.
Run with: uvicorn app:app --reload
Then open http://localhost:8000
"""
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

DB_PATH = Path(__file__).parent / "gutcheck.db"
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5"  # fast and cheap, plenty smart for structured extraction

# Categories we let the LLM use. Anything else gets mapped to "other".
CATEGORIES = {"gut", "inflammation", "lethargy", "anxiety", "period", "other"}

# Initialize the RAG schema alongside the existing tables (same DB file)
from rag.store import init_rag_db
from rag.ingest import ingest_file, UPLOADS_DIR as _RAG_UPLOADS  # noqa
from agents.orchestrator import run_orchestrator

app = FastAPI(title="HealthLog")


# ---------- database ----------

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables and gently migrate older schemas (adds new columns if missing)."""
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                raw_messages TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                occurred_at TEXT,
                end_date TEXT,          -- YYYY-MM-DD; when set, entry spans occurred_at -> end_date
                category TEXT,          -- gut|inflammation|lethargy|anxiety|period|other
                symptoms TEXT,          -- JSON list
                severity INTEGER,       -- 1-10
                location TEXT,
                triggers TEXT,          -- JSON list
                foods TEXT,             -- JSON list of foods consumed
                duration TEXT,
                notes TEXT,
                raw_text TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_entries_when
                ON entries (COALESCE(occurred_at, created_at) DESC);
            CREATE TABLE IF NOT EXISTS periods (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                created_at TEXT NOT NULL,
                start_date TEXT NOT NULL,   -- YYYY-MM-DD
                end_date TEXT,              -- YYYY-MM-DD, NULL = ongoing
                flow TEXT,                  -- light|medium|heavy|spotting (optional)
                notes TEXT,
                raw_text TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_periods_start
                ON periods (start_date DESC);
        """)
        # Migration: add columns if they don't already exist
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(entries)").fetchall()}
        if "category" not in cols:
            conn.execute("ALTER TABLE entries ADD COLUMN category TEXT")
        if "foods" not in cols:
            conn.execute("ALTER TABLE entries ADD COLUMN foods TEXT")
        if "end_date" not in cols:
            conn.execute("ALTER TABLE entries ADD COLUMN end_date TEXT")


init_db()
init_rag_db()


# ---------- schemas ----------

class StartReq(BaseModel):
    text: str


class ReplyReq(BaseModel):
    session_id: int
    text: str


# ---------- LLM logic ----------

SYSTEM_PROMPT = """You are GutCheck, a calm and efficient health logger for someone with chronic stomach issues.

Your job: turn casual free-text into structured data with MINIMAL friction. A single user message may describe one event, many events, a weekly food summary, AND/OR a menstrual period.

You produce TWO kinds of output:

==========
1) ENTRIES — point-in-time events (symptoms, meals causing trouble, fatigue, etc.)
==========
Extract ONE entry per distinct event. Multiple symptoms at different times → multiple entries. Same window → one entry.

CATEGORIES (pick ONE per entry):
- "gut": stomach pain, cramping, bloating, nausea, diarrhea, constipation, reflux
- "inflammation": joint pain, swelling, skin flares, general inflammation
- "lethargy": fatigue, brain fog, exhaustion, low energy
- "anxiety": anxious feelings, stress, panic, racing thoughts
- "period": menstrual-related symptoms LIKE cramps or PMS that the user wants logged as a discrete event. Do NOT use this for the menstrual period itself — that's a PERIOD, not an entry.
- "other": headache, sleep issue, etc.

FIELDS per entry:
- symptoms: list (e.g. ["cramping", "bloating"])
- severity: 1-10 (mild=2-3, moderate=5-6, bad=7-8, severe=9-10)
- occurred_at: ISO 8601 if resolvable. "yesterday" + current date → previous date. Noon if only a date is known.
- foods: list of what the user consumed around this event (separate from triggers — just what they ate)
- triggers: list of suspected causes (often a food, but could be stress, an activity)
- location, duration, notes: optional

==========
2) PERIODS — menstrual period DATE RANGES
==========
A "period" is a multi-day event with a start_date and (usually) end_date. Output it as a PERIOD object, not an entry.

Triggers to create a period:
- "started my period [date]"
- "got my period [date]"
- "I'm on my period"
- "period from X to Y"
- "period ended [date]"

PERIOD FIELDS:
- start_date: YYYY-MM-DD (REQUIRED)
- end_date: YYYY-MM-DD (NULL = ongoing/unknown)
- flow: "light" | "medium" | "heavy" | "spotting" | null
- notes: optional

If the user mentions a period but you cannot determine the start_date, ask. If start is known but end is not AND they didn't say it's still ongoing, ask whether it's still ongoing or when it ended. ALWAYS ask for missing period dates — they matter for the calendar view.

==========
3) WEEKLY FOOD LOGS — loose summaries of what was eaten across several days
==========
Triggered by messages like "this week I had pasta Monday, pizza Wednesday, sushi Friday"
or "I've been eating a lot of dairy and gluten lately" or "foods this week: eggs, bread, wine."

Rules:
- Create ONE entry per distinct day/meal grouping mentioned.
- category: "other"
- foods: populated from what the user said
- symptoms: []          ← always empty; user is logging food, not a symptom episode
- severity: null        ← never ask for severity on a food log
- occurred_at: infer from the day name using the current date (see INFERENCE RULES below).
  If no specific day is given, use today's date.
- done: true immediately — do NOT ask follow-up questions for food-only entries.

INFERENCE RULES:
- "May 20th-22nd" or "from Monday to Wednesday" → occurred_at = start date, end_date = end date (YYYY-MM-DD)
- Single-day events → end_date = null (do not repeat occurred_at as end_date)
- Day-of-week names → resolve to the most recent past occurrence using the current date.
  Example: current date is Sunday 2025-06-01; "Monday" → 2025-05-26, "Friday" → 2025-05-30.
  "this Monday" or "last Monday" always = the Monday that just passed.
- "really bad" → severity 8-9
- "mild" → 2-3, "moderate" → 5-6
- "after the burrito" → foods: ["burrito"], triggers: ["burrito"]
- "ate pasta for lunch, fine afterward" → foods: ["pasta"], triggers: []
- "yesterday" + current date → previous calendar date
- "started period yesterday" → period start_date = yesterday's date

WHEN TO ASK A FOLLOW-UP (done=false):
- A period's start_date is missing or ambiguous → ASK
- A period is extracted AND end_date is null AND the user did not say "still going", "ongoing", "not sure when it'll end" → ALWAYS ASK "how long did/will your period last?" — even if start_date is clear. The end date is important for the calendar.
- Critical entry fields (symptoms / severity / occurred_at / category) are missing AND uninferable → ASK
- Food-only entries (no symptoms mentioned) → NEVER ask for symptoms or severity. Set done=true.
- ONE question per turn covering all gaps. Warm but brief.

Output JSON ONLY in this exact schema:
{
  "done": bool,
  "message": str,
  "entries": [
    {
      "category": "gut" | "inflammation" | "lethargy" | "anxiety" | "period" | "other",
      "symptoms": [str],
      "severity": int | null,
      "occurred_at": str | null,   // ISO 8601 start datetime (or YYYY-MM-DD if time unknown)
      "end_date": str | null,      // YYYY-MM-DD; set ONLY when the entry spans multiple days (e.g. "May 20th-22nd" → end_date "2025-05-22"). Leave null for single-day events.
      "foods": [str],
      "triggers": [str],
      "location": str | null,
      "duration": str | null,
      "notes": str | null
    }
  ],
  "periods": [
    {
      "start_date": str,        // YYYY-MM-DD
      "end_date": str | null,   // YYYY-MM-DD or null (ongoing)
      "flow": str | null,       // light|medium|heavy|spotting
      "notes": str | null
    }
  ]
}

If done=false, still include best-effort entries and periods so far (with whatever dates you have).
Never give medical advice.
"""


def call_llm(messages: list[dict]) -> dict:
    """Call Claude with `{` prefilled. Retries once if JSON is malformed."""
    now_str = datetime.now().strftime("%A %Y-%m-%d %H:%M").strip()
    system = SYSTEM_PROMPT + f"\n\nCurrent time: {now_str}"

    def _one_call(extra_system: str = "") -> str:
        api_messages = [*messages, {"role": "assistant", "content": "{"}]
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system + extra_system,
            messages=api_messages,
            temperature=0.3,
        )
        raw = "{" + resp.content[0].text
        end = raw.rfind("}")
        if end != -1:
            raw = raw[: end + 1]
        return raw

    raw = _one_call()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Log so we can see what Claude actually returned
        print(f"[call_llm] JSON parse failed: {e}")
        print(f"[call_llm] Raw output was:\n{raw}\n---")
        # Retry once with an explicit reminder about escaping
        retry_raw = _one_call(
            "\n\nIMPORTANT: Return STRICT JSON. Escape any inner double-quotes "
            "in string values with a backslash. Do NOT include any text outside "
            "the JSON object."
        )
        try:
            return json.loads(retry_raw)
        except json.JSONDecodeError:
            print(f"[call_llm] Retry also failed. Raw:\n{retry_raw}")
            # Last resort: return a graceful failure shape the frontend can handle
            return {
                "done": False,
                "message": "I had trouble parsing my own response — could you try rephrasing?",
                "entries": [],
                "periods": [],
            }


# ---------- routes ----------

@app.get("/")
def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/log/start")
def start_log(req: StartReq):
    messages = [{"role": "user", "content": req.text}]
    result = call_llm(messages)
    messages.append({"role": "assistant", "content": json.dumps(result)})

    with db() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (created_at, raw_messages) VALUES (?, ?)",
            (datetime.now().isoformat(), json.dumps(messages)),
        )
        session_id = cur.lastrowid

    if result.get("done"):
        entry_ids, period_ids = finalize_session(
            session_id, req.text, result.get("entries", []), result.get("periods", [])
        )
        return {
            "session_id": session_id,
            "done": True,
            "message": result["message"],
            "entries": result.get("entries", []),
            "periods": result.get("periods", []),
            "entry_ids": entry_ids,
            "period_ids": period_ids,
        }

    return {
        "session_id": session_id,
        "done": False,
        "message": result["message"],
        "entries": result.get("entries", []),
        "periods": result.get("periods", []),
    }


@app.post("/api/log/reply")
def reply_log(req: ReplyReq):
    with db() as conn:
        row = conn.execute(
            "SELECT raw_messages, status FROM sessions WHERE id = ?", (req.session_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "session not found")
    if row["status"] == "finalized":
        raise HTTPException(400, "session already finalized")

    messages = json.loads(row["raw_messages"])
    messages.append({"role": "user", "content": req.text})

    result = call_llm(messages)
    messages.append({"role": "assistant", "content": json.dumps(result)})

    with db() as conn:
        conn.execute(
            "UPDATE sessions SET raw_messages = ? WHERE id = ?",
            (json.dumps(messages), req.session_id),
        )

    if result.get("done"):
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        raw_text = " | ".join(user_turns)
        entry_ids, period_ids = finalize_session(
            req.session_id, raw_text, result.get("entries", []), result.get("periods", [])
        )
        return {
            "session_id": req.session_id,
            "done": True,
            "message": result["message"],
            "entries": result.get("entries", []),
            "periods": result.get("periods", []),
            "entry_ids": entry_ids,
            "period_ids": period_ids,
        }

    return {
        "session_id": req.session_id,
        "done": False,
        "message": result["message"],
        "entries": result.get("entries", []),
        "periods": result.get("periods", []),
    }


def finalize_session(
    session_id: int,
    raw_text: str,
    entries: list[dict],
    periods: list[dict],
) -> tuple[list[int], list[int]]:
    """Persist all extracted entries AND periods from this session."""
    entry_ids = []
    period_ids = []
    now_iso = datetime.now().isoformat()
    with db() as conn:
        for e in entries:
            cat = (e.get("category") or "other").lower()
            if cat not in CATEGORIES:
                cat = "other"
            cur = conn.execute(
                """INSERT INTO entries
                   (session_id, created_at, occurred_at, end_date, category, symptoms,
                    severity, location, triggers, foods, duration, notes, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    now_iso,
                    e.get("occurred_at"),
                    e.get("end_date"),
                    cat,
                    json.dumps(e.get("symptoms") or []),
                    e.get("severity"),
                    e.get("location"),
                    json.dumps(e.get("triggers") or []),
                    json.dumps(e.get("foods") or []),
                    e.get("duration"),
                    e.get("notes"),
                    raw_text,
                ),
            )
            entry_ids.append(cur.lastrowid)

        for p in periods:
            start = p.get("start_date")
            if not start:
                continue  # skip invalid; LLM should have asked
            end = p.get("end_date")
            cur = conn.execute(
                """INSERT INTO periods
                   (session_id, created_at, start_date, end_date, flow, notes, raw_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, now_iso, start, end, p.get("flow"), p.get("notes"), raw_text),
            )
            period_ids.append(cur.lastrowid)

        conn.execute(
            "UPDATE sessions SET status = 'finalized' WHERE id = ?", (session_id,)
        )
    return entry_ids, period_ids


def row_to_entry(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "occurred_at": r["occurred_at"],
        "end_date": r["end_date"],
        "category": r["category"] or "other",
        "symptoms": json.loads(r["symptoms"] or "[]"),
        "severity": r["severity"],
        "location": r["location"],
        "triggers": json.loads(r["triggers"] or "[]"),
        "foods": json.loads(r["foods"] or "[]"),
        "duration": r["duration"],
        "notes": r["notes"],
        "raw_text": r["raw_text"],
    }


@app.get("/api/entries")
def list_entries(limit: int = 500):
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY COALESCE(occurred_at, created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [row_to_entry(r) for r in rows]


class EntryDatesPatch(BaseModel):
    occurred_at: Optional[str] = None  # ISO datetime OR YYYY-MM-DD; we'll normalize
    end_date: Optional[str] = None     # YYYY-MM-DD or null to clear (single-day entry)


class EntryUpdate(BaseModel):
    occurred_at: Optional[str] = None
    end_date: Optional[str] = None
    category: Optional[str] = None
    symptoms: Optional[list[str]] = None
    severity: Optional[int] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    notes: Optional[str] = None
    foods: Optional[list[str]] = None
    triggers: Optional[list[str]] = None


def _normalize_occurred(occurred: Optional[str], previous: Optional[str]) -> Optional[str]:
    """Accept either a full ISO datetime or a YYYY-MM-DD date.
    If only a date is given and we have a previous datetime, preserve its time-of-day."""
    if not occurred:
        return None
    if "T" in occurred:
        return occurred  # already a full ISO datetime
    # date-only — preserve previous time if any, else default to noon
    time_part = "T12:00:00"
    if previous and "T" in previous:
        time_part = "T" + previous.split("T", 1)[1]
    return occurred[:10] + time_part


@app.patch("/api/entries/{entry_id}")
def patch_entry(entry_id: int, body: EntryDatesPatch):
    """Update an entry's dates (used by drag-and-resize on the calendar)."""
    with db() as conn:
        row = conn.execute(
            "SELECT occurred_at FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "entry not found")
        new_occurred = _normalize_occurred(body.occurred_at, row["occurred_at"])
        # Sanity: if end_date is before occurred date, drop end_date
        new_end = body.end_date
        if new_end and new_occurred and new_end < new_occurred[:10]:
            new_end = None
        conn.execute(
            "UPDATE entries SET occurred_at = ?, end_date = ? WHERE id = ?",
            (new_occurred, new_end, entry_id),
        )
    return {"ok": True}


@app.put("/api/entries/{entry_id}")
def update_entry(entry_id: int, body: EntryUpdate):
    """Full update of an entry's fields from the manual edit UI."""
    with db() as conn:
        row = conn.execute(
            "SELECT occurred_at FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "entry not found")
        new_occurred = _normalize_occurred(body.occurred_at, row["occurred_at"])
        cat = (body.category or "other").lower()
        if cat not in CATEGORIES:
            cat = "other"
        new_end = body.end_date or None
        if new_end and new_occurred and new_end < new_occurred[:10]:
            new_end = None
        conn.execute(
            """UPDATE entries SET
               occurred_at = ?, end_date = ?, category = ?,
               symptoms = ?, severity = ?, location = ?,
               duration = ?, notes = ?, foods = ?, triggers = ?
               WHERE id = ?""",
            (
                new_occurred,
                new_end,
                cat,
                json.dumps(body.symptoms or []),
                body.severity,
                body.location,
                body.duration,
                body.notes,
                json.dumps(body.foods or []),
                json.dumps(body.triggers or []),
                entry_id,
            ),
        )
    return {"ok": True}


@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int):
    with db() as conn:
        conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    return {"ok": True}


# ---------- periods ----------

class PeriodUpsert(BaseModel):
    start_date: str               # YYYY-MM-DD
    end_date: Optional[str] = None
    flow: Optional[str] = None
    notes: Optional[str] = None


def row_to_period(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "start_date": r["start_date"],
        "end_date": r["end_date"],
        "flow": r["flow"],
        "notes": r["notes"],
        "raw_text": r["raw_text"],
    }


@app.get("/api/periods")
def list_periods():
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM periods ORDER BY start_date DESC"
        ).fetchall()
    return [row_to_period(r) for r in rows]


@app.post("/api/periods")
def create_period(p: PeriodUpsert):
    now_iso = datetime.now().isoformat()
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO periods (created_at, start_date, end_date, flow, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (now_iso, p.start_date, p.end_date, p.flow, p.notes),
        )
        pid = cur.lastrowid
    return {"id": pid, "ok": True}


@app.put("/api/periods/{period_id}")
def update_period(period_id: int, p: PeriodUpsert):
    with db() as conn:
        conn.execute(
            """UPDATE periods
               SET start_date = ?, end_date = ?, flow = ?, notes = ?
               WHERE id = ?""",
            (p.start_date, p.end_date, p.flow, p.notes, period_id),
        )
    return {"ok": True}


@app.delete("/api/periods/{period_id}")
def delete_period(period_id: int):
    with db() as conn:
        conn.execute("DELETE FROM periods WHERE id = ?", (period_id,))
    return {"ok": True}


@app.get("/api/stats")
def stats():
    """Rich rollup for the trends view.

    Returns:
      - gut_severity_60d: per-day max severity over the last 60 days
      - category_counts, top_foods/triggers/symptoms, total_entries (60-day window)
      - baseline_severity: avg gut severity per day (over last 90 days, days with any
        gut entry only) — the "normal" your foods get compared against
      - food_correlations: list of {item, kind, occurrences, avg_severity, delta}
        where delta = avg_severity - baseline_severity. Positive = worse than baseline.
        Wider 90-day window for more signal. Only items with ≥2 occurrences shown.
    """
    from datetime import timedelta
    with db() as conn:
        rows_90 = conn.execute(
            """SELECT * FROM entries
               WHERE COALESCE(occurred_at, created_at) >= date('now', '-90 days')"""
        ).fetchall()

    today = datetime.now().date()
    cutoff_60 = today - timedelta(days=60)
    cutoff_90 = today - timedelta(days=90)

    # Build a per-day gut severity map across the full 90-day window
    # so correlations can look up severity for "day X" and "day X+1"
    gut_by_day_90: dict[str, int] = {}

    cat_counts: dict[str, int] = {}
    food_counts_60: dict[str, int] = {}
    trigger_counts_60: dict[str, int] = {}
    symptom_counts_60: dict[str, int] = {}
    total_60 = 0

    # For correlation: collect (food/trigger, list_of_days_consumed)
    # day index = list of dates a given item appeared on
    food_days: dict[str, set] = {}
    trigger_days: dict[str, set] = {}

    for r in rows_90:
        cat = (r["category"] or "other").lower()
        start_iso = (r["occurred_at"] or r["created_at"])[:10]
        end_iso = r["end_date"] or start_iso
        try:
            start_d = datetime.strptime(start_iso, "%Y-%m-%d").date()
            end_d = datetime.strptime(end_iso, "%Y-%m-%d").date()
        except ValueError:
            continue
        if end_d < start_d:
            end_d = start_d

        # Walk every day this entry covers, within 90-day window
        sev = r["severity"]
        cur = max(start_d, cutoff_90)
        last = min(end_d, today)
        entry_days = []
        while cur <= last:
            entry_days.append(cur.isoformat())
            if cat == "gut" and sev is not None:
                gut_by_day_90[cur.isoformat()] = max(gut_by_day_90.get(cur.isoformat(), 0), sev)
            cur += timedelta(days=1)

        # 60-day-window-only counters
        if start_d >= cutoff_60:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            total_60 += 1
            for f in json.loads(r["foods"] or "[]"):
                food_counts_60[f.lower()] = food_counts_60.get(f.lower(), 0) + 1
            for t in json.loads(r["triggers"] or "[]"):
                trigger_counts_60[t.lower()] = trigger_counts_60.get(t.lower(), 0) + 1
            for s in json.loads(r["symptoms"] or "[]"):
                symptom_counts_60[s.lower()] = symptom_counts_60.get(s.lower(), 0) + 1

        # Correlation tracking: which days each food/trigger appeared on (90-day window)
        for f in json.loads(r["foods"] or "[]"):
            food_days.setdefault(f.lower().strip(), set()).update(entry_days)
        for t in json.loads(r["triggers"] or "[]"):
            trigger_days.setdefault(t.lower().strip(), set()).update(entry_days)

    # 60-day series for the line chart
    gut_series = []
    for offset in range(59, -1, -1):
        d = (today - timedelta(days=offset)).isoformat()
        gut_series.append({"day": d, "severity": gut_by_day_90.get(d, 0)})

    # Baseline: avg severity across all days that had any gut severity recorded
    nonzero = [v for v in gut_by_day_90.values() if v > 0]
    baseline = round(sum(nonzero) / len(nonzero), 2) if nonzero else 0.0

    def day_severity(day_iso: str) -> int:
        """Max severity on the given day OR the following day (delayed reactions)."""
        sev_today = gut_by_day_90.get(day_iso, 0)
        # parse + 1 day
        try:
            d = datetime.strptime(day_iso, "%Y-%m-%d").date() + timedelta(days=1)
            sev_next = gut_by_day_90.get(d.isoformat(), 0)
        except ValueError:
            sev_next = 0
        return max(sev_today, sev_next)

    def build_correlations(item_days: dict[str, set], kind: str) -> list[dict]:
        out = []
        for item, days in item_days.items():
            if len(days) < 2:
                continue  # need at least 2 occurrences for a pattern
            severities = [day_severity(d) for d in days]
            # Only count days where there's any signal at all (avoid penalizing items
            # that appear on quiet days the user didn't bother logging gut data for)
            scored = [s for s in severities if s > 0]
            if not scored:
                # Item never coincides with a logged gut day — record as 0 with low impact
                avg = 0.0
            else:
                avg = round(sum(scored) / len(scored), 2)
            out.append({
                "item": item,
                "kind": kind,
                "occurrences": len(days),
                "scored_days": len(scored),
                "avg_severity": avg,
                "delta": round(avg - baseline, 2) if avg > 0 else 0.0,
            })
        return out

    correlations = build_correlations(food_days, "food") + build_correlations(trigger_days, "trigger")
    # Worst offenders: highest avg severity, then most occurrences as tiebreaker
    worst = sorted(
        [c for c in correlations if c["avg_severity"] > 0],
        key=lambda x: (-x["avg_severity"], -x["scored_days"]),
    )[:10]
    # Calm-day items: items that consistently appear on low-severity days
    # (only useful if they have several scored days and avg below baseline)
    calm = sorted(
        [c for c in correlations if c["scored_days"] >= 2 and c["avg_severity"] < baseline],
        key=lambda x: (x["avg_severity"], -x["scored_days"]),
    )[:10]

    return {
        "gut_severity_60d": gut_series,
        "category_counts": cat_counts,
        "top_foods": sorted(food_counts_60.items(), key=lambda x: -x[1])[:10],
        "top_triggers": sorted(trigger_counts_60.items(), key=lambda x: -x[1])[:10],
        "top_symptoms": sorted(symptom_counts_60.items(), key=lambda x: -x[1])[:10],
        "total_entries": total_60,
        "baseline_severity": baseline,
        "worst_offenders": worst,
        "calm_day_foods": calm,
    }


@app.get("/api/export.csv")
def export_csv():
    import csv, io
    with db() as conn:
        rows = conn.execute("SELECT * FROM entries ORDER BY created_at DESC").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "occurred_at", "category", "symptoms",
                "severity", "location", "triggers", "foods", "duration",
                "notes", "raw_text"])
    for r in rows:
        w.writerow([r["id"], r["created_at"], r["occurred_at"], r["category"],
                    r["symptoms"], r["severity"], r["location"], r["triggers"],
                    r["foods"], r["duration"], r["notes"], r["raw_text"]])
    return JSONResponse(
        content=buf.getvalue(),
        headers={"Content-Type": "text/csv",
                 "Content-Disposition": "attachment; filename=gutcheck-export.csv"},
    )


# ---------- documents (RAG corpus) ----------

ALLOWED_SOURCE_TYPES = {"pdf", "text", "journal", "article", "doctor_note", "lab"}


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    source_type: str = Form("text"),
    doc_date: Optional[str] = Form(None),
):
    """Upload a file (PDF or text), ingest into the RAG store."""
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise HTTPException(400, f"source_type must be one of: {sorted(ALLOWED_SOURCE_TYPES)}")
    if doc_date:
        try:
            datetime.strptime(doc_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "doc_date must be YYYY-MM-DD")

    # Persist the uploaded file with a timestamped name to avoid collisions
    safe_name = Path(file.filename or "upload").name
    dest = UPLOADS_DIR / f"{int(datetime.now().timestamp())}_{safe_name}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        doc_id = ingest_file(dest, source_type=source_type, doc_date=doc_date)
    except RuntimeError as e:
        # Most likely: VOYAGE_API_KEY missing
        dest.unlink(missing_ok=True)
        raise HTTPException(500, str(e))
    except ValueError as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(e))

    return {"id": doc_id, "filename": safe_name, "ok": True}


@app.get("/api/documents")
def list_docs():
    from rag.store import list_documents
    return list_documents()


@app.delete("/api/documents/{doc_id}")
def remove_doc(doc_id: int):
    from rag.store import delete_document, db as ragdb
    # Also remove the file on disk if we have its path
    with ragdb() as conn:
        row = conn.execute("SELECT raw_path FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row and row["raw_path"]:
        try:
            Path(row["raw_path"]).unlink()
        except OSError:
            pass
    delete_document(doc_id)
    return {"ok": True}


# ---------- agent chat ----------

# In-memory conversation store per session. Lightweight; resets when server restarts.
# For persistence, this could move into SQLite — left simple for now.
_chat_sessions: dict[str, list[dict]] = {}


class ChatReq(BaseModel):
    session_id: Optional[str] = None
    message: str


@app.post("/api/chat")
def chat(req: ChatReq):
    """Ask the multi-agent system a question about your health data + documents."""
    sid = req.session_id or f"sess_{int(datetime.now().timestamp() * 1000)}"
    history = _chat_sessions.setdefault(sid, [])

    # The orchestrator handles routing + final synthesis
    result = run_orchestrator(req.message, conversation_history=history)

    # Append both user message and final assistant text to the history for next turn.
    # (We don't store tool-use turns — keeps the history compact and the next
    # turn re-decides routing based on what the user actually asked.)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": result["answer"]})

    return {
        "session_id": sid,
        "answer": result["answer"],
        "agent_trace": result["agent_trace"],
        "citations": result["citations"],
    }


@app.post("/api/chat/reset")
def chat_reset(session_id: str):
    _chat_sessions.pop(session_id, None)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
