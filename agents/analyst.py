"""
Analyst Agent: runs structured analyses over the entries/periods database.

Exposes a SAFE, FIXED set of analyses (not arbitrary SQL) — the orchestrator
picks one or more and the analyst computes them and returns results. This
sandbox is important because docs in the RAG corpus could contain prompt
injections; we don't want a doc to talk the LLM into running DELETE.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "gutcheck.db"


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _entries_in_range(date_from: Optional[str], date_to: Optional[str]) -> list[sqlite3.Row]:
    parts = []
    params: list = []
    if date_from:
        parts.append("COALESCE(occurred_at, created_at) >= ?")
        params.append(date_from)
    if date_to:
        parts.append("COALESCE(occurred_at, created_at) <= ? || 'T23:59:59'")
        params.append(date_to)
    where = ("WHERE " + " AND ".join(parts)) if parts else ""
    with db() as conn:
        return conn.execute(
            f"SELECT * FROM entries {where} ORDER BY COALESCE(occurred_at, created_at)",
            params,
        ).fetchall()


def _walk_days(start_iso: str, end_iso: str, end_date_iso: Optional[str]) -> list[str]:
    """Days an entry covers (occurred_at .. end_date inclusive)."""
    try:
        s = datetime.strptime(start_iso[:10], "%Y-%m-%d").date()
        e = datetime.strptime((end_date_iso or end_iso)[:10], "%Y-%m-%d").date()
    except ValueError:
        return []
    if e < s:
        e = s
    out = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def analyze_overview(date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """High-level snapshot."""
    rows = _entries_in_range(date_from, date_to)
    if not rows:
        return {"empty": True, "message": "No entries in this date range."}

    by_cat = {}
    sev_by_day = {}
    food_count = {}
    trigger_count = {}
    symptom_count = {}
    for r in rows:
        cat = (r["category"] or "other").lower()
        by_cat[cat] = by_cat.get(cat, 0) + 1
        for d in _walk_days(r["occurred_at"] or r["created_at"], r["occurred_at"] or r["created_at"], r["end_date"]):
            if cat == "gut" and r["severity"] is not None:
                sev_by_day[d] = max(sev_by_day.get(d, 0), r["severity"])
        for f in json.loads(r["foods"] or "[]"):
            food_count[f.lower()] = food_count.get(f.lower(), 0) + 1
        for t in json.loads(r["triggers"] or "[]"):
            trigger_count[t.lower()] = trigger_count.get(t.lower(), 0) + 1
        for s in json.loads(r["symptoms"] or "[]"):
            symptom_count[s.lower()] = symptom_count.get(s.lower(), 0) + 1

    nonzero = [v for v in sev_by_day.values() if v > 0]
    avg = round(sum(nonzero) / len(nonzero), 2) if nonzero else 0.0
    max_day = max(sev_by_day.items(), key=lambda x: x[1]) if sev_by_day else None

    return {
        "total_entries": len(rows),
        "by_category": by_cat,
        "avg_gut_severity": avg,
        "worst_day": {"date": max_day[0], "severity": max_day[1]} if max_day else None,
        "top_foods": sorted(food_count.items(), key=lambda x: -x[1])[:5],
        "top_triggers": sorted(trigger_count.items(), key=lambda x: -x[1])[:5],
        "top_symptoms": sorted(symptom_count.items(), key=lambda x: -x[1])[:5],
    }


def analyze_correlations(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_occurrences: int = 2,
) -> dict:
    """Food/trigger -> gut severity (same-day or next-day). Same logic as /api/stats."""
    rows = _entries_in_range(date_from, date_to)
    gut_by_day = {}
    food_days: dict[str, set] = {}
    trigger_days: dict[str, set] = {}

    for r in rows:
        cat = (r["category"] or "other").lower()
        days = _walk_days(r["occurred_at"] or r["created_at"], r["occurred_at"] or r["created_at"], r["end_date"])
        if cat == "gut" and r["severity"] is not None:
            for d in days:
                gut_by_day[d] = max(gut_by_day.get(d, 0), r["severity"])
        for f in json.loads(r["foods"] or "[]"):
            food_days.setdefault(f.lower().strip(), set()).update(days)
        for t in json.loads(r["triggers"] or "[]"):
            trigger_days.setdefault(t.lower().strip(), set()).update(days)

    def day_sev(d_iso: str) -> int:
        sev_today = gut_by_day.get(d_iso, 0)
        try:
            nxt = (datetime.strptime(d_iso, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
            sev_next = gut_by_day.get(nxt, 0)
        except ValueError:
            sev_next = 0
        return max(sev_today, sev_next)

    nonzero = [v for v in gut_by_day.values() if v > 0]
    baseline = round(sum(nonzero) / len(nonzero), 2) if nonzero else 0.0

    def build(item_days: dict, kind: str) -> list[dict]:
        out = []
        for item, days in item_days.items():
            if len(days) < min_occurrences:
                continue
            scored = [day_sev(d) for d in days if day_sev(d) > 0]
            if not scored:
                continue
            avg = round(sum(scored) / len(scored), 2)
            out.append({
                "item": item,
                "kind": kind,
                "occurrences": len(days),
                "scored_days": len(scored),
                "avg_severity": avg,
                "delta_vs_baseline": round(avg - baseline, 2),
            })
        return out

    correlations = build(food_days, "food") + build(trigger_days, "trigger")
    worst = sorted(correlations, key=lambda x: (-x["avg_severity"], -x["scored_days"]))[:10]
    return {"baseline_severity": baseline, "worst_offenders": worst}


def analyze_period_overlap(date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """Does gut severity differ during periods vs outside?"""
    rows = _entries_in_range(date_from, date_to)
    with db() as conn:
        period_rows = conn.execute("SELECT * FROM periods").fetchall()

    period_days: set = set()
    for p in period_rows:
        try:
            s = datetime.strptime(p["start_date"], "%Y-%m-%d").date()
            e = datetime.strptime(p["end_date"], "%Y-%m-%d").date() if p["end_date"] else datetime.now().date()
        except ValueError:
            continue
        cur = s
        while cur <= e:
            period_days.add(cur.isoformat())
            cur += timedelta(days=1)

    gut_by_day = {}
    for r in rows:
        if (r["category"] or "").lower() != "gut" or r["severity"] is None:
            continue
        for d in _walk_days(r["occurred_at"] or r["created_at"], r["occurred_at"] or r["created_at"], r["end_date"]):
            gut_by_day[d] = max(gut_by_day.get(d, 0), r["severity"])

    on, off = [], []
    for d, sev in gut_by_day.items():
        if sev <= 0:
            continue
        (on if d in period_days else off).append(sev)

    def stats(xs: list[int]) -> dict:
        if not xs:
            return {"n_days": 0, "avg": 0.0, "max": 0}
        return {"n_days": len(xs), "avg": round(sum(xs) / len(xs), 2), "max": max(xs)}

    return {
        "during_period": stats(on),
        "outside_period": stats(off),
        "period_count": len(period_rows),
    }


def analyze_trend(date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    """Is gut severity trending up, down, or steady? Compare halves of the window."""
    rows = _entries_in_range(date_from, date_to)
    gut_by_day = {}
    all_days = set()
    for r in rows:
        days = _walk_days(r["occurred_at"] or r["created_at"], r["occurred_at"] or r["created_at"], r["end_date"])
        for d in days:
            all_days.add(d)
        if (r["category"] or "").lower() == "gut" and r["severity"] is not None:
            for d in days:
                gut_by_day[d] = max(gut_by_day.get(d, 0), r["severity"])
    if len(all_days) < 14:
        return {"message": "Not enough data for a trend (need ≥14 days).", "trend": "unknown"}

    sorted_days = sorted(all_days)
    half = len(sorted_days) // 2
    first_half = sorted_days[:half]
    second_half = sorted_days[half:]
    fh = [gut_by_day.get(d, 0) for d in first_half if gut_by_day.get(d, 0) > 0]
    sh = [gut_by_day.get(d, 0) for d in second_half if gut_by_day.get(d, 0) > 0]
    avg_first = sum(fh) / len(fh) if fh else 0
    avg_second = sum(sh) / len(sh) if sh else 0
    delta = avg_second - avg_first
    if abs(delta) < 0.5:
        trend = "steady"
    elif delta > 0:
        trend = "worsening"
    else:
        trend = "improving"
    return {
        "first_half": {"days": len(first_half), "avg_severity": round(avg_first, 2)},
        "second_half": {"days": len(second_half), "avg_severity": round(avg_second, 2)},
        "delta": round(delta, 2),
        "trend": trend,
    }


# Dispatch table — orchestrator picks one of these analyses by name
ANALYSES = {
    "overview": analyze_overview,
    "correlations": analyze_correlations,
    "period_overlap": analyze_period_overlap,
    "trend": analyze_trend,
}


def run_analyst(analysis: str, date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
    fn = ANALYSES.get(analysis)
    if not fn:
        return {"error": f"unknown analysis: {analysis}. Available: {list(ANALYSES)}"}
    return fn(date_from=date_from, date_to=date_to)
