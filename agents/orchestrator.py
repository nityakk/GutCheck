"""
Orchestrator: receives a user query, decides which agents to invoke via
Claude's tool-use loop, accumulates findings, and produces a final answer.

The orchestrator itself does NOT answer questions — its job is purely routing.
The final user-facing reply comes from the Reflection agent, which sees the
gathered material from Retriever and Analyst.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from anthropic import Anthropic

from .analyst import ANALYSES, run_analyst
from .reflection import run_reflection
from .retriever import run_retriever

MODEL = "claude-sonnet-4-6" 
_client: Optional[Anthropic] = None


def _ant() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


ORCHESTRATOR_SYSTEM = """You are the Orchestrator of a personal health-intelligence app called HealthLog.

You DO NOT answer the user directly. Your only job is to decide which specialist
agents to call (via tools) to gather material, and then call the synthesize_answer
tool to produce the final response.

You have three tools:
1. search_documents — RAG over the user's uploaded documents (journals, lab
   PDFs, doctor notes, articles). Doctor notes may contain diagnoses, treatment
   recommendations, dietary restrictions, test results, and other context that
   can explain patterns in the symptom log. ALWAYS search documents when doing
   any health trend, correlation, or overview analysis.
2. analyze_data — runs a fixed analysis over the user's structured symptom log
   (entries with categories, severities, foods, triggers, periods). Use for
   "what's been worst lately", correlations, period comparisons, trend over time.
3. synthesize_answer — finalize the response. ALWAYS call this last, exactly
   once. It returns the answer that goes to the user.

Routing guidelines:
- COMBINE data and documents for any substantive health question. Document context
  (doctor notes, labs) is essential for interpreting what the symptom numbers mean.
- If the question is purely conversational ("hi", "thanks"), skip straight to
  synthesize_answer with no findings.
- For all other questions, follow this two-phase pattern:

  PHASE 1 — gather data first:
  • "how have I been?" → analyze_data(analysis="overview")
  • food/symptom correlations → analyze_data(analysis="correlations")
  • getting better/worse → analyze_data(analysis="trend")
  • period/cycle questions → analyze_data(analysis="period_overlap")
  • specific document question ("what did my doctor say about X") → search_documents only

  PHASE 2 — enrich with documents:
  After analyze_data returns, call search_documents with a query derived from
  what the data revealed. Examples:
  • Data showed dairy and gluten as top triggers → search_documents(query="dairy gluten intolerance sensitivity doctor recommendation")
  • Data showed worsening trend → search_documents(query="treatment plan symptom management prognosis")
  • Data showed high gut severity → search_documents(query="gut pain diagnosis IBS Crohn treatment")
  • Overview analysis → search_documents(query="dietary recommendations lifestyle changes doctor notes")
  Always use source_types=["doctor_note","lab"] as a first pass to prioritize
  clinical context; broaden to all types if fewer than 2 passages come back.

- After calling tools, ALWAYS call synthesize_answer with the user's original
  question and any findings you gathered.
"""


TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Semantic search over the user's uploaded documents (PDFs of lab "
            "results, doctor notes, journal entries, articles they've saved). "
            "Returns the most relevant passages. Use this alongside analyze_data "
            "to cross-reference what doctors, labs, or journals say about the "
            "patterns found in the symptom log — e.g. if the data shows dairy "
            "as a top trigger, search for what the doctor said about dairy. "
            "Prefer source_types=['doctor_note','lab'] to surface clinical context first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for. Use the user's words or a focused rephrasing."},
                "source_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["pdf", "text", "journal", "article", "doctor_note", "lab"]},
                    "description": "Optional: filter to specific document types.",
                },
                "date_from": {"type": "string", "description": "Optional YYYY-MM-DD filter."},
                "date_to": {"type": "string", "description": "Optional YYYY-MM-DD filter."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_data",
        "description": (
            "Run a structured analysis over the user's symptom log. Available "
            "analyses: 'overview' (snapshot of categories, top foods/triggers, "
            "worst day), 'correlations' (foods/triggers vs gut severity), "
            "'period_overlap' (gut severity during vs outside menstrual periods), "
            "'trend' (is severity worsening / improving / steady)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "enum": list(ANALYSES.keys()),
                    "description": "Which analysis to run.",
                },
                "date_from": {"type": "string", "description": "Optional YYYY-MM-DD window start."},
                "date_to": {"type": "string", "description": "Optional YYYY-MM-DD window end."},
            },
            "required": ["analysis"],
        },
    },
    {
        "name": "synthesize_answer",
        "description": (
            "Produce the final answer to the user. Call this LAST, exactly once. "
            "Pass through the user's original question; the Reflection agent will "
            "synthesize the answer from gathered findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_query": {"type": "string", "description": "The user's original question, unchanged."},
            },
            "required": ["user_query"],
        },
    },
]


def run_orchestrator(
    user_query: str,
    *,
    conversation_history: Optional[list[dict]] = None,
    max_iters: int = 6,
) -> dict:
    """Main entry point. Returns:
        {
          "answer": str,                # Final user-facing reply
          "agent_trace": list[dict],    # What the orchestrator did, for the UI
          "citations": list[dict],      # Documents/passages cited (for the UI)
        }
    """
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_query})

    # Accumulators across the tool-use loop
    gathered_passages: list[dict] = []
    gathered_analysis: dict = {}
    trace: list[dict] = []

    for _ in range(max_iters):
        resp = _ant().messages.create(
            model=MODEL,
            max_tokens=1024,
            system=ORCHESTRATOR_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Add the assistant turn (with tool_use blocks) to the conversation
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            # Orchestrator decided to talk directly — shouldn't normally happen
            # given the system prompt, but handle it gracefully.
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return {
                "answer": text or "(no response)",
                "agent_trace": trace,
                "citations": [],
            }

        # Process tool calls
        tool_results = []
        final_answer = None
        for block in resp.content:
            if block.type != "tool_use":
                continue
            name = block.name
            args = block.input

            if name == "search_documents":
                trace.append({"agent": "retriever", "input": args})
                result = run_retriever(
                    query=args["query"],
                    source_types=args.get("source_types"),
                    date_from=args.get("date_from"),
                    date_to=args.get("date_to"),
                )
                gathered_passages.extend(result.get("passages", []))
                trace[-1]["output_summary"] = result.get("summary")
                # Return a SHORT summary to the orchestrator — full text goes to
                # the Reflection agent later. Keeps context window manageable.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({
                        "n_passages": len(result.get("passages", [])),
                        "summary": result.get("summary"),
                        "previews": [
                            {"filename": p["filename"], "date": p.get("doc_date"), "snippet": p["text"][:160]}
                            for p in result.get("passages", [])
                        ],
                    }),
                })

            elif name == "analyze_data":
                trace.append({"agent": "analyst", "input": args})
                result = run_analyst(
                    analysis=args["analysis"],
                    date_from=args.get("date_from"),
                    date_to=args.get("date_to"),
                )
                gathered_analysis[args["analysis"]] = result
                trace[-1]["output_summary"] = f"ran {args['analysis']}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            elif name == "synthesize_answer":
                trace.append({"agent": "reflection", "input": {"user_query": args["user_query"]}})
                # Dedupe passages by chunk_id while preserving order
                seen = set()
                deduped = []
                for p in gathered_passages:
                    if p["chunk_id"] not in seen:
                        seen.add(p["chunk_id"])
                        deduped.append(p)
                answer = run_reflection(
                    user_query=args["user_query"],
                    retrieved_passages=deduped,
                    analyst_findings=gathered_analysis if gathered_analysis else None,
                    conversation_history=conversation_history,
                )
                final_answer = answer
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "answer produced",
                })
                trace[-1]["output_summary"] = "synthesized final answer"

            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps({"error": f"unknown tool: {name}"}),
                    "is_error": True,
                })

        # Feed all tool results back as a single user turn
        messages.append({"role": "user", "content": tool_results})

        if final_answer is not None:
            citations = [
                {
                    "filename": p["filename"],
                    "title": p.get("title"),
                    "doc_date": p.get("doc_date"),
                    "page": p.get("page"),
                    "snippet": p["text"][:240],
                }
                for p in gathered_passages
            ]
            return {
                "answer": final_answer,
                "agent_trace": trace,
                "citations": citations,
            }

    # Safety fallback if we hit max_iters without synthesizing
    return {
        "answer": (
            "I gathered some information but couldn't finalize an answer. "
            "Try asking again, perhaps with a more specific question."
        ),
        "agent_trace": trace,
        "citations": [],
    }
