"""LangGraph orchestrator — state machine over the agents.

Stages (matches the sequence diagram):
  accounting → ml → rag → narrative → guardrails
                                          │
                                 pass ────┤
                                 retry ───┘ (<= MAX_NARRATIVE_RETRIES back to narrative)
                                 escalate ─┘ (end with status=escalated)
  (no violations) → await_approval → finalise
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

import pandas as pd

from agents.accounting import build_snapshot
from agents.ml_agents import detect_anomalies, forecast_next
from agents.rag_agent import retrieve
from agents.narrative_agent import generate_narrative
from agents.guardrails import check as guardrails_check

from storage.db import (
    init_db, start_run, complete_run, audit,
    store_tb, store_gl, save_snapshot, seal_snapshot,
    load_tb, load_gl,
)
from config import MAX_NARRATIVE_RETRIES, RAG_CONFIDENCE_THRESHOLD


class State(TypedDict, total=False):
    run_id: str
    period: str
    snapshot: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    forecasts: Dict[str, Any]
    policy_hits: List[Dict[str, Any]]
    rag_confidence: float
    rag_query: str
    narrative: str
    guardrail: Dict[str, Any]
    retries: int
    status: str
    error: str


# ---------- Nodes ----------
def node_accounting(state: State) -> State:
    run_id, period = state["run_id"], state["period"]
    audit(run_id, "accounting.start", {"period": period})
    tb = load_tb(run_id)
    gl = load_gl(run_id)
    snap = build_snapshot(tb, gl, period)
    save_snapshot(run_id, snap, sealed=False)
    audit(run_id, "accounting.done", {
        "tb_diff": snap["tb_totals"]["diff"],
        "reconciled": snap["reconciliation"]["matched"],
    })
    return {"snapshot": snap}


def node_ml(state: State) -> State:
    run_id = state["run_id"]
    tb = load_tb(run_id)
    anomalies = detect_anomalies(tb, state["period"])
    forecasts = forecast_next(tb, state["period"])
    audit(run_id, "ml.done", {
        "anomaly_count": len(anomalies),
        "forecast_keys": list(forecasts.keys()),
    })
    return {"anomalies": anomalies, "forecasts": forecasts}


def node_rag(state: State) -> State:
    run_id = state["run_id"]
    res = retrieve(state["snapshot"], state.get("anomalies", []))
    audit(run_id, "rag.done", {
        "query": res["query"], "avg_confidence": res["avg_confidence"],
        "hits": len(res["hits"]),
    })
    return {
        "policy_hits": res["hits"],
        "rag_confidence": res["avg_confidence"],
        "rag_query": res["query"],
    }


def node_narrative(state: State) -> State:
    run_id = state["run_id"]
    feedback = state.get("guardrail", {}).get("feedback", "")
    narrative = generate_narrative(
        state["snapshot"],
        state.get("anomalies", []),
        state.get("forecasts", {}),
        state.get("policy_hits", []),
        retry_feedback=feedback,
    )
    audit(run_id, "narrative.done", {
        "chars": len(narrative), "retry": state.get("retries", 0),
    })
    return {"narrative": narrative}


def node_guardrails(state: State) -> State:
    run_id = state["run_id"]
    gr = guardrails_check(
        state.get("narrative", ""),
        state["snapshot"],
        state.get("anomalies", []),
        state.get("forecasts", {}),
        state.get("policy_hits", []),
    )
    audit(run_id, "guardrails.done", {
        "status": gr["status"], "violations": len(gr["violations"]),
    })
    return {"guardrail": gr, "retries": state.get("retries", 0) + (1 if gr["status"] == "retry" else 0)}


def node_finalise(state: State) -> State:
    run_id = state["run_id"]
    payload = {
        "period": state["period"],
        "snapshot": state["snapshot"],
        "anomalies": state.get("anomalies", []),
        "forecasts": state.get("forecasts", {}),
        "policy_hits": state.get("policy_hits", []),
        "rag_query": state.get("rag_query", ""),
        "rag_confidence": state.get("rag_confidence", 0.0),
        "narrative": state.get("narrative", ""),
        "guardrail": state.get("guardrail", {}),
    }
    save_snapshot(run_id, payload, sealed=False)
    audit(run_id, "finalise.pending_approval", {})
    return {"status": "awaiting_approval"}


# ---------- Routing ----------
def route_after_rag(state: State) -> str:
    if state.get("rag_confidence", 0.0) < RAG_CONFIDENCE_THRESHOLD:
        audit(state["run_id"], "rag.low_confidence",
              {"confidence": state.get("rag_confidence")})
        # Still proceed — narrative will note weak policy grounding.
    return "narrative"


def route_after_guardrails(state: State) -> str:
    gr = state.get("guardrail", {})
    status = gr.get("status", "pass")
    if status == "pass":
        return "finalise"
    if status == "retry" and state.get("retries", 0) < MAX_NARRATIVE_RETRIES:
        return "narrative"
    # escalate or retries exhausted
    return "escalated"


def node_escalated(state: State) -> State:
    audit(state["run_id"], "guardrails.escalated", {
        "retries": state.get("retries", 0),
        "violations": state.get("guardrail", {}).get("violations", []),
    })
    return {"status": "escalated"}


# ---------- Graph ----------
def build_graph():
    g = StateGraph(State)
    g.add_node("accounting", node_accounting)
    g.add_node("ml", node_ml)
    g.add_node("rag", node_rag)
    g.add_node("narrative", node_narrative)
    g.add_node("guardrails", node_guardrails)
    g.add_node("finalise", node_finalise)
    g.add_node("escalated", node_escalated)

    g.set_entry_point("accounting")
    g.add_edge("accounting", "ml")
    g.add_edge("ml", "rag")
    g.add_conditional_edges("rag", route_after_rag, {"narrative": "narrative"})
    g.add_edge("narrative", "guardrails")
    g.add_conditional_edges(
        "guardrails", route_after_guardrails,
        {"finalise": "finalise", "narrative": "narrative", "escalated": "escalated"},
    )
    g.add_edge("finalise", END)
    g.add_edge("escalated", END)
    return g.compile()


# ---------- Public entry points ----------
def stage_datasets(run_id: str, period: str,
                   tb_df: pd.DataFrame, gl_df: pd.DataFrame) -> None:
    """Persist the uploaded TB/GL into SQLite so the graph can read them."""
    init_db()
    start_run(run_id, period)
    audit(run_id, "ingestion.start",
          {"tb_rows": len(tb_df), "gl_rows": len(gl_df), "period": period})
    store_tb(run_id, tb_df)
    store_gl(run_id, gl_df)
    audit(run_id, "ingestion.done", {})


def run_pipeline(run_id: str, period: str) -> State:
    graph = build_graph()
    initial: State = {
        "run_id": run_id, "period": period,
        "retries": 0, "status": "running",
    }
    final = graph.invoke(initial)
    return final


def approve_and_seal(run_id: str, approver_id: str) -> None:
    audit(run_id, "human.approved", {"approver_id": approver_id})
    seal_snapshot(run_id)
    duration = complete_run(run_id, "completed")
    audit(run_id, "run.completed", {"duration_ms": duration})


def new_run_id() -> str:
    return str(uuid.uuid4())[:8]
