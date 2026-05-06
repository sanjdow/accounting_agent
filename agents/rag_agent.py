"""RAG agent — turns snapshot signals into a targeted policy query."""
from __future__ import annotations
from typing import Any, Dict, List

from storage.vector_store import query_policies
from config import RAG_TOP_K


def build_query(snapshot: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> str:
    """Assemble a contextual query string from the most material signals."""
    parts: List[str] = []

    recon = snapshot.get("reconciliation", {})
    if not recon.get("matched", True):
        parts.append("TB to GL reconciliation differences")

    if snapshot.get("accrual_candidates"):
        parts.append("accrual recognition timing differences period cutoff")

    if anomalies:
        kinds = {a.get("type") for a in anomalies}
        if "historical_zscore" in kinds:
            parts.append("material variance prior period account movement explanation")
        if "isolation_forest" in kinds:
            parts.append("unusual account balance outlier")

    top = snapshot.get("top_accounts") or []
    if top:
        # surface a couple of account names to bias retrieval
        names = [t.get("account_name") for t in top[:3] if t.get("account_name")]
        if names:
            parts.append(" ".join(str(n) for n in names))

    if not parts:
        parts.append("month-end close standard controls balance sheet review")

    return " | ".join(parts)


def retrieve(snapshot: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
    q = build_query(snapshot, anomalies)
    result = query_policies(q, k=RAG_TOP_K)
    result["query"] = q
    return result
