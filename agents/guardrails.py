"""Guardrails — rule-based, no LLM.

Four checks:
  1. Schema:        narrative is non-empty text, reasonable length.    (CRITICAL)
  2. Grounding:     every number in narrative appears in snapshot.     (WARNING)
  3. Hallucination: no forbidden phrases ("I think", "probably", etc.) (WARNING)
  4. Policy cite:   if policy hits provided, narrative mentions >=1.   (WARNING)

Output:
  status in {"pass", "retry", "escalate"}
  violations: list of dicts
  feedback: string suitable for retry prompt
"""
from __future__ import annotations
import re
from typing import Any, Dict, List

FORBIDDEN_PHRASES = [
    "i think", "i believe", "probably", "possibly", "maybe",
    "it seems", "might be", "could be around",
]


def _extract_numbers(text: str) -> List[float]:
    # Match integers/decimals with optional separators (1,234.56 / 1234.56)
    rx = re.compile(r"-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")
    out: List[float] = []
    for m in rx.findall(text):
        try:
            out.append(float(m.replace(",", "").replace(" ", "")))
        except ValueError:
            continue
    return out


def _snapshot_numbers(snapshot: Dict[str, Any],
                      anomalies: List[Dict[str, Any]],
                      forecasts: Dict[str, Any]) -> set:
    nums = set()

    def add(v):
        try:
            nums.add(round(float(v), 2))
        except (TypeError, ValueError):
            return

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, (int, float)):
            add(obj)

    walk(snapshot)
    walk(anomalies)
    walk(forecasts)

    # Also add small integers and common tolerances so "4 paragraphs" etc don't trip
    for n in range(0, 30):
        nums.add(float(n))
    return nums


def check(
    narrative: str,
    snapshot: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    forecasts: Dict[str, Any],
    policy_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []

    # 1. Schema
    text = (narrative or "").strip()
    if len(text) < 100:
        violations.append({
            "check": "schema", "severity": "CRITICAL",
            "detail": f"Narrative too short ({len(text)} chars).",
        })
    elif len(text) > 6000:
        violations.append({
            "check": "schema", "severity": "CRITICAL",
            "detail": f"Narrative too long ({len(text)} chars).",
        })

    # 2. Grounding — numbers in narrative must appear in snapshot
    valid_nums = _snapshot_numbers(snapshot, anomalies, forecasts)
    narr_nums = _extract_numbers(text)
    ungrounded: List[float] = []
    for n in narr_nums:
        n_r = round(n, 2)
        # tolerate near-match within 0.5% or 1.0 absolute
        if any(abs(n_r - v) <= max(1.0, abs(v) * 0.005) for v in valid_nums):
            continue
        ungrounded.append(n_r)
    if ungrounded:
        violations.append({
            "check": "grounding", "severity": "WARNING",
            "detail": f"{len(ungrounded)} numbers not traceable to snapshot: "
                      f"{ungrounded[:5]}",
        })

    # 3. Hallucination language
    lower = text.lower()
    hits = [p for p in FORBIDDEN_PHRASES if p in lower]
    if hits:
        violations.append({
            "check": "hallucination_language", "severity": "WARNING",
            "detail": f"Hedging phrases found: {hits}",
        })

    # 4. Policy citation
    if policy_hits:
        sources = {h["source"] for h in policy_hits}
        if not any(src in text for src in sources):
            violations.append({
                "check": "policy_citation", "severity": "WARNING",
                "detail": f"No policy source cited. Expected one of: {sorted(sources)}",
            })

    # Decide status
    has_critical = any(v["severity"] == "CRITICAL" for v in violations)
    has_warning = any(v["severity"] == "WARNING" for v in violations)

    if has_critical:
        status = "escalate"
    elif has_warning:
        status = "retry"
    else:
        status = "pass"

    feedback = "; ".join(f"[{v['severity']}] {v['check']}: {v['detail']}"
                        for v in violations)
    return {"status": status, "violations": violations, "feedback": feedback}
