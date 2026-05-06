"""Narrative agent — provider-agnostic via LiteLLM.

The prompt is strictly grounded: the LLM is told to explain, never compute.
All numbers in the narrative must come from the snapshot we pass in.
"""
from __future__ import annotations
import json
from typing import Any, Dict, List

from litellm import completion

from config import LLM_MODEL, LLM_API_BASE


SYSTEM = """You are an accounting close commentator.
Rules:
- Use ONLY the figures provided in the snapshot JSON. Do NOT invent numbers.
- Cite policy sources by their filename when you refer to them.
- Keep it concise: 4-6 short paragraphs.
- Structure: (1) period summary, (2) reconciliation status,
  (3) notable account movements / anomalies, (4) accrual considerations,
  (5) forecast outlook, (6) policy references used.
- If a section has no material content, say so in one sentence.
- Do not give advice; describe the facts and flag items needing review.
"""


def _compact_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Strip bulky fields to keep the prompt small."""
    s = {k: v for k, v in snapshot.items() if k != "account_balances"}
    s["top_accounts"] = s.get("top_accounts", [])[:8]
    s["accrual_candidates"] = s.get("accrual_candidates", [])[:8]
    return s


def generate_narrative(
    snapshot: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    forecasts: Dict[str, Any],
    policy_hits: List[Dict[str, Any]],
    retry_feedback: str = "",
) -> str:
    context = {
        "snapshot": _compact_snapshot(snapshot),
        "anomalies": anomalies[:10],
        "forecasts": forecasts,
        "policy_excerpts": [
            {"source": h["source"], "text": h["text"][:500]}
            for h in policy_hits[:4]
        ],
    }

    user_msg = (
        "Write the month-end close narrative for the data below. "
        "Return plain text only.\n\n"
        f"```json\n{json.dumps(context, default=str, indent=2)}\n```"
    )
    if retry_feedback:
        user_msg += f"\n\nGuardrail feedback from previous draft:\n{retry_feedback}"

    kwargs: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.2,
        "max_tokens": 1000,
    }
    if LLM_API_BASE:
        kwargs["api_base"] = LLM_API_BASE

    resp = completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()
