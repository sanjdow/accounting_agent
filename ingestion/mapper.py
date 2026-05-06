"""Auto-detect mapping from user columns -> canonical schema.

Strategy: for each canonical column, score each user column by
(a) exact match, (b) normalised exact match, (c) substring/alias match.
Return best candidate with a confidence score so the UI can show
high-confidence fields pre-filled and let the user fix the rest.
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional

# Aliases keyed by canonical column name. Lowercase, no punctuation.
ALIASES: Dict[str, List[str]] = {
    # shared
    "period":       ["period", "posting period", "fiscal period", "month",
                     "period id", "per", "yyyymm", "year month", "close period"],
    "entity":       ["entity", "company", "company code", "legal entity",
                     "buk", "bukrs", "subsidiary", "org", "unit", "business unit"],
    "account":      ["account", "gl account", "account number", "acct",
                     "account id", "account code", "hkont", "nominal code"],
    "account_name": ["account name", "account description", "account desc",
                     "gl name", "nominal name", "name"],
    "debit":        ["debit", "debit amount", "dr", "debits",
                     "debit value", "dr amount"],
    "credit":       ["credit", "credit amount", "cr", "credits",
                     "credit value", "cr amount"],
    # GL-only
    "txn_date":     ["date", "posting date", "transaction date", "doc date",
                     "document date", "txn date", "gl date", "budat"],
    "journal_id":   ["journal id", "journal", "document number", "doc no",
                     "belnr", "entry id", "voucher", "je id", "je number"],
    "description":  ["description", "text", "memo", "narration",
                     "narrative", "sgtxt", "line desc", "line description"],
}


@dataclass
class MappingSuggestion:
    canonical: str
    user_column: Optional[str]
    confidence: float  # 0..1


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch == " ").strip()


def _score(user_col_norm: str, alias: str) -> float:
    if user_col_norm == alias:
        return 1.0
    if alias in user_col_norm or user_col_norm in alias:
        return 0.85
    return SequenceMatcher(None, user_col_norm, alias).ratio()


def suggest_mapping(
    user_columns: List[str],
    canonical_columns: List[str],
) -> List[MappingSuggestion]:
    """For each canonical column, pick the best user column.

    Each user column is used at most once (greedy best-score assignment).
    """
    norm_user = {c: _norm(str(c)) for c in user_columns}
    used: set[str] = set()
    results: List[MappingSuggestion] = []

    # Pre-compute a (canonical, user_col, score) matrix
    scored: List[tuple[str, str, float]] = []
    for canon in canonical_columns:
        aliases = ALIASES.get(canon, [canon])
        for uc in user_columns:
            best = max(_score(norm_user[uc], a) for a in aliases)
            scored.append((canon, uc, best))

    # Greedy by highest score, one-to-one
    scored.sort(key=lambda t: t[2], reverse=True)
    assigned_canon: Dict[str, tuple[str, float]] = {}
    for canon, uc, sc in scored:
        if canon in assigned_canon:
            continue
        if uc in used:
            continue
        if sc < 0.55:
            continue  # below threshold -> leave unassigned
        assigned_canon[canon] = (uc, sc)
        used.add(uc)

    for canon in canonical_columns:
        if canon in assigned_canon:
            uc, sc = assigned_canon[canon]
            results.append(MappingSuggestion(canon, uc, round(sc, 2)))
        else:
            results.append(MappingSuggestion(canon, None, 0.0))

    return results
