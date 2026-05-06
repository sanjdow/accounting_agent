"""Apply user-confirmed mapping, coerce types, and run integrity checks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    df: pd.DataFrame
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _coerce_numeric(s: pd.Series) -> pd.Series:
    # Handle strings like "1.234,56" (DE) and "1,234.56" (EN)
    if s.dtype.kind in ("i", "f"):
        return s.astype(float).fillna(0.0)
    cleaned = (
        s.astype(str)
         .str.replace(r"[^\d,\.\-]", "", regex=True)
         .str.replace(",", "", regex=False)   # naive EN-format assumption
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _coerce_period(s: pd.Series) -> pd.Series:
    """Normalise period to YYYY-MM strings."""
    out = []
    for v in s.astype(str):
        v = v.strip()
        parsed = pd.to_datetime(v, errors="coerce")
        if pd.notna(parsed):
            out.append(parsed.strftime("%Y-%m"))
            continue
        # try YYYYMM
        digits = "".join(ch for ch in v if ch.isdigit())
        if len(digits) == 6:
            out.append(f"{digits[:4]}-{digits[4:]}")
        else:
            out.append(v)
    return pd.Series(out, index=s.index)


def apply_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],   # canonical -> user_col
    canonical_cols: List[str],
    required_cols: List[str],
) -> pd.DataFrame:
    """Rename user columns to canonical and keep only canonicals.
    Missing optional canonicals are added as empty columns."""
    # invert: user_col -> canonical
    rename = {v: k for k, v in mapping.items() if v is not None and v in df.columns}
    out = df.rename(columns=rename).copy()
    for canon in canonical_cols:
        if canon not in out.columns:
            out[canon] = np.nan
    # enforce required are present after rename
    missing = [c for c in required_cols if c not in rename.values()]
    if missing:
        raise ValueError(f"Required columns not mapped: {missing}")
    return out[canonical_cols]


def validate_tb(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(df=df.copy())
    try:
        res.df["debit"] = _coerce_numeric(res.df["debit"])
        res.df["credit"] = _coerce_numeric(res.df["credit"])
        res.df["period"] = _coerce_period(res.df["period"])
        res.df["account"] = res.df["account"].astype(str).str.strip()
    except KeyError as e:
        res.errors.append(f"Missing required column after mapping: {e}")
        return res

    # Balance check per period
    by_period = res.df.groupby("period")[["debit", "credit"]].sum()
    by_period["diff"] = (by_period["debit"] - by_period["credit"]).round(2)
    unbalanced = by_period[by_period["diff"].abs() > 0.01]
    if not unbalanced.empty:
        for p, row in unbalanced.iterrows():
            res.warnings.append(
                f"TB not balanced for period {p}: debit - credit = {row['diff']:,.2f}"
            )

    # Empty rows
    empty_accts = res.df["account"].isin(["", "nan", "None"]).sum()
    if empty_accts:
        res.warnings.append(f"{empty_accts} rows have empty account codes.")

    res.stats = {
        "rows": len(res.df),
        "accounts": res.df["account"].nunique(),
        "periods": res.df["period"].nunique(),
        "total_debit": float(res.df["debit"].sum()),
        "total_credit": float(res.df["credit"].sum()),
    }
    return res


def validate_gl(df: pd.DataFrame) -> ValidationResult:
    res = ValidationResult(df=df.copy())
    try:
        res.df["debit"] = _coerce_numeric(res.df["debit"])
        res.df["credit"] = _coerce_numeric(res.df["credit"])
        res.df["period"] = _coerce_period(res.df["period"])
        res.df["account"] = res.df["account"].astype(str).str.strip()
        res.df["txn_date"] = pd.to_datetime(res.df["txn_date"], errors="coerce")
    except KeyError as e:
        res.errors.append(f"Missing required column after mapping: {e}")
        return res

    bad_dates = res.df["txn_date"].isna().sum()
    if bad_dates:
        res.warnings.append(f"{bad_dates} rows have unparseable txn_date.")

    # Each journal should balance (sum debits == sum credits)
    if "journal_id" in res.df.columns and res.df["journal_id"].notna().any():
        by_je = res.df.groupby("journal_id")[["debit", "credit"]].sum()
        by_je["diff"] = (by_je["debit"] - by_je["credit"]).round(2)
        unbalanced = by_je[by_je["diff"].abs() > 0.01]
        if not unbalanced.empty:
            res.warnings.append(
                f"{len(unbalanced)} journals are not balanced "
                f"(first: {unbalanced.index[0]})."
            )

    res.stats = {
        "rows": len(res.df),
        "accounts": res.df["account"].nunique(),
        "periods": res.df["period"].nunique(),
        "journals": int(res.df.get("journal_id", pd.Series([])).nunique()),
        "total_debit": float(res.df["debit"].sum()),
        "total_credit": float(res.df["credit"].sum()),
    }
    return res
