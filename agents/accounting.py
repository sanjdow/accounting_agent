"""Deterministic accounting engine.

Builds a close snapshot from the user-uploaded TB and GL:
- period totals and account balances
- cross-check TB vs GL (sum of GL debits/credits should reconcile to TB)
- top-N accounts by net movement
- entity-level rollup
- simple accrual candidates (GL txns posted in period X but dated in period Y)
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd


def build_snapshot(tb: pd.DataFrame, gl: pd.DataFrame, period: str) -> Dict[str, Any]:
    tb_p = tb[tb["period"] == period].copy()
    gl_p = gl[gl["period"] == period].copy()

    if tb_p.empty:
        raise ValueError(f"No TB rows for period {period}")

    # --- TB summary
    tb_p["net"] = tb_p["debit"] - tb_p["credit"]
    tb_totals = {
        "debit": float(tb_p["debit"].sum()),
        "credit": float(tb_p["credit"].sum()),
        "diff": float(round(tb_p["debit"].sum() - tb_p["credit"].sum(), 2)),
    }

    # Account balances (TB)
    acct_bal = (
        tb_p.groupby(["account", "account_name"], dropna=False)[["debit", "credit", "net"]]
        .sum()
        .reset_index()
        .sort_values("net", key=lambda s: s.abs(), ascending=False)
    )
    top_accounts = acct_bal.head(10).to_dict(orient="records")

    # Entity rollup (TB)
    entity_rollup = []
    if tb_p["entity"].notna().any():
        er = tb_p.groupby("entity", dropna=False)[["debit", "credit", "net"]].sum().reset_index()
        entity_rollup = er.to_dict(orient="records")

    # --- GL cross-check
    gl_totals = {"debit": 0.0, "credit": 0.0, "rows": 0}
    reconciliation = {"matched": True, "tb_minus_gl_debit": 0.0, "tb_minus_gl_credit": 0.0}
    accrual_candidates: list[dict] = []

    if not gl_p.empty:
        gl_totals = {
            "debit": float(gl_p["debit"].sum()),
            "credit": float(gl_p["credit"].sum()),
            "rows": int(len(gl_p)),
        }
        diff_d = round(tb_totals["debit"] - gl_totals["debit"], 2)
        diff_c = round(tb_totals["credit"] - gl_totals["credit"], 2)
        reconciliation = {
            "matched": abs(diff_d) < 0.01 and abs(diff_c) < 0.01,
            "tb_minus_gl_debit": diff_d,
            "tb_minus_gl_credit": diff_c,
        }

        # Accrual candidates: txn_date falls in period P-1 but posted to period P
        gl_p["txn_date"] = pd.to_datetime(gl_p["txn_date"], errors="coerce")
        gl_p["txn_period"] = gl_p["txn_date"].dt.strftime("%Y-%m")
        cross = gl_p[gl_p["txn_period"].notna() & (gl_p["txn_period"] != gl_p["period"])]
        if not cross.empty:
            a = (
                cross.groupby(["journal_id", "txn_period", "period"], dropna=False)
                .agg(debit=("debit", "sum"), credit=("credit", "sum"),
                     rows=("account", "count"))
                .reset_index()
                .sort_values("rows", ascending=False)
                .head(15)
            )
            accrual_candidates = a.to_dict(orient="records")

    return {
        "period": period,
        "tb_totals": tb_totals,
        "gl_totals": gl_totals,
        "reconciliation": reconciliation,
        "top_accounts": top_accounts,
        "entity_rollup": entity_rollup,
        "accrual_candidates": accrual_candidates,
        "account_balances": acct_bal.to_dict(orient="records"),
    }
