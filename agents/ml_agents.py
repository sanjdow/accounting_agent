"""ML agents: Isolation Forest anomaly detection + YoY/linear forecasting.

Both work off the TB history the user uploaded (multiple periods ideally).
Single-period uploads still produce valid output (anomalies only vs
current-period distribution; forecasts return 'insufficient history').
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import ANOMALY_CONTAMINATION


def detect_anomalies(tb: pd.DataFrame, period: str) -> List[Dict[str, Any]]:
    """Flag unusual account movements for the current period.

    Approach: if we have >=3 historical periods per account, compare current-period
    net vs that account's historical mean+stdev (z-score). Separately, run
    IsolationForest over (debit, credit, net) for the current period to catch
    outliers that don't need history.
    """
    flags: List[Dict[str, Any]] = []

    tb = tb.copy()
    tb["net"] = tb["debit"] - tb["credit"]

    # --- Historical z-score (if we have history)
    hist = tb[tb["period"] < period]
    curr = tb[tb["period"] == period].copy()

    if not hist.empty and not curr.empty:
        stats = hist.groupby("account")["net"].agg(["mean", "std", "count"]).reset_index()
        stats = stats[stats["count"] >= 3]
        merged = curr.merge(stats, on="account", how="inner")
        merged["z"] = (merged["net"] - merged["mean"]) / merged["std"].replace(0, np.nan)
        outliers = merged[merged["z"].abs() >= 2.5].sort_values(
            "z", key=lambda s: s.abs(), ascending=False
        )
        for _, r in outliers.head(20).iterrows():
            flags.append({
                "type": "historical_zscore",
                "account": r["account"],
                "account_name": r.get("account_name", ""),
                "current_net": float(r["net"]),
                "historical_mean": float(r["mean"]),
                "z_score": round(float(r["z"]), 2),
                "severity": "high" if abs(r["z"]) > 4 else "medium",
            })

    # --- IsolationForest over current-period distribution
    if len(curr) >= 10:
        X = curr[["debit", "credit", "net"]].values
        clf = IsolationForest(
            contamination=ANOMALY_CONTAMINATION, random_state=42, n_estimators=100,
        )
        labels = clf.fit_predict(X)
        scores = clf.score_samples(X)
        curr["_anom_label"] = labels
        curr["_anom_score"] = scores
        iso_hits = curr[curr["_anom_label"] == -1].sort_values("_anom_score")
        known = {f["account"] for f in flags}
        for _, r in iso_hits.head(10).iterrows():
            if r["account"] in known:
                continue
            flags.append({
                "type": "isolation_forest",
                "account": r["account"],
                "account_name": r.get("account_name", ""),
                "current_net": float(r["net"]),
                "anomaly_score": round(float(r["_anom_score"]), 3),
                "severity": "medium",
            })

    return flags


def forecast_next(tb: pd.DataFrame, period: str) -> Dict[str, Any]:
    """Produce a naive forecast for the next period per entity-total.

    Strategy:
    - linear regression on (period_index, total_net) per entity
    - plus YoY growth rate if we have >=13 periods
    Returns dict: {entity_or_total: {method, forecast, history_points}}.
    """
    tb = tb.copy()
    tb["net"] = tb["debit"] - tb["credit"]

    # Build a monthly ordered period index
    periods = sorted(tb["period"].unique())
    if len(periods) < 3:
        return {"_note": "Need >=3 historical periods for a meaningful forecast.",
                "periods_available": len(periods)}

    p_idx = {p: i for i, p in enumerate(periods)}
    tb["p_idx"] = tb["period"].map(p_idx)
    current_idx = p_idx[period]
    next_idx = current_idx + 1

    out: Dict[str, Any] = {}

    def _lin_forecast(series: pd.DataFrame, key: str) -> None:
        s = series.groupby("p_idx")["net"].sum().sort_index()
        if len(s) < 3:
            return
        x = s.index.values
        y = s.values
        # simple OLS
        slope, intercept = np.polyfit(x, y, 1)
        fc_linear = float(slope * next_idx + intercept)
        entry = {
            "method": "linear",
            "forecast": round(fc_linear, 2),
            "history_points": int(len(s)),
            "slope_per_period": round(float(slope), 2),
        }
        if len(s) >= 13:
            # YoY: same month last year -> this month, apply growth
            yoy_rate = (s.iloc[-1] - s.iloc[-13]) / abs(s.iloc[-13]) if s.iloc[-13] else 0
            entry["yoy_rate"] = round(float(yoy_rate), 4)
            entry["forecast_yoy"] = round(float(s.iloc[-12] * (1 + yoy_rate)), 2)
        out[key] = entry

    if tb["entity"].notna().any():
        for ent, sub in tb.groupby("entity", dropna=False):
            _lin_forecast(sub, str(ent))
    _lin_forecast(tb, "TOTAL")

    return out
