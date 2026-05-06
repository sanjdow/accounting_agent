"""SQLite-backed audit log, TB/GL dataset storage, and sealed snapshots."""
from __future__ import annotations
import sqlite3
import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    period         TEXT NOT NULL,
    started_at     REAL NOT NULL,
    completed_at   REAL,
    status         TEXT NOT NULL,
    duration_ms    INTEGER
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    ts         REAL NOT NULL,
    event      TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_log(run_id);

CREATE TABLE IF NOT EXISTS tb_rows (
    run_id       TEXT NOT NULL,
    period       TEXT NOT NULL,
    entity       TEXT,
    account      TEXT NOT NULL,
    account_name TEXT,
    debit        REAL NOT NULL,
    credit       REAL NOT NULL,
    PRIMARY KEY (run_id, period, entity, account)
);

CREATE TABLE IF NOT EXISTS gl_rows (
    run_id       TEXT NOT NULL,
    row_id       INTEGER NOT NULL,
    period       TEXT NOT NULL,
    entity       TEXT,
    txn_date     TEXT,
    journal_id   TEXT,
    account      TEXT NOT NULL,
    account_name TEXT,
    description  TEXT,
    debit        REAL NOT NULL,
    credit       REAL NOT NULL,
    PRIMARY KEY (run_id, row_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    run_id   TEXT PRIMARY KEY,
    sealed   INTEGER NOT NULL DEFAULT 0,
    payload  TEXT NOT NULL   -- JSON: snapshot summary, anomalies, forecasts, narrative
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


# ---------- Runs ----------
def start_run(run_id: str, period: str) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO runs(run_id, period, started_at, status) VALUES (?,?,?,?)",
            (run_id, period, time.time(), "running"),
        )


def complete_run(run_id: str, status: str) -> int:
    now = time.time()
    with connect() as con:
        row = con.execute(
            "SELECT started_at FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if not row:
            return 0
        duration_ms = int((now - row[0]) * 1000)
        con.execute(
            "UPDATE runs SET completed_at=?, status=?, duration_ms=? WHERE run_id=?",
            (now, status, duration_ms, run_id),
        )
        return duration_ms


# ---------- Audit ----------
def audit(run_id: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO audit_log(run_id, ts, event, payload) VALUES (?,?,?,?)",
            (run_id, time.time(), event, json.dumps(payload or {}, default=str)),
        )


def get_audit(run_id: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT ts, event, payload FROM audit_log WHERE run_id=? ORDER BY id",
            con, params=(run_id,),
        )


# ---------- TB / GL ingestion ----------
def store_tb(run_id: str, df: pd.DataFrame) -> None:
    d = df.copy()
    d["run_id"] = run_id
    cols = ["run_id", "period", "entity", "account", "account_name", "debit", "credit"]
    with connect() as con:
        d[cols].to_sql("tb_rows", con, if_exists="append", index=False)


def store_gl(run_id: str, df: pd.DataFrame) -> None:
    d = df.copy().reset_index(drop=True)
    d["run_id"] = run_id
    d["row_id"] = d.index
    d["txn_date"] = d["txn_date"].astype(str)
    cols = ["run_id", "row_id", "period", "entity", "txn_date", "journal_id",
            "account", "account_name", "description", "debit", "credit"]
    with connect() as con:
        d[cols].to_sql("gl_rows", con, if_exists="append", index=False)


def load_tb(run_id: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM tb_rows WHERE run_id=?", con, params=(run_id,)
        )


def load_gl(run_id: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM gl_rows WHERE run_id=?", con, params=(run_id,)
        )


# ---------- Snapshots ----------
def save_snapshot(run_id: str, payload: Dict[str, Any], sealed: bool = False) -> None:
    with connect() as con:
        con.execute(
            "INSERT OR REPLACE INTO snapshots(run_id, sealed, payload) VALUES (?,?,?)",
            (run_id, 1 if sealed else 0, json.dumps(payload, default=str)),
        )


def load_snapshot(run_id: str) -> Optional[Dict[str, Any]]:
    with connect() as con:
        row = con.execute(
            "SELECT payload, sealed FROM snapshots WHERE run_id=?", (run_id,)
        ).fetchone()
    if not row:
        return None
    snap = json.loads(row[0])
    snap["_sealed"] = bool(row[1])
    return snap


def seal_snapshot(run_id: str) -> None:
    with connect() as con:
        con.execute("UPDATE snapshots SET sealed=1 WHERE run_id=?", (run_id,))


def list_runs() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT run_id, period, started_at, completed_at, status, duration_ms "
            "FROM runs ORDER BY started_at DESC",
            con,
        )
