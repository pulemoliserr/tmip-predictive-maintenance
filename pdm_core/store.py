"""SQLite decision-audit store: engineer decisions, operating policy, fleet state and shift log."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    shift         INTEGER NOT NULL,
    machine_id    TEXT NOT NULL,
    line          TEXT,
    risk          REAL NOT NULL,
    threshold     REAL NOT NULL,
    priority      REAL,
    top_drivers   TEXT,
    rule_flags    TEXT,
    decision      TEXT NOT NULL CHECK (decision IN ('Inspect now','Schedule at next stop','Dismiss – false alarm','Escalate to OEM')),
    finding       TEXT,
    comment       TEXT,
    reviewer      TEXT NOT NULL,
    role          TEXT,
    model_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, ts_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fleet (
    machine_id TEXT PRIMARY KEY, line TEXT, type TEXT, air_temp REAL, process_temp REAL,
    rpm REAL, torque REAL, tool_wear REAL, last_service_shift INTEGER, status_note TEXT);
CREATE TABLE IF NOT EXISTS shifts (shift INTEGER PRIMARY KEY, ts_utc TEXT, alerts INTEGER, mean_risk REAL);
"""

DECISIONS = ["Inspect now", "Schedule at next stop", "Dismiss – false alarm", "Escalate to OEM"]
FINDINGS = ["—", "Confirmed: worn tool replaced", "Confirmed: cooling / heat issue fixed",
            "Confirmed: power / drive issue fixed", "Confirmed: overstrain – load reduced",
            "No defect found", "Not yet inspected"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def conn(path=DB_PATH):
    c = sqlite3.connect(path, timeout=10)
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


# ---- settings -------------------------------------------------------------
def get_setting(key, default=None):
    with conn() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else default


def set_setting(key, value):
    with conn() as c:
        c.execute("INSERT INTO settings(key,value,ts_utc) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts_utc=excluded.ts_utc",
                  (key, json.dumps(value), now()))


# ---- decisions -------------------------------------------------------------
def log_decision(**kw):
    cols = ["ts_utc", "shift", "machine_id", "line", "risk", "threshold", "priority", "top_drivers", "rule_flags",
            "decision", "finding", "comment", "reviewer", "role", "model_version"]
    kw.setdefault("ts_utc", now())
    with conn() as c:
        c.execute(f"INSERT INTO decisions({','.join(cols)}) VALUES({','.join('?' * len(cols))})", [kw.get(k) for k in cols])


def decisions_df() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", c)


def reviewed_this_shift(shift: int) -> set:
    with conn() as c:
        rows = c.execute("SELECT machine_id FROM decisions WHERE shift=?", (shift,)).fetchall()
    return {r[0] for r in rows}


# ---- fleet ---------------------------------------------------------------
def save_fleet(df: pd.DataFrame):
    with conn() as c:
        c.execute("DELETE FROM fleet")
        df.to_sql("fleet", c, if_exists="append", index=False)


def load_fleet() -> pd.DataFrame | None:
    with conn() as c:
        df = pd.read_sql_query("SELECT * FROM fleet", c)
    return df if len(df) else None


def log_shift(shift, alerts, mean_risk):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO shifts VALUES(?,?,?,?)", (shift, now(), int(alerts), float(mean_risk)))


def shifts_df():
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM shifts ORDER BY shift", c)


def reset_all():
    with conn() as c:
        c.executescript("DELETE FROM decisions; DELETE FROM fleet; DELETE FROM shifts; DELETE FROM settings;")
