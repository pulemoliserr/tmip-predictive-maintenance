"""Shared state, caching and UI helpers for all pages."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from pdm_core import store
from pdm_core.config import (MODEL_PATH, FLEET_SEED_PATH, TEST_PRED_PATH, TRAIN_STATS_PATH, LINES, STATUS,
                             SERIES, INK, INK2, MUTED, GRID)
from pdm_core.features import rule_checks, LABELS
from pdm_core.model import RiskModel

# ---- Plotly theme (validated categorical order, recessive grid) ------------
pio.templates["pdm"] = go.layout.Template(layout=dict(
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
    title=dict(font=dict(color=INK, size=15)), colorway=SERIES,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    yaxis=dict(gridcolor=GRID, linecolor="#c3c2b7", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(bgcolor="white", font_color=INK),
    legend=dict(orientation="h", y=1.08, x=0)))
pio.templates.default = "plotly_white+pdm"

DEFAULT_POLICY = {"threshold": 0.22, "watch": 0.10, "cost_fp": 100, "cost_fn": 400}


@st.cache_resource(show_spinner="Loading model …")
def get_model() -> RiskModel:
    return RiskModel(MODEL_PATH)


@st.cache_data
def test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PRED_PATH)


@st.cache_data
def train_reference() -> pd.DataFrame:
    return pd.read_csv(TRAIN_STATS_PATH)


def get_policy() -> dict:
    pol = store.get_setting("policy")
    if pol is None:
        m = get_model()
        pol = {**DEFAULT_POLICY, "threshold": float(m.cfg.get("threshold", 0.22)),
               "cost_fp": m.cfg.get("cost_fp", 100), "cost_fn": m.cfg.get("cost_fn", 400)}
        store.set_setting("policy", pol)
    return pol


def get_shift() -> int:
    return int(store.get_setting("shift", 1))


def get_fleet() -> pd.DataFrame:
    f = store.load_fleet()
    if f is None:
        f = pd.read_csv(FLEET_SEED_PATH)
        store.save_fleet(f)
        store.set_setting("shift", 1)
    return f


def _hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


@st.cache_data(show_spinner=False)
def _score(fleet_hash: str, fleet: pd.DataFrame, threshold: float, watch: float) -> pd.DataFrame:
    m = get_model()
    f = fleet.copy()
    f["risk"] = m.risk(f)
    f["rules_breached"] = [sum(c["breached"] for c in rule_checks(r)) for _, r in f.iterrows()]
    f["rules_near"] = [sum(c["near"] and not c["breached"] for c in rule_checks(r)) for _, r in f.iterrows()]
    f["status"] = np.where(f.risk >= threshold, "critical", np.where(f.risk >= watch, "warning", "good"))
    f["criticality"] = f["line"].map(LINES).fillna(1.0)
    f["priority"] = (f["risk"] * f["criticality"] * (1 + 0.25 * f["rules_breached"])).round(3)
    sv = m.shap_values(f)
    f["drivers"] = [", ".join(LABELS.get(c, c).split(" [")[0] for c in r[r > 0].sort_values(ascending=False).head(2).index) or "—"
                    for _, r in sv.iterrows()]
    return f


def scored_fleet() -> pd.DataFrame:
    pol = get_policy()
    f = get_fleet()
    return _score(_hash(f), f, pol["threshold"], pol["watch"])


def status_text(s: str) -> str:
    return f"{STATUS[s]['icon']} {STATUS[s]['label']}"


def user() -> tuple[str, str]:
    return st.session_state.get("reviewer", "Engineer"), st.session_state.get("role", "Maintenance Engineer")


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def fmt_eur(x: float) -> str:
    return f"€{x:,.0f}"
