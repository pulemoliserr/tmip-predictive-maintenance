"""Plant fleet: initial state from unseen test data and a shift-by-shift digital simulation.

The AI4I data has no time axis, so the prototype simulates operation: each shift tool wear
accumulates (by quality type), the ambient temperature random-walks and operating points drift.
Maintenance actions logged in the review queue feed back into the machine state (closed loop).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import LINES, SEED

WEAR_PER_SHIFT = {"L": 10, "M": 15, "H": 25}   # 2/3/5 min per process × ~5 processes per shift
FLEET_COLS = ["machine_id", "line", "type", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
              "last_service_shift", "status_note"]


def initial_fleet(test_rows: pd.DataFrame, n=150, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = test_rows.sample(n=min(n, len(test_rows)), random_state=seed).reset_index(drop=True)
    lines = list(LINES)
    fleet = pd.DataFrame({
        "machine_id": [f"{t}-{100 + i:03d}" for i, t in enumerate(pick["type"])],
        "line": [lines[i % len(lines)] for i in rng.permutation(len(pick))],
        "type": pick["type"].values,
        "air_temp": pick["air_temp"].values, "process_temp": pick["process_temp"].values,
        "rpm": pick["rpm"].values.astype(float), "torque": pick["torque"].values,
        "tool_wear": pick["tool_wear"].values.astype(float),
        "last_service_shift": 0, "status_note": ""})
    return fleet[FLEET_COLS]


def advance_shift(fleet: pd.DataFrame, shift: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else SEED + shift)
    f = fleet.copy()
    n = len(f)
    ambient_shift = rng.normal(0, 0.5)                      # plant-wide ambient drift
    offset = f["process_temp"] - f["air_temp"]
    f["air_temp"] = np.clip(f["air_temp"] + ambient_shift + rng.normal(0, 0.25, n), 295.0, 305.0)
    f["process_temp"] = f["air_temp"] + offset + rng.normal(0, 0.15, n)
    f["tool_wear"] = f["tool_wear"] + f["type"].map(WEAR_PER_SHIFT).values * rng.uniform(0.7, 1.3, n)
    f["rpm"] = np.clip(f["rpm"] + rng.normal(0, 20, n), 1150, 2900)
    f["torque"] = np.clip(f["torque"] + rng.normal(0, 1.5, n), 3, 77)
    # occasional job change: new operating point
    job = rng.random(n) < 0.06
    tq = rng.normal(40, 10, job.sum()).clip(4, 76)                  # AI4I: torque ~ N(40, 10)
    power = rng.normal(6400, 900, job.sum())                          # roughly constant mechanical power
    f.loc[job, "torque"] = tq
    f.loc[job, "rpm"] = (power / (tq * 2 * np.pi / 60)).clip(1170, 2880)
    f["status_note"] = ""
    return f


def apply_maintenance(fleet: pd.DataFrame, machine_id: str, finding: str, shift: int) -> pd.DataFrame:
    f = fleet.copy()
    m = f["machine_id"] == machine_id
    finding = finding.lower()
    if "tool" in finding:
        f.loc[m, "tool_wear"] = 0.0
    if "heat" in finding:
        f.loc[m, "process_temp"] = f.loc[m, "air_temp"] + 10.5
    if "power" in finding:
        f.loc[m, ["rpm", "torque"]] = [1500.0, 40.0]
    if "overstrain" in finding:
        f.loc[m, "torque"] = f.loc[m, "torque"] * 0.8
        f.loc[m, "tool_wear"] = 0.0
    if finding.startswith("confirmed"):
        f.loc[m, "last_service_shift"] = shift
        f.loc[m, "status_note"] = f"serviced in shift {shift}"
    return f


def psi(expected: np.ndarray, actual: np.ndarray, bins=10) -> float:
    """Population Stability Index between training reference and current fleet (drift monitor)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
