"""Domain feature engineering and transparent physics rules (AI4I 2020)."""
from __future__ import annotations
import numpy as np
import pandas as pd

RENAME = {"UDI": "udi", "Product ID": "product_id", "Type": "type",
          "Air temperature [K]": "air_temp", "Process temperature [K]": "process_temp",
          "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque", "Tool wear [min]": "tool_wear",
          "Machine failure": "failure", "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf"}
SENSORS = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
ENGINEERED = ["temp_diff", "power", "strain"]
CONT = SENSORS + ENGINEERED
ALL_FEATURES = CONT + ["type_L", "type_M", "type_H"]

LABELS = {"air_temp": "Air temperature [K]", "process_temp": "Process temperature [K]",
          "rpm": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]",
          "temp_diff": "Process–air ΔT [K]", "power": "Mechanical power [W]", "strain": "Strain wear×torque [min·Nm]",
          "type_L": "Type L", "type_M": "Type M", "type_H": "Type H"}

MODE_NAMES = {"twf": "Tool wear failure", "hdf": "Heat dissipation failure", "pwf": "Power failure",
              "osf": "Overstrain failure", "rnf": "Random failure"}

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=RENAME)
    return df


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature construction (no fitted statistics -> no leakage)."""
    X = frame[SENSORS].astype(float).copy()
    X["temp_diff"] = frame["process_temp"] - frame["air_temp"]
    X["power"] = frame["torque"] * frame["rpm"] * 2 * np.pi / 60
    X["strain"] = frame["tool_wear"] * frame["torque"]
    for t in ["L", "M", "H"]:
        X[f"type_{t}"] = (frame["type"] == t).astype(int)
    X["type"] = frame["type"].values          # raw category, one-hot encoded inside the model pipeline
    return X[ALL_FEATURES + ["type"]]


def rule_checks(row: pd.Series) -> list[dict]:
    """Transparent engineering rules (Matzka 2020) with the margin to each limit.

    Shown next to the ML score so engineers can see *why* a machine is close to a failure regime.
    """
    t = row.get("type", "L")
    dT = row["process_temp"] - row["air_temp"]
    power = row["torque"] * row["rpm"] * 2 * np.pi / 60
    strain = row["tool_wear"] * row["torque"]
    lim = OSF_LIMIT.get(t, 11000)
    checks = [
        dict(rule="Heat dissipation", condition="ΔT < 8.6 K and speed < 1 380 rpm",
             value=f"ΔT {dT:.1f} K · {row['rpm']:.0f} rpm", breached=bool(dT < 8.6 and row["rpm"] < 1380),
             near=bool(dT < 9.3 and row["rpm"] < 1450)),
        dict(rule="Power window", condition="3 500 W ≤ power ≤ 9 000 W",
             value=f"{power:,.0f} W", breached=bool(power < 3500 or power > 9000),
             near=bool(power < 4000 or power > 8500)),
        dict(rule="Overstrain", condition=f"wear × torque < {lim:,} min·Nm (type {t})",
             value=f"{strain:,.0f} min·Nm", breached=bool(strain > lim), near=bool(strain > 0.85 * lim)),
        dict(rule="Tool life", condition="tool wear < 200 min (TWF window 200–240)",
             value=f"{row['tool_wear']:.0f} min", breached=bool(row["tool_wear"] >= 200),
             near=bool(row["tool_wear"] >= 180)),
    ]
    return checks
