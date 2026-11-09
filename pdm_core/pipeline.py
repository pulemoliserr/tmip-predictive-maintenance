"""Preprocessing and pipeline factory — mirrors the CRISP-DM notebook (v2: no imputation, no Boruta)."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier


class ThreeSigmaClip(BaseEstimator, TransformerMixin):
    """3σ rule fitted on training data. mode: 'clip' (substitute by the 3σ boundary) | 'none'."""
    def __init__(self, mode="clip"):
        self.mode = mode

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        m, s = X.mean(axis=0), X.std(axis=0, ddof=1)
        self.lo_, self.hi_ = m - 3 * s, m + 3 * s
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_) if self.mode == "clip" else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


NUMERIC = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "temp_diff", "power", "strain"]

DEFAULT_CONFIG = {
    "model_name": "Random Forest",
    "model_params": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None},
    "sampler": "none", "outlier_mode": "clip", "numeric_features": NUMERIC,
    "features": NUMERIC + ["type_L", "type_M"],
    "calibration": "none", "threshold": 0.35, "cost_fp": 100, "cost_fn": 400, "seed": 42,
}

SAMPLERS = {
    "none": lambda seed: "passthrough", "class_weight": lambda seed: "passthrough",
    "under": lambda seed: RandomUnderSampler(random_state=seed), "over": lambda seed: RandomOverSampler(random_state=seed),
    "rose": lambda seed: RandomOverSampler(shrinkage=0.5, random_state=seed),
    "smote": lambda seed: SMOTE(random_state=seed), "adasyn": lambda seed: ADASYN(random_state=seed),
}


def make_estimator(cfg: dict):
    seed = cfg.get("seed", 42)
    p = dict(cfg.get("model_params", {}))
    name = cfg["model_name"]
    if name == "Random Forest":
        return RandomForestClassifier(**p, n_jobs=-1, random_state=seed)
    if name == "XGBoost":
        return XGBClassifier(**p, eval_metric="logloss", n_jobs=2, random_state=seed)
    if name == "Logistic Regression":
        return LogisticRegression(**p)
    if name.startswith("Neural Network"):
        if "hidden_layer_sizes" in p:
            p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(**p, random_state=seed)
    raise ValueError(f"unknown model {name}")


def make_prep(cfg: dict) -> ColumnTransformer:
    num = cfg.get("numeric_features", NUMERIC)
    return ColumnTransformer([
        ("num", SkPipeline([("clip", ThreeSigmaClip(cfg.get("outlier_mode", "clip"))), ("scale", MinMaxScaler())]), num),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), ["type"]),
    ])


def make_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("prep", make_prep(cfg)),
        ("balance", SAMPLERS[cfg.get("sampler", "none")](cfg.get("seed", 42))),
        ("model", make_estimator(cfg)),
    ])


def make_calibrated(cfg: dict):
    """The serving estimator: isotonic-calibrated pipeline if the notebook chose calibration, else the pipeline itself."""
    if cfg.get("calibration", "none") in (None, "none", False):
        return make_pipeline(cfg)
    return CalibratedClassifierCV(make_pipeline(cfg), method=cfg["calibration"],
                                  cv=StratifiedKFold(3, shuffle=True, random_state=cfg.get("seed", 42)))
