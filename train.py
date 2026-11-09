"""Training job: rebuilds the model selected in the CRISP-DM notebook and writes serving artifacts.

    python train.py            # uses artifacts/model_config.json exported by the notebook (§6.1)

Steps (identical to the notebook v2): stratified 80/20 split (seed 42) -> physics features ->
3σ clipping -> min-max + one-hot type -> balancing (train only, if selected) -> model -> calibration (if selected).
No imputation (no missing values) and no Boruta (only 6 raw predictors). The locked test set is scored once and stored for the Model Card
and the Operating-Policy page.
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pdm_core.config import (ARTIFACTS, DATA_PATH, MODEL_CONFIG, MODEL_PATH, METRICS_PATH, TEST_PRED_PATH,
                             FLEET_SEED_PATH, TRAIN_STATS_PATH, SEED)
from pdm_core.features import load_raw, build_features, MODES, SENSORS
from pdm_core.pipeline import DEFAULT_CONFIG, make_calibrated, make_pipeline
from pdm_core.economics import metrics_at
from pdm_core.fleet import initial_fleet
from sklearn.metrics import roc_auc_score, average_precision_score


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = {**DEFAULT_CONFIG, **(json.loads(MODEL_CONFIG.read_text()) if MODEL_CONFIG.exists() else {})}
    df = load_raw(DATA_PATH)
    X, y = build_features(df), df["failure"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"train {len(y_tr)} ({y_tr.sum()} failures) | test {len(y_te)} ({y_te.sum()} failures)")

    cal = make_calibrated(cfg).fit(X_tr, y_tr)
    exp_pipe = make_pipeline(cfg).fit(X_tr, y_tr)
    p = cal.predict_proba(X_te)[:, 1]
    thr = float(cfg.get("threshold", 0.35))
    m = metrics_at(y_te, p, thr, cfg["cost_fp"], cfg["cost_fn"])
    m.update(roc_auc=roc_auc_score(y_te, p), pr_auc=average_precision_score(y_te, p),
             n_test=int(len(y_te)), n_train=int(len(y_tr)), failures_test=int(y_te.sum()))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    version = "pdm-" + hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    background = exp_pipe.named_steps["prep"].transform(X_tr.sample(100, random_state=SEED))   # for non-tree SHAP explainers
    joblib.dump(dict(calibrated=cal, explain_pipe=exp_pipe, config=cfg, version=version, trained_at=trained_at,
                     background=background), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(dict(version=version, trained_at=trained_at, config=cfg, test=m), indent=2, default=float))

    te = df.loc[X_te.index]
    pd.DataFrame({"udi": te["udi"], "type": te["type"], "y": y_te.values, "p": p, **{k: te[k].values for k in MODES}}
                 ).to_csv(TEST_PRED_PATH, index=False)
    df.loc[X_tr.index, ["type"] + SENSORS].to_csv(TRAIN_STATS_PATH, index=False)
    initial_fleet(te).to_csv(FLEET_SEED_PATH, index=False)
    print(f"saved {MODEL_PATH.name} ({version})")


if __name__ == "__main__":
    main()
