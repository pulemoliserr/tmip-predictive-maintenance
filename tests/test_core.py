"""Unit tests for the core library:  python -m pytest -q  (run from app/)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdm_core.features import build_features, rule_checks, ALL_FEATURES
from pdm_core.pipeline import ThreeSigmaClip, make_pipeline, DEFAULT_CONFIG
from pdm_core.economics import npv, irr, payback, metrics_at, BusinessCase
from pdm_core.fleet import psi, apply_maintenance

ROW = pd.DataFrame([{"type": "L", "air_temp": 300.0, "process_temp": 309.0, "rpm": 1300, "torque": 60.0, "tool_wear": 210}])


def test_features_rowwise():
    X = build_features(ROW)
    assert list(X.columns) == ALL_FEATURES + ["type"]
    assert X.temp_diff.iloc[0] == 9.0 and X.strain.iloc[0] == 12600 and X.type_L.iloc[0] == 1


def test_rules():
    c = {r["rule"]: r for r in rule_checks(ROW.iloc[0])}
    assert c["Overstrain"]["breached"] and c["Tool life"]["breached"] and not c["Heat dissipation"]["breached"]


def test_outlier_modes_fit_on_train_only():
    tr = np.r_[np.zeros(99), 100.0].reshape(-1, 1)
    c = ThreeSigmaClip("clip").fit(tr)
    assert c.transform([[1e6]])[0, 0] == c.hi_[0]
    assert ThreeSigmaClip("none").fit(tr).transform([[1e6]])[0, 0] == 1e6


def test_pipeline_builds_for_every_sampler():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"type": rng.choice(list("LMH"), 300), "air_temp": rng.normal(300, 2, 300),
                          "process_temp": rng.normal(310, 1, 300), "rpm": rng.normal(1500, 150, 300),
                          "torque": rng.normal(40, 10, 300), "tool_wear": rng.uniform(0, 250, 300)})
    y = (rng.random(300) < 0.15).astype(int)
    for s in ["none", "class_weight", "under", "over", "rose", "smote", "adasyn"]:
        p = make_pipeline({**DEFAULT_CONFIG, "sampler": s, "model_params": {"n_estimators": 10}}).fit(build_features(frame), y)
        assert p.predict_proba(build_features(frame)).shape == (300, 2)


def test_cost_and_finance():
    m = metrics_at([0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], 0.5, 100, 400)
    assert (m["FP"], m["FN"], m["cost"]) == (1, 1, 500)
    assert abs(npv(0.1, [110]) - 100) < 1e-9
    assert abs(irr([-100, 110]) - 0.10) < 1e-5
    assert abs(payback([-360986, 393468]) - 1.917) < 1e-3   # course example: 1.92 years
    s = BusinessCase().summary()
    assert np.isfinite(s["npv"])


def test_psi_and_maintenance():
    a = np.random.default_rng(1).normal(size=5000)
    assert psi(a, a) < 0.01 and psi(a, a + 2) > 0.25
    f = pd.DataFrame([{"machine_id": "L-1", "tool_wear": 220.0, "air_temp": 300, "process_temp": 309, "rpm": 1400,
                       "torque": 50, "last_service_shift": 0, "status_note": ""}])
    assert apply_maintenance(f, "L-1", "Confirmed: worn tool replaced", 3).tool_wear.iloc[0] == 0
