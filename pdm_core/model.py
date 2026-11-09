"""Model serving: load the trained artifact, score machines, explain predictions with SHAP."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import MODEL_PATH, METRICS_PATH
from .features import build_features


class RiskModel:
    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.calibrated = bundle["calibrated"]
        self.explain_pipe = bundle["explain_pipe"]
        self.cfg = bundle["config"]
        self.version = bundle["version"]
        self.trained_at = bundle["trained_at"]
        self._pre = self.explain_pipe.named_steps["prep"]          # 3σ clip -> min-max, one-hot type
        self.features = [n.split("__")[-1] for n in self._pre.get_feature_names_out()]
        self._model = self.explain_pipe.named_steps["model"]
        self._background = bundle.get("background")                 # transformed training sample (non-tree models)
        self._explainer = None
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated failure probability for each row of a raw-sensor frame."""
        return self.calibrated.predict_proba(build_features(frame))[:, 1]

    @property
    def explainer(self):
        if self._explainer is None:
            if isinstance(self._model, (RandomForestClassifier, XGBClassifier)):
                self._explainer = shap.TreeExplainer(self._model)
            elif isinstance(self._model, LogisticRegression):
                self._explainer = shap.LinearExplainer(self._model, self._background)
            else:   # MLP: model-agnostic explainer on the positive-class probability
                self._explainer = shap.Explainer(lambda z: self._model.predict_proba(z)[:, 1], self._background)
        return self._explainer

    def shap_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        Z = self._pre.transform(build_features(frame))
        ex = self.explainer
        sv = ex.shap_values(Z) if hasattr(ex, "shap_values") and not isinstance(ex, shap.explainers.Permutation) else ex(Z).values
        sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
        return pd.DataFrame(sv, columns=self.features, index=frame.index)

    def top_drivers(self, frame: pd.DataFrame, k=3) -> list[str]:
        sv = self.shap_values(frame)
        out = []
        for _, r in sv.iterrows():
            top = r[r > 0].sort_values(ascending=False).head(k)
            out.append(", ".join(top.index) if len(top) else "—")
        return out
