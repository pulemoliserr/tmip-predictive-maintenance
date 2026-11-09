"""Paths and global constants."""
from pathlib import Path
import os

APP_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS = APP_DIR / "artifacts"
DATA_PATH = Path(os.environ.get("PDM_DATA", APP_DIR.parent / "data" / "ai4i2020.csv"))
DB_PATH = Path(os.environ.get("PDM_DB", ARTIFACTS / "pdm_audit.sqlite"))

MODEL_PATH = ARTIFACTS / "model.joblib"
MODEL_CONFIG = ARTIFACTS / "model_config.json"   # exported by the CRISP-DM notebook (§6.1)
METRICS_PATH = ARTIFACTS / "metrics.json"
TEST_PRED_PATH = ARTIFACTS / "test_predictions.csv"
FLEET_SEED_PATH = ARTIFACTS / "fleet_seed.csv"
TRAIN_STATS_PATH = ARTIFACTS / "train_reference.csv"

SEED = 42
LINES = {"Line A – Milling": 1.3, "Line B – Turning": 1.0, "Line C – Drilling": 0.9, "Line D – Finishing": 1.1}
# Line criticality multiplies the priority score (bottleneck lines first)

# Status palette (reserved; always shipped with icon + label)
STATUS = {
    "critical": {"color": "#d03b3b", "icon": "🔴", "label": "Critical"},
    "warning": {"color": "#fab219", "icon": "🟠", "label": "Watch"},
    "good": {"color": "#0ca30c", "icon": "🟢", "label": "Normal"},
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
