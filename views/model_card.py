import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

from pdm_core.economics import metrics_at
from pdm_core.features import SENSORS, LABELS, MODES, MODE_NAMES
from pdm_core.fleet import psi
from views.common import get_model, get_policy, test_predictions, train_reference, scored_fleet, page_header

m, pol = get_model(), get_policy()
tp = test_predictions()
page_header("🧪 Model Card & Monitoring", "What the model is, how well it works on unseen data, where it fails, and whether today's data still looks like the training data.")

cfg = m.cfg
a, b = st.columns([1, 1.4])
with a.container(border=True):
    st.markdown(f"#### {cfg['model_name']} · `{m.version}`")
    st.markdown(f"""
- **Intended use:** decision support for maintenance engineers — ranking machines by failure risk. **Not** for autonomous shutdown.
- **Data:** UCI AI4I 2020 (10 000 synthetic machine records), stratified 80/20 split, seed {cfg['seed']}.
- **Features:** {', '.join(cfg['features'])} — 6 raw predictors + 3 Ishikawa physics features (kept after a CV ablation); IDs & failure-mode flags excluded to prevent leakage.
- **Pipeline:** 3σ outliers `{cfg['outlier_mode']}` → min-max + one-hot type → balancing `{cfg['sampler']}` (train only) → {cfg['model_name']} {cfg.get('model_params', {})} → calibration `{cfg.get('calibration', 'none')}`. No imputation (no missing values).
- **Selection:** leakage-free tournament (4 models × 7 balancing strategies, 5-fold CV on training data) → grid search → threshold minimising 100 € × Type I + 400 € × Type II on out-of-fold predictions.
- **Trained:** {m.trained_at} UTC.""")
with b:
    t = metrics_at(tp.y, tp.p, pol["threshold"])
    k = st.columns(4)
    k[0].metric("ROC-AUC", f"{roc_auc_score(tp.y, tp.p):.3f}", border=True)
    k[1].metric("PR-AUC", f"{average_precision_score(tp.y, tp.p):.3f}", border=True, help="baseline = prevalence 0.034")
    k[2].metric("Sensitivity", f"{t['sensitivity']:.1%}", border=True)
    k[3].metric("Precision", f"{t['precision']:.1%}", border=True)
    k = st.columns(4)
    k[0].metric("Specificity", f"{t['specificity']:.1%}", border=True)
    k[1].metric("F1", f"{t['f1']:.3f}", border=True)
    k[2].metric("Accuracy", f"{t['accuracy']:.1%}", border=True, help="Misleading under imbalance: 'always OK' scores 96.6 %")
    k[3].metric("Test cost", f"€{t['cost']:,}", border=True)
    st.caption(f"Held-out test set: {len(tp)} machines, {int(tp.y.sum())} failures · threshold {pol['threshold']:.2f}")

c1, c2, c3 = st.columns(3)
fpr, tpr, _ = roc_curve(tp.y, tp.p)
fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#2a78d6", width=2.5)))
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#898781", width=1), showlegend=False)
fig.update_layout(title="ROC curve", height=300, showlegend=False, xaxis_title="false positive rate", yaxis_title="sensitivity")
c1.plotly_chart(fig, config={"displayModeBar": False})
pr, rc, _ = precision_recall_curve(tp.y, tp.p)
fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(color="#2a78d6", width=2.5)))
fig.add_hline(y=tp.y.mean(), line_dash="dash", line_color="#898781")
fig.update_layout(title="Precision–recall curve", height=300, showlegend=False, xaxis_title="recall", yaxis_title="precision")
c2.plotly_chart(fig, config={"displayModeBar": False})
bands = pd.cut(tp.p, [-0.01, 0.01, 0.1, pol["threshold"], 1.0],
                labels=["< 1 %", "1–10 %", f"10–{pol['threshold']:.0%}", f"≥ {pol['threshold']:.0%}"])
cal = tp.assign(band=bands).groupby("band", observed=False).agg(machines=("y", "size"), expected=("p", "sum"), observed=("y", "sum")).reset_index()
fig = go.Figure()
fig.add_bar(x=cal.band.astype(str), y=cal.expected, name="expected (Σ predicted prob.)", marker_color="#86b6ef")
fig.add_bar(x=cal.band.astype(str), y=cal.observed, name="observed failures", marker_color="#2a78d6")
fig.update_layout(title="Calibration by risk band (test)", height=300, barmode="group", bargap=0.35,
                  xaxis_title="predicted risk band", yaxis_title="failures", legend=dict(y=1.18))
c3.plotly_chart(fig, config={"displayModeBar": False})

st.subheader("Where does the model fail? — recall by failure mode")
pred = tp.p >= pol["threshold"]
rows = []
for md in MODES:
    s = (tp[md] == 1) & (tp.y == 1)
    rows.append(dict(mode=MODE_NAMES[md], failures=int(s.sum()), caught=int((s & pred).sum()),
                     recall=(s & pred).sum() / s.sum() if s.sum() else np.nan))
st.dataframe(pd.DataFrame(rows), hide_index=True,
             column_config={"recall": st.column_config.ProgressColumn("recall", format="percent", min_value=0, max_value=1)})
st.caption("Tool-wear failures occur randomly in the 200–240 min window and are hard to predict → keep a fixed tool-change rule "
           "as a complement. Random failures are unpredictable by design.")

st.subheader("Data drift monitor — current fleet vs. training data (PSI)")
ref, cur = train_reference(), scored_fleet()
d = pd.DataFrame([dict(feature=LABELS[c], psi=psi(ref[c], cur[c])) for c in SENSORS])
d["state"] = np.where(d.psi > 0.25, "🔴 significant drift", np.where(d.psi > 0.1, "🟠 moderate", "🟢 stable"))
st.dataframe(d, hide_index=True, column_config={"psi": st.column_config.NumberColumn("PSI", format="%.3f")})
st.caption("PSI < 0.1 stable · 0.1–0.25 watch · > 0.25 investigate & consider retraining. Tool wear drifts naturally as the "
           "simulated shifts advance without tool changes — a realistic retraining / maintenance trigger.")

with st.expander("Limitations & ethical considerations"):
    st.markdown("""
- **Synthetic data:** AI4I 2020 is generated from physics rules; real plants have sensor noise, missing data, drift and different failure modes. A shadow-mode pilot on plant data is mandatory.
- **Label noise:** 27 of 10 000 records are inconsistent between target and failure-mode flags.
- **Unpredictable failures:** random (RNF) and tool-wear (TWF) failures limit achievable recall.
- **Human oversight:** the model ranks; engineers decide. All decisions are logged with model version for accountability.
- **Workforce data:** no operator-level performance data is used (works-council agreement).""")
