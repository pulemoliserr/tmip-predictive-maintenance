import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm_core import store
from pdm_core.economics import threshold_sweep, metrics_at
from views.common import get_policy, test_predictions, page_header, user, fmt_eur

pol = get_policy()
tp = test_predictions()
page_header("⚖️ Operating Policy — cost-sensitive alert threshold",
            "Turns the confusion matrix into money. Evaluated on the 2 000 held-out test machines (68 real failures) "
            "that were never used for training or tuning.")

c1, c2, c3 = st.columns(3)
c_fp = c1.number_input("Type I cost — false alarm (€)", 10, 10_000, int(pol["cost_fp"]), 10,
                       help="Technician visit / unnecessary inspection")
c_fn = c2.number_input("Type II cost — missed failure (€)", 10, 100_000, int(pol["cost_fn"]), 10,
                       help="Breakdown: emergency repair, downtime, secondary damage")
sw = threshold_sweep(tp.y, tp.p, c_fp, c_fn)
opt = sw.loc[sw.cost.idxmin()]
thr = c3.slider("Alert threshold", 0.02, 0.98, float(pol["threshold"]), 0.01,
                help=f"Cost-optimal for these costs: {opt.threshold:.2f}")
cur = metrics_at(tp.y, tp.p, thr, c_fp, c_fn)

k = st.columns(6)
k[0].metric("Total cost (test)", fmt_eur(cur["cost"]), delta=f"+{fmt_eur(cur['cost'] - opt.cost)} vs min",
            delta_color="inverse", border=True)
k[1].metric("Sensitivity", f"{cur['sensitivity']:.1%}", border=True, help="share of real failures caught")
k[2].metric("Precision", f"{cur['precision']:.1%}", border=True, help="share of alerts that are real → alert fatigue")
k[3].metric("Specificity", f"{cur['specificity']:.1%}", border=True)
k[4].metric("False alarms (Type I)", cur["FP"], border=True)
k[5].metric("Missed failures (Type II)", cur["FN"], border=True)

left, right = st.columns([1.6, 1])
with left:
    fig = go.Figure()
    fig.add_scatter(x=sw.threshold, y=sw.cost, mode="lines", name="total cost", line=dict(color="#2a78d6", width=2.5),
                    hovertemplate="thr %{x:.2f}<br>cost €%{y:,.0f}<extra></extra>")
    fig.add_scatter(x=sw.threshold, y=c_fp * sw.FP, mode="lines", name="Type I cost (false alarms)", line=dict(color="#eda100", width=2, dash="dot"))
    fig.add_scatter(x=sw.threshold, y=c_fn * sw.FN, mode="lines", name="Type II cost (missed failures)", line=dict(color="#eb6834", width=2, dash="dot"))
    fig.add_vline(x=thr, line_color="#0b0b0b", line_width=1.5, annotation_text=f"selected {thr:.2f}")
    fig.add_scatter(x=[opt.threshold], y=[opt.cost], mode="markers", name=f"optimum {opt.threshold:.2f}",
                    marker=dict(size=12, color="#1baf7a", line=dict(color="white", width=2)))
    fig.update_layout(title="Cost curve on the held-out test set", height=380, xaxis_title="alert threshold",
                      yaxis=dict(title="cost (€)", rangemode="tozero"))
    st.plotly_chart(fig, config={"displayModeBar": False})
with right:
    z = [[cur["TN"], cur["FP"]], [cur["FN"], cur["TP"]]]
    txt = [[f"TN<br>{cur['TN']}", f"FP · Type I<br>{cur['FP']}<br>€{c_fp * cur['FP']:,}"],
           [f"FN · Type II<br>{cur['FN']}<br>€{c_fn * cur['FN']:,}", f"TP<br>{cur['TP']}"]]
    fig = go.Figure(go.Heatmap(z=np.log1p(z), x=["predicted OK", "predicted failure"], y=["actual OK", "actual failure"],
                               text=txt, texttemplate="%{text}", textfont=dict(size=14), showscale=False,
                               colorscale=[[0, "#eef4fc"], [1, "#2a78d6"]], hoverinfo="skip", xgap=3, ygap=3))
    fig.update_layout(title=f"Confusion matrix at {thr:.2f}", height=380, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, config={"displayModeBar": False})

st.info(f"**Decision theory check.** For perfectly calibrated probabilities the cost-optimal threshold is "
        f"c_FP / (c_FP + c_FN) = **{c_fp / (c_fp + c_fn):.2f}**. The plant policy ({pol['threshold']:.2f}) was chosen on "
        f"out-of-fold *training* predictions (notebook §4.8–4.9). The test-set optimum ({opt.threshold:.2f}) is shown for "
        "transparency only — re-tuning the threshold on the test set would overfit it (68 failures) and leak information. "
        "In operation the threshold is re-validated on reviewed alerts from the audit log.", icon="📐")

st.subheader("Operating points — the menu for the policy decision")
pts = sorted(set([0.05, 0.10, 0.20, round(float(opt.threshold), 2), 0.50, 0.70, round(thr, 2)]))
ops = pd.DataFrame([metrics_at(tp.y, tp.p, t, c_fp, c_fn) for t in pts])
st.dataframe(ops[["threshold", "sensitivity", "precision", "specificity", "FP", "FN", "cost"]], hide_index=True,
             column_config={"sensitivity": st.column_config.NumberColumn(format="percent"),
                            "precision": st.column_config.NumberColumn(format="percent"),
                            "specificity": st.column_config.NumberColumn(format="percent"),
                            "cost": st.column_config.NumberColumn("total cost €", format="€%d")})

st.subheader("Cost scenarios — replacing interview-based cost estimates")
st.caption("Practitioner interviews could not be scheduled, so the cost of a missed failure is unknown. These scenarios span the range "
           "from a single cell's direct cost (€400) to the business-case value incl. downtime, scrap and expediting (€4,700). "
           "Optimal thresholds are searched on the test set and are illustrative only.")
scen = [("A — current setting", 100, 400), ("B — moderate", 100, 1_000), ("C — high", 150, 2_000), ("D — business-case scale", 100, 4_700)]
rows = []
for name, cfp_s, cfn_s in scen:
    s_sw = threshold_sweep(tp.y, tp.p, cfp_s, cfn_s); s_opt = s_sw.loc[s_sw.cost.idxmin()]
    at_pol = metrics_at(tp.y, tp.p, pol["threshold"], cfp_s, cfn_s)
    reactive = int(tp.y.sum()) * cfn_s
    rows.append(dict(scenario=name, c_fp=cfp_s, c_fn=cfn_s, opt_threshold=float(s_opt.threshold),
                     caught_opt=f"{int(s_opt.TP)} / {int(s_opt.TP + s_opt.FN)}", false_alarms_opt=int(s_opt.FP),
                     cost_opt=float(s_opt.cost), cost_policy=float(at_pol["cost"]), cost_reactive=float(reactive),
                     saving_vs_reactive=1 - float(s_opt.cost) / reactive))
st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
    "c_fp": st.column_config.NumberColumn("€ / false alarm", format="€%d"),
    "c_fn": st.column_config.NumberColumn("€ / missed failure", format="€%d"),
    "opt_threshold": st.column_config.NumberColumn("optimal threshold", format="%.2f"),
    "caught_opt": "caught (optimal)", "false_alarms_opt": "false alarms (optimal)",
    "cost_opt": st.column_config.NumberColumn("cost at optimum", format="€%d"),
    "cost_policy": st.column_config.NumberColumn(f"cost at plant policy", format="€%d"),
    "cost_reactive": st.column_config.NumberColumn("cost reactive", format="€%d"),
    "saving_vs_reactive": st.column_config.NumberColumn("saving vs reactive", format="percent")})
st.caption("Reading: the model saves 84–91 % against reactive maintenance in every scenario. The optimal threshold only drops "
           "(more alerts, more false alarms) when a missed failure costs more than ~20–40× a false alarm — so the real inspection "
           "cost must be measured in the pilot before the policy is changed (cf. McKinsey 2021 on false positives).")

st.divider()
w = st.slider("Watch-list threshold (non-urgent queue)", 0.01, float(thr), float(min(pol["watch"], thr)), 0.01)
reviewer, role = user()
can = role in ("Plant Manager", "Data Scientist")
if st.button("Apply as plant operating policy", type="primary", disabled=not can,
             help=None if can else "Only Plant Managers / Data Scientists can change the plant policy"):
    store.set_setting("policy", {"threshold": float(thr), "watch": float(w), "cost_fp": int(c_fp), "cost_fn": int(c_fn)})
    store.set_setting("policy_changed_by", {"by": reviewer, "role": role, "ts": store.now()})
    st.cache_data.clear()
    st.toast("Policy applied — fleet re-scored with the new thresholds.", icon="⚖️")
    st.rerun()
if not can:
    st.caption("Switch the role in the sidebar to *Plant Manager* to change the policy (governance: four-eyes principle).")
chg = store.get_setting("policy_changed_by")
if chg:
    st.caption(f"Current policy last changed by {chg['by']} ({chg['role']}) at {chg['ts']} UTC.")
