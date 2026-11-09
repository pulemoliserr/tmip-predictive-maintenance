import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm_core.config import STATUS
from pdm_core.features import rule_checks, LABELS, SENSORS, build_features
from views.common import scored_fleet, get_model, get_policy, status_text, page_header, train_reference

pol, m = get_policy(), get_model()
f = scored_fleet().sort_values("priority", ascending=False)
page_header("🔍 Machine Diagnosis",
            "Why is this machine at risk? Calibrated probability, SHAP drivers, transparent engineering rules and a what-if simulator.")

ids = f.machine_id.tolist()
default = st.session_state.get("machine", ids[0])
mid = st.selectbox("Machine (sorted by priority)", ids, index=ids.index(default) if default in ids else 0,
                   format_func=lambda x: f"{x}  ·  {status_text(f.set_index('machine_id').loc[x, 'status'])}  ·  "
                                         f"{f.set_index('machine_id').loc[x, 'risk']:.1%}")
st.session_state["machine"] = mid
row = f[f.machine_id == mid].iloc[0]
raw = f[f.machine_id == mid][["type"] + SENSORS]


def recommendation(r, checks):
    if r.status == "critical":
        hit = [c["rule"] for c in checks if c["breached"] or c["near"]]
        focus = {"Tool life": "replace tool", "Overstrain": "reduce load", "Power window": "check drive & spindle load",
                 "Heat dissipation": "check cooling / heat exchange"}
        acts = ", ".join(dict.fromkeys(focus[h] for h in hit)) if hit else "inspect at next opportunity"
        return "critical", f"**Inspect now** – {acts}. Send to review queue for engineer decision."
    if r.status == "warning":
        return "warning", "**Watch** – schedule inspection at next planned stop; re-check next shift."
    return "good", "**No action** – continue operation."


checks = rule_checks(row)
lvl, rec = recommendation(row, checks)

c1, c2 = st.columns([1, 1.5])
with c1:
    with st.container(border=True):
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=row.risk * 100, number=dict(suffix=" %", valueformat=".1f"),
            gauge=dict(axis=dict(range=[0, 100], ticksuffix="%"), bar=dict(color=STATUS[row.status]["color"], thickness=0.3),
                       steps=[dict(range=[0, pol["watch"] * 100], color="#eef5ee"),
                              dict(range=[pol["watch"] * 100, pol["threshold"] * 100], color="#fdf3dc"),
                              dict(range=[pol["threshold"] * 100, 100], color="#f9e2e2")],
                       threshold=dict(line=dict(color="#0b0b0b", width=3), value=pol["threshold"] * 100)),
            title=dict(text=f"{mid} · {row.line}<br><span style='font-size:0.8em'>{status_text(row.status)}</span>")))
        g.update_layout(height=270, margin=dict(t=70, b=10, l=30, r=30))
        st.plotly_chart(g, config={"displayModeBar": False})
        {"critical": st.error, "warning": st.warning, "good": st.success}[lvl](rec)
        st.caption(f"Priority score {row.priority:.3f} (line criticality ×{row.criticality}). Probability is isotonic-calibrated: "
                   "a value of 30 % means roughly 3 in 10 comparable machines fail.")
with c2:
    sv = m.shap_values(raw).iloc[0].sort_values(key=abs)
    vals = build_features(raw).iloc[0]
    fig = go.Figure(go.Bar(
        x=sv.values, y=[f"{LABELS[c]} = {vals[c]:,.1f}" for c in sv.index], orientation="h",
        marker_color=["#eb6834" if v > 0 else "#2a78d6" for v in sv.values],
        hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_width=1)
    fig.update_layout(title="Local explanation (SHAP) — what pushes the risk up (orange) or down (blue)", height=360,
                      xaxis_title="contribution to failure probability (Random Forest, before calibration)", bargap=0.35)
    st.plotly_chart(fig, config={"displayModeBar": False})

st.subheader("Transparent engineering rules")
st.caption("Physics-based limits of the failure modes (Matzka 2020). Shown next to the ML score so that the engineer can verify it.")
rc = pd.DataFrame(checks)
rc["State"] = np.where(rc.breached, "❌ breached", np.where(rc.near, "⚠️ near limit", "✅ ok"))
st.dataframe(rc[["State", "rule", "condition", "value"]], hide_index=True,
             column_config={"rule": "Rule", "condition": "Limit", "value": "Current value"})

ref = train_reference()
pct = {c: (ref[c] < row[c]).mean() * 100 for c in SENSORS}
st.subheader("Sensor readings vs. training population")
cols = st.columns(5)
for col, c in zip(cols, SENSORS):
    col.metric(LABELS[c], f"{row[c]:,.1f}", delta=f"P{pct[c]:.0f} of fleet history", delta_color="off", border=True)

with st.expander("🧪 What-if simulator — how would the risk change?", expanded=False):
    st.caption("Change operating parameters to test countermeasures before acting (e.g. reduce torque, replace tool).")
    w1, w2, w3 = st.columns(3)
    t_wear = w1.slider("Tool wear [min]", 0, 260, int(row.tool_wear), key=f"wi_wear_{mid}")
    t_torque = w2.slider("Torque [Nm]", 3.0, 77.0, float(round(row.torque, 1)), 0.5, key=f"wi_tq_{mid}")
    t_rpm = w3.slider("Rotational speed [rpm]", 1150, 2900, int(row.rpm), 10, key=f"wi_rpm_{mid}")
    w4, w5, _ = st.columns(3)
    t_air = w4.slider("Air temperature [K]", 295.0, 305.0, float(round(row.air_temp, 1)), 0.1, key=f"wi_air_{mid}")
    t_proc = w5.slider("Process temperature [K]", 305.0, 314.0, float(round(row.process_temp, 1)), 0.1, key=f"wi_pr_{mid}")
    sim = pd.DataFrame([{"type": row.type, "air_temp": t_air, "process_temp": t_proc, "rpm": t_rpm, "torque": t_torque, "tool_wear": t_wear}])
    r_new = float(m.risk(sim)[0])
    a, b = st.columns(2)
    a.metric("Simulated risk", f"{r_new:.1%}", delta=f"{(r_new - row.risk) * 100:+.1f} pp vs. now", delta_color="inverse", border=True)
    s_new = "critical" if r_new >= pol["threshold"] else "warning" if r_new >= pol["watch"] else "good"
    b.metric("Simulated status", status_text(s_new), border=True)
