import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm_core import store
from pdm_core.config import STATUS, FLEET_SEED_PATH
from pdm_core.fleet import advance_shift
from views.common import (scored_fleet, get_policy, get_shift, get_fleet, status_text, page_header, fmt_eur)

pol, shift = get_policy(), get_shift()
page_header(f"🏭 Plant Control Room — Shift {shift}",
            "Live risk scoring of every monitored machine. Calibrated failure probability per machine; "
            "alerts above the cost-optimal threshold go to the human review queue.")

c1, c2, _ = st.columns([1.3, 1, 3])
if c1.button("▶ Run next shift", type="primary", help="Simulate one production shift: tool wear grows, temperatures and load drift"):
    f = scored_fleet()
    store.log_shift(shift, int((f.status == "critical").sum()), float(f.risk.mean()))
    store.save_fleet(advance_shift(get_fleet(), shift + 1))
    store.set_setting("shift", shift + 1)
    st.toast(f"Shift {shift + 1} started — fleet re-scored.", icon="⏱️")
    st.rerun()
if c2.button("↺ Reset plant", help="Restore the initial fleet and clear the shift counter (audit log is kept)"):
    store.save_fleet(pd.read_csv(FLEET_SEED_PATH)); store.set_setting("shift", 1)
    with store.conn() as c:
        c.execute("DELETE FROM shifts")
    st.rerun()

f = scored_fleet()
reviewed = store.reviewed_this_shift(shift)
crit = f[f.status == "critical"]
open_alerts = crit[~crit.machine_id.isin(reviewed)]

k = st.columns(5)
k[0].metric("Machines monitored", len(f), border=True)
k[1].metric(f"{STATUS['critical']['icon']} Critical alerts", len(crit), border=True,
            help=f"risk ≥ {pol['threshold']:.2f} (cost-optimal threshold)")
k[2].metric(f"{STATUS['warning']['icon']} Watch list", int((f.status == "warning").sum()), border=True,
            help=f"{pol['watch']:.2f} ≤ risk < {pol['threshold']:.2f}")
k[3].metric("Open (unreviewed) alerts", len(open_alerts), border=True)
k[4].metric("Expected failures", f"{f.risk.sum():.1f}",
            delta=fmt_eur(f.risk.sum() * pol["cost_fn"]) + " exposure", delta_color="off", delta_arrow="off", border=True,
            help="Sum of calibrated probabilities = expected number of failures; × Type-II cost")

if len(open_alerts):
    st.warning(f"**{len(open_alerts)} alert(s) waiting for engineering review** — open the Review Queue. "
               f"Highest priority: **{open_alerts.sort_values('priority', ascending=False).machine_id.iloc[0]}**.", icon="📋")
    st.page_link("views/review_queue.py", label="Go to Review Queue", icon="📋")

left, right = st.columns([1.6, 1])
with left:
    fig = go.Figure()
    for s in ["good", "warning", "critical"]:
        d = f[f.status == s]
        fig.add_scatter(x=d.tool_wear, y=d.risk.clip(lower=0.0005), mode="markers", name=status_text(s),
                        marker=dict(color=STATUS[s]["color"], size=10 if s != "good" else 8,
                                    symbol={"good": "circle", "warning": "diamond", "critical": "triangle-up"}[s],
                                    line=dict(color="white", width=1.5)),
                        customdata=np.c_[d.machine_id, d.line, d.drivers],
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>risk %{y:.1%} · wear %{x:.0f} min"
                                      "<br>drivers: %{customdata[2]}<extra></extra>")
    fig.add_hline(y=pol["threshold"], line_dash="dash", line_color="#52514e",
                  annotation_text=f"alert threshold {pol['threshold']:.2f}", annotation_position="top left")
    fig.update_layout(title="Fleet risk map — failure probability vs. tool wear", height=400,
                      yaxis=dict(title="calibrated failure probability (log scale)", type="log", range=[-3.4, 0.12],
                                 tickvals=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1], ticktext=["0.1%", "0.3%", "1%", "3%", "10%", "30%", "100%"]),
                      xaxis_title="tool wear [min]")
    st.plotly_chart(fig, config={"displayModeBar": False})
with right:
    by_line = f.groupby("line").agg(machines=("machine_id", "size"), critical=("status", lambda s: (s == "critical").sum()),
                                    watch=("status", lambda s: (s == "warning").sum()), exp_fail=("risk", "sum")).reset_index()
    fig = go.Figure()
    fig.add_bar(y=by_line.line, x=by_line.critical, orientation="h", name=status_text("critical"), marker_color=STATUS["critical"]["color"])
    fig.add_bar(y=by_line.line, x=by_line.watch, orientation="h", name=status_text("warning"), marker_color=STATUS["warning"]["color"])
    fig.update_layout(barmode="stack", title="Alerts by production line", height=250, bargap=0.45,
                      xaxis=dict(title="machines", dtick=1), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, config={"displayModeBar": False})
    hist = store.shifts_df()
    if len(hist):
        hist = pd.concat([hist, pd.DataFrame([{"shift": shift, "alerts": len(crit), "mean_risk": f.risk.mean()}])])
        fig = go.Figure(go.Scatter(x=hist["shift"], y=hist["alerts"], mode="lines+markers", line=dict(width=2, color="#2a78d6")))
        fig.update_layout(title="Critical alerts per shift", height=200, xaxis=dict(title="shift", dtick=1), yaxis_title="alerts")
        st.plotly_chart(fig, config={"displayModeBar": False})
    else:
        st.info("Run a few shifts to see how tool wear drives risk — and how maintenance decisions bring it down.", icon="⏱️")

st.subheader("Fleet status")
flt = st.segmented_control("Show", ["All", "Critical", "Watch", "Normal"], default="All")
view = f.copy()
if flt and flt != "All":
    view = view[view.status == {"Critical": "critical", "Watch": "warning", "Normal": "good"}[flt]]
view = view.sort_values("priority", ascending=False)
view["Status"] = view.status.map(status_text)
view["Reviewed"] = np.where(view.machine_id.isin(reviewed), "✔ this shift", "")
sel = st.dataframe(
    view[["Status", "machine_id", "line", "type", "risk", "priority", "drivers", "rules_breached", "tool_wear",
          "torque", "rpm", "process_temp", "Reviewed"]],
    hide_index=True, height=420, on_select="rerun", selection_mode="single-row",
    column_config={
        "machine_id": "Machine", "line": "Line", "type": "Type",
        "risk": st.column_config.ProgressColumn("Failure risk", format="percent", min_value=0, max_value=1),
        "priority": st.column_config.NumberColumn("Priority", help="risk × line criticality × (1 + 0.25 × rules breached)", format="%.3f"),
        "drivers": "Top SHAP drivers", "rules_breached": st.column_config.NumberColumn("Rules breached", format="%d"),
        "tool_wear": st.column_config.NumberColumn("Wear [min]", format="%.0f"),
        "torque": st.column_config.NumberColumn("Torque [Nm]", format="%.1f"),
        "rpm": st.column_config.NumberColumn("Speed [rpm]", format="%.0f"),
        "process_temp": st.column_config.NumberColumn("Proc. temp [K]", format="%.1f")})
rows = sel.selection.rows if sel and sel.selection else []
if rows:
    mid = view.iloc[rows[0]].machine_id
    st.session_state["machine"] = mid
    st.page_link("views/diagnosis.py", label=f"Open diagnosis for {mid}", icon="🔍")
