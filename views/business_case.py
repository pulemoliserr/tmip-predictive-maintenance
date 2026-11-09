import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm_core.economics import BusinessCase, metrics_at, sensitivity
from views.common import get_policy, test_predictions, page_header, fmt_eur

pol = get_policy()
tp = test_predictions()
cur = metrics_at(tp.y, tp.p, pol["threshold"])
page_header("💶 Business Case — ROI, NPV and break-even",
            "Discounted-cash-flow business case (course method: benefits vs. costs, cumulative net, NPV at 7.5 %, IRR, amortisation). "
            "Model quality enters through sensitivity and precision at the current operating policy.")

with st.form("bc"):
    st.markdown("**Plant & failure assumptions**")
    a = st.columns(4)
    machines = a[0].number_input("Monitored machines", 10, 5000, 150, 10)
    fpy = a[1].number_input("Unplanned failures / machine / year (today)", 0.1, 10.0, 0.8, 0.1)
    dt_h = a[2].number_input("Downtime per unplanned failure (h)", 0.5, 72.0, 6.0, 0.5)
    c_h = a[3].number_input("Cost of downtime (€/h)", 50, 50_000, 800, 50)
    b = st.columns(4)
    em = b[0].number_input("Emergency repair cost (€)", 0, 200_000, 3500, 100)
    pl = b[1].number_input("Planned repair cost (€)", 0, 200_000, 2000, 100)
    pl_h = b[2].number_input("Planned downtime (h)", 0.0, 24.0, 2.0, 0.5)
    insp = b[3].number_input("Inspection cost per false alarm (€)", 0, 10_000, 350, 10)
    st.markdown("**Model performance & project costs**")
    c = st.columns(4)
    rec = c[0].number_input("Sensitivity (recall)", 0.05, 0.99, round(cur["sensitivity"], 3), 0.01,
                            help="Default: test-set value at the current operating policy")
    prec = c[1].number_input("Precision", 0.05, 0.99, round(cur["precision"], 3), 0.01)
    capex = c[2].number_input("Implementation CAPEX year 1 (€)", 0, 10_000_000, 350_000, 10_000,
                              help="IoT retrofit, data platform, integration with CMMS, project team")
    opex = c[3].number_input("Running cost per year (€)", 0, 5_000_000, 90_000, 5_000,
                             help="Cloud & licences, 0.5 FTE data scientist, model monitoring & retraining")
    d = st.columns(4)
    rate = d[0].number_input("Discount rate", 0.0, 0.3, 0.075, 0.005, format="%.3f")
    ad1 = d[1].slider("Adoption year 1", 0.0, 1.0, 0.5, 0.05)
    ad2 = d[2].slider("Adoption year 2", 0.0, 1.0, 0.85, 0.05)
    years = d[3].selectbox("Horizon (years)", [3, 4, 5], index=1)
    st.form_submit_button("Recalculate", type="primary")

bc = BusinessCase(machines=machines, failures_per_machine_year=fpy, downtime_h_per_failure=dt_h, cost_per_downtime_h=c_h,
                  emergency_repair_cost=em, planned_repair_cost=pl, planned_downtime_h=pl_h, inspection_cost=insp,
                  recall=rec, precision=prec, capex=capex, opex_year=opex, adoption=(ad1, ad2, 1.0, 1.0, 1.0),
                  discount_rate=rate, years=years)
y, s, ue = bc.yearly(), bc.summary(), bc.unit_economics()

k = st.columns(5)
k[0].metric(f"NPV ({rate:.1%})", fmt_eur(s["npv"]), border=True)
k[1].metric("IRR", f"{s['irr']:.0%}" if np.isfinite(s["irr"]) else "n/a", border=True)
k[2].metric("Amortisation", f"{s['payback_years']:.2f} years" if np.isfinite(s["payback_years"]) else "> horizon", border=True)
k[3].metric("ROI (net / total cost)", f"{s['roi']:.0%}", border=True)
k[4].metric("Saving per caught failure", fmt_eur(ue["saving_per_catch"]), border=True,
            help=f"unplanned {fmt_eur(ue['cost_unplanned'])} − planned {fmt_eur(ue['cost_planned'])}")

left, right = st.columns([1.5, 1])
with left:
    fig = go.Figure()
    fig.add_bar(x=y.index, y=y.benefit, name="Benefit (avoided breakdown cost)", marker_color="#1baf7a")
    fig.add_bar(x=y.index, y=-y.total_cost, name="Total cost (capex, opex, false alarms)", marker_color="#eb6834")
    fig.add_scatter(x=y.index, y=y.cumulative, name="Cumulative net", mode="lines+markers", line=dict(color="#2a78d6", width=3))
    fig.add_hline(y=0, line_color="#52514e", line_width=1)
    fig.update_layout(barmode="relative", title="Benefits, costs and cumulative net cash flow", height=380,
                      xaxis=dict(title="year", dtick=1), yaxis_title="€", bargap=0.45)
    st.plotly_chart(fig, config={"displayModeBar": False})
with right:
    tab = pd.DataFrame({"Benefit": y.benefit, "Total cost": -y.total_cost, "Net": y.net, "Cumulative net": y.cumulative,
                        "NPV to date": y.npv_to_date, "Failures caught": y.failures_caught, "False alarms": y.false_alarms}).T
    tab.columns = [f"Year {c}" for c in tab.columns]
    st.markdown("**Discounted cash-flow table**")
    st.dataframe(tab.style.format("{:,.0f}"))

st.subheader("Maintenance strategy comparison (annual, steady state)")
failures = machines * fpy
reactive = failures * ue["cost_unplanned"]
preventive_share = 0.45  # fixed-interval maintenance prevents ~45 % of failures at the cost of extra planned stops
preventive = failures * (1 - preventive_share) * ue["cost_unplanned"] + machines * 2 * (1.0 * c_h + 400)
caught = failures * rec
predictive = (failures - caught) * ue["cost_unplanned"] + caught * ue["cost_planned"] + caught * (1 / prec - 1) * insp + opex
strat = pd.DataFrame({"strategy": ["L1 Reactive (run-to-failure)", "L2 Preventive (fixed interval, 2×/yr)", "L4 Predictive (this system)"],
                      "annual cost": [reactive, preventive, predictive]})
fig = go.Figure(go.Bar(x=strat["annual cost"], y=strat.strategy, orientation="h",
                       marker_color=["#c3c2b7", "#86b6ef", "#2a78d6"], text=[fmt_eur(v) for v in strat["annual cost"]],
                       textposition="outside", cliponaxis=False))
fig.update_layout(height=230, yaxis=dict(autorange="reversed"), xaxis=dict(title="€ per year", range=[0, strat["annual cost"].max() * 1.25]),
                  bargap=0.4, margin=dict(r=60))
st.plotly_chart(fig, config={"displayModeBar": False})
st.caption("Preventive assumption: two planned service stops per machine and year (1 h downtime + €400 service kit) "
           "avoid ~45 % of failures. Predictive = missed failures at unplanned cost + caught failures at planned cost "
           "+ false-alarm inspections + running cost.")

st.subheader("Sensitivity analysis (±20 %) — what drives the NPV?")
params = ["failures_per_machine_year", "cost_per_downtime_h", "emergency_repair_cost", "recall", "precision",
          "inspection_cost", "capex", "opex_year"]
sens, base = sensitivity(bc, params)
names = {"failures_per_machine_year": "Failure rate", "cost_per_downtime_h": "Downtime cost/h", "emergency_repair_cost": "Emergency repair",
         "recall": "Model sensitivity", "precision": "Model precision", "inspection_cost": "Inspection cost",
         "capex": "CAPEX", "opex_year": "Running cost"}
fig = go.Figure()
fig.add_bar(y=sens.parameter.map(names), x=sens.low, orientation="h", name="−20 %", marker_color="#eb6834")
fig.add_bar(y=sens.parameter.map(names), x=sens.high, orientation="h", name="+20 %", marker_color="#2a78d6")
fig.update_layout(barmode="overlay", height=340, title=f"Tornado: change of NPV vs. base {fmt_eur(base)}", xaxis_title="Δ NPV (€)", bargap=0.35)
st.plotly_chart(fig, config={"displayModeBar": False})
st.caption("Assumptions are illustrative defaults for a mid-sized machining plant and must be validated with plant controlling "
           "before an investment decision (see risk matrix in the paper).")
