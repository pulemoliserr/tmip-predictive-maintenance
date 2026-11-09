import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm_core import store
from pdm_core.features import rule_checks
from pdm_core.fleet import apply_maintenance
from views.common import scored_fleet, get_model, get_policy, get_shift, get_fleet, status_text, page_header, user

pol, shift, m = get_policy(), get_shift(), get_model()
page_header("📋 Review Queue & Decision Audit",
            "Human-in-the-loop: every alert is reviewed by an engineer. Decisions are written to the SQLite audit store "
            "and confirmed findings update the machine state (closed loop).")

tab_q, tab_log, tab_fb = st.tabs(["Open queue", "Audit log", "Feedback analytics"])

with tab_q:
    f = scored_fleet()
    reviewed = store.reviewed_this_shift(shift)
    inc_watch = st.toggle("Include watch-list machines", value=False)
    q = f[(f.status == "critical") | (inc_watch & (f.status == "warning"))]
    q = q[~q.machine_id.isin(reviewed)].sort_values("priority", ascending=False)
    st.caption(f"Shift {shift} · {len(q)} open item(s) · sorted by priority = risk × line criticality × rule breaches")
    if q.empty:
        st.success("Queue is empty for this shift. Run the next shift in the Control Room.", icon="✅")
    reviewer, role = user()
    for _, r in q.head(12).iterrows():
        checks = rule_checks(r)
        flags = [c["rule"] for c in checks if c["breached"]]
        with st.expander(f"{status_text(r.status)}  ·  **{r.machine_id}**  ·  {r.line}  ·  risk {r.risk:.1%}  ·  priority {r.priority:.3f}"):
            a, b = st.columns([1, 1.3])
            a.markdown(f"**Top SHAP drivers:** {r.drivers}  \n**Rules breached:** {', '.join(flags) or 'none'}  \n"
                       f"**Wear** {r.tool_wear:.0f} min · **Torque** {r.torque:.1f} Nm · **Speed** {r.rpm:.0f} rpm · "
                       f"**ΔT** {r.process_temp - r.air_temp:.1f} K")
            with b.form(f"form_{r.machine_id}_{shift}", border=False):
                dec = st.radio("Decision", store.DECISIONS, horizontal=True, key=f"d_{r.machine_id}")
                fnd = st.selectbox("Finding", store.FINDINGS, key=f"f_{r.machine_id}",
                                   help="Confirmed findings update the machine state (e.g. worn tool replaced → wear = 0)")
                com = st.text_input("Comment", key=f"c_{r.machine_id}")
                if st.form_submit_button("Submit decision", type="primary"):
                    store.log_decision(shift=shift, machine_id=r.machine_id, line=r.line, risk=float(r.risk),
                                       threshold=pol["threshold"], priority=float(r.priority), top_drivers=r.drivers,
                                       rule_flags=json.dumps(flags), decision=dec, finding=fnd, comment=com,
                                       reviewer=reviewer, role=role, model_version=m.version)
                    if fnd.startswith("Confirmed") and dec in ("Inspect now", "Escalate to OEM"):
                        store.save_fleet(apply_maintenance(get_fleet(), r.machine_id, fnd, shift))
                    st.toast(f"Decision for {r.machine_id} logged by {reviewer}.", icon="✅")
                    st.rerun()
    if len(q) > 12:
        st.caption(f"+ {len(q) - 12} more in the queue …")

with tab_log:
    log = store.decisions_df()
    if log.empty:
        st.info("No decisions logged yet.")
    else:
        c1, c2 = st.columns(2)
        dsel = c1.multiselect("Decision", store.DECISIONS, default=store.DECISIONS)
        rsel = c2.multiselect("Reviewer", sorted(log.reviewer.unique()), default=sorted(log.reviewer.unique()))
        v = log[log.decision.isin(dsel) & log.reviewer.isin(rsel)]
        st.dataframe(v, hide_index=True, column_config={"risk": st.column_config.NumberColumn(format="%.3f"),
                                                        "priority": st.column_config.NumberColumn(format="%.3f")})
        st.download_button("Download audit log (CSV)", v.to_csv(index=False).encode(), "pdm_audit_log.csv", "text/csv")
        st.caption("Each record stores model version, threshold, risk, drivers and reviewer — traceable for ISO 55000 / audit purposes.")

with tab_fb:
    log = store.decisions_df()
    if log.empty:
        st.info("Feedback analytics appear once decisions are logged.")
    else:
        conf = log.finding.str.startswith("Confirmed").sum()
        nodef = (log.finding == "No defect found").sum() + (log.decision == "Dismiss – false alarm").sum()
        k = st.columns(3)
        k[0].metric("Decisions logged", len(log), border=True)
        k[1].metric("Confirmed defects", int(conf), border=True)
        k[2].metric("Field precision", f"{conf / max(1, conf + nodef):.0%}", border=True,
                    help="Confirmed / (confirmed + no defect + dismissed) — compare with the model's test precision")
        cnt = log.decision.value_counts().reindex(store.DECISIONS, fill_value=0)
        fig = go.Figure(go.Bar(x=cnt.values, y=cnt.index, orientation="h", marker_color="#2a78d6"))
        fig.update_layout(title="Decisions by type", height=260, yaxis=dict(autorange="reversed"), bargap=0.45)
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption("Reviewed alerts are new labelled data → the next CRISP-DM iteration (retraining on plant data).")
