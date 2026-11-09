import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from views.common import page_header

page_header("🧭 Technology Management toolbox",
            "How the course methodology (assessment, problem-solving, business-case and PM tools) frames the data-mining solution. "
            "Project type: **Applying** — a chosen technology (ML-based predictive maintenance) applied to a specific business problem "
            "(unplanned stops) at a reference company: a typical German precision-machining supplier. Practitioner interviews were "
            "replaced by structured desk research (see tab *Desk research*).")

tabs = st.tabs(["Scoping", "Maturity assessment", "Ishikawa", "Morphological box", "Stakeholders", "Process As-Is / To-Be",
                "SWOT", "Desk research", "Roadmap & risks", "CRISP-DM"])

# ---------------------------------------------------------------- Scoping
with tabs[0]:
    st.markdown("**Problem statement.** Unscheduled machine failures cause unplanned downtime, emergency repair costs, secondary "
                "damage and inefficient technician dispatching. **Objective:** a hybrid, human-in-the-loop decision-support "
                "system that flags high-risk machines early while keeping false alarms manageable.")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("#### ✅ In scope")
        st.markdown("- Binary failure classification on UCI AI4I 2020\n- Tournament: LR, RF, XGBoost, MLP × 7 balancing strategies\n"
                    "- SHAP local explanations\n- Streamlit prototype with review queue & SQLite audit log\n"
                    "- ROI model (avoided breakdowns vs. false-alert inspections)")
    with b.container(border=True):
        st.markdown("#### ⛔ Out of scope")
        st.markdown("- Live PLC / gateway integration and sensor wiring\n- Autonomous machine shutdown\n"
                    "- Remaining-Useful-Life regression\n- Automated workforce & spare-parts scheduling")
    st.caption("Scoping is an ongoing process — harmonised with stakeholders at each milestone gate (scope freeze in W2).")

# ---------------------------------------------------------------- Maturity
with tabs[1]:
    levels = ["1 Beginner", "2 Minded", "3 Ready", "4 Integrated", "5 Excellence"]
    layers = {
        "Maintenance strategy": ["Reactive: repair after failure", "Fixed-interval preventive", "Condition-based limits",
                                 "ML risk scoring, cost-optimal policy", "Prescriptive, self-optimising"],
        "Processes & workflow": ["Ad-hoc call-outs", "Planned service calendar", "Alarm handling procedure",
                                 "Human-in-the-loop review queue, audit trail", "Closed-loop CMMS automation"],
        "Data & IT/OT": ["Paper logs", "Isolated machine counters", "SCADA / historian, basic telemetry",
                         "Integrated sensor + maintenance data pipeline", "Plant-wide digital twin"],
        "Analytics & AI": ["None", "Spreadsheet KPIs", "Static thresholds, dashboards",
                           "Calibrated ML + SHAP explanations", "Continuous learning, prescriptive AI"],
        "People & skills": ["Craft knowledge only", "Planners & schedulers", "Condition-monitoring specialists",
                            "Engineers trained in data-driven decisions", "Data culture across the plant"],
        "Governance & economics": ["No cost tracking", "Maintenance budget", "Downtime KPIs (OEE)",
                                   "Cost-based decisions (Type I/II), ROI tracking", "Value-based asset management"],
    }
    st.markdown("**Maturity matrix (the 'ruler')** — levels × layers of action, filled independently of any specific solution.")
    st.dataframe(pd.DataFrame(layers, index=levels).T, column_config={l: st.column_config.TextColumn(l, width="medium") for l in levels})
    st.markdown("**Assessment** — set the As-Is profile (workshop with plant experts) and the To-Be target.")
    cols = st.columns(len(layers))
    asis_def, tobe_def = [2, 2, 3, 1, 2, 2], [4, 4, 4, 4, 3, 4]
    asis, tobe = [], []
    for c, (lay, a0, t0) in zip(cols, zip(layers, asis_def, tobe_def)):
        c.caption(lay)
        asis.append(c.number_input("As-Is", 1, 5, a0, key=f"as_{lay}"))
        tobe.append(c.number_input("To-Be", 1, 5, t0, key=f"tb_{lay}"))
    theta = list(layers) + [list(layers)[0]]
    fig = go.Figure()
    fig.add_scatterpolar(r=asis + asis[:1], theta=theta, name="As-Is", line=dict(color="#eb6834", width=2.5), fill="toself", fillcolor="rgba(235,104,52,.12)")
    fig.add_scatterpolar(r=tobe + tobe[:1], theta=theta, name="To-Be", line=dict(color="#2a78d6", width=2.5), fill="toself", fillcolor="rgba(42,120,214,.10)")
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 5], tickvals=[1, 2, 3, 4, 5])), height=430, title="Spider diagram: As-Is vs. To-Be")
    st.plotly_chart(fig, config={"displayModeBar": False})
    gaps = [t - a for a, t in zip(asis, tobe)]
    if max(gaps) > 2:
        st.warning("Steps of more than two levels are rarely realistic in one programme — consider an intermediate target.", icon="⚠️")
    st.markdown("**Evolution path:** Level 2/3 → **Level 4 Integrated** via (1) sensor data pipeline, (2) calibrated risk model with SHAP, "
                "(3) review queue & audit trail, (4) cost-based operating policy, (5) training of maintenance engineers.")

# ---------------------------------------------------------------- Ishikawa
with tabs[2]:
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.axis("off"); fig.patch.set_alpha(0)
    ax.annotate("", xy=(13.2, 0), xytext=(-0.5, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#0b0b0b"))
    ax.add_patch(mpatches.FancyBboxPatch((13.25, -0.45), 1.6, 0.9, boxstyle="round,pad=0.05", fc="#eb6834", ec="none"))
    ax.text(14.05, 0, "Unplanned\nmachine\nfailure", ha="center", va="center", color="white", weight="bold", fontsize=9)
    bones = {"Equipment / Technical": (2.3, 1, ["Tool wear accumulation → tool_wear, strain", "Torque spikes → torque", "Speed anomalies → rpm, power"]),
             "Environment / Heat": (8.4, 1, ["Process temperature overload", "Low air-process ΔT → temp_diff", "Ambient fluctuation → air_temp"]),
             "Operation / Human": (2.3, -1, ["High duty-cycle strain (overstrain)", "Product quality variant L/M/H", "Delayed routine inspections"]),
             "Data / Analytics": (8.4, -1, ["No real-time risk visibility", "Uncalibrated thresholds → alert fatigue", "Random failures (not predictable)"])}
    for title, (x, s, causes) in bones.items():
        ax.plot([x, x + 2.3], [s * 2.1, 0], color="#52514e", lw=1.6)
        ax.text(x - 0.1, s * 2.35, title, weight="bold", color="#2a78d6", fontsize=10)
        for j, c in enumerate(causes):
            t = (j + 1) / 4; px, py = x + 2.3 * t, s * 2.1 * (1 - t)
            ax.plot([px - 1.0, px], [py, py], color="#e1e0d9", lw=1); ax.text(px - 1.05, py, c, ha="right", va="center", fontsize=8)
    ax.set_xlim(-2.6, 15); ax.set_ylim(-2.7, 2.7)
    st.pyplot(fig)
    st.caption("Each technical cause category became an engineered model feature (ΔT, power, strain) — "
               "the Ishikawa analysis directly improved the model (see notebook §3.2).")

# ---------------------------------------------------------------- Morphological box
with tabs[3]:
    box = [("Data source", ["Real sensor gateway", "UCI AI4I benchmark", "Synthetic stream generator"], 1),
           ("Predictive model", ["Logistic regression", "Random Forest (selected) / XGBoost", "Neural network (MLP)"], 1),
           ("Explainability", ["Global feature importance", "SHAP local drivers", "LIME"], 1),
           ("User interface", ["Static PDF reports", "Streamlit web app", "Power BI dashboard"], 1),
           ("Decision rule", ["Fixed 0.5 threshold", "Cost-sensitive calibrated policy", "Dynamic adaptive threshold"], 1),
           ("Imbalance handling", ["None (selected by tournament)", "Class weights / over- / undersampling", "SMOTE / ADASYN / ROSE"], 0),
           ("Outlier treatment", ["Remove + impute", "3σ clipping (selected)", "Do nothing"], 1),
           ("Audit store", ["CSV files", "SQLite", "Enterprise CMMS"], 1)]
    html = "<table style='width:100%;border-collapse:separate;border-spacing:4px'><tr><th style='text-align:left'>Dimension</th>" + \
           "".join(f"<th style='text-align:left'>Option {c}</th>" for c in "ABC") + "</tr>"
    for dim, opts, sel in box:
        html += f"<tr><td style='font-weight:600;padding:6px'>{dim}</td>"
        for i, o in enumerate(opts):
            sty = "background:#2a78d6;color:white;font-weight:600" if i == sel else "background:rgba(127,127,127,.08)"
            html += f"<td style='{sty};padding:8px;border-radius:6px'>{'✔ ' if i == sel else ''}{o}</td>"
        html += "</tr>"
    st.markdown(html + "</table>", unsafe_allow_html=True)
    st.caption("Highlighted = selected configuration. The data-mining dimensions (imbalance, outliers) were decided by the leakage-free tournament and grid search (notebook §4.4–4.7).")

# ---------------------------------------------------------------- Stakeholders
with tabs[4]:
    sh = pd.DataFrame([
        ("Plant manager", 0.85, 0.80, "Sponsor: budget, ROI, policy"), ("Maintenance engineers", 0.75, 0.55, "Primary users: review alerts"),
        ("Machine operators", 0.45, 0.35, "Data & feedback on site"), ("IT/OT team", 0.70, 0.40, "Data pipeline, security"),
        ("Works council", 0.70, 0.25, "Monitoring of employees, acceptance"), ("CFO / controlling", 0.65, 0.60, "Business case validation"),
        ("Health & safety officer", 0.55, 0.70, "No autonomous shutdown"), ("Machine OEM", 0.35, 0.45, "Escalations, warranty"),
        ("Data science team", 0.40, 0.90, "Model, monitoring, retraining")], columns=["stakeholder", "impact", "commitment", "role"])
    fig = go.Figure()
    for x0, x1, y0, y1, lab in [(0, .5, .5, 1, "Risk! Explain & mobilise"), (.5, 1, .5, 1, "Ensure sustainable sponsorship"),
                                (0, .5, 0, .5, "Train & involve where necessary"), (.5, 1, 0, .5, "Keep informed, communication network")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(color="#e1e0d9"), fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=(x0 + x1) / 2, y=y1 - 0.04, text=lab, showarrow=False, font=dict(size=11, color="#898781"))
    fig.add_scatter(x=sh.commitment, y=sh.impact, mode="markers+text", text=sh.stakeholder, textposition="bottom center",
                    marker=dict(size=14, color="#2a78d6", line=dict(color="white", width=2)), customdata=sh.role,
                    hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>")
    fig.update_layout(height=480, xaxis=dict(title="commitment →", range=[0, 1], showgrid=False, showticklabels=False),
                      yaxis=dict(title="impact →", range=[0, 1], showgrid=False, showticklabels=False), title="Stakeholder map (impact × commitment)")
    st.plotly_chart(fig, config={"displayModeBar": False})
    st.caption("Communication plan: works council and IT/OT sit in the risk quadrant → early involvement (data-protection agreement, "
               "no individual performance monitoring; OT security review). Since the 2021 Betriebsrätemodernisierungsgesetz the employer must "
               "inform the works council in good time about planned AI use (§ 90(1) no. 3 BetrVG) and the council may call in an AI expert "
               "(§ 80(3) BetrVG) — both are planned and budgeted.")

# ---------------------------------------------------------------- Process
with tabs[5]:
    a, b = st.columns(2)
    with a:
        st.markdown("#### As-Is — reactive / fixed interval")
        st.graphviz_chart("""digraph{rankdir=TB; node[shape=box,style="rounded,filled",fillcolor="#f3f2ee",color="#c3c2b7",fontname="Helvetica",fontsize=11];
        a[label="Machine runs"]; b[label="Unexpected breakdown",fillcolor="#f9e2e2"]; c[label="Operator calls maintenance"];
        d[label="Emergency diagnosis\\n(waiting time)",fillcolor="#fdf3dc"]; e[label="Spare parts?\\n(waiting time)",fillcolor="#fdf3dc"];
        f[label="Emergency repair (overtime)"]; g[label="Restart, scrap & rework"]; a->b->c->d->e->f->g; g->a[style=dashed];}""")
    with b:
        st.markdown("#### To-Be — predictive, human-in-the-loop")
        st.graphviz_chart("""digraph{rankdir=TB; node[shape=box,style="rounded,filled",fillcolor="#eef4fc",color="#86b6ef",fontname="Helvetica",fontsize=11];
        a[label="Sensor telemetry each shift"]; b[label="Calibrated risk score + SHAP"]; c[label="Risk ≥ cost-optimal threshold?",shape=diamond,fillcolor="white"];
        d[label="Review queue (priority)"]; e[label="Engineer decision\\n(inspect / schedule / dismiss)"]; f[label="Planned repair at next stop",fillcolor="#e6f5ec"];
        g[label="Audit log (SQLite)"]; h[label="Retraining with confirmed labels"];
        a->b->c; c->d[label="yes"]; c->a[label="no",style=dashed]; d->e->f; e->g->h; h->b[style=dashed];}""")
    st.caption("Saving potential: waiting times for diagnosis and parts disappear, repair becomes planned, every decision is traceable.")

# ---------------------------------------------------------------- SWOT
with tabs[6]:
    sw = {"Strengths (internal)": ["Explainable, calibrated risk scores", "Cost-based policy links ML to € impact",
                                   "Human-in-the-loop keeps accountability", "Low-cost open-source stack"],
          "Weaknesses (internal)": ["Trained on synthetic benchmark data", "Random / tool-wear failures hard to predict",
                                    "No live PLC integration yet", "Small team, 10-week horizon"],
          "Opportunities (external)": ["IoT retrofits getting cheaper", "Labelled feedback from review queue",
                                       "Extension to RUL & spare-parts planning", "Industry 4.0 funding programmes"],
          "Threats (external)": ["Alert fatigue → users ignore system", "Works-council / data-protection concerns",
                                 "OT cyber-security requirements", "Concept drift after process changes"]}
    cols = st.columns(2)
    for i, (k, v) in enumerate(sw.items()):
        with cols[i % 2].container(border=True):
            st.markdown(f"#### {k}"); st.markdown("\n".join(f"- {x}" for x in v))

# ---------------------------------------------------------------- Desk research
with tabs[7]:
    st.markdown("**Structured desk research (replaces practitioner interviews and survey).** Each question from the planned interview "
                "guide was answered from at least two independent sources where possible — legal texts and peer-reviewed studies first, "
                "then association surveys, then consultancy studies — and given a confidence rating.")
    desk = pd.DataFrame([
        ("How is maintenance done today?", "25 % visual inspection, 42 % instrument inspection, 20 % real-time condition monitoring, 11 % PdM 4.0 "
         "(268 firms BE/DE/NL); Germany 9 % at PdM 4.0", "PwC & Mainnovation 2018; Bitkom 2025", "Medium", "Level 2–3 baseline is typical"),
        ("How often / how costly are stops?", "27 h and 25 incidents per month in large plants; poor maintenance cuts capacity 5–20 %",
         "Siemens 2024; Deloitte 2017", "Medium", "Value per stop is firm-specific → cost scenarios"),
        ("What causes failures?", "18 % age-related, 82 % random failure patterns", "ARC Advisory Group 2015 (NASA, US Navy data)",
         "Medium", "Fixed intervals miss most failures; condition-based detection needed"),
        ("Which data are available?", "Data availability is the #1 success factor (67 %); ~20–30 recorded cases needed per failure type",
         "PwC & Mainnovation 2018; Deloitte 2017", "High", "Start with controller data and disciplined failure logging"),
        ("What blocks adoption?", "No business case 63 %, no data 23 %, no analytics capability 8 %; 42 % lack AI expertise",
         "PwC & Mainnovation 2018; Bitkom 2025", "High", "Transparent business case and small pilot matter most"),
        ("Will staff trust AI advice?", "Explanations demanded; positive attitude to XAI ≠ acceptance", "Heider et al. 2026; Zhao & Turkay 2023",
         "Low–medium", "SHAP + rule checks + human decision, co-designed with engineers"),
        ("Works council role?", "Co-determination § 87(1) no. 6; inform about AI § 90(1) no. 3; AI expert § 80(3) BetrVG", "BetrVG; "
         "Betriebsrätemodernisierungsgesetz 2021", "High", "Inform early, budget expert, no individual monitoring"),
        ("Expected benefits?", "Downtime −30–50 %, life +20–40 %; uptime +10–20 %, cost −5–10 %; adopters: uptime +9 %, cost −12 %",
         "McKinsey 2017; Deloitte 2017; PwC & Mainnovation 2018", "Medium", "Plant-level gains far below test-set failure-cost reduction"),
        ("False alarm vs. missed failure?", "10 % false-positive rate erased > $1 M of projected savings", "McKinsey 2021", "Medium",
         "Threshold must follow the cost ratio → Operating Policy page"),
    ], columns=["Question (from interview guide)", "Evidence", "Sources", "Confidence", "Implication for the reference company"])
    st.dataframe(desk, hide_index=True, column_config={
        "Evidence": st.column_config.TextColumn(width="large"),
        "Implication for the reference company": st.column_config.TextColumn(width="large")})
    st.caption("Limits: several surveys date from 2017–2018, cover large firms or the Benelux and are not specific to machining SMEs. "
               "Firm-specific values (failure rate, cost per stop) are therefore treated as scenarios and measured in the pilot.")

# ---------------------------------------------------------------- Roadmap & risks
with tabs[8]:
    plan = pd.DataFrame([
        ("Scope framing & research", 1, 2, "Project"), ("Data audit & pipeline", 3, 4, "Project"), ("Model training & benchmark", 5, 6, "Project"),
        ("Explainability & dashboard", 7, 8, "Project"), ("Business case & handover", 9, 10, "Project"),
        ("Pilot: shadow mode on one line", 11, 18, "Deployment"), ("Assisted operation + review queue", 19, 30, "Deployment"),
        ("Scale-up & CMMS integration", 31, 44, "Deployment")], columns=["task", "start", "end", "phase"])
    fig = go.Figure()
    for _, r in plan.iterrows():
        fig.add_bar(y=[r.task], x=[r.end - r.start + 1], base=[r.start - 1], orientation="h",
                    marker_color="#2a78d6" if r.phase == "Project" else "#1baf7a", showlegend=False,
                    hovertemplate=f"{r.task}<br>week {r.start}–{r.end}<extra></extra>")
    for w, lab in [(2, "Scope freeze"), (4, "Data readiness"), (6, "Model shortlist"), (8, "Operating policy"), (10, "Final handover")]:
        fig.add_vline(x=w, line_dash="dot", line_color="#898781")
        fig.add_annotation(x=w, y=1.02, yref="paper", text=lab, showarrow=False, font=dict(size=10), textangle=-30)
    fig.update_layout(height=380, yaxis=dict(autorange="reversed"), xaxis=dict(title="week", dtick=4), bargap=0.35,
                      title="Work plan (blue: 10-week course project) and deployment roadmap (green)", margin=dict(t=90))
    st.plotly_chart(fig, config={"displayModeBar": False})
    risks = pd.DataFrame([
        ("Class imbalance (3.4 % failures)", 5, 4, "7 balancing strategies tested in CV; PR-AUC & cost metrics"),
        ("Data leakage via failure-mode flags", 3, 5, "Flags & IDs excluded; locked test set with hash check"),
        ("Alert fatigue", 4, 4, "Cost-optimal threshold, precision KPI, watch list"),
        ("Synthetic data — external validity", 4, 4, "Shadow-mode pilot on plant data before go-live"),
        ("Scope creep (10 weeks)", 3, 3, "Scope freeze gate, out-of-scope list"),
        ("User acceptance / works council", 3, 4, "Early involvement, no personal monitoring"),
        ("Concept drift", 3, 3, "PSI drift monitor, retraining trigger"),
        ("Desk evidence not specific to machining SMEs", 3, 3, "≥ 2 independent sources per question, confidence ratings, cost scenarios")], columns=["risk", "probability", "impact", "mitigation"])
    fig = go.Figure()
    for x0, y0, col in [(0.5, 0.5, "rgba(12,163,12,.07)"), (3, 3, "rgba(250,178,25,.12)"), (4, 4, "rgba(208,59,59,.10)")]:
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=5.5, y1=5.5, line_width=0, fillcolor=col, layer="below")
    fig.add_scatter(x=risks.probability, y=risks.impact, mode="markers+text", text=[f"R{i+1}" for i in range(len(risks))],
                    textposition="middle right", marker=dict(size=14, color="#eb6834", line=dict(color="white", width=2)),
                    customdata=risks[["risk", "mitigation"]], hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>")
    fig.update_layout(height=360, title="Risk matrix", xaxis=dict(title="probability", range=[0.5, 5.5], dtick=1),
                      yaxis=dict(title="impact", range=[0.5, 5.5], dtick=1))
    a, b = st.columns([1, 1.2])
    a.plotly_chart(fig, config={"displayModeBar": False})
    risks.index = [f"R{i+1}" for i in range(len(risks))]
    b.dataframe(risks, column_config={"mitigation": st.column_config.TextColumn(width="large")})

# ---------------------------------------------------------------- CRISP-DM
with tabs[9]:
    st.graphviz_chart("""digraph{rankdir=LR; node[shape=box,style="rounded,filled",fillcolor="#2a78d6",fontcolor=white,color=white,fontname="Helvetica",fontsize=11];
    BU[label="1 Business\\nUnderstanding"]; DU[label="2 Data\\nUnderstanding"]; DP[label="3 Data\\nPreparation"]; M[label="4 Modeling"];
    E[label="5 Evaluation",fillcolor="#eb6834"]; D[label="6 Deployment\\n(this app)",fillcolor="#1baf7a"];
    BU->DU->DP->M->E->D; DU->BU[style=dashed,color="#eb6834"]; M->DP[style=dashed,color="#eb6834",label="grid search"]; E->BU[style=dashed,color="#eb6834"];
    D->BU[style=dashed,color="#1baf7a",label="review-queue labels"];}""")
    st.markdown("| Phase | Where in this project |\n|---|---|\n"
                "| Business understanding | Proposal, scoping, Ishikawa, maturity, cost scheme (Type I 100 € / Type II 400 €) |\n"
                "| Data understanding | Notebook §2: imbalance 3.39 %, no missing values, outliers = failure signal |\n"
                "| Data preparation | Notebook §3: leakage control, stratified split, 3σ clipping, physics-feature ablation (no imputation, no Boruta needed) |\n"
                "| Modeling | Notebook §4: leakage-free tournament (27 configs), grid search, cost-optimal threshold, calibration |\n"
                "| Evaluation | Notebook §5: locked test set, cost vs. maturity-level strategies, SHAP |\n"
                "| Deployment | This Streamlit app: control room, review queue, audit store, policy, ROI, monitoring |")
