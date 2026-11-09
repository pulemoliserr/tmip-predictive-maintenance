"""Predictive Maintenance Decision-Support System — Streamlit entry point.

    streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from pdm_core.config import MODEL_PATH, APP_DIR  # noqa: E402

st.set_page_config(page_title="PdM Decision Support", page_icon="🛠️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.7rem; }
  .block-container { padding-top: 2.2rem; }
  div[data-testid="stExpander"] details summary p { font-weight: 600; }
  .pdm-chip { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
              border:1px solid rgba(11,11,11,.12); margin-right:6px; }
</style>""", unsafe_allow_html=True)

# ---- First start: train the model if the artifact is missing ---------------
if not MODEL_PATH.exists():
    with st.spinner("First start: training the model from the CRISP-DM configuration (≈ 2 min) …"):
        r = subprocess.run([sys.executable, "train.py"], cwd=APP_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        st.error("Training failed — see log below."); st.code(r.stderr[-3000:]); st.stop()
    st.rerun()

from views.common import get_model, get_policy, get_shift  # noqa: E402

# ---- Sidebar: identity & context ------------------------------------------
with st.sidebar:
    st.markdown("### 🛠️ PdM Decision Support")
    st.session_state.setdefault("reviewer", "J. Doe")
    st.session_state.setdefault("role", "Maintenance Engineer")
    # widget keys are re-bound on every page so the values survive page switches
    st.session_state.reviewer = st.session_state.reviewer
    st.session_state.role = st.session_state.role
    st.text_input("Your name", key="reviewer")
    st.selectbox("Role", ["Maintenance Engineer", "Plant Manager", "Data Scientist"], key="role")
    m, pol = get_model(), get_policy()
    st.markdown(
        f"<span class='pdm-chip'>Shift {get_shift()}</span>"
        f"<span class='pdm-chip'>Alert ≥ {pol['threshold']:.2f}</span>"
        f"<span class='pdm-chip'>Watch ≥ {pol['watch']:.2f}</span>", unsafe_allow_html=True)
    st.caption(f"Model `{m.version}` · {m.cfg['model_name']} · trained {m.trained_at[:10]}")
    st.caption("Human-in-the-loop: the system recommends, engineers decide. No autonomous shutdown.")

pages = {
    "Operations": [
        st.Page("views/control_room.py", title="Control Room", icon="🏭", default=True),
        st.Page("views/diagnosis.py", title="Machine Diagnosis", icon="🔍"),
        st.Page("views/review_queue.py", title="Review Queue & Audit", icon="📋"),
    ],
    "Management": [
        st.Page("views/policy.py", title="Operating Policy", icon="⚖️"),
        st.Page("views/business_case.py", title="Business Case (ROI)", icon="💶"),
        st.Page("views/tm_tools.py", title="Technology Management", icon="🧭"),
    ],
    "Model": [
        st.Page("views/model_card.py", title="Model Card & Monitoring", icon="🧪"),
    ],
}
st.navigation(pages).run()
