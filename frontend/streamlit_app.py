"""Home dashboard (Member 1). Run from the repo root:  streamlit run frontend/streamlit_app.py

Other members add pages in frontend/pages/ (single prediction, batch upload, insights).
Streamlit shows them automatically in the sidebar.
"""
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOG = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"

st.set_page_config(page_title="Term Deposit Call Prioritisation", page_icon="📞", layout="wide")

st.title("Term Deposit Call Prioritisation")
st.caption("IT3051 Fundamentals of Data Mining - Bank Marketing Project")

st.markdown(
    "Telemarketing agents have limited time. This tool estimates **how likely a client is to subscribe to a "
    "term deposit before the call is made**, so the team can start with the most promising clients. "
    "It supports staff decisions - it does not replace them."
)

# ---------------------------------------------------------------- system status
st.subheader("System status")
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health.get("model_loaded"):
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("API", "Online")
        c2.metric("Model", info["model_name"])
        c3.metric("Call threshold", f"{info['threshold']:.2f}")
    else:
        st.warning("API is running but no model is loaded yet.")
except requests.RequestException:
    st.error(f"Prediction API not reachable at {API_URL}. Start it with: uvicorn backend.app:app --reload")

# ---------------------------------------------------------------- model results
st.subheader("Model results")
if LOG.exists() and LOG.stat().st_size > 0:
    log = pd.read_csv(LOG)
    if not log.empty:
        best = log.sort_values("test_pr_auc", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model (test PR-AUC)", str(best["model"]))
        c2.metric("PR-AUC", f"{best['test_pr_auc']:.3f}")
        c3.metric("Recall (yes)", f"{best['recall']:.2f}")
        c4.metric("Precision (yes)", f"{best['precision']:.2f}")
        with st.expander("All logged experiments"):
            st.dataframe(log, width="stretch")
    else:
        st.info("No experiments logged yet.")
else:
    st.info("No experiment results found yet.")

# ---------------------------------------------------------------- how to use
st.subheader("How to use")
c1, c2, c3 = st.columns(3)
c1.markdown("**1. Enter a client**\n\nUse the *Single prediction* page and fill in what is known before the call.")
c2.markdown("**2. Read the result**\n\nYou get a probability, a Call / Do not call recommendation and a priority level.")
c3.markdown("**3. Plan the call list**\n\nUse the *Batch* page to rank many clients and call the top of the list first.")

st.subheader("Good to know")
st.markdown(
    "- The call length (`duration`) is **never used**, because it is only known after the call.\n"
    "- The data comes from a Portuguese bank (2008-2010); results may differ for other banks or years.\n"
    "- A high probability is not consent. Always check contact preferences and opt-outs."
)
