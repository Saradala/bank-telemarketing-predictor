"""Home dashboard (Member 1, restyled by Member 3 to match the Figma design).
Run from the repo root:  streamlit run frontend/streamlit_app.py

Other members add pages in frontend/pages/ (single prediction, batch upload, insights).
Streamlit shows them automatically in the sidebar.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st

from frontend.theme import inject_css, status_badge

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOG = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"

st.set_page_config(page_title="Term Deposit Call Prioritisation", page_icon="\U0001F4DE", layout="wide")
inject_css()

st.markdown("OVERVIEW")
st.title("Term Deposit Call Prioritisation")
st.caption("Find the customers most likely to subscribe, so your team can focus its calls.")

# ---------------------------------------------------------------- system status
st.subheader("System status")
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    model_loaded = health.get("model_loaded")
    model_name = threshold = None
    if model_loaded:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        model_name, threshold = info["model_name"], info["threshold"]
except requests.RequestException:
    health, model_loaded, model_name, threshold = None, False, None, None

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.caption("API connection")
        st.markdown(f"### {'Online' if health is not None else 'Offline'}")
        st.caption("Sample connection status" if health is not None else f"Not reachable at {API_URL}")
with c2:
    with st.container(border=True):
        st.caption("Loaded model")
        st.markdown(f"### {model_name or 'Not loaded'}")
        st.caption("Illustrative selected model" if model_name else
                  "Run `python -m src.export_model ...` to load one")
with c3:
    with st.container(border=True):
        st.caption("Current call threshold")
        st.markdown(f"### {f'{threshold:.0%}' if threshold is not None else '—'}")
        st.caption(f"Call when probability is {threshold:.0%} or higher" if threshold is not None
                  else "Available after a model is loaded")

if health is None:
    st.warning(f"Prediction API not reachable at {API_URL}. Start it with: `uvicorn backend.app:app --reload`")
elif not model_loaded:
    st.warning("The API is running, but no model has been loaded yet.")

st.write("")

# ---------------------------------------------------------------- getting started
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("#### \U0001F464 Check one customer")
        st.write("Enter a customer's details and review a clear call recommendation.")
        st.page_link("pages/1_Single_Prediction.py", label="Check one customer →")
with c2:
    with st.container(border=True):
        st.markdown("#### \U0001F4CB Build a prioritised call list")
        st.write("Upload your CSV to rank customers and plan your next round of calls.")
        st.page_link("pages/2_Batch_Upload.py", label="Upload a customer list →")

st.write("")

# ---------------------------------------------------------------- model performance
st.subheader("Model performance")
if LOG.exists() and LOG.stat().st_size > 0:
    log = pd.read_csv(LOG)
    has_test = log["test_pr_auc"].notna()
    if has_test.any():
        best = log[has_test].sort_values("test_pr_auc", ascending=False).iloc[0]
        st.caption(f"Best model so far: {best['model']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.caption("Best model · PR-AUC")
                st.markdown(f"### {best['test_pr_auc']:.2f}")
                st.caption("How well the model finds subscribers across thresholds.")
        with c2:
            with st.container(border=True):
                st.caption("Recall")
                st.markdown(f"### {best['recall']:.0%}")
                st.caption("Share of all subscribers the call recommendation finds.")
        with c3:
            with st.container(border=True):
                st.caption("Precision")
                st.markdown(f"### {best['precision']:.0%}")
                st.caption("Share of recommended customers who subscribe.")
        with st.expander("All logged experiments"):
            st.dataframe(log, width="stretch")
    else:
        st.info("No evaluated results logged yet.")
else:
    st.info("No experiment results found yet.")

st.write("")

# ---------------------------------------------------------------- how to use
with st.container(border=True):
    st.markdown("#### How to use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<span class="step-marker">1</span> **Choose your customers**', unsafe_allow_html=True)
        st.caption("Check one customer or upload a CSV list.")
    with c2:
        st.markdown('<span class="step-marker">2</span> **Review recommendations**', unsafe_allow_html=True)
        st.caption("Start with High priority; check the probability.")
    with c3:
        st.markdown('<span class="step-marker">3</span> **Call responsibly**', unsafe_allow_html=True)
        st.caption("Review opt-outs, then download your call list.")

st.divider()
st.caption("\U0001F6E1️ Decision support only. Check contact preferences and opt-outs before calling.")
