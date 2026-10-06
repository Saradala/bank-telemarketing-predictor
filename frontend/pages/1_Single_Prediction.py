"""Single Prediction page (Member 3). Enter one client's pre-call details, get a Call / Do not
call recommendation back from the live model via POST /predict.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

from frontend.components import call_predict, check_api_health, priority_badge, render_client_form

st.set_page_config(page_title="Single Prediction", page_icon="\U0001F4DE", layout="wide")

st.title("Single Prediction")
st.caption("Enter one client's details to see their estimated subscription probability.")

reachable, loaded, info = check_api_health()
if not reachable:
    st.error("Prediction API is not reachable. Start it with: `uvicorn backend.app:app --reload`")
    st.stop()
if not loaded:
    st.warning("The API is running, but no model has been loaded yet "
              "(`python -m src.export_model ...` to export one).")
    st.stop()

st.info(f"Using model: **{info['model_name']}** (call threshold {info['threshold']:.2f})")

with st.form("single_prediction_form"):
    values = render_client_form(key_prefix="single")
    submitted = st.form_submit_button("Predict", type="primary", width="stretch")

if submitted:
    with st.spinner("Scoring..."):
        try:
            result = call_predict(values)
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
            st.error(f"The API rejected this request: {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Subscription probability", f"{result['probability']:.1%}")
            c2.metric("Predicted class", result["predicted_class"].upper())
            c3.metric("Recommendation", result["recommendation"])
            with c4:
                st.markdown("**Priority**")
                st.markdown(priority_badge(result["priority"]), unsafe_allow_html=True)
            st.caption(result["note"])

            if result["recommendation"] == "Call":
                st.success("This client is worth calling, based on the model's estimate.")
            else:
                st.info("This client is a lower priority for a call right now.")
