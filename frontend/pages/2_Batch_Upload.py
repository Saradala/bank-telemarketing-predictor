"""Batch Upload page (Member 3). Upload a CSV of clients, get them back ranked by subscription
probability (highest first) via POST /predict-batch - the real call-list feature for campaign managers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import requests
import streamlit as st

from frontend.components import call_predict_batch, check_api_health, download_template_csv

st.set_page_config(page_title="Batch Upload", page_icon="\U0001F4CB", layout="wide")

st.title("Batch Upload - Call List")
st.caption("Upload a CSV of clients and get back a ranked call list, highest priority first.")

reachable, loaded, info = check_api_health()
if not reachable:
    st.error("Prediction API is not reachable. Start it with: `uvicorn backend.app:app --reload`")
    st.stop()
if not loaded:
    st.warning("The API is running, but no model has been loaded yet "
              "(`python -m src.export_model ...` to export one).")
    st.stop()

st.info(f"Using model: **{info['model_name']}** (call threshold {info['threshold']:.2f})")

with st.expander("Instructions and template", expanded=False):
    st.markdown(
        "- One row = one client. `client_id` is optional, and is not used by the model, "
        "but is kept in the results so you can match a row back to a person.\n"
        "- `duration` is **not** a column - call length is only known after the call, so it is never used.\n"
        "- Up to 5,000 rows per upload."
    )
    try:
        template_bytes = download_template_csv()
        st.download_button("Download CSV template", data=template_bytes, file_name="batch_template.csv",
                           mime="text/csv")
    except requests.RequestException:
        st.caption("(Could not fetch the template from the API right now.)")

uploaded = st.file_uploader("Upload a CSV of clients", type=["csv"])

if uploaded is not None:
    with st.spinner("Scoring every client..."):
        try:
            response = call_predict_batch(uploaded.getvalue(), uploaded.name)
        except requests.HTTPError as exc:
            detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
            if isinstance(detail, dict):
                st.error(detail.get("message", "The file failed validation."))
                if detail.get("errors"):
                    st.dataframe(pd.DataFrame(detail["errors"]), width="stretch")
                if detail.get("total_errors", 0) > len(detail.get("errors", [])):
                    st.caption(f"... and {detail['total_errors'] - len(detail['errors'])} more row(s) with errors.")
            else:
                st.error(f"The API rejected this file: {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
        else:
            results = pd.DataFrame(response["results"])
            st.divider()

            counts = results["priority"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Clients ranked", response["count"])
            c2.metric("High priority", int(counts.get("High", 0)))
            c3.metric("Medium priority", int(counts.get("Medium", 0)))
            c4.metric("Low priority", int(counts.get("Low", 0)))

            priority_filter = st.radio("Filter by priority", ["All", "High", "Medium", "Low"],
                                       horizontal=True)
            shown = results if priority_filter == "All" else results[results["priority"] == priority_filter]

            st.dataframe(
                shown.style.format({"probability": "{:.1%}"}),
                width="stretch", hide_index=True,
            )

            st.download_button("Download ranked list (CSV)", data=shown.to_csv(index=False).encode(),
                               file_name="ranked_call_list.csv", mime="text/csv")
            st.caption(response["note"])
