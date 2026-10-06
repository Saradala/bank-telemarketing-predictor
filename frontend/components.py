"""Shared pieces for every frontend page (Member 3).

Keeps the field list, dropdown options and API calls in ONE place, so the Single Prediction
and Batch Upload pages (and Home) stay consistent and never duplicate the same values twice.

Field options are imported directly from backend/schemas.py rather than re-typed here, so the
dropdowns can never silently drift out of sync with what the API actually accepts.
"""
import sys
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so `backend.schemas` can be imported from frontend/

import requests
import streamlit as st

from backend.schemas import ClientFeatures, Contact, Education, Job, Marital, Month, Poutcome, Weekday, YesNoUnknown

# ------------------------------------------------------------------ dropdown options, from the API's own schema
JOB_OPTIONS = list(get_args(Job))
MARITAL_OPTIONS = list(get_args(Marital))
EDUCATION_OPTIONS = list(get_args(Education))
YES_NO_UNKNOWN_OPTIONS = list(get_args(YesNoUnknown))
CONTACT_OPTIONS = list(get_args(Contact))
MONTH_OPTIONS = list(get_args(Month))
WEEKDAY_OPTIONS = list(get_args(Weekday))
POUTCOME_OPTIONS = list(get_args(Poutcome))

REQUIRED_COLUMNS = list(ClientFeatures.model_fields)    # the 19 raw field names the API expects

PRIORITY_COLOR = {"High": "#1baf7a", "Medium": "#eda100", "Low": "#8a8d93"}   # dataviz skill status palette


def api_url() -> str:
    """The backend's base URL - same env var convention as frontend/streamlit_app.py."""
    import os
    return os.getenv("API_URL", "http://127.0.0.1:8000")


def check_api_health():
    """Returns (is_reachable, model_loaded, model_info_dict_or_None)."""
    try:
        health = requests.get(f"{api_url()}/health", timeout=3).json()
    except requests.RequestException:
        return False, False, None
    if not health.get("model_loaded"):
        return True, False, None
    info = requests.get(f"{api_url()}/model-info", timeout=3).json()
    return True, True, info


def priority_badge(priority: str) -> str:
    """A small coloured HTML badge for a priority level, for st.markdown(..., unsafe_allow_html=True)."""
    color = PRIORITY_COLOR.get(priority, "#8a8d93")
    return (f'<span style="background-color:{color}; color:white; padding:2px 10px; '
            f'border-radius:10px; font-size:0.85em; font-weight:600;">{priority}</span>')


def render_client_form(key_prefix: str = "single") -> dict:
    """Renders the 19 pre-call input fields, grouped into sections. Returns a dict of raw values.

    Does NOT include `duration` - it is deliberately never asked for, because it is only known
    after the call ends (see src/features.py and the dataset proposal's leakage discussion).
    """
    values = {}

    st.markdown("**Client profile**")
    c1, c2, c3, c4 = st.columns(4)
    values["age"] = c1.number_input("Age", min_value=17, max_value=100, value=40, key=f"{key_prefix}_age")
    values["job"] = c2.selectbox("Job", JOB_OPTIONS, key=f"{key_prefix}_job")
    values["marital"] = c3.selectbox("Marital status", MARITAL_OPTIONS, key=f"{key_prefix}_marital")
    values["education"] = c4.selectbox("Education", EDUCATION_OPTIONS, key=f"{key_prefix}_education")

    st.markdown("**Credit and loan status**")
    c1, c2, c3 = st.columns(3)
    values["default"] = c1.selectbox("Has credit in default?", YES_NO_UNKNOWN_OPTIONS, key=f"{key_prefix}_default")
    values["housing"] = c2.selectbox("Has a housing loan?", YES_NO_UNKNOWN_OPTIONS, key=f"{key_prefix}_housing")
    values["loan"] = c3.selectbox("Has a personal loan?", YES_NO_UNKNOWN_OPTIONS, key=f"{key_prefix}_loan")

    st.markdown("**Planned contact**")
    c1, c2, c3 = st.columns(3)
    values["contact"] = c1.selectbox("Contact channel", CONTACT_OPTIONS, key=f"{key_prefix}_contact")
    values["month"] = c2.selectbox("Month of contact", MONTH_OPTIONS, index=MONTH_OPTIONS.index("may"), key=f"{key_prefix}_month")
    values["day_of_week"] = c3.selectbox("Day of the week", WEEKDAY_OPTIONS, key=f"{key_prefix}_dow")

    st.markdown("**Campaign history**")
    c1, c2, c3, c4 = st.columns(4)
    values["campaign"] = c1.number_input("Contacts this campaign", min_value=1, max_value=60, value=1,
                                         key=f"{key_prefix}_campaign",
                                         help="Including the call about to be made")
    values["pdays"] = c2.number_input("Days since last contact", min_value=0, max_value=999, value=999,
                                      key=f"{key_prefix}_pdays", help="999 = never contacted before")
    values["previous"] = c3.number_input("Contacts before this campaign", min_value=0, max_value=20, value=0,
                                         key=f"{key_prefix}_previous")
    values["poutcome"] = c4.selectbox("Previous campaign outcome", POUTCOME_OPTIONS, key=f"{key_prefix}_poutcome")

    st.markdown("**Economic context** _(same for every client on a given day - ask your campaign manager)_")
    c1, c2, c3, c4, c5 = st.columns(5)
    values["emp_var_rate"] = c1.number_input("Employment variation rate", min_value=-5.0, max_value=3.0,
                                             value=1.1, step=0.1, format="%.1f", key=f"{key_prefix}_evr")
    values["cons_price_idx"] = c2.number_input("Consumer price index", min_value=90.0, max_value=96.0,
                                               value=93.2, step=0.1, format="%.1f", key=f"{key_prefix}_cpi")
    values["cons_conf_idx"] = c3.number_input("Consumer confidence index", min_value=-60.0, max_value=-20.0,
                                              value=-36.4, step=0.1, format="%.1f", key=f"{key_prefix}_cci")
    values["euribor3m"] = c4.number_input("3-month Euribor rate", min_value=0.0, max_value=6.0,
                                          value=4.9, step=0.1, format="%.1f", key=f"{key_prefix}_euribor")
    values["nr_employed"] = c5.number_input("Number employed (thousands)", min_value=4700.0, max_value=5400.0,
                                            value=5191.0, step=1.0, format="%.1f", key=f"{key_prefix}_nremp")
    return values


def call_predict(client_values: dict) -> dict:
    """POST one client to /predict. Raises requests.HTTPError with a readable message on failure."""
    resp = requests.post(f"{api_url()}/predict", json=client_values, timeout=10)
    resp.raise_for_status()
    return resp.json()


def call_predict_batch(file_bytes: bytes, filename: str) -> dict:
    """POST an uploaded CSV to /predict-batch. Raises requests.HTTPError with a readable message on failure."""
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{api_url()}/predict-batch", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_template_csv() -> bytes:
    """GET the blank CSV template from the API (client_id + the 19 required columns)."""
    resp = requests.get(f"{api_url()}/predict-batch/template", timeout=5)
    resp.raise_for_status()
    return resp.content
