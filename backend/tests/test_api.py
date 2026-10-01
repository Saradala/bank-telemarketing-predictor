"""API tests.

The tests build a tiny, fast pipeline in memory (like tests/test_xgb_model.py) and load it
directly into the app's `state`, instead of needing a real models/final_model.joblib on disk.
This keeps the tests fast and independent of whichever model the team ends up training.
"""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.app as app_module
from src.xgb_model import build_xgb_pipeline, compute_scale_pos_weight

_LEVELS = {"job": ["admin.", "retired", "unknown"], "marital": ["married", "single"],
          "education": ["university.degree", "high.school", "unknown"], "default": ["no", "unknown"],
          "housing": ["yes", "no"], "loan": ["yes", "no"], "contact": ["cellular", "telephone"],
          "month": ["may", "jun", "mar"], "day_of_week": ["mon", "tue", "wed"],
          "poutcome": ["nonexistent", "failure", "success"]}
INPUT_COLUMNS = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
                 "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
                 "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

# maps the API's snake_case client field to the raw training column name and a valid sample value
CLIENT_ROW = {"age": 35, "job": "admin.", "marital": "married", "education": "university.degree",
             "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
             "day_of_week": "mon", "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
             "emp_var_rate": 1.1, "cons_price_idx": 93.2, "cons_conf_idx": -36.4, "euribor3m": 4.9,
             "nr_employed": 5191.0}


def _synthetic_training_data(n=200, seed=0):
    """Small synthetic frame with the 19 raw pre-call columns, for fitting a fast test pipeline."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in _LEVELS.items()})
    df["age"] = rng.integers(18, 90, n)
    df["campaign"] = rng.integers(1, 15, n)
    df["previous"] = np.where(df["poutcome"] == "nonexistent", 0, rng.integers(1, 4, n))
    df["pdays"] = np.where(df["previous"] > 0, rng.integers(0, 20, n), 999)
    for c in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        df[c] = rng.normal(size=n)
    y = pd.Series(((df["poutcome"] == "success") | (rng.random(n) < 0.1)).astype(int))
    return df[INPUT_COLUMNS], y


@pytest.fixture
def client():
    """A TestClient whose app has a small, fast, fitted pipeline loaded - no file on disk needed."""
    X, y = _synthetic_training_data()
    pipeline = build_xgb_pipeline(compute_scale_pos_weight(y), n_estimators=10, max_depth=2).fit(X, y)
    app_module.state["bundle"] = {"pipeline": pipeline, "threshold": 0.3,
                                  "input_columns": INPUT_COLUMNS, "model_name": "XGBoost (test)"}
    yield TestClient(app_module.app)
    app_module.state.clear()


@pytest.fixture
def client_no_model():
    """A TestClient with no model loaded, to test the 503 'not ready' path."""
    app_module.state.clear()
    yield TestClient(app_module.app)


def _csv_bytes(rows: list[dict]) -> bytes:
    """Turn a list of row-dicts into the raw bytes an uploaded CSV file would contain."""
    return pd.DataFrame(rows).to_csv(index=False).encode()


# ---------------------------------------------------------------- /health and /model-info

def test_health_reports_model_loaded_true_when_a_bundle_is_present(client):
    """/health says model_loaded: true once a bundle is in state."""
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_health_reports_model_loaded_false_when_no_bundle(client_no_model):
    """/health still responds (status ok) even with no model loaded, but says model_loaded: false."""
    assert client_no_model.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_model_info_returns_name_threshold_and_columns(client):
    """/model-info echoes back exactly what is in the loaded bundle."""
    body = client.get("/model-info").json()
    assert body["model_name"] == "XGBoost (test)"
    assert body["threshold"] == 0.3
    assert body["input_columns"] == INPUT_COLUMNS


def test_model_info_is_503_when_no_model_loaded(client_no_model):
    """/model-info cannot answer without a loaded model, so it fails clearly instead of crashing."""
    assert client_no_model.get("/model-info").status_code == 503


# ---------------------------------------------------------------- /predict (single)

def test_predict_returns_a_valid_probability_and_recommendation(client):
    """/predict gives a 0-1 probability and a recommendation consistent with the bundle's threshold."""
    body = client.post("/predict", json=CLIENT_ROW).json()
    assert 0 <= body["probability"] <= 1
    assert body["predicted_class"] in ("yes", "no")
    assert body["recommendation"] == ("Call" if body["probability"] >= 0.3 else "Do not call")


def test_predict_rejects_duration_field(client):
    """duration is deliberately not a field on ClientFeatures - sending it must be rejected."""
    bad = {**CLIENT_ROW, "duration": 120}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_an_out_of_range_value(client):
    """Pydantic's field validation (e.g. age >= 17) is enforced, not silently accepted."""
    bad = {**CLIENT_ROW, "age": 5}   # below the allowed minimum of 17
    assert client.post("/predict", json=bad).status_code == 422


# ---------------------------------------------------------------- /predict-batch

def test_predict_batch_template_has_client_id_and_every_required_column(client):
    """The downloadable template has the optional client_id column plus every ClientFeatures field."""
    resp = client.get("/predict-batch/template")
    assert resp.status_code == 200
    header = resp.text.strip().split(",")
    assert header[0] == "client_id"
    assert set(header[1:]) == set(CLIENT_ROW)


def test_predict_batch_ranks_clients_by_probability_descending(client):
    """Results come back sorted highest-probability first, ranked 1..n, with the client_id column preserved."""
    rows = [{"client_id": "a", **CLIENT_ROW}, {"client_id": "b", **CLIENT_ROW, "poutcome": "success", "previous": 2}]
    files = {"file": ("clients.csv", _csv_bytes(rows), "text/csv")}
    body = client.post("/predict-batch", files=files).json()

    assert body["count"] == 2
    probs = [r["probability"] for r in body["results"]]
    assert probs == sorted(probs, reverse=True)                       # ranked, highest first
    assert [r["rank"] for r in body["results"]] == [1, 2]
    assert {r["client_id"] for r in body["results"]} == {"a", "b"}     # the pass-through column survived


def test_predict_batch_rejects_a_non_csv_file(client):
    """A file that is not named .csv is rejected before anything is parsed."""
    files = {"file": ("clients.txt", b"not a csv", "text/plain")}
    assert client.post("/predict-batch", files=files).status_code == 400


def test_predict_batch_rejects_a_missing_required_column(client):
    """A CSV missing a required column (age) is rejected with that column named in the error."""
    rows = [{k: v for k, v in CLIENT_ROW.items() if k != "age"}]     # age column dropped entirely
    files = {"file": ("clients.csv", _csv_bytes(rows), "text/csv")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 422
    assert "age" in resp.json()["detail"]


def test_predict_batch_reports_row_level_validation_errors(client):
    """One bad row is reported with its real CSV row number (accounting for the header row)."""
    rows = [CLIENT_ROW, {**CLIENT_ROW, "age": 200}]                   # 2nd row: age out of range
    files = {"file": ("clients.csv", _csv_bytes(rows), "text/csv")}
    resp = client.post("/predict-batch", files=files)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["total_errors"] == 1
    assert detail["errors"][0]["row"] == 3                            # header (row 1) + valid row (row 2) + bad row


def test_predict_batch_rejects_an_empty_file(client):
    """An empty CSV (no rows) is rejected instead of silently returning an empty ranked list."""
    files = {"file": ("clients.csv", b"", "text/csv")}
    assert client.post("/predict-batch", files=files).status_code in (400, 422)


def test_predict_batch_is_503_when_no_model_loaded(client_no_model):
    """A valid file still cannot be scored if no model is loaded - fails clearly, not silently."""
    files = {"file": ("clients.csv", _csv_bytes([CLIENT_ROW]), "text/csv")}
    assert client_no_model.post("/predict-batch", files=files).status_code == 503
