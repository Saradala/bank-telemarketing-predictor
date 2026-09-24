"""FastAPI service: loads models/final_model.joblib and serves predictions.

Run from the repo root:   uvicorn backend.app:app --reload
Docs (auto-generated):    http://127.0.0.1:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # so the pickled pipeline can import `src.*`

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.schemas import COLUMN_MAP, ClientFeatures, PredictionResponse
from src.config import FINAL_MODEL_PATH

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path(FINAL_MODEL_PATH).exists():
        state["bundle"] = joblib.load(FINAL_MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Bank Term-Deposit Call Prioritisation API", version="1.0", lifespan=lifespan)


def get_bundle():
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m src.export_model ...")
    return state["bundle"]


def to_frame(client: ClientFeatures, columns) -> pd.DataFrame:
    """API field names -> training column names, in the order the pipeline expects."""
    row = {COLUMN_MAP.get(k, k): v for k, v in client.model_dump().items()}
    return pd.DataFrame([row])[list(columns)]


def priority_for(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "High"
    return "Medium" if prob >= threshold / 2 else "Low"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info")
def model_info():
    b = get_bundle()
    return {"model_name": b["model_name"], "threshold": b["threshold"], "input_columns": b["input_columns"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    b = get_bundle()
    prob = float(b["pipeline"].predict_proba(to_frame(client, b["input_columns"]))[0, 1])
    call = prob >= b["threshold"]
    return PredictionResponse(probability=round(prob, 4), predicted_class="yes" if call else "no",
                              recommendation="Call" if call else "Do not call",
                              priority=priority_for(prob, b["threshold"]),
                              threshold=round(b["threshold"], 4), model_name=b["model_name"])
