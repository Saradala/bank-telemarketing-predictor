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

import io

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.schemas import COLUMN_MAP, BatchPredictionResponse, ClientFeatures, PredictionResponse
from src.config import FINAL_MODEL_PATH

state = {}
REQUIRED_COLUMNS = list(ClientFeatures.model_fields)   # single source of truth, shared with /predict
MAX_BATCH_ROWS = 5000                                   # a soft cap so one upload cannot hang the server
MAX_ROW_ERRORS_SHOWN = 20                                # avoid a huge error payload on a badly-formed file

# The Next.js frontend (frontend-web/) runs on a different port in development, so the browser
# blocks requests to this API unless it is explicitly allowed. Next.js picks a different port
# (3001, 3002, ...) if 3000 is already taken by another project, so any localhost port is
# allowed in dev rather than hardcoding one. A deployed frontend's real URL would need adding
# as a fixed origin, since this regex only matches localhost/127.0.0.1.
DEV_FRONTEND_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1):\d+"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if Path(FINAL_MODEL_PATH).exists():
        state["bundle"] = joblib.load(FINAL_MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="Bank Term-Deposit Call Prioritisation API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=DEV_FRONTEND_ORIGIN_REGEX, allow_methods=["*"], allow_headers=["*"])


def get_bundle():
    """Return the loaded model bundle, or a 503 if no model has been loaded yet."""
    if "bundle" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m src.export_model ...")
    return state["bundle"]


def map_client(client: ClientFeatures) -> dict:
    """API field names -> training column names (e.g. emp_var_rate -> emp.var.rate)."""
    return {COLUMN_MAP.get(k, k): v for k, v in client.model_dump().items()}


def to_frame(client: ClientFeatures, columns) -> pd.DataFrame:
    """One client -> a one-row DataFrame, in the column order the pipeline expects."""
    return pd.DataFrame([map_client(client)])[list(columns)]


def to_frame_batch(clients: list[ClientFeatures], columns) -> pd.DataFrame:
    """Several clients -> one DataFrame, in the column order the pipeline expects."""
    return pd.DataFrame([map_client(c) for c in clients])[list(columns)]


def priority_for(prob: float, threshold: float) -> str:
    """High at/above the call threshold, Low below half of it, Medium in between."""
    if prob >= threshold:
        return "High"
    return "Medium" if prob >= threshold / 2 else "Low"


@app.get("/health")
def health():
    """Simple liveness/readiness check: is the API up, and is a model loaded?"""
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info")
def model_info():
    """Which model is currently loaded, its call threshold, and the columns it expects."""
    b = get_bundle()
    return {"model_name": b["model_name"], "threshold": b["threshold"], "input_columns": b["input_columns"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientFeatures):
    """Predict the subscription probability for one client and return a Call / Do not call result."""
    b = get_bundle()
    prob = float(b["pipeline"].predict_proba(to_frame(client, b["input_columns"]))[0, 1])
    call = prob >= b["threshold"]
    return PredictionResponse(probability=round(prob, 4), predicted_class="yes" if call else "no",
                              recommendation="Call" if call else "Do not call",
                              priority=priority_for(prob, b["threshold"]),
                              threshold=round(b["threshold"], 4), model_name=b["model_name"])


@app.get("/predict-batch/template")
def predict_batch_template():
    """A blank CSV with the required column headers, for a campaign manager to fill in and upload.

    `client_id` is an optional extra column: not required, not used by the model, but carried
    through into the ranked results unchanged so a row in the output can be matched back to a person.
    """
    header = ",".join(["client_id", *REQUIRED_COLUMNS])
    return Response(content=header + "\n", media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=batch_template.csv"})


def _read_batch_csv(raw_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, accepting either a comma or a semicolon as the separator."""
    return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python")


def _validate_batch_rows(df: pd.DataFrame) -> tuple[list[ClientFeatures], list[dict]]:
    """Validate every row against ClientFeatures (the same rules /predict uses). Returns (clients, errors)."""
    clients, errors = [], []
    for i, row in df.iterrows():
        try:
            clients.append(ClientFeatures(**{c: row[c] for c in REQUIRED_COLUMNS}))
        except ValidationError as exc:
            for err in exc.errors():
                field = ".".join(str(p) for p in err["loc"])
                errors.append({"row": i + 2, "field": field, "message": err["msg"],
                               "value": str(err.get("input", ""))})   # +2: header row, 1-indexed
    return clients, errors


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Upload a CSV of clients, get them back ranked by subscription probability, highest first."""
    b = get_bundle()

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read()
    try:
        df = _read_batch_csv(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the file as a CSV")

    if df.empty:
        raise HTTPException(status_code=422, detail="The uploaded file has no rows")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=422, detail=f"Too many rows ({len(df)}); the limit per upload is {MAX_BATCH_ROWS}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required column(s): {', '.join(missing)}")

    clients, errors = _validate_batch_rows(df)
    if errors:
        raise HTTPException(status_code=422, detail={"message": f"{len(errors)} row(s) failed validation",
                                                      "errors": errors[:MAX_ROW_ERRORS_SHOWN],
                                                      "total_errors": len(errors)})

    extra_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]   # e.g. client_id: passed through, not validated
    proba = b["pipeline"].predict_proba(to_frame_batch(clients, b["input_columns"]))[:, 1]

    ranked = sorted(range(len(clients)), key=lambda i: proba[i], reverse=True)
    results = []
    for rank, i in enumerate(ranked, start=1):
        prob = float(proba[i])
        call = prob >= b["threshold"]
        row = {c: df.iloc[i][c] for c in extra_columns}
        row.update(rank=rank, probability=round(prob, 4), predicted_class="yes" if call else "no",
                   recommendation="Call" if call else "Do not call", priority=priority_for(prob, b["threshold"]))
        results.append(row)

    return BatchPredictionResponse(count=len(results), threshold=round(b["threshold"], 4),
                                   model_name=b["model_name"], results=results)
