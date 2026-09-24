"""Export the final model for the API (Member 1).

Saves ONE file, models/final_model.joblib, holding a dict:
    pipeline       - the full fitted pipeline (preprocessing + model), takes RAW pre-call columns
    threshold      - probability cut-off used for the 'Call' recommendation
    input_columns  - the raw columns the pipeline expects (no duration)
    model_name     - label shown by the API

Usage (from the repo root):
    python -m src.export_model --model models/experiments/lr_tuned.joblib --threshold 0.60 --name "Logistic Regression"
    python -m src.export_model --model models/experiments/lr_tuned.joblib --use-budget-threshold
"""
import argparse
from pathlib import Path

import joblib

from src.config import FINAL_MODEL_PATH, MODELS_DIR
from src.lr_model import INPUT_COLUMNS


def export_model(pipeline, threshold, model_name, out_path=FINAL_MODEL_PATH):
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("the model must support predict_proba (needed for the call-priority probability)")
    bundle = {"pipeline": pipeline, "threshold": float(threshold),
              "input_columns": list(INPUT_COLUMNS), "model_name": model_name}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Export the final model bundle for the API.")
    ap.add_argument("--model", required=True, help="path to a fitted pipeline (.joblib)")
    ap.add_argument("--threshold", type=float, help="probability cut-off for 'Call'")
    ap.add_argument("--use-budget-threshold", action="store_true",
                    help="use the 20%% call-budget threshold saved by the LR notebook")
    ap.add_argument("--name", default="Logistic Regression")
    ap.add_argument("--out", default=str(FINAL_MODEL_PATH))
    a = ap.parse_args()

    if a.use_budget_threshold:
        thresholds = joblib.load(MODELS_DIR / "experiments" / "lr_thresholds.joblib")
        thr = thresholds["thr_budget"]
    elif a.threshold is not None:
        thr = a.threshold
    else:
        ap.error("give --threshold or --use-budget-threshold")

    pipe = joblib.load(a.model)
    path = export_model(pipe, thr, a.name, a.out)
    print(f"Saved {path}  (model: {a.name}, threshold: {thr:.3f})")


if __name__ == "__main__":
    main()
