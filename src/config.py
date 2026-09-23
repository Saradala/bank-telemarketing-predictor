"""Shared paths, constants and feature lists."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "bank-additional-full.csv"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
EXPERIMENT_LOG = ROOT / "results" / "experiment_log.csv"

TARGET = "y"
# Known only after the call ends -> leakage; excluded from the deployable model.
LEAKAGE_COLUMNS = ["duration"]

# Fill in during preprocessing.
NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []
