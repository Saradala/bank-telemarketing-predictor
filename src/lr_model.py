"""Logistic Regression pipeline for Member 1 (PE2).

This file is self-contained on purpose: it lets Member 1 train and tune Logistic Regression
without waiting for the team's shared split/preprocessing. When the shared pipeline is ready,
replace `load_data`, `chronological_split` and `build_lr_pipeline` with the team versions.

The pipeline takes RAW pre-call columns (INPUT_COLUMNS) and does all feature work inside,
so the same object can be saved and served by the API.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.client_features import AgeGrouper, UnknownHandler
from src.config import LEAKAGE_COLUMNS, RANDOM_STATE, RAW_DATA, TARGET
from src.data_cleaning import remove_duplicates


# raw columns available BEFORE a call (duration is excluded - it is leakage)
INPUT_COLUMNS = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
                 "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
                 "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact",
               "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "previous", "was_previously_contacted",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
NO_PREVIOUS_CONTACT = 999   # pdays sentinel: client was not contacted before


class PdaysFlag(BaseEstimator, TransformerMixin):
    """pdays == 999 means 'never contacted before' - turn it into a 0/1 flag instead of a day count."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["was_previously_contacted"] = (X["pdays"] != NO_PREVIOUS_CONTACT).astype(int)
        return X


def load_data():
    """Read the raw CSV, remove duplicates (before dropping duration, per team's data_cleaning.py),
    then drop leakage columns. Returns X (raw columns), y (0/1)."""
    with open(RAW_DATA, encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    df = pd.read_csv(RAW_DATA, sep=sep)
    df = remove_duplicates(df)                            # dedup on ALL columns first (matches proposal: 12 dupes)
    df = df.drop(columns=LEAKAGE_COLUMNS, errors="ignore")  # THEN drop duration (leakage)
    y = (df[TARGET] == "yes").astype(int)
    return df[INPUT_COLUMNS], y

def chronological_split(X, y, test_fraction=0.20):
    """Rows are ordered by date, so the LAST rows are the test set (no shuffling)."""
    cut = int(len(X) * (1 - test_fraction))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def build_lr_pipeline(C=1.0, l1_ratio=0.0, class_weight="balanced",
                      unknown_strategy="category", use_age_group=True):
    """Raw columns -> pdays flag -> unknown handling -> age group -> encode + scale -> LogisticRegression.

    l1_ratio=0 means L2 penalty (Ridge-style), l1_ratio=1 means L1 penalty (Lasso-style, can zero out features).
    """
    cat = CATEGORICAL + (["age_group"] if use_age_group else [])
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), NUMERIC),
    ])
    clf = LogisticRegression(C=C, l1_ratio=l1_ratio, solver="liblinear", class_weight=class_weight,
                             max_iter=1000, random_state=RANDOM_STATE)
    return Pipeline([
        ("pdays", PdaysFlag()),
        ("unknown", UnknownHandler(strategy=unknown_strategy)),
        ("age", AgeGrouper()),
        ("prep", prep),
        ("clf", clf),
    ])
