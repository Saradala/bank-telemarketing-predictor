import pandas as pd

from src.lr_model import INPUT_COLUMNS, PdaysFlag, build_lr_pipeline, chronological_split


def _raw(n=60):
    rows = []
    for i in range(n):
        rows.append({"age": 20 + i % 60, "job": ["admin.", "student", "unknown"][i % 3],
                     "marital": ["married", "single"][i % 2], "education": ["high.school", "unknown"][i % 2],
                     "default": "no", "housing": ["yes", "no"][i % 2], "loan": "no", "contact": "cellular",
                     "month": "may", "day_of_week": "mon", "campaign": 1 + i % 5,
                     "pdays": 999 if i % 4 else 5, "previous": 0 if i % 4 else 1,
                     "poutcome": "nonexistent" if i % 4 else "success",
                     "emp.var.rate": 1.1, "cons.price.idx": 93.9, "cons.conf.idx": -36.4,
                     "euribor3m": 4.8, "nr.employed": 5191.0})
    return pd.DataFrame(rows)[INPUT_COLUMNS]


def test_pdays_flag():
    out = PdaysFlag().fit_transform(_raw(8))
    assert out["was_previously_contacted"].tolist() == [1, 0, 0, 0, 1, 0, 0, 0]


def test_chronological_split_keeps_order_and_size():
    X = _raw(100); y = pd.Series(range(100))
    Xtr, Xte, ytr, yte = chronological_split(X, y, 0.2)
    assert len(Xtr) == 80 and len(Xte) == 20
    assert ytr.max() < yte.min()          # test rows come AFTER train rows (no shuffling)


def test_pipeline_fits_and_predicts_probabilities_from_raw_columns():
    X = _raw(80); y = pd.Series([i % 4 == 0 for i in range(80)]).astype(int)
    pipe = build_lr_pipeline().fit(X, y)
    proba = pipe.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2) and ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_unseen_category_in_prediction():
    X = _raw(80); y = pd.Series([i % 4 == 0 for i in range(80)]).astype(int)
    pipe = build_lr_pipeline().fit(X, y)
    new = X.iloc[:1].copy(); new["job"] = "astronaut"
    assert pipe.predict_proba(new).shape == (1, 2)      # handle_unknown='ignore'
