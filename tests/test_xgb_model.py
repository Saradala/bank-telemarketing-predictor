import numpy as np
import pandas as pd
import pytest

from src.features import CampaignFeatures
from src.lr_model import CATEGORICAL, INPUT_COLUMNS
from src.xgb_model import (build_xgb_pipeline, compute_scale_pos_weight, cv_summary,
                           evaluate_at_threshold)

_LEVELS = {"job": ["admin.", "retired", "unknown"], "marital": ["married", "single"],
           "education": ["university.degree", "high.school", "unknown"], "default": ["no", "unknown"],
           "housing": ["yes", "no"], "loan": ["yes", "no"], "contact": ["cellular", "telephone"],
           "month": ["may", "jun", "mar"], "day_of_week": ["mon", "tue", "wed"],
           "poutcome": ["nonexistent", "failure", "success"]}


def _data(n=300, seed=0):
    """Small synthetic frame with the 19 pre-call columns; subscribing depends on poutcome and campaign."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in _LEVELS.items()})
    df["age"] = rng.integers(18, 90, n)
    df["campaign"] = rng.integers(1, 30, n)
    df["previous"] = np.where(df["poutcome"] == "nonexistent", 0, rng.integers(1, 4, n))
    df["pdays"] = np.where(df["previous"] > 0, rng.integers(0, 20, n), 999)
    for c in ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]:
        df[c] = rng.normal(size=n)
    y = pd.Series(((df["poutcome"] == "success") | (rng.random(n) < 0.1)).astype(int))
    return df[INPUT_COLUMNS], y


def _small_pipeline(y):
    """Tiny, fast pipeline for tests."""
    return build_xgb_pipeline(compute_scale_pos_weight(y), n_estimators=5, max_depth=2)


def test_scale_pos_weight_is_negatives_over_positives():
    """The weight of the 'yes' class is the ratio of negatives to positives."""
    assert compute_scale_pos_weight(pd.Series([0] * 8 + [1] * 2)) == pytest.approx(4.0)


def test_scale_pos_weight_needs_positive_examples():
    """A training set without any subscriber has no sensible weight."""
    with pytest.raises(ValueError):
        compute_scale_pos_weight(pd.Series([0, 0, 0]))


def test_pipeline_starts_with_campaign_features_and_has_no_duration_input():
    """CampaignFeatures is the first step, and the model works with the 19 pre-call columns only."""
    X, y = _data()
    pipe = _small_pipeline(y)
    assert isinstance(pipe.steps[0][1], CampaignFeatures)
    assert "duration" not in X.columns
    pipe.fit(X, y)                                   # would fail if the pipeline needed a duration column


def test_predict_proba_returns_valid_probabilities():
    """One probability per row, between 0 and 1, for the 'yes' class."""
    X, y = _data()
    proba = _small_pipeline(y).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unseen_category_at_prediction_time_does_not_crash():
    """A category never seen in training (e.g. a new job) is ignored by the encoder, not an error."""
    X, y = _data()
    pipe = _small_pipeline(y).fit(X, y)
    new = X.head(3).copy()
    new["job"] = "astronaut"
    assert pipe.predict_proba(new).shape == (3, 2)


def test_search_style_parameter_names_reach_the_classifier():
    """Tuning searches use clf__<name>; the classifier must receive them."""
    X, y = _data()
    pipe = _small_pipeline(y)
    pipe.set_params(clf__max_depth=3, clf__learning_rate=0.2)
    assert pipe.named_steps["clf"].max_depth == 3
    assert pipe.named_steps["clf"].learning_rate == 0.2


def test_cv_summary_returns_scores_between_zero_and_one():
    """cv_summary gives mean and std of PR-AUC and ROC-AUC from stratified folds on the training data."""
    X, y = _data()
    out = cv_summary(_small_pipeline(y), X, y, n_splits=3)
    assert set(out) == {"cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "cv_roc_auc_std"}
    assert 0 < out["cv_pr_auc_mean"] <= 1 and 0 < out["cv_roc_auc_mean"] <= 1


def test_evaluate_at_threshold_on_a_perfect_ranking():
    """A perfect ranking gives PR-AUC = ROC-AUC = 1, and the confusion matrix counts add up."""
    y = np.array([0, 0, 0, 1, 1])
    m = evaluate_at_threshold(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9]), threshold=0.5)
    assert m["pr_auc"] == m["roc_auc"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 0, 0, 2)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0


def test_evaluate_at_threshold_counts_errors():
    """A threshold that lets in one false positive and misses one subscriber is reflected in the counts."""
    y = np.array([0, 0, 1, 1])
    m = evaluate_at_threshold(y, np.array([0.6, 0.1, 0.4, 0.9]), threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
