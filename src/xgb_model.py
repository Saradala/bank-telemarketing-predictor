"""XGBoost pipeline for Member 3 (PE2).

Self-contained on purpose (like src/lr_model.py): it reuses the team's data loading and chronological
split, and adds only what is specific to XGBoost.

The pipeline takes the RAW pre-call columns (INPUT_COLUMNS, no `duration`) and does all feature work
inside, so the same object can be saved and served by the API:

    CampaignFeatures -> one-hot encoding of the categorical columns -> XGBClassifier

Trees do not need scaling, so there is no scaler. 'unknown' stays a category of its own (team decision).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE
from src.features import CampaignFeatures
from src.lr_model import CATEGORICAL


def compute_scale_pos_weight(y_train):
    """negatives / positives, computed from the TRAINING labels only (weight of the 'yes' class)."""
    positives = int((y_train == 1).sum())
    if positives == 0:
        raise ValueError("y_train has no positive (subscribed) examples")
    return float((y_train == 0).sum() / positives)


def build_xgb_pipeline(scale_pos_weight, cap_quantile=0.99, **params):
    """Raw columns -> CampaignFeatures -> one-hot encode -> XGBClassifier.

    scale_pos_weight: use compute_scale_pos_weight(y_train). Extra keyword arguments go to XGBClassifier,
    so a tuning search can use the names clf__max_depth, clf__learning_rate, ...
    """
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",          # numeric columns and the CampaignFeatures flags go straight to XGBoost
    )
    clf = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
                        random_state=RANDOM_STATE, n_jobs=-1, **params)
    return Pipeline([("features", CampaignFeatures(cap_quantile=cap_quantile)), ("prep", prep), ("clf", clf)])


def cv_summary(pipeline, X_train, y_train, n_splits=CV_FOLDS):
    """Stratified k-fold CV on the TRAINING part only. Returns mean and std of PR-AUC and ROC-AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, n_jobs=1,
                            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"})
    return {"cv_pr_auc_mean": scores["test_pr_auc"].mean(), "cv_pr_auc_std": scores["test_pr_auc"].std(),
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(), "cv_roc_auc_std": scores["test_roc_auc"].std()}


def evaluate_at_threshold(y_true, proba, threshold=0.5):
    """Ranking metrics (PR-AUC, ROC-AUC) plus precision / recall / F1 and the confusion matrix at a threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"pr_auc": average_precision_score(y_true, proba), "roc_auc": roc_auc_score(y_true, proba),
            "threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
