"""XGBoost pipeline for Member 3 (PE2).

Self-contained on purpose (like src/lr_model.py): it reuses the team's data loading and chronological
split, and adds only what is specific to XGBoost.

The pipeline takes the RAW pre-call columns (INPUT_COLUMNS, no `duration`) and does all feature work
inside, so the same object can be saved and served by the API:

    CampaignFeatures -> one-hot encoding of the categorical columns -> XGBClassifier

Trees do not need scaling, so there is no scaler. 'unknown' stays a category of its own (team decision).
"""
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.config import CV_FOLDS, RANDOM_STATE
from src.features import CampaignFeatures
from src.lr_model import CATEGORICAL

# Search space for hyperparameter tuning (RandomizedSearchCV and Optuna both use this range,
# so the two search methods are compared fairly). Keys use the clf__ prefix so they reach the
# XGBClassifier step inside the pipeline built by build_xgb_pipeline().
SEARCH_SPACE = {
    "clf__n_estimators": randint(200, 1001),
    "clf__learning_rate": loguniform(0.01, 0.3),
    "clf__max_depth": randint(3, 11),
    "clf__min_child_weight": randint(1, 11),
    "clf__subsample": uniform(0.6, 0.4),          # sampled range: 0.6 to 1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),   # sampled range: 0.6 to 1.0
    "clf__gamma": uniform(0, 5),
    "clf__reg_lambda": loguniform(0.1, 10),
}


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


def random_search_xgb(X_train, y_train, scale_pos_weight, n_iter=40, cap_quantile=0.99,
                      search_space=SEARCH_SPACE, n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    """Search SEARCH_SPACE with RandomizedSearchCV, scored on PR-AUC, using stratified CV on TRAINING data.

    Tries n_iter random combinations (default 40, matching the team's Optuna budget so the two search
    methods can be compared fairly). Returns the fitted RandomizedSearchCV object: .best_params_,
    .best_score_ (mean CV PR-AUC of the best combination) and .best_estimator_ (refit on all of X_train).
    """
    pipeline = build_xgb_pipeline(scale_pos_weight, cap_quantile=cap_quantile)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(pipeline, search_space, n_iter=n_iter, cv=cv, scoring="average_precision",
                                random_state=random_state, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search


def search_space_bounds(search_space=SEARCH_SPACE):
    """Read (low, high, integer, log_scale) out of each scipy distribution in SEARCH_SPACE.

    Used to build the Optuna objective from the SAME numbers as RandomizedSearchCV, so the two
    search methods sample from identical ranges and the comparison between them is fair.
    """
    bounds = {}
    for name, dist in search_space.items():
        param = name.removeprefix("clf__")
        a, b = dist.args
        kind = dist.dist.__class__.__name__
        if kind == "randint_gen":                     # scipy randint(low, high): high is EXCLUSIVE
            bounds[param] = {"low": a, "high": b - 1, "int": True, "log": False}
        elif kind == "uniform_gen":                    # scipy uniform(loc, scale): range is [loc, loc + scale]
            bounds[param] = {"low": a, "high": a + b, "int": False, "log": False}
        elif kind == "reciprocal_gen":                  # scipy loguniform(a, b): range is [a, b], sampled on a log scale
            bounds[param] = {"low": a, "high": b, "int": False, "log": True}
        else:
            raise ValueError(f"Unknown distribution type for {name}: {kind}")
    return bounds


def optuna_search_xgb(X_train, y_train, scale_pos_weight, n_trials=40, cap_quantile=0.99,
                      search_space=SEARCH_SPACE, n_splits=CV_FOLDS, random_state=RANDOM_STATE):
    """Search the SAME SEARCH_SPACE with Optuna's TPE sampler, scored on PR-AUC, stratified CV on TRAINING data.

    Tries n_trials combinations (default 40, matching random_search_xgb's budget). Unlike
    RandomizedSearchCV, Optuna picks each new trial using what it learned from earlier trials.
    Returns the fitted optuna.Study: .best_value (mean CV PR-AUC) and .best_params.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep 40 trials of output out of notebooks

    bounds = search_space_bounds(search_space)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        """One Optuna trial: sample one set of hyperparameters and return its mean CV PR-AUC."""
        params = {}
        for param, b in bounds.items():
            if b["int"]:
                params[param] = trial.suggest_int(param, b["low"], b["high"])
            else:
                params[param] = trial.suggest_float(param, b["low"], b["high"], log=b["log"])
        pipeline = build_xgb_pipeline(scale_pos_weight, cap_quantile=cap_quantile, **params)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return study


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
