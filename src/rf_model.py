"""Random Forest pipelines for Member 2 (PE2).

This module builds leakage-safe Random Forest pipelines for comparing:
1. No class-imbalance handling
2. Class-weight handling
3. SMOTE oversampling

SMOTE and preprocessing are included inside the pipeline so that they are
applied only to the training folds during cross-validation.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from src.client_features import AgeGrouper, UnknownHandler
from src.config import RANDOM_STATE
from src.lr_model import (
    CATEGORICAL,
    NUMERIC,
    PdaysFlag,
    chronological_split,
    load_data,
)


def build_rf_pipeline(
    imbalance_method: str = "none",
    n_estimators: int = 300,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
):
    """Build a leakage-safe Random Forest pipeline.

    Parameters
    ----------
    imbalance_method:
        "none" uses the original class distribution.
        "class_weight" uses balanced_subsample class weights.
        "smote" applies SMOTE only inside the training pipeline.
    """

    valid_methods = {"none", "class_weight", "smote"}

    if imbalance_method not in valid_methods:
        raise ValueError(
            "imbalance_method must be 'none', 'class_weight', or 'smote'."
        )

    categorical_features = CATEGORICAL + ["age_group"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC,
            ),
        ]
    )

    class_weight = (
        "balanced_subsample"
        if imbalance_method == "class_weight"
        else None
    )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline_steps = [
        ("pdays", PdaysFlag()),
        ("unknown", UnknownHandler(strategy="category")),
        ("age", AgeGrouper()),
        ("preprocessor", preprocessor),
    ]

    if imbalance_method == "smote":
        pipeline_steps.append(
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE),
            )
        )

    pipeline_steps.append(("classifier", classifier))

    return Pipeline(pipeline_steps)