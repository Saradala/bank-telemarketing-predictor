"""Reusable preprocessing utilities for numerical features.(Member 4)"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MACRO_FEATURES = [
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


def find_high_correlations(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Return feature pairs whose absolute correlation exceeds the threshold."""
    selected_data = data[columns] if columns is not None else data.select_dtypes(
        include=np.number
    )

    correlation_matrix = selected_data.corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    correlation_pairs = upper_triangle.stack().reset_index()
    correlation_pairs.columns = ["feature_1", "feature_2", "correlation"]

    return (
        correlation_pairs[
            correlation_pairs["correlation"] >= threshold
        ]
        .sort_values("correlation", ascending=False)
        .reset_index(drop=True)
    )


def build_numeric_pipeline(
    scale: bool = True,
    k_best: Union[int, str] = "all",
) -> Pipeline:
    """
    Build a leakage-safe numerical preprocessing pipeline.

    The pipeline imputes missing numerical values, optionally scales features,
    and optionally applies mutual-information feature selection. It must be
    fitted only on training data or inside cross-validation.
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale:
        steps.append(("scaler", StandardScaler()))

    if k_best != "all":
        if not isinstance(k_best, int) or k_best < 1:
            raise ValueError("k_best must be 'all' or a positive integer.")

        steps.append(
            (
                "feature_selection",
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=k_best,
                ),
            )
        )

    return Pipeline(steps)