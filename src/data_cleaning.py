"""Data cleaning and leakage-safe splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and reset the index."""
    return data.drop_duplicates().reset_index(drop=True)


def prepare_features_and_target(
    data: pd.DataFrame,
    target_column: str = "y",
    leakage_columns: tuple[str, ...] = ("duration",),
):
    """Separate the target and remove leakage features from predictors."""

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    columns_to_drop = [
        column
        for column in leakage_columns
        if column in data.columns
    ]

    X = data.drop(
        columns=[target_column, *columns_to_drop]
    ).copy()

    y = data[target_column].copy()

    return X, y


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Create a reproducible stratified train-test split."""

    if len(X) != len(y):
        raise ValueError(
            "The feature data and target must contain the same number of rows."
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def clean_and_split(
    data: pd.DataFrame,
    target_column: str = "y",
    leakage_columns: tuple[str, ...] = ("duration",),
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Remove duplicate records, exclude leakage features,
    and create a stratified train-test split.
    """

    clean_data = remove_duplicates(data)

    X, y = prepare_features_and_target(
        clean_data,
        target_column=target_column,
        leakage_columns=leakage_columns,
    )

    return create_stratified_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )