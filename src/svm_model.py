import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import RAW_DATA, RANDOM_STATE, LEAKAGE_COLUMNS, TARGET


def load_data():
    """
    Load the bank marketing dataset, remove exact duplicates,
    remove leakage columns, and convert the target to binary.
    """

    df = pd.read_csv(RAW_DATA, sep=";")

    # Remove exact duplicates BEFORE dropping duration
    df = df.drop_duplicates().reset_index(drop=True)

    # Separate target
    y = (df[TARGET] == "yes").astype(int)

    # Remove target and leakage columns such as duration
    X = df.drop(
        columns=[TARGET] + list(LEAKAGE_COLUMNS),
        errors="ignore"
    )

    return X, y


def chronological_split(X, y, test_fraction=0.20):
    """
    Use the last 20% of observations as the chronological test set.
    """

    split_index = int(len(X) * (1 - test_fraction))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return X_train, X_test, y_train, y_test


def build_svm_pipeline(
    C=1.0,
    gamma="scale",
    kernel="rbf",
    class_weight="balanced",
    probability=False
):
    """
    Build the preprocessing + SVM pipeline.

    Numerical features:
        StandardScaler

    Categorical features:
        OneHotEncoder

    Classifier:
        Support Vector Classifier (SVC)

    probability=False is used by default because probability
    calibration makes SVM training significantly slower.
    """

    X, _ = load_data()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_features
            ),
        ]
    )

    svm = SVC(
        C=C,
        gamma=gamma,
        kernel=kernel,
        class_weight=class_weight,
        probability=probability,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", svm)
        ]
    )

    return pipeline