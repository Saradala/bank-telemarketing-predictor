# tests/test_svm_model.py

import sys
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.svm_model import (
    load_data,
    chronological_split,
    build_svm_pipeline,
)


def test_load_data():
    """Dataset should load correctly without target/leakage columns."""

    X, y = load_data()

    assert len(X) == len(y)
    assert len(X) > 0

    assert "y" not in X.columns
    assert "duration" not in X.columns

    assert set(y.unique()).issubset({0, 1})


def test_chronological_split():
    """Chronological split should create non-empty 80/20 partitions."""

    X, y = load_data()

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        test_fraction=0.20
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    assert len(X_train) > 0
    assert len(X_test) > 0

    assert len(X_train) + len(X_test) == len(X)


def test_svm_pipeline():
    """SVM builder should return the expected sklearn pipeline."""

    model = build_svm_pipeline()

    assert isinstance(model, Pipeline)

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    assert isinstance(
        model.named_steps["classifier"],
        SVC
    )


def test_svm_configuration():
    """Check the baseline SVM configuration."""

    model = build_svm_pipeline(
        C=1.0,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced"
    )

    classifier = model.named_steps["classifier"]

    assert classifier.C == 1.0
    assert classifier.gamma == "scale"
    assert classifier.kernel == "rbf"
    assert classifier.class_weight == "balanced"