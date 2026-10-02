"""Tests for the Member 2 Random Forest pipelines."""

import pytest

from src.rf_model import build_rf_pipeline


def test_rf_pipeline_without_imbalance_handling():
    """The baseline pipeline should not contain SMOTE or class weights."""
    pipeline = build_rf_pipeline(imbalance_method="none")

    assert "smote" not in pipeline.named_steps
    assert pipeline.named_steps["classifier"].class_weight is None


def test_rf_pipeline_uses_class_weight():
    """The class-weight strategy should use balanced subsample weights."""
    pipeline = build_rf_pipeline(imbalance_method="class_weight")

    assert "smote" not in pipeline.named_steps
    assert (
        pipeline.named_steps["classifier"].class_weight
        == "balanced_subsample"
    )


def test_rf_pipeline_places_smote_before_classifier():
    """SMOTE must be applied inside the pipeline before classification."""
    pipeline = build_rf_pipeline(imbalance_method="smote")

    step_names = list(pipeline.named_steps)

    assert "smote" in step_names
    assert step_names.index("smote") < step_names.index("classifier")


def test_rf_pipeline_rejects_invalid_imbalance_method():
    """An unsupported imbalance strategy should raise a clear error."""
    with pytest.raises(ValueError, match="imbalance_method"):
        build_rf_pipeline(imbalance_method="invalid")