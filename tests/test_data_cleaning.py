import pandas as pd

from src.data_cleaning import (
    clean_and_split,
    create_stratified_split,
    prepare_features_and_target,
    remove_duplicates,
)


def create_sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "job": ["admin"] * 10 + ["student"] * 10,
            "duration": list(range(100, 120)),
            "y": ["no"] * 10 + ["yes"] * 10,
        }
    )


def test_remove_duplicates():
    data = create_sample_data()

    data_with_duplicate = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    cleaned_data = remove_duplicates(data_with_duplicate)

    assert len(cleaned_data) == 20
    assert cleaned_data.duplicated().sum() == 0


def test_prepare_features_and_target():
    data = create_sample_data()

    X, y = prepare_features_and_target(data)

    assert "y" not in X.columns
    assert "duration" not in X.columns
    assert len(X) == len(y)
    assert y.name == "y"


def test_clean_and_split_uses_80_20_split():
    data = create_sample_data()

    X_train, X_test, y_train, y_test = clean_and_split(data)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_clean_and_split_is_stratified():
    data = create_sample_data()

    _, _, y_train, y_test = clean_and_split(data)

    assert y_train.value_counts().to_dict() == {
        "no": 8,
        "yes": 8,
    }

    assert y_test.value_counts().to_dict() == {
        "no": 2,
        "yes": 2,
    }


def test_clean_and_split_is_reproducible():
    data = create_sample_data()

    first_split = clean_and_split(data)
    second_split = clean_and_split(data)

    for first_result, second_result in zip(
        first_split,
        second_split,
    ):
        assert first_result.equals(second_result)


def test_create_stratified_split_passes_target_to_stratify(
    monkeypatch,
):
    """Ensure the target is explicitly passed to stratify."""

    data = create_sample_data()
    X, y = prepare_features_and_target(data)

    captured_arguments = {}

    def fake_train_test_split(X_data, y_data, **kwargs):
        captured_arguments.update(kwargs)
        return X_data, X_data, y_data, y_data

    monkeypatch.setattr(
        "src.data_cleaning.train_test_split",
        fake_train_test_split,
    )

    create_stratified_split(X, y)

    assert captured_arguments["stratify"] is y
    assert captured_arguments["test_size"] == 0.20
    assert captured_arguments["random_state"] == 42