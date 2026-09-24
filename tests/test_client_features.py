import pandas as pd
import pytest

from src.client_features import AgeGrouper, UnknownHandler


def _df():
    return pd.DataFrame({
        "age": [17, 30, 50, 70, 98],
        "job": ["admin.", "admin.", "unknown", "retired", "admin."],
        "marital": ["married", "unknown", "single", "married", "married"],
        "education": ["unknown", "high.school", "high.school", "unknown", "university.degree"],
    })


def test_category_strategy_keeps_unknown():
    out = UnknownHandler(strategy="category").fit_transform(_df())
    assert (out["job"] == "unknown").sum() == 1


def test_mode_strategy_replaces_unknown_with_most_frequent_known():
    out = UnknownHandler(strategy="mode").fit_transform(_df())
    assert (out[["job", "marital", "education"]] == "unknown").sum().sum() == 0
    assert out.loc[2, "job"] == "admin."          # most frequent known job
    assert out.loc[0, "education"] == "high.school"


def test_mode_is_learned_on_train_only_and_reused_on_test():
    train, test = _df(), pd.DataFrame({"age": [40], "job": ["unknown"], "marital": ["unknown"], "education": ["unknown"]})
    handler = UnknownHandler(strategy="mode").fit(train)
    assert handler.transform(test).loc[0, "job"] == "admin."   # value from TRAIN, not from test


def test_transform_does_not_change_input():
    df = _df()
    UnknownHandler(strategy="mode").fit_transform(df)
    assert (df["job"] == "unknown").sum() == 1


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        UnknownHandler(strategy="drop").fit(_df())


def test_age_group_labels_and_edges():
    out = AgeGrouper().fit_transform(_df())
    assert out["age_group"].tolist() == ["<25", "25-34", "45-54", "65+", "65+"]
    assert out["age_group"].isna().sum() == 0
    assert "age" in out.columns                 # numeric age is kept


def test_age_group_works_in_a_sklearn_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    ct = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["job", "marital", "education", "age_group"])])
    pipe = Pipeline([("unknown", UnknownHandler()), ("age", AgeGrouper()), ("prep", ct)])
    X = pipe.fit_transform(_df())
    assert X.shape[0] == 5
