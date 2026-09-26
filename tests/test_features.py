import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from src.features import CampaignFeatures


def _train():
    """100 clients: campaign = 1..100, most never contacted before, a few with a recorded pdays."""
    return pd.DataFrame({
        "campaign": list(range(1, 101)),
        "pdays": [999] * 90 + [3, 6, 10, 12, 15, 5, 7, 9, 11, 4],
        "previous": [0] * 90 + [1, 1, 2, 1, 3, 1, 2, 1, 1, 1],
    })


def test_previously_contacted_follows_previous_not_pdays():
    """previously_contacted uses `previous`; a client with pdays = 999 but previous = 1 was contacted."""
    df = pd.DataFrame({"campaign": [1, 1, 1], "pdays": [999, 999, 5], "previous": [0, 1, 2]})
    out = CampaignFeatures().fit_transform(df)
    # 2nd row: pdays is the 999 code but previous = 1 -> the client WAS contacted before
    assert out["previously_contacted"].tolist() == [0, 1, 1]


def test_pdays_known_flags_the_999_code():
    """pdays_known is 0 for the 999 code and 1 for a real day count."""
    df = pd.DataFrame({"campaign": [1, 1], "pdays": [999, 5], "previous": [0, 1]})
    out = CampaignFeatures().fit_transform(df)
    assert out["pdays_known"].tolist() == [0, 1]


def test_missing_pdays_is_not_marked_as_known():
    """A missing pdays has no real day count, so pdays_known must be 0 (not 1) and the value stays missing."""
    df = pd.DataFrame({"campaign": [1, 1, 1], "pdays": [999, 5, np.nan], "previous": [0, 1, 0]})
    out = CampaignFeatures().fit_transform(df)
    assert out["pdays_known"].tolist() == [0, 1, 0]
    assert np.isnan(out.loc[2, "pdays"])


def test_pdays_999_is_recoded_and_real_values_are_kept():
    """999 becomes -1; real values (including 0 = same day) are unchanged."""
    df = pd.DataFrame({"campaign": [1, 1, 1], "pdays": [999, 0, 21], "previous": [0, 1, 1]})
    out = CampaignFeatures().fit_transform(df)
    assert out["pdays"].tolist() == [-1, 0, 21]        # 0 is a real value (same-day), not a code


def test_campaign_is_capped_at_the_training_quantile():
    """campaign values above the 99th percentile of the training data are clipped to it."""
    tf = CampaignFeatures(cap_quantile=0.99).fit(_train())
    assert tf.campaign_cap_ == pytest.approx(99.01)
    out = tf.transform(pd.DataFrame({"campaign": [1, 50, 100], "pdays": [999] * 3, "previous": [0] * 3}))
    assert out["campaign"].max() == pytest.approx(99.01)
    assert out["campaign"].tolist()[:2] == [1, 50]      # values below the cap are untouched


def test_cap_is_learned_on_train_only_and_reused_on_test():
    """The cap comes from the training data; an extreme test value is clipped to it, not used to change it."""
    tf = CampaignFeatures(cap_quantile=0.99).fit(_train())
    test = pd.DataFrame({"campaign": [5000], "pdays": [999], "previous": [0]})
    out = tf.transform(test)
    assert out.loc[0, "campaign"] == pytest.approx(tf.campaign_cap_)   # cap from TRAIN, not from test


def test_no_rows_are_removed():
    """Outliers are capped, never deleted: the row count is unchanged."""
    df = _train()
    assert len(CampaignFeatures().fit_transform(df)) == len(df)


def test_transform_does_not_change_input():
    """transform works on a copy, so the caller's DataFrame is untouched."""
    df = _train()
    before = df.copy()
    CampaignFeatures().fit_transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_other_columns_are_passed_through():
    """Columns that CampaignFeatures does not use (contact, month, ...) stay in the output."""
    df = _train().assign(contact="cellular", month="may")
    out = CampaignFeatures().fit_transform(df)
    assert {"contact", "month"} <= set(out.columns)


def test_missing_required_column_raises():
    """A clear ValueError names the missing column."""
    with pytest.raises(ValueError, match="pdays"):
        CampaignFeatures().fit(pd.DataFrame({"campaign": [1], "previous": [0]}))


def test_invalid_cap_quantile_raises():
    """cap_quantile must be in (0, 1]."""
    with pytest.raises(ValueError):
        CampaignFeatures(cap_quantile=0).fit(_train())


@pytest.mark.parametrize("campaign_values", [[], [np.nan, np.nan]])
def test_fit_rejects_empty_or_all_missing_campaign(campaign_values):
    """No valid campaign values means no cap can be learned; fit must fail instead of storing NaN."""
    df = pd.DataFrame({
        "campaign": pd.Series(campaign_values, dtype=float),
        "pdays": pd.Series([999] * len(campaign_values), dtype=float),
        "previous": pd.Series([0] * len(campaign_values), dtype=float),
    })
    with pytest.raises(ValueError, match="no valid values"):
        CampaignFeatures().fit(df)


def test_transform_before_fit_raises_not_fitted_error():
    """Calling transform without fit raises sklearn's NotFittedError, not an AttributeError."""
    with pytest.raises(NotFittedError):
        CampaignFeatures().transform(_train())


def test_works_inside_a_pipeline_and_can_be_cloned():
    """The transformer follows the sklearn API: it can be cloned and used as a Pipeline step."""
    pipe = Pipeline([("features", CampaignFeatures(cap_quantile=0.95))])
    assert clone(pipe).get_params()["features__cap_quantile"] == 0.95
    out = pipe.fit_transform(_train())
    assert out["campaign"].max() <= pipe.named_steps["features"].campaign_cap_
