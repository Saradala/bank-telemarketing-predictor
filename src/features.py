"""Campaign feature engineering (Member 3).

CampaignFeatures is a small sklearn-style transformer for the campaign / previous-contact columns:

* previously_contacted - 1 if `previous` > 0, else 0.
* pdays_known          - 1 if `pdays` holds a real day count, 0 if it holds the code 999.
* pdays                - the code 999 is replaced by -1 (it is a code, not "999 days").
* campaign             - capped at a high quantile learned in fit() on TRAINING data only.

Why `previous` and not `pdays` for "was this client contacted before"? In the data, pdays = 999
does NOT reliably mean "never contacted": some clients with pdays = 999 have poutcome = 'failure'
(they were contacted, but the gap was not recorded). `previous` > 0 matches poutcome exactly.
See notebooks/03_eda_campaign.ipynb, section 6.

Use it as the FIRST step of a pipeline, before the ColumnTransformer:

    Pipeline([("features", CampaignFeatures()), ("prep", column_transformer), ("clf", model)])
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

PDAYS_NOT_RECORDED = 999    # code used in the raw data for "no previous contact recorded"
PDAYS_RECODED = -1          # value that replaces the code (trees split on it cleanly)
REQUIRED_COLUMNS = ["campaign", "pdays", "previous"]


class CampaignFeatures(BaseEstimator, TransformerMixin):
    """Campaign feature engineering + outlier capping of `campaign`.

    cap_quantile: `campaign` values above this quantile of the training data are clipped to it.
    Outliers are capped, not removed - they are real customers.
    """

    def __init__(self, cap_quantile=0.99):
        """Store the quantile used for the cap (no data is looked at here)."""
        self.cap_quantile = cap_quantile

    def _check_columns(self, X):
        """Raise a clear error if a required column is missing."""
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"CampaignFeatures needs these columns: {missing}")

    def fit(self, X, y=None):
        """Learn the `campaign` cap from the training data."""
        if not 0.0 < self.cap_quantile <= 1.0:
            raise ValueError("cap_quantile must be in (0, 1]")
        self._check_columns(X)
        # the cap is learned from TRAINING data only (a cap from the test set would be leakage)
        cap = float(X["campaign"].quantile(self.cap_quantile))
        if np.isnan(cap):
            # an empty / all-missing column gives a NaN quantile, and clip(upper=NaN) would cap nothing
            raise ValueError("`campaign` has no valid values in the training data, so no cap can be learned")
        self.campaign_cap_ = cap
        return self

    def transform(self, X):
        """Add the two flags, recode pdays 999 to -1 and cap `campaign`; returns a new DataFrame."""
        check_is_fitted(self, "campaign_cap_")     # sklearn's NotFittedError if fit() was not called
        self._check_columns(X)
        X = X.copy()
        # 1 only when a positive count is recorded; a missing `previous` is not treated as a contact
        X["previously_contacted"] = (X["previous"] > 0).astype(int)
        # "known" means a real day count: neither the 999 code nor a missing value
        X["pdays_known"] = (X["pdays"].notna() & (X["pdays"] != PDAYS_NOT_RECORDED)).astype(int)
        X["pdays"] = X["pdays"].replace(PDAYS_NOT_RECORDED, PDAYS_RECODED)
        X["campaign"] = X["campaign"].clip(upper=self.campaign_cap_)
        return X
