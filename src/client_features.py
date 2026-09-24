"""Client-profile feature handling (Member 1).

Two small sklearn-style transformers that work on a pandas DataFrame:

* UnknownHandler - deals with the coded-missing value 'unknown' in job, marital, education.
* AgeGrouper     - adds an `age_group` column from `age`.

Both follow the fit / transform rule: anything learned from data (e.g. the most
frequent value) is learned in fit() on TRAINING data only, then reused in transform().
Use them BEFORE the ColumnTransformer, e.g.:

    Pipeline([("unknown", UnknownHandler()), ("age", AgeGrouper()), ("prep", column_transformer)])
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

UNKNOWN = "unknown"
UNKNOWN_COLUMNS = ["job", "marital", "education"]

AGE_BINS = [16, 24, 34, 44, 54, 64, 100]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


class UnknownHandler(BaseEstimator, TransformerMixin):
    """Handle 'unknown' in categorical client columns.

    strategy="category" (default): keep 'unknown' as its own category (no change).
    strategy="mode": replace 'unknown' by the most frequent KNOWN value, learned in fit().
    """

    def __init__(self, columns=None, strategy="category"):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy not in ("category", "mode"):
            raise ValueError("strategy must be 'category' or 'mode'")
        self.columns_ = list(self.columns) if self.columns is not None else list(UNKNOWN_COLUMNS)
        self.modes_ = {}
        if self.strategy == "mode":
            for col in self.columns_:
                known = X.loc[X[col] != UNKNOWN, col]
                self.modes_[col] = known.mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        if self.strategy == "mode":
            for col in self.columns_:
                X[col] = X[col].replace(UNKNOWN, self.modes_[col])
        return X


class AgeGrouper(BaseEstimator, TransformerMixin):
    """Add an `age_group` column (string labels) and keep the numeric `age` too."""

    def __init__(self, bins=None, labels=None):
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        self.bins_ = list(self.bins) if self.bins is not None else list(AGE_BINS)
        self.labels_ = list(self.labels) if self.labels is not None else list(AGE_LABELS)
        return self

    def transform(self, X):
        X = X.copy()
        groups = pd.cut(X["age"], bins=self.bins_, labels=self.labels_)
        X["age_group"] = groups.astype(str)   # plain strings so OneHotEncoder can use it
        return X
