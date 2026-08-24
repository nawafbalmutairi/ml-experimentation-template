"""Encode a grouping column by the target rate of the *other* rows in its group.

People who travel, shop or transact together often share an outcome. A column that names the
group — a household, a ticket, an order, a device — carries that signal, but it is useless as a
raw feature: it is high-cardinality, and the groups in new data are mostly not the ones seen in
training.

What generalises is the group's observed outcome rate. Computing it is a well-known way to leak,
so this does it as a fitted transformer:

- `fit` reads the rate from the rows it is given, and nothing else.
- `fit_transform` leaves each row out of its own group's rate, so a row never sees its own label.
- `transform` scores new rows from the rates learned at fit time, and reports nothing for a group
  it has never seen.

Inside a `Pipeline`, that means a cross-validation fold learns rates from its training rows only,
and a genuine holdout is scored from training rows only. Computing the same statistic over a whole
dataset before splitting inflates every estimate that follows.

The rate covers every member of the group, which is the safe general form and not always the most
informative one. Where only part of a group shares its outcome, the rest dilute the signal: on the
Titanic data, families determined the fate of women and children but not of adult men, and a
project-specific rate restricted to the former beat this encoder by 1.2 points on the leaderboard.
There is deliberately no filter parameter here — one project has wanted it, and the template adds a
capability when two have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

UNKNOWN_SUFFIX = "_known"


class GroupTargetEncoder(BaseEstimator, TransformerMixin):
    """Two columns per group key: the group's target rate, and whether the group was seen.

    The second column matters as much as the first. A missing rate means "no labelled member of
    this group", which is a fact about the row, not an absence to be quietly imputed away.
    """

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> GroupTargetEncoder:
        if y is None:
            raise ValueError(f"{type(self).__name__} needs the target to encode '{self.column}'")

        target = pd.Series(np.asarray(y), index=X.index, name="_target")
        grouped = target.groupby(X[self.column])
        self.sums_ = grouped.sum()
        self.counts_ = grouped.size()
        self.feature_names_ = [self.column, f"{self.column}{UNKNOWN_SUFFIX}"]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        keys = X[self.column]
        sums = keys.map(self.sums_)
        counts = keys.map(self.counts_)
        return self._stack(sums, counts)

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None, **fit_params: object
    ) -> np.ndarray:
        """Leave each row out of its own group, so no row is encoded with its own label."""
        self.fit(X, y)

        keys = X[self.column]
        target = pd.Series(np.asarray(y), index=X.index)
        sums = keys.map(self.sums_) - target
        counts = keys.map(self.counts_) - 1
        return self._stack(sums, counts)

    @staticmethod
    def _stack(sums: pd.Series, counts: pd.Series) -> np.ndarray:
        known = counts.fillna(0) > 0
        rates = np.where(known, sums / counts.where(counts > 0), np.nan)
        return np.column_stack([rates, known.astype(float)])

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
