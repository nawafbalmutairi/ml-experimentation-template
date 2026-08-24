from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_template.group_encoder import GroupTargetEncoder


def split_features_and_target(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    return frame[list(feature_columns)], frame[target]


def build_preprocessor(
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    group_keys: Sequence[str] = (),
) -> ColumnTransformer:
    """Build an unfitted preprocessor.

    It is returned unfitted on purpose: callers put it in a Pipeline that is fitted on the
    training split only, so validation and test data never influence imputation, scaling or
    encoding statistics. That matters most for `group_keys`, which are encoded by the target rate
    of their group: computing those rates before the split would leak labels into every fold.
    """
    if not numerical_features and not categorical_features and not group_keys:
        raise ValueError("At least one numerical or categorical feature is required")

    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numerical", numerical_pipeline, list(numerical_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
            # One transformer per key: each learns its group rates from the training rows of the
            # fold it is fitted on, so a fold never encodes a row with a label it should not see.
            *(
                (
                    f"group:{key}",
                    Pipeline(
                        [
                            ("encode", GroupTargetEncoder(key)),
                            # A group absent from training has no rate. The flag column records
                            # that, so imputing the rate does not erase the fact.
                            ("impute", SimpleImputer(strategy="median")),
                        ]
                    ),
                    [key],
                )
                for key in group_keys
            ),
        ],
        remainder="drop",
    )
