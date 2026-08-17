from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_template.feature_engineer import build_preprocessor, split_features_and_target

NUMERICAL = ["age"]
CATEGORICAL = ["city"]


def frame_of(ages: list[float | None], cities: list[str | None]) -> pd.DataFrame:
    return pd.DataFrame({"age": ages, "city": cities})


def test_split_features_and_target_separates_columns() -> None:
    frame = pd.DataFrame({"age": [30, 40], "city": ["a", "b"], "label": [1, 0]})

    features, target = split_features_and_target(frame, ["age", "city"], "label")

    assert list(features.columns) == ["age", "city"]
    assert target.tolist() == [1, 0]


def test_preprocessor_standardises_numerical_features() -> None:
    train = frame_of([10.0, 20.0, 30.0, 40.0], ["a", "b", "a", "b"])
    preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)

    transformed = preprocessor.fit_transform(train)

    assert transformed[:, 0].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed[:, 0].std() == pytest.approx(1.0, abs=1e-9)


def test_preprocessor_one_hot_encodes_categorical_features() -> None:
    train = frame_of([10.0, 20.0], ["a", "b"])
    preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)

    transformed = preprocessor.fit_transform(train)

    assert transformed.shape == (2, 3)


def test_preprocessor_uses_training_statistics_only() -> None:
    train = frame_of([10.0, 20.0, 30.0, 40.0], ["a", "b", "a", "b"])
    test = frame_of([1000.0], ["a"])
    preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)
    preprocessor.fit(train)

    scaled_test = preprocessor.transform(test)[0, 0]
    refitted_scaled = build_preprocessor(NUMERICAL, CATEGORICAL).fit_transform(
        pd.concat([train, test])
    )[-1, 0]

    assert scaled_test > 10
    assert scaled_test != pytest.approx(refitted_scaled)


def test_preprocessor_imputes_missing_values_from_training_data() -> None:
    train = frame_of([10.0, 20.0, 30.0, None], ["a", "b", "a", None])
    preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)

    transformed = preprocessor.fit_transform(train)

    assert not np.isnan(transformed).any()


def test_preprocessor_ignores_unseen_categories() -> None:
    train = frame_of([10.0, 20.0], ["a", "b"])
    preprocessor = build_preprocessor(NUMERICAL, CATEGORICAL)
    preprocessor.fit(train)

    transformed = preprocessor.transform(frame_of([15.0], ["unseen"]))

    assert transformed.shape == (1, 3)
    assert transformed[0, 1:].sum() == 0.0


def test_preprocessor_requires_at_least_one_feature() -> None:
    with pytest.raises(ValueError, match="feature"):
        build_preprocessor([], [])
