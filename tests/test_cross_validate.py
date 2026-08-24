from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

from ml_template.config import TEMPORAL_SPLIT, Config
from ml_template.data_processor import save_dataset
from scripts.cross_validate import SCORERS, build_splitter, prepared_data


def test_temporal_strategy_never_trains_on_the_future(config: Config) -> None:
    """Shuffling a temporal problem into random folds trains on rows it is then scored against."""
    splitter = build_splitter(
        replace(config, split=replace(config.split, strategy=TEMPORAL_SPLIT)), 5, 1
    )

    assert isinstance(splitter, TimeSeriesSplit)

    positions = pd.DataFrame({"a": range(50)})
    for train_index, test_index in splitter.split(positions):
        assert max(train_index) < min(test_index)


def test_classification_folds_are_stratified(config: Config) -> None:
    assert isinstance(build_splitter(config, 5, 1), RepeatedStratifiedKFold)


def test_regression_folds_are_not_stratified(config: Config) -> None:
    regression = replace(config, model=replace(config.model, task="regression"))

    assert isinstance(build_splitter(regression, 5, 1), RepeatedKFold)


def test_repeats_multiply_the_number_of_folds(config: Config) -> None:
    frame = pd.DataFrame({"a": range(50), "b": [0, 1] * 25})

    single = build_splitter(config, 5, 1).get_n_splits(frame, frame["b"])
    repeated = build_splitter(config, 5, 3).get_n_splits(frame, frame["b"])

    assert single == 5
    assert repeated == 15


def test_every_reportable_metric_has_a_scorer() -> None:
    """A metric the template can report but not cross-validate would fail only at run time."""
    import numpy as np

    from ml_template.evaluate import classification_metrics, regression_metrics

    reportable = set(
        classification_metrics(
            pd.Series([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
        )
    ) | set(regression_metrics(pd.Series([1.0, 2.0]), np.array([1.0, 2.0])))

    assert reportable <= set(SCORERS), f"no scorer for {reportable - set(SCORERS)}"


def test_prepared_data_applies_the_same_cleaning_as_training(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """Cross-validating rows that training would have dropped would report a different problem."""
    contaminated = pd.concat([training_frame, training_frame.head(3)], ignore_index=True)
    save_dataset(contaminated, config.data.raw_path)

    features, target = prepared_data(config)

    assert len(features) == len(target) == len(training_frame)
    assert list(features.columns) == config.features.all_features


def test_unknown_metric_is_rejected_before_fitting(config: Config) -> None:
    assert "lift" not in SCORERS
    with pytest.raises(KeyError):
        _ = SCORERS["lift"]
