"""Cross-validate the configured model, respecting the configured split strategy.

    python scripts/cross_validate.py --config config/config.yaml --folds 5 --repeats 1

Read the result as an estimate of fit to the *training distribution*, which is not the same thing
as performance on new data. Every fold resamples the same rows, so cross-validation cannot see a
difference between the data you have and the data you will meet. On a small or shifting dataset it
can be confidently wrong, and repeating it only narrows the error bars around the wrong number.
It has ranked two models backwards on a real project here while overestimating both by 5-7 points.

Use it to compare models that differ a lot, not to choose between models that differ a little, and
prefer a genuine holdout whenever the data allows one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from ml_template.config import (
    CLASSIFICATION,
    TEMPORAL_SPLIT,
    Config,
    default_config_path,
    load_config,
)
from ml_template.data_processor import clean_dataset, encode_target, load_dataset
from ml_template.feature_engineer import build_preprocessor, split_features_and_target
from ml_template.model import create_model

# Our metric names to scikit-learn's. Errors are negated by sklearn, and reported back positive.
SCORERS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def build_splitter(config: Config, folds: int, repeats: int) -> BaseCrossValidator:
    """Fold generator matching the configured split strategy.

    A temporal strategy gets a rolling origin: every fold trains on rows preceding the ones it is
    scored on. Shuffling it into random folds would train on the future to predict the past, the
    exact leak the strategy exists to prevent, and would report a better number for doing so.
    """
    if config.split.strategy == TEMPORAL_SPLIT:
        return TimeSeriesSplit(n_splits=folds)
    if config.model.task == CLASSIFICATION:
        return RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=config.seed)
    return RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=config.seed)


def prepared_data(config: Config) -> tuple[pd.DataFrame, pd.Series]:
    raw = load_dataset(config.data.raw_path, config.data.separator)
    raw = encode_target(raw, config.data.target, config.data.target_mapping)
    cleaned = clean_dataset(
        raw, config.features.all_features, config.data.target, config.validation.value_ranges
    )
    return split_features_and_target(cleaned, config.features.all_features, config.data.target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1, help="ignored for a temporal strategy")
    args = parser.parse_args()

    config = load_config(args.config)
    features, target = prepared_data(config)

    unknown = [name for name in config.evaluation.metrics if name not in SCORERS]
    if unknown:
        raise SystemExit(f"No cross-validation scorer for {unknown}. Known: {sorted(SCORERS)}")

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(config.features.numerical, config.features.categorical),
            ),
            ("model", create_model(config.model, config.seed)),
        ]
    )
    splitter = build_splitter(config, args.folds, args.repeats)
    results = cross_validate(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring={name: SCORERS[name] for name in config.evaluation.metrics},
    )

    evaluations = len(results[f"test_{config.evaluation.metrics[0]}"])
    print(f"{type(splitter).__name__}, {evaluations} evaluations on {len(features)} rows\n")
    print(f"{'metric':20} {'mean':>9} {'std':>9} {'min':>9} {'max':>9}")
    print("-" * 60)
    for name in config.evaluation.metrics:
        scores = np.abs(results[f"test_{name}"])
        print(
            f"{name:20} {scores.mean():9.4f} {scores.std():9.4f} "
            f"{scores.min():9.4f} {scores.max():9.4f}"
        )

    print(
        "\nThis measures fit to the training distribution. It cannot detect a difference between\n"
        "this data and the data the model will meet, so treat a small gap between two models as\n"
        "unresolved rather than decided. See the module docstring."
    )


if __name__ == "__main__":
    main()
