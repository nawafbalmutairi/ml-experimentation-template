from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml_template.config import (
    CLASSIFICATION,
    TEMPORAL_SPLIT,
    Config,
    default_config_path,
    load_config,
)
from ml_template.data_processor import clean_dataset, encode_target, load_dataset, save_dataset
from ml_template.data_validator import (
    PROCESSED_STAGE,
    RAW_STAGE,
    SCHEMA_CHECK,
    DataQualityError,
    ValidationReport,
    validate_dataset,
    write_quality_report,
)
from ml_template.evaluate import evaluate_model, save_metrics
from ml_template.feature_engineer import build_preprocessor, split_features_and_target
from ml_template.model import create_model


def build_pipeline(config: Config) -> Pipeline:
    preprocessor = build_preprocessor(config.features.numerical, config.features.categorical)
    model = create_model(config.model, config.seed)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def split_train_test(
    features: pd.DataFrame, target: pd.Series, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold out a test set, either at random or by taking the most recent rows.

    The temporal strategy assumes the dataset is already in chronological order and keeps that
    order, so the model is only ever fitted on rows that precede the ones it is scored on. It
    cannot stratify: forcing the class balance would mean reaching into the future.
    """
    if config.split.strategy == TEMPORAL_SPLIT:
        train_rows = len(features) - round(len(features) * config.split.test_size)
        return (
            features.iloc[:train_rows],
            features.iloc[train_rows:],
            target.iloc[:train_rows],
            target.iloc[train_rows:],
        )

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=config.split.test_size,
        random_state=config.seed,
        stratify=target if config.model.task == CLASSIFICATION else None,
    )
    return features_train, features_test, target_train, target_test


def quality_error(report: ValidationReport, report_path: Path) -> DataQualityError:
    details = "\n".join(f"- {issue.check}: {issue.message}" for issue in report.errors)
    return DataQualityError(
        f"{report.stage.capitalize()} data failed {len(report.errors)} quality check(s). "
        f"See {report_path}\n{details}"
    )


def run_training(config: Config) -> dict[str, float]:
    raw = load_dataset(config.data.raw_path, config.data.separator)
    raw = encode_target(raw, config.data.target, config.data.target_mapping)
    raw_report = validate_dataset(raw, config, RAW_STAGE)
    if raw_report.has_error(SCHEMA_CHECK):
        raise quality_error(raw_report, write_quality_report([raw_report], config.validation))

    cleaned = clean_dataset(
        raw, config.features.all_features, config.data.target, config.validation.value_ranges
    )
    processed_report = validate_dataset(cleaned, config, PROCESSED_STAGE)
    report_path = write_quality_report([raw_report, processed_report], config.validation)
    if not processed_report.passed:
        raise quality_error(processed_report, report_path)

    save_dataset(cleaned, config.data.processed_path)

    features, target = split_features_and_target(
        cleaned, config.features.all_features, config.data.target
    )
    features_train, features_test, target_train, target_test = split_train_test(
        features, target, config
    )

    pipeline = build_pipeline(config)
    pipeline.fit(features_train, target_train)

    metrics = evaluate_model(
        pipeline, features_test, target_test, config.model.task, config.evaluation.metrics
    )
    save_metrics(
        metrics,
        config.evaluation,
        context={
            "model_type": config.model.type,
            "task": config.model.task,
            "params": config.model.params,
            "split_strategy": config.split.strategy,
            "train_rows": len(features_train),
            "test_rows": len(features_test),
        },
    )
    save_pipeline(pipeline, config.model.output_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the configured model pipeline.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = run_training(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
