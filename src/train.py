from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CLASSIFICATION, Config, default_config_path, load_config
from src.data_processor import clean_dataset, load_dataset, save_dataset
from src.evaluate import evaluate_model, save_metrics
from src.feature_engineer import build_preprocessor, split_features_and_target
from src.model import create_model


def build_pipeline(config: Config) -> Pipeline:
    preprocessor = build_preprocessor(config.features.numerical, config.features.categorical)
    model = create_model(config.model, config.seed)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def run_training(config: Config) -> dict[str, float]:
    raw = load_dataset(config.data.raw_path)
    cleaned = clean_dataset(raw, config.features.all_features, config.data.target)
    save_dataset(cleaned, config.data.processed_path)

    features, target = split_features_and_target(
        cleaned, config.features.all_features, config.data.target
    )
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=config.split.test_size,
        random_state=config.seed,
        stratify=target if config.model.task == CLASSIFICATION else None,
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
