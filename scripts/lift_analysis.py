"""Rank the held-out set by predicted probability and report decile lift.

Accuracy says little about a targeting problem. This answers the question a campaign actually
asks: if we act on the top N% of the ranking, how much of the positive class do we reach?

    python scripts/lift_analysis.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.config import Config, default_config_path, load_config
from ml_template.data_processor import clean_dataset, encode_target, load_dataset
from ml_template.feature_engineer import split_features_and_target
from ml_template.predict import load_pipeline, positive_class_scores
from ml_template.train import split_train_test

DECILES = 10


def held_out_scores(config: Config) -> pd.DataFrame:
    """Rebuild the exact split training used, then score its test half."""
    raw = load_dataset(config.data.raw_path, config.data.separator)
    raw = encode_target(raw, config.data.target, config.data.target_mapping)
    cleaned = clean_dataset(
        raw, config.features.all_features, config.data.target, config.validation.value_ranges
    )
    features, target = split_features_and_target(
        cleaned, config.features.all_features, config.data.target
    )
    _, features_test, _, target_test = split_train_test(features, target, config)

    pipeline = load_pipeline(config.model.output_path)
    scores = positive_class_scores(pipeline, features_test)
    if scores is None:
        raise SystemExit("Lift needs a binary classifier that can produce probabilities.")

    # The positive class is the model's highest label, matching the column the scores come from.
    # Reading the labels as 0/1 would call class 2 of a {1, 2} target a positive twice over.
    positive_label = pipeline[-1].classes_[-1]
    actual = (target_test.to_numpy() == positive_label).astype(int)
    return pd.DataFrame({"score": scores, "actual": actual}).sort_values(
        "score", ascending=False, ignore_index=True
    )


def render_lift_table(ranked: pd.DataFrame) -> str:
    if len(ranked) < DECILES:
        raise SystemExit(
            f"Decile lift needs at least {DECILES} held-out rows, got {len(ranked)}. "
            "Raise split.test_size, or use more data."
        )
    total_positives = int(ranked["actual"].sum())
    if not total_positives:
        raise SystemExit(
            "The held-out set contains no positives, so there is no lift to measure. "
            "Re-split the data, or use a larger test set."
        )
    base_rate = float(ranked["actual"].mean())

    lines = [
        f"Held-out rows: {len(ranked)}, positives: {total_positives}, base rate: {base_rate:.2%}",
        "",
        "| Decile | Rows | Positives | Response rate | Lift | Cumulative % of positives |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    cumulative = 0
    for index, positions in enumerate(np.array_split(np.arange(len(ranked)), DECILES), start=1):
        chunk = ranked.iloc[positions]
        positives = int(chunk["actual"].sum())
        cumulative += positives
        rate = positives / len(chunk)
        lines.append(
            f"| {index} | {len(chunk)} | {positives} | {rate:.1%} | {rate / base_rate:.2f}x "
            f"| {cumulative / total_positives:.1%} |"
        )

    lines.append("")
    for fraction in (0.1, 0.2, 0.3, 0.5):
        top = ranked.head(round(len(ranked) * fraction))
        lines.append(
            f"Top {fraction:.0%} ({len(top)} rows): reaches "
            f"{int(top['actual'].sum()) / total_positives:.1%} of positives at a "
            f"{float(top['actual'].mean()):.1%} response rate "
            f"({float(top['actual'].mean()) / base_rate:.2f}x base)"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=default_config_path())
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config: {args.config} (split: {config.split.strategy})\n")
    print(render_lift_table(held_out_scores(config)))


if __name__ == "__main__":
    main()
