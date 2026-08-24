from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ml_template.config import TEMPORAL_SPLIT, Config, ValueRange
from ml_template.data_processor import save_dataset
from ml_template.data_validator import DataQualityError
from ml_template.feature_engineer import split_features_and_target
from ml_template.predict import load_pipeline, predict, run_prediction
from ml_template.train import run_training, split_train_test


def train_on(config: Config, frame: pd.DataFrame) -> dict[str, float]:
    save_dataset(frame, config.data.raw_path)
    return run_training(config)


def test_training_reports_the_configured_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    metrics = train_on(config, training_frame)

    assert list(metrics) == config.evaluation.metrics
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_training_learns_a_signal(config: Config, training_frame: pd.DataFrame) -> None:
    metrics = train_on(config, training_frame)

    assert metrics["accuracy"] > 0.8


def test_training_writes_processed_data_and_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    metrics = train_on(config, training_frame)

    assert config.data.processed_path.is_file()

    result_files = list(config.evaluation.results_dir.glob("metrics-*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["metrics"] == metrics
    assert payload["model_type"] == config.model.type
    assert payload["train_rows"] + payload["test_rows"] == len(training_frame)


def test_saved_pipeline_predicts_after_reloading(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)
    assert config.model.output_path.is_file()

    pipeline = load_pipeline(config.model.output_path)
    unseen = training_frame.drop(columns=[config.data.target]).head(5)
    predicted = predict(pipeline, unseen, config.features.all_features)

    assert len(predicted) == 5
    assert set(predicted["prediction"]).issubset({0, 1})


def test_prediction_scores_rank_clients_for_targeting(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """A label cannot say who to contact first; the score is what a call list is sorted on."""
    train_on(config, training_frame)
    pipeline = load_pipeline(config.model.output_path)
    unseen = training_frame.drop(columns=[config.data.target])

    predicted = predict(pipeline, unseen, config.features.all_features)

    assert "score" in predicted.columns
    assert predicted["score"].between(0.0, 1.0).all()
    # The score has to agree with the label, or ranking by it would contradict the model.
    assert predicted.loc[predicted["prediction"] == 1, "score"].min() > 0.5
    assert predicted.loc[predicted["prediction"] == 0, "score"].max() <= 0.5
    assert predicted["score"].nunique() > 1


def test_regression_predictions_have_no_score_column(
    config: Config, training_frame: pd.DataFrame
) -> None:
    regression_config = replace(
        config,
        data=replace(config.data, target="monthly_charges"),
        features=replace(config.features, numerical=["tenure_months"]),
        model=replace(config.model, task="regression"),
        evaluation=replace(config.evaluation, metrics=["mae"]),
    )
    train_on(regression_config, training_frame)
    pipeline = load_pipeline(regression_config.model.output_path)

    predicted = predict(
        pipeline,
        training_frame.drop(columns=["monthly_charges"]),
        regression_config.features.all_features,
    )

    assert "score" not in predicted.columns
    assert "prediction" in predicted.columns


def test_prediction_writes_a_csv_with_the_original_columns(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)
    input_path = config.data.raw_path.parent / "inference.csv"
    output_path = config.data.raw_path.parent / "predictions.csv"
    save_dataset(training_frame.drop(columns=[config.data.target]).head(10), input_path)

    run_prediction(config, input_path, output_path)

    written = pd.read_csv(output_path)
    assert len(written) == 10
    assert "prediction" in written.columns


def temporal(config: Config) -> Config:
    return replace(config, split=replace(config.split, strategy=TEMPORAL_SPLIT))


def test_temporal_split_holds_out_the_last_rows_in_order(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame, config.features.all_features, config.data.target
    )

    features_train, features_test, _, target_test = split_train_test(
        features, target, temporal(config)
    )

    assert len(features_test) == 30
    assert len(features_train) == 90
    assert features_test.index.tolist() == list(range(90, 120))
    assert features_train.index.tolist() == list(range(90))
    assert target_test.index.tolist() == features_test.index.tolist()


def test_temporal_split_does_not_reorder_or_overlap(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame, config.features.all_features, config.data.target
    )

    features_train, features_test, _, _ = split_train_test(features, target, temporal(config))

    assert set(features_train.index).isdisjoint(features_test.index)
    assert max(features_train.index) < min(features_test.index)
    assert features_test["tenure_months"].tolist() == (
        training_frame["tenure_months"].iloc[90:].tolist()
    )


def sized(config: Config, test_size: float) -> Config:
    return replace(config, split=replace(config.split, test_size=test_size))


def test_temporal_split_always_holds_out_at_least_one_row(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """0.01 of 50 rows once rounded to nothing, failing later inside SimpleImputer instead."""
    features, target = split_features_and_target(
        training_frame.head(50), config.features.all_features, config.data.target
    )

    _, features_test, _, target_test = split_train_test(
        features, target, temporal(sized(config, 0.01))
    )

    assert len(features_test) == len(target_test) == 1


def test_temporal_split_rejects_a_test_size_that_leaves_nothing_to_train_on(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame.head(50), config.features.all_features, config.data.target
    )

    with pytest.raises(ValueError, match="test_size"):
        split_train_test(features, target, temporal(sized(config, 0.99)))


def test_both_split_strategies_hold_out_the_same_number_of_rows(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """30 rows at 0.15 is 4.5: rounding to nearest and rounding up must not disagree."""
    awkward = sized(config, 0.15)
    features, target = split_features_and_target(
        training_frame.head(30), config.features.all_features, config.data.target
    )

    _, random_test, _, _ = split_train_test(features, target, awkward)
    _, temporal_test, _, _ = split_train_test(features, target, temporal(awkward))

    assert len(temporal_test) == len(random_test) == 5


def test_random_split_preserves_the_class_balance(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """20% positives overall, so a stratified quarter holds exactly 6 of the 24."""
    imbalanced = training_frame.assign(churned=[1] * 24 + [0] * 96)
    features, target = split_features_and_target(
        imbalanced, config.features.all_features, config.data.target
    )

    _, _, target_train, target_test = split_train_test(features, target, config)

    assert int(target_test.sum()) == 6
    assert int(target_train.sum()) == 18


def test_random_split_repeats_exactly_for_a_seed_and_changes_with_it(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame, config.features.all_features, config.data.target
    )
    other_seed = replace(config, seed=config.seed + 1)

    first = split_train_test(features, target, config)[1].index.tolist()
    again = split_train_test(features, target, config)[1].index.tolist()
    elsewhere = split_train_test(features, target, other_seed)[1].index.tolist()

    assert first == again
    assert first != elsewhere


def test_random_split_shuffles_rather_than_taking_the_tail(
    config: Config, training_frame: pd.DataFrame
) -> None:
    features, target = split_features_and_target(
        training_frame, config.features.all_features, config.data.target
    )

    _, features_test, _, _ = split_train_test(features, target, config)

    assert features_test.index.tolist() != list(range(90, 120))


def test_temporal_training_records_the_strategy_it_used(
    config: Config, training_frame: pd.DataFrame
) -> None:
    metrics = train_on(temporal(config), training_frame)

    assert set(metrics) == set(config.evaluation.metrics)

    payload = json.loads(
        next(config.evaluation.results_dir.glob("metrics-*.json")).read_text(encoding="utf-8")
    )
    assert payload["split_strategy"] == TEMPORAL_SPLIT
    assert payload["train_rows"] == 90
    assert payload["test_rows"] == 30


def quality_reports(config: Config) -> list[Path]:
    return sorted(config.validation.report_dir.glob("data-quality-*.md"))


def test_training_writes_a_data_quality_report(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)

    reports = quality_reports(config)
    assert len(reports) == 1
    content = reports[0].read_text(encoding="utf-8")
    assert "## Stage: raw" in content
    assert "## Stage: processed" in content


def test_training_stops_when_the_processed_data_fails_the_gate(
    config: Config, training_frame: pd.DataFrame
) -> None:
    demanding = replace(config, validation=replace(config.validation, min_rows=10_000))

    with pytest.raises(DataQualityError, match="minimum_rows"):
        train_on(demanding, training_frame)

    assert not demanding.model.output_path.exists()
    assert not demanding.data.processed_path.exists()


def test_failed_gate_still_writes_the_quality_report(
    config: Config, training_frame: pd.DataFrame
) -> None:
    demanding = replace(config, validation=replace(config.validation, min_rows=10_000))

    with pytest.raises(DataQualityError):
        train_on(demanding, training_frame)

    reports = quality_reports(demanding)
    assert len(reports) == 1
    assert "FAILED" in reports[0].read_text(encoding="utf-8")


def test_training_stops_before_cleaning_when_columns_are_missing(
    config: Config, training_frame: pd.DataFrame
) -> None:
    with pytest.raises(DataQualityError, match="monthly_charges"):
        train_on(config, training_frame.drop(columns=["monthly_charges"]))

    content = quality_reports(config)[0].read_text(encoding="utf-8")
    assert "## Stage: raw" in content
    assert "## Stage: processed" not in content


def test_raw_duplicates_over_the_threshold_block_training(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """Cleaning erases duplicates, so the raw stage is the only place the threshold can fire."""
    duplicated = pd.concat([training_frame, training_frame], ignore_index=True)

    with pytest.raises(DataQualityError, match="duplicates"):
        train_on(config, duplicated)

    assert not config.model.output_path.exists()
    assert not config.data.processed_path.exists()


def test_raw_duplicates_under_the_threshold_are_cleaned_away(
    config: Config, training_frame: pd.DataFrame
) -> None:
    tolerant = replace(config, validation=replace(config.validation, max_duplicate_fraction=0.5))
    duplicated = pd.concat([training_frame, training_frame], ignore_index=True)

    metrics = train_on(tolerant, duplicated)

    assert metrics["accuracy"] > 0.8
    assert len(pd.read_csv(tolerant.data.processed_path)) == len(training_frame)


def test_the_quality_report_shows_how_many_rows_cleaning_removed(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """A silent 800-row loss between the two stages is the failure this line makes visible."""
    tolerant = replace(config, validation=replace(config.validation, max_duplicate_fraction=0.5))
    duplicated = pd.concat([training_frame, training_frame], ignore_index=True)

    train_on(tolerant, duplicated)

    content = quality_reports(tolerant)[0].read_text(encoding="utf-8")
    assert f"- Rows removed since raw: {len(training_frame)}" in content


def test_duplicates_counted_at_the_raw_stage_are_the_rows_cleaning_removes(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """An id column makes every row unique, so nothing is duplicated and nothing may be dropped."""
    identified = training_frame.assign(customer_id=range(len(training_frame)))
    doubled = pd.concat([identified, identified.assign(customer_id=range(120, 240))])

    train_on(config, doubled.reset_index(drop=True))

    assert len(pd.read_csv(config.data.processed_path)) == len(doubled)


def test_boolean_class_labels_train_successfully(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """A True/False column is what pandas reads such a CSV as, and XGBoost accepts it."""
    boolean = training_frame.assign(churned=training_frame["churned"].astype(bool))

    metrics = train_on(config, boolean)

    assert metrics["accuracy"] > 0.8
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_string_class_labels_are_rejected_at_the_gate(
    config: Config, training_frame: pd.DataFrame
) -> None:
    """The booster's own error is unreadable, so the gate has to catch this first."""
    labelled = training_frame.assign(
        churned=training_frame["churned"].map({0: "active", 1: "churned"})
    )

    with pytest.raises(DataQualityError, match="integer-encoded"):
        train_on(config, labelled)

    assert not config.model.output_path.exists()


def test_multiclass_training_completes(config: Config, training_frame: pd.DataFrame) -> None:
    multiclass = replace(config, evaluation=replace(config.evaluation, metrics=["accuracy", "f1"]))
    frame = training_frame.copy()
    frame["churned"] = [index % 3 for index in range(len(frame))]

    metrics = train_on(multiclass, frame)

    assert set(metrics) == {"accuracy", "f1"}
    assert 0.0 <= metrics["f1"] <= 1.0


def test_invalid_rows_are_cleaned_so_the_gate_passes(
    config: Config, training_frame: pd.DataFrame
) -> None:
    bounded = replace(
        config,
        validation=replace(
            config.validation, value_ranges={"tenure_months": ValueRange(minimum=0, maximum=72)}
        ),
    )
    contaminated = training_frame.copy()
    contaminated.loc[0, "tenure_months"] = -999.0

    metrics = train_on(bounded, contaminated)

    assert metrics["accuracy"] > 0.8
    processed = pd.read_csv(bounded.data.processed_path)
    assert processed["tenure_months"].min() >= 0

    content = quality_reports(bounded)[0].read_text(encoding="utf-8")
    assert "invalid_values" in content


def test_regression_training_reports_regression_metrics(
    config: Config, training_frame: pd.DataFrame
) -> None:
    regression_config = replace(
        config,
        data=replace(config.data, target="monthly_charges"),
        features=replace(config.features, numerical=["tenure_months"]),
        model=replace(config.model, task="regression"),
        evaluation=replace(config.evaluation, metrics=["mae", "rmse", "r2"]),
    )

    metrics = train_on(regression_config, training_frame)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert "accuracy" not in metrics
