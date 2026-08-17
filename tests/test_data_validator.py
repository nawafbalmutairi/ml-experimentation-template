from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ml_template.config import Config, ValueRange
from ml_template.data_validator import (
    ERROR,
    PROCESSED_STAGE,
    RAW_STAGE,
    SCHEMA_CHECK,
    WARNING,
    ValidationIssue,
    ValidationReport,
    check_data_types,
    check_duplicates,
    check_invalid_values,
    check_minimum_rows,
    check_missingness,
    check_schema,
    check_target_quality,
    validate_dataset,
    write_quality_report,
)

FEATURES = ["age", "city"]
TARGET = "label"


def valid_frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": np.linspace(20, 60, rows),
            "city": ["a", "b"] * (rows // 2),
            TARGET: [0, 1] * (rows // 2),
        }
    )


def severities(issues: list[ValidationIssue]) -> set[str]:
    return {issue.severity for issue in issues}


def test_check_schema_accepts_a_complete_frame() -> None:
    assert check_schema(valid_frame(), [*FEATURES, TARGET]) == []


def test_check_schema_reports_every_missing_column() -> None:
    issues = check_schema(pd.DataFrame({"age": [1]}), [*FEATURES, TARGET])

    assert len(issues) == 1
    assert issues[0].check == SCHEMA_CHECK
    assert issues[0].severity == ERROR
    assert "city" in issues[0].message
    assert TARGET in issues[0].message


def test_check_schema_ignores_extra_columns() -> None:
    frame = valid_frame().assign(unused="x")

    assert check_schema(frame, [*FEATURES, TARGET]) == []


def test_check_data_types_accepts_matching_dtypes() -> None:
    assert check_data_types(valid_frame(), ["age"], ["city"]) == []


def test_check_data_types_rejects_non_numeric_numerical_feature() -> None:
    frame = valid_frame().assign(age="not a number")

    issues = check_data_types(frame, ["age"], ["city"])

    assert severities(issues) == {ERROR}
    assert "age" in issues[0].message


def test_check_data_types_warns_about_numeric_categorical_feature() -> None:
    frame = valid_frame().assign(city=1)

    issues = check_data_types(frame, ["age"], ["city"])

    assert severities(issues) == {WARNING}


def test_check_data_types_skips_absent_columns() -> None:
    assert check_data_types(pd.DataFrame({"age": [1.0]}), ["age", "absent"], ["city"]) == []


def test_check_missingness_accepts_values_under_the_threshold() -> None:
    frame = valid_frame(20)
    frame.loc[0, "age"] = None

    assert check_missingness(frame, FEATURES, max_missing_fraction=0.2) == []


def test_check_missingness_rejects_columns_over_the_threshold() -> None:
    frame = valid_frame(20)
    frame.loc[0:9, "age"] = None

    issues = check_missingness(frame, FEATURES, max_missing_fraction=0.2)

    assert severities(issues) == {ERROR}
    assert "age" in issues[0].message


def test_check_missingness_reports_each_failing_column() -> None:
    frame = valid_frame(20)
    frame.loc[0:9, "age"] = None
    frame.loc[0:9, "city"] = None

    assert len(check_missingness(frame, FEATURES, max_missing_fraction=0.2)) == 2


def test_check_missingness_tolerates_an_empty_frame() -> None:
    assert check_missingness(pd.DataFrame(columns=FEATURES), FEATURES, 0.2) == []


def test_check_duplicates_accepts_unique_rows() -> None:
    assert check_duplicates(valid_frame(), max_duplicate_fraction=0.05) == []


def test_check_duplicates_rejects_frames_over_the_threshold() -> None:
    frame = pd.concat([valid_frame(10)] * 2, ignore_index=True)

    issues = check_duplicates(frame, max_duplicate_fraction=0.05)

    assert severities(issues) == {ERROR}
    assert "50.0%" in issues[0].message


def test_check_invalid_values_accepts_values_inside_the_configured_range() -> None:
    ranges = {"age": ValueRange(minimum=0, maximum=120)}

    assert check_invalid_values(valid_frame(), ["age"], ranges) == []


def test_check_invalid_values_rejects_values_below_the_minimum() -> None:
    frame = valid_frame()
    frame.loc[0, "age"] = -5

    issues = check_invalid_values(frame, ["age"], {"age": ValueRange(minimum=0)})

    assert severities(issues) == {ERROR}
    assert "below the minimum" in issues[0].message


def test_check_invalid_values_rejects_values_above_the_maximum() -> None:
    frame = valid_frame()
    frame.loc[0, "age"] = 500

    issues = check_invalid_values(frame, ["age"], {"age": ValueRange(maximum=120)})

    assert severities(issues) == {ERROR}
    assert "above the maximum" in issues[0].message


def test_check_invalid_values_rejects_infinite_values() -> None:
    frame = valid_frame()
    frame.loc[0, "age"] = np.inf

    issues = check_invalid_values(frame, ["age"], {})

    assert severities(issues) == {ERROR}
    assert "infinite" in issues[0].message


def test_check_invalid_values_ignores_missing_values() -> None:
    frame = valid_frame()
    frame.loc[0, "age"] = None

    assert check_invalid_values(frame, ["age"], {"age": ValueRange(minimum=0)}) == []


def test_check_invalid_values_skips_columns_without_a_configured_range() -> None:
    assert check_invalid_values(valid_frame(), ["age"], {}) == []


def test_check_target_quality_accepts_a_balanced_target() -> None:
    assert check_target_quality(valid_frame(), TARGET, "classification", 0.05) == []


def test_check_target_quality_reports_an_absent_target_column() -> None:
    issues = check_target_quality(pd.DataFrame({"age": [1]}), TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert TARGET in issues[0].message


def test_check_target_quality_rejects_missing_target_values() -> None:
    frame = valid_frame()
    frame.loc[0, TARGET] = None

    issues = check_target_quality(frame, TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert "missing" in issues[0].message


def test_check_target_quality_rejects_a_constant_target() -> None:
    frame = valid_frame().assign(**{TARGET: 1})

    issues = check_target_quality(frame, TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert "constant" in issues[0].message


def test_check_target_quality_rejects_classes_below_the_minimum_share() -> None:
    frame = valid_frame(100)
    frame[TARGET] = [1] + [0] * 99

    issues = check_target_quality(frame, TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert "1.0%" in issues[0].message


def test_check_target_quality_rejects_string_class_labels() -> None:
    frame = valid_frame().assign(**{TARGET: ["active", "churned"] * 10})

    issues = check_target_quality(frame, TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert "integer-encoded" in issues[0].message


def test_check_target_quality_rejects_labels_that_do_not_start_at_zero() -> None:
    frame = valid_frame().assign(**{TARGET: [1, 2] * 10})

    issues = check_target_quality(frame, TARGET, "classification", 0.05)

    assert severities(issues) == {ERROR}
    assert "[1, 2]" in issues[0].message


def test_check_target_quality_accepts_whole_number_float_labels() -> None:
    frame = valid_frame().assign(**{TARGET: [0.0, 1.0] * 10})

    assert check_target_quality(frame, TARGET, "classification", 0.05) == []


def test_check_target_quality_accepts_encoded_multiclass_labels() -> None:
    frame = valid_frame(30).assign(**{TARGET: [0, 1, 2] * 10})

    assert check_target_quality(frame, TARGET, "classification", 0.05) == []


def test_check_target_quality_accepts_a_continuous_regression_target() -> None:
    frame = valid_frame().assign(**{TARGET: np.linspace(1.0, 10.0, 20)})

    assert check_target_quality(frame, TARGET, "regression", 0.05) == []


def test_check_target_quality_rejects_a_non_numeric_regression_target() -> None:
    frame = valid_frame().assign(**{TARGET: ["a", "b"] * 10})

    issues = check_target_quality(frame, TARGET, "regression", 0.05)

    assert severities(issues) == {ERROR}
    assert "non-numeric" in issues[0].message


def test_check_target_quality_ignores_class_balance_for_regression() -> None:
    frame = valid_frame(100)
    frame[TARGET] = [1.0] + [0.0] * 99

    assert check_target_quality(frame, TARGET, "regression", 0.05) == []


def test_check_minimum_rows_accepts_a_large_enough_frame() -> None:
    assert check_minimum_rows(valid_frame(20), min_rows=20) == []


def test_check_minimum_rows_rejects_a_short_frame() -> None:
    issues = check_minimum_rows(valid_frame(10), min_rows=20)

    assert severities(issues) == {ERROR}
    assert "10 rows" in issues[0].message


def test_report_passes_when_only_warnings_are_present() -> None:
    report = ValidationReport(
        stage=RAW_STAGE,
        row_count=1,
        column_count=1,
        issues=[ValidationIssue("data_types", WARNING, "numeric category")],
    )

    assert report.passed
    assert report.warnings
    assert not report.has_error("data_types")


def test_validate_dataset_passes_on_clean_data(
    config: Config, training_frame: pd.DataFrame
) -> None:
    report = validate_dataset(training_frame, config, RAW_STAGE)

    assert report.passed
    assert report.stage == RAW_STAGE
    assert report.row_count == len(training_frame)
    assert report.column_count == len(training_frame.columns)


def test_validate_dataset_collects_issues_from_several_checks(
    config: Config, training_frame: pd.DataFrame
) -> None:
    broken = training_frame.drop(columns=["monthly_charges"]).head(5)

    report = validate_dataset(broken, config, PROCESSED_STAGE)

    assert not report.passed
    assert report.has_error(SCHEMA_CHECK)
    assert report.has_error("minimum_rows")


def test_validate_dataset_applies_configured_value_ranges(
    config: Config, training_frame: pd.DataFrame
) -> None:
    narrow = replace(
        config,
        validation=replace(
            config.validation, value_ranges={"tenure_months": ValueRange(maximum=5)}
        ),
    )

    report = validate_dataset(training_frame, narrow, PROCESSED_STAGE)

    assert report.has_error("invalid_values")


def test_write_quality_report_records_every_stage(
    config: Config, training_frame: pd.DataFrame
) -> None:
    reports = [
        validate_dataset(training_frame, config, RAW_STAGE),
        validate_dataset(training_frame.head(2), config, PROCESSED_STAGE),
    ]

    path = write_quality_report(reports, config.validation)

    content = path.read_text(encoding="utf-8")
    assert path.parent == config.validation.report_dir
    assert "## Stage: raw" in content
    assert "## Stage: processed" in content
    assert "PASSED" in content
    assert "FAILED" in content
    assert "minimum_rows" in content


def test_write_quality_report_states_when_nothing_is_wrong(
    config: Config, training_frame: pd.DataFrame
) -> None:
    path = write_quality_report(
        [validate_dataset(training_frame, config, RAW_STAGE)], config.validation
    )

    assert "No issues found." in path.read_text(encoding="utf-8")
