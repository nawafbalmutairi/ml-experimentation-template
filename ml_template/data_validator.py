from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.config import CLASSIFICATION, Config, ValidationConfig, ValueRange
from ml_template.model import LABEL_ENCODED_MODEL_TYPES

ERROR = "error"
WARNING = "warning"

RAW_STAGE = "raw"
PROCESSED_STAGE = "processed"

SCHEMA_CHECK = "schema"
DUPLICATES_CHECK = "duplicates"

# Raw-stage errors that have to be judged before cleaning runs. Without the configured columns
# there is nothing to clean; and cleaning erases the duplicates, so by the processed stage the
# share is always 0% and max_duplicate_fraction could never fire. Everything else the raw stage
# reports is advisory, because the processed stage still sees it if cleaning left it behind.
BLOCKING_RAW_CHECKS = (SCHEMA_CHECK, DUPLICATES_CHECK)


class DataQualityError(RuntimeError):
    """Raised when a dataset fails a blocking quality gate."""


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    severity: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    stage: str
    row_count: int
    column_count: int
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def has_error(self, check: str) -> bool:
        return any(issue.check == check for issue in self.errors)


def check_schema(frame: pd.DataFrame, required_columns: Sequence[str]) -> list[ValidationIssue]:
    missing = [column for column in required_columns if column not in frame.columns]
    if not missing:
        return []
    return [ValidationIssue(SCHEMA_CHECK, ERROR, f"Missing required columns: {sorted(missing)}")]


def check_data_types(
    frame: pd.DataFrame,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> list[ValidationIssue]:
    issues = []
    for column in numerical_features:
        if column in frame.columns and not pd.api.types.is_numeric_dtype(frame[column]):
            issues.append(
                ValidationIssue(
                    "data_types",
                    ERROR,
                    f"Numerical feature '{column}' has non-numeric dtype {frame[column].dtype}",
                )
            )
    for column in categorical_features:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            issues.append(
                ValidationIssue(
                    "data_types",
                    WARNING,
                    f"Categorical feature '{column}' has numeric dtype {frame[column].dtype}",
                )
            )
    return issues


def check_missingness(
    frame: pd.DataFrame,
    columns: Sequence[str],
    max_missing_fraction: float,
) -> list[ValidationIssue]:
    if frame.empty:
        return []
    issues = []
    for column in columns:
        if column not in frame.columns:
            continue
        fraction = float(frame[column].isna().mean())
        if fraction > max_missing_fraction:
            issues.append(
                ValidationIssue(
                    "missingness",
                    ERROR,
                    f"Column '{column}' is {fraction:.1%} missing, "
                    f"above the {max_missing_fraction:.1%} threshold",
                )
            )
    return issues


def check_duplicates(
    frame: pd.DataFrame,
    max_duplicate_fraction: float,
) -> list[ValidationIssue]:
    """Count rows identical across every column — exactly the rows cleaning removes."""
    if frame.empty:
        return []
    fraction = float(frame.duplicated().mean())
    if fraction > max_duplicate_fraction:
        return [
            ValidationIssue(
                DUPLICATES_CHECK,
                ERROR,
                f"{fraction:.1%} of rows are duplicates, "
                f"above the {max_duplicate_fraction:.1%} threshold",
            )
        ]
    return []


def check_invalid_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    value_ranges: dict[str, ValueRange],
) -> list[ValidationIssue]:
    """Non-numeric columns are skipped, so this covers exactly what cleaning drops rows for."""
    issues = []
    for column in columns:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column]
        infinite_count = int(np.isinf(values.to_numpy(dtype="float64", na_value=np.nan)).sum())
        if infinite_count:
            issues.append(
                ValidationIssue(
                    "invalid_values",
                    ERROR,
                    f"Column '{column}' contains {infinite_count} infinite values",
                )
            )
        issues.extend(_out_of_range_issues(values, column, value_ranges.get(column)))
    return issues


def check_target_quality(
    frame: pd.DataFrame,
    target: str,
    task: str,
    min_class_fraction: float,
    require_encoded_labels: bool = True,
) -> list[ValidationIssue]:
    if target not in frame.columns:
        return [ValidationIssue("target_quality", ERROR, f"Target column '{target}' is missing")]

    values = frame[target]
    missing_count = int(values.isna().sum())
    if missing_count:
        return [
            ValidationIssue(
                "target_quality",
                ERROR,
                f"Target '{target}' has {missing_count} missing values",
            )
        ]

    observed = values.dropna()
    if observed.empty:
        return [ValidationIssue("target_quality", ERROR, f"Target '{target}' has no values")]
    if observed.nunique() < 2:
        return [
            ValidationIssue(
                "target_quality", ERROR, f"Target '{target}' is constant across all rows"
            )
        ]

    if task == CLASSIFICATION:
        encoding_issues = _label_encoding_issues(observed, target) if require_encoded_labels else []
        return encoding_issues or _rare_class_issues(observed, target, min_class_fraction)
    if not pd.api.types.is_numeric_dtype(observed):
        return [
            ValidationIssue(
                "target_quality",
                ERROR,
                f"Regression target '{target}' has non-numeric dtype {observed.dtype}",
            )
        ]
    return []


def check_minimum_rows(frame: pd.DataFrame, min_rows: int) -> list[ValidationIssue]:
    if len(frame) >= min_rows:
        return []
    return [
        ValidationIssue(
            "minimum_rows",
            ERROR,
            f"Dataset has {len(frame)} rows, below the required minimum of {min_rows}",
        )
    ]


def validate_dataset(frame: pd.DataFrame, config: Config, stage: str) -> ValidationReport:
    validation = config.validation
    features = config.features
    modelling_columns = [*features.all_features, config.data.target]

    # A duplicate record is only recognisable while the columns that tell records apart are still
    # there. Once cleaning has narrowed the frame to the modelling columns, identical-looking rows
    # are distinct observations that happen to share features, so the raw stage is where this is
    # both measurable and enforced.
    duplicates = (
        check_duplicates(frame, validation.max_duplicate_fraction) if stage == RAW_STAGE else []
    )

    issues = [
        *check_schema(frame, modelling_columns),
        *check_minimum_rows(frame, validation.min_rows),
        *check_data_types(frame, features.numerical, features.categorical),
        *check_missingness(frame, features.all_features, validation.max_missing_fraction),
        *duplicates,
        *check_invalid_values(frame, modelling_columns, validation.value_ranges),
        *check_target_quality(
            frame,
            config.data.target,
            config.model.task,
            validation.min_class_fraction,
            require_encoded_labels=config.model.type in LABEL_ENCODED_MODEL_TYPES,
        ),
    ]
    return ValidationReport(
        stage=stage,
        row_count=len(frame),
        column_count=len(frame.columns),
        issues=issues,
    )


def write_quality_report(reports: Sequence[ValidationReport], config: ValidationConfig) -> Path:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    output_path = config.report_dir / f"data-quality-{timestamp:%Y%m%dT%H%M%SZ}.md"
    output_path.write_text(_render_report(reports, config, timestamp), encoding="utf-8")
    return output_path


def _render_report(
    reports: Sequence[ValidationReport],
    config: ValidationConfig,
    timestamp: datetime,
) -> str:
    lines = [
        "# Data quality report",
        "",
        f"Generated: {timestamp.isoformat()}",
        "",
        "## Thresholds",
        "",
        f"- Minimum rows: {config.min_rows}",
        f"- Maximum missing fraction per column: {config.max_missing_fraction:.1%}",
        f"- Maximum duplicate row fraction: {config.max_duplicate_fraction:.1%}",
        f"- Minimum class fraction: {config.min_class_fraction:.1%}",
    ]

    for position, report in enumerate(reports):
        lines += [
            "",
            f"## Stage: {report.stage}",
            "",
            f"- Result: {'PASSED' if report.passed else 'FAILED'}",
            f"- Rows: {report.row_count}",
        ]
        if position:
            previous = reports[position - 1]
            lines.append(
                f"- Rows removed since {previous.stage}: {previous.row_count - report.row_count}"
            )
        lines += [
            f"- Columns: {report.column_count}",
            f"- Errors: {len(report.errors)}",
            f"- Warnings: {len(report.warnings)}",
            "",
        ]
        if not report.issues:
            lines.append("No issues found.")
            continue
        lines += ["| Check | Severity | Detail |", "| --- | --- | --- |"]
        lines += [
            f"| {issue.check} | {issue.severity} | {issue.message} |" for issue in report.issues
        ]

    return "\n".join(lines) + "\n"


def _out_of_range_issues(
    values: pd.Series,
    column: str,
    value_range: ValueRange | None,
) -> list[ValidationIssue]:
    if value_range is None:
        return []
    issues = []
    if value_range.minimum is not None:
        below = int((values < value_range.minimum).sum())
        if below:
            issues.append(
                ValidationIssue(
                    "invalid_values",
                    ERROR,
                    f"Column '{column}' has {below} values below the minimum {value_range.minimum}",
                )
            )
    if value_range.maximum is not None:
        above = int((values > value_range.maximum).sum())
        if above:
            issues.append(
                ValidationIssue(
                    "invalid_values",
                    ERROR,
                    f"Column '{column}' has {above} values above the maximum {value_range.maximum}",
                )
            )
    return issues


def _label_encoding_issues(observed: pd.Series, target: str) -> list[ValidationIssue]:
    """Some model families, XGBoost among them, require labels already encoded as 0..n-1.

    Catching it here turns a confusing error from deep inside the booster into a clear failure
    before any training time is spent.
    """
    advice = (
        f"Classification target '{target}' must be integer-encoded as 0..n-1 for model type"
        " 'xgboost', for example {'no': 0, 'yes': 1}"
    )
    labels = _encoded_labels(observed)
    if labels is None:
        found = sorted(pd.unique(observed).tolist(), key=repr)[:5]
        return [ValidationIssue("target_quality", ERROR, f"{advice}. Found labels: {found}")]
    if labels != list(range(len(labels))):
        return [ValidationIssue("target_quality", ERROR, f"{advice}. Found labels: {labels}")]
    return []


def _encoded_labels(observed: pd.Series) -> list[int] | None:
    """The integers behind the labels, or None when they are not whole numbers.

    Booleans and numeric categoricals are already 0/1 to the estimators, so they count as encoded.
    """
    labels = pd.Series(pd.unique(observed))
    if isinstance(labels.dtype, pd.CategoricalDtype):
        labels = labels.astype(labels.cat.categories.dtype)
    if pd.api.types.is_bool_dtype(labels):
        labels = labels.astype(int)
    if not pd.api.types.is_numeric_dtype(labels) or not bool((labels % 1 == 0).all()):
        return None
    return sorted(int(label) for label in labels)


def _rare_class_issues(
    observed: pd.Series,
    target: str,
    min_class_fraction: float,
) -> list[ValidationIssue]:
    fractions = observed.value_counts(normalize=True)
    rare = fractions[fractions < min_class_fraction]
    if rare.empty:
        return []
    detail = ", ".join(f"{label}={fraction:.1%}" for label, fraction in rare.items())
    return [
        ValidationIssue(
            "target_quality",
            ERROR,
            f"Target '{target}' has classes below the {min_class_fraction:.1%} "
            f"minimum share: {detail}",
        )
    ]
