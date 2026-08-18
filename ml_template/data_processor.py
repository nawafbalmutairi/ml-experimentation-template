from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ml_template.config import ValueRange


def load_dataset(path: Path, separator: str = ",") -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, sep=separator)


def validate_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def encode_target(frame: pd.DataFrame, target: str, mapping: dict[str, int]) -> pd.DataFrame:
    """Replace class labels with the integers the estimator expects.

    Unmapped labels are an error rather than a silent NaN, which cleaning would drop as a
    missing target and quietly shrink the dataset.
    """
    if not mapping:
        return frame

    validate_columns(frame, [target])
    unmapped = set(frame[target].dropna().unique()) - set(mapping)
    if unmapped:
        raise ValueError(
            f"data.target_mapping has no entry for labels {sorted(unmapped, key=repr)} "
            f"in column '{target}'"
        )
    return frame.assign(**{target: frame[target].map(mapping)})


def drop_rows_with_missing_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame.dropna(subset=[target])


def drop_rows_with_invalid_values(
    frame: pd.DataFrame,
    value_ranges: dict[str, ValueRange],
) -> pd.DataFrame:
    """Drop rows holding an infinite value, or a value outside a configured bound.

    Missing values are kept: they are imputed inside the preprocessing pipeline.
    """
    numeric = frame.select_dtypes(include="number")
    keep = ~np.isinf(numeric).any(axis=1) if not numeric.empty else pd.Series(True, frame.index)

    for column, bounds in value_ranges.items():
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column]
        if bounds.minimum is not None:
            keep &= values.isna() | (values >= bounds.minimum)
        if bounds.maximum is not None:
            keep &= values.isna() | (values <= bounds.maximum)

    return frame[keep]


def clean_dataset(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
    value_ranges: dict[str, ValueRange] | None = None,
) -> pd.DataFrame:
    """Return the modelling columns with unusable rows removed.

    Missing feature values are deliberately left in place: they are imputed inside the
    preprocessing pipeline so that imputation statistics are learned from training data only.
    """
    validate_columns(frame, [*feature_columns, target])
    selected = frame[[*feature_columns, target]]
    cleaned = drop_rows_with_missing_target(selected, target)
    cleaned = drop_rows_with_invalid_values(cleaned, value_ranges or {})
    return cleaned.drop_duplicates().reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
