from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_columns(frame: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def drop_rows_with_missing_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    return frame.dropna(subset=[target])


def clean_dataset(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
) -> pd.DataFrame:
    """Return the modelling columns with unusable rows removed.

    Missing feature values are deliberately left in place: they are imputed inside the
    preprocessing pipeline so that imputation statistics are learned from training data only.
    """
    validate_columns(frame, [*feature_columns, target])
    selected = frame[[*feature_columns, target]]
    cleaned = drop_rows_with_missing_target(selected, target)
    return cleaned.drop_duplicates().reset_index(drop=True)


def save_dataset(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
