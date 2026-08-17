from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ValueRange
from src.data_processor import (
    clean_dataset,
    drop_rows_with_invalid_values,
    load_dataset,
    save_dataset,
    validate_columns,
)

FEATURES = ["age", "city"]
TARGET = "label"


def test_load_dataset_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("age,label\n30,1\n40,0\n", encoding="utf-8")

    frame = load_dataset(path)

    assert list(frame.columns) == ["age", "label"]
    assert len(frame) == 2


def test_load_dataset_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "absent.csv")


def test_validate_columns_lists_every_missing_column() -> None:
    frame = pd.DataFrame({"age": [1]})

    with pytest.raises(ValueError, match="city"):
        validate_columns(frame, ["age", "city", "label"])


def test_clean_dataset_drops_rows_without_target() -> None:
    frame = pd.DataFrame(
        {"age": [30, 40, 50], "city": ["a", "b", "c"], TARGET: [1, None, 0]},
    )

    cleaned = clean_dataset(frame, FEATURES, TARGET)

    assert len(cleaned) == 2
    assert cleaned[TARGET].notna().all()


def test_clean_dataset_keeps_rows_with_missing_features() -> None:
    frame = pd.DataFrame({"age": [30, None], "city": ["a", "b"], TARGET: [1, 0]})

    cleaned = clean_dataset(frame, FEATURES, TARGET)

    assert len(cleaned) == 2


def test_clean_dataset_removes_duplicates_and_unused_columns() -> None:
    frame = pd.DataFrame(
        {
            "age": [30, 30, 40],
            "city": ["a", "a", "b"],
            TARGET: [1, 1, 0],
            "note": ["keep", "out", "of it"],
        }
    )

    cleaned = clean_dataset(frame, FEATURES, TARGET)

    assert list(cleaned.columns) == [*FEATURES, TARGET]
    assert len(cleaned) == 2


def test_clean_dataset_rejects_missing_target_column() -> None:
    frame = pd.DataFrame({"age": [30], "city": ["a"]})

    with pytest.raises(ValueError, match=TARGET):
        clean_dataset(frame, FEATURES, TARGET)


def test_drop_rows_with_invalid_values_removes_out_of_range_rows() -> None:
    frame = pd.DataFrame({"age": [30.0, -5.0, 200.0], "city": ["a", "b", "c"]})

    kept = drop_rows_with_invalid_values(frame, {"age": ValueRange(minimum=0, maximum=120)})

    assert kept["age"].tolist() == [30.0]


def test_drop_rows_with_invalid_values_removes_infinities_without_configured_ranges() -> None:
    frame = pd.DataFrame({"age": [30.0, np.inf, -np.inf], "city": ["a", "b", "c"]})

    kept = drop_rows_with_invalid_values(frame, {})

    assert kept["age"].tolist() == [30.0]


def test_drop_rows_with_invalid_values_keeps_missing_values() -> None:
    frame = pd.DataFrame({"age": [30.0, None], "city": ["a", "b"]})

    kept = drop_rows_with_invalid_values(frame, {"age": ValueRange(minimum=0)})

    assert len(kept) == 2


def test_drop_rows_with_invalid_values_ignores_unconfigured_and_absent_columns() -> None:
    frame = pd.DataFrame({"age": [-5.0, 30.0], "city": ["a", "b"]})

    kept = drop_rows_with_invalid_values(frame, {"absent": ValueRange(minimum=0)})

    assert len(kept) == 2


def test_clean_dataset_removes_rows_outside_the_configured_range() -> None:
    frame = pd.DataFrame(
        {"age": [30.0, -5.0], "city": ["a", "b"], TARGET: [1, 0]},
    )

    cleaned = clean_dataset(frame, FEATURES, TARGET, {"age": ValueRange(minimum=0)})

    assert cleaned["age"].tolist() == [30.0]


def test_clean_dataset_keeps_every_row_when_no_ranges_are_configured() -> None:
    frame = pd.DataFrame({"age": [30.0, -5.0], "city": ["a", "b"], TARGET: [1, 0]})

    assert len(clean_dataset(frame, FEATURES, TARGET)) == 2


def test_save_dataset_creates_parent_directories(tmp_path: Path) -> None:
    frame = pd.DataFrame({"age": [30]})
    path = tmp_path / "nested" / "out.csv"

    save_dataset(frame, path)

    assert load_dataset(path).equals(frame)
