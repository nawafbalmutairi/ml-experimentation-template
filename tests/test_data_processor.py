from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_processor import (
    clean_dataset,
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


def test_save_dataset_creates_parent_directories(tmp_path: Path) -> None:
    frame = pd.DataFrame({"age": [30]})
    path = tmp_path / "nested" / "out.csv"

    save_dataset(frame, path)

    assert load_dataset(path).equals(frame)
