from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ml_template.config import (
    PROJECT_ROOT_ENV,
    RANDOM_SPLIT,
    TEMPORAL_SPLIT,
    load_config,
    project_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_CONFIG: dict[str, Any] = {
    "seed": 7,
    "data": {
        "raw_path": "data/raw/example.csv",
        "processed_path": "data/processed/example.csv",
        "target": "label",
    },
    "features": {"numerical": ["age"], "categorical": ["city"]},
    "split": {"test_size": 0.25},
    "model": {
        "task": "classification",
        "type": "xgboost",
        "output_path": "models/example.joblib",
        "params": {"n_estimators": 5},
    },
    "evaluation": {"results_dir": "experiments/results", "metrics": ["accuracy"]},
    "validation": {
        "min_rows": 10,
        "max_missing_fraction": 0.2,
        "max_duplicate_fraction": 0.05,
        "min_class_fraction": 0.05,
        "report_dir": "reports/data_quality",
        "value_ranges": {"age": {"min": 0, "max": 120}},
    },
}


def write_config(tmp_path: Path, **overrides: Any) -> Path:
    raw = {**VALID_CONFIG, **overrides}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_load_config_reads_every_section(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.seed == 7
    assert config.data.target == "label"
    assert config.features.all_features == ["age", "city"]
    assert config.split.test_size == 0.25
    assert config.model.params == {"n_estimators": 5}
    assert config.evaluation.metrics == ["accuracy"]
    assert config.validation.min_rows == 10
    assert config.validation.value_ranges["age"].maximum == 120


def test_load_config_resolves_relative_paths_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copied project must resolve its own data, wherever the package is installed."""
    project = tmp_path / "some-project"
    project.mkdir()
    monkeypatch.chdir(project)

    config = load_config(write_config(tmp_path))

    assert config.data.raw_path == project / "data" / "raw" / "example.csv"
    assert config.model.output_path.is_absolute()


def test_relative_paths_follow_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path)
    first = tmp_path / "project-one"
    second = tmp_path / "project-two"
    first.mkdir()
    second.mkdir()

    monkeypatch.chdir(first)
    from_first = load_config(config_path).data.raw_path
    monkeypatch.chdir(second)
    from_second = load_config(config_path).data.raw_path

    assert from_first != from_second
    assert from_first.is_relative_to(first)
    assert from_second.is_relative_to(second)


def test_project_root_env_var_overrides_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(elsewhere))

    assert project_root() == elsewhere.resolve()
    assert load_config(write_config(tmp_path)).data.raw_path.is_relative_to(elsewhere)


def test_load_config_keeps_absolute_paths(tmp_path: Path) -> None:
    absolute = (tmp_path / "elsewhere.csv").resolve()
    data = {**VALID_CONFIG["data"], "raw_path": str(absolute)}

    config = load_config(write_config(tmp_path, data=data))

    assert config.data.raw_path == absolute


def test_load_config_defaults_the_separator_to_a_comma(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path))

    assert config.data.separator == ","
    assert config.data.target_mapping == {}


def test_load_config_reads_the_separator_and_target_mapping(tmp_path: Path) -> None:
    data = {**VALID_CONFIG["data"], "separator": ";", "target_mapping": {"no": 0, "yes": 1}}

    config = load_config(write_config(tmp_path, data=data))

    assert config.data.separator == ";"
    assert config.data.target_mapping == {"no": 0, "yes": 1}


def test_load_config_rejects_unquoted_yaml_boolean_label_keys(tmp_path: Path) -> None:
    """Bare yes/no in YAML parse as booleans, which would never match the CSV's labels."""
    path = tmp_path / "config.yaml"
    raw = {**VALID_CONFIG, "data": {**VALID_CONFIG["data"], "target_mapping": {False: 0, True: 1}}}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="quoted"):
        load_config(path)


def test_load_config_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_load_config_rejects_a_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_rejects_a_missing_section(tmp_path: Path) -> None:
    incomplete = {key: value for key, value in VALID_CONFIG.items() if key != "validation"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(incomplete), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        load_config(path)


def test_load_config_rejects_an_unsupported_task(tmp_path: Path) -> None:
    model = {**VALID_CONFIG["model"], "task": "clustering"}

    with pytest.raises(ValueError, match="clustering"):
        load_config(write_config(tmp_path, model=model))


def test_load_config_defaults_the_split_strategy_to_random(tmp_path: Path) -> None:
    assert load_config(write_config(tmp_path)).split.strategy == RANDOM_SPLIT


def test_load_config_reads_a_temporal_split_strategy(tmp_path: Path) -> None:
    split = {**VALID_CONFIG["split"], "strategy": "temporal"}

    assert load_config(write_config(tmp_path, split=split)).split.strategy == TEMPORAL_SPLIT


def test_load_config_rejects_an_unsupported_split_strategy(tmp_path: Path) -> None:
    split = {**VALID_CONFIG["split"], "strategy": "grouped"}

    with pytest.raises(ValueError, match="grouped"):
        load_config(write_config(tmp_path, split=split))


@pytest.mark.parametrize("test_size", [0.0, 1.0, 1.5, -0.2])
def test_load_config_rejects_an_out_of_range_test_size(tmp_path: Path, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        load_config(write_config(tmp_path, split={"test_size": test_size}))


@pytest.mark.parametrize(
    "field", ["max_missing_fraction", "max_duplicate_fraction", "min_class_fraction"]
)
def test_load_config_rejects_out_of_range_validation_fractions(tmp_path: Path, field: str) -> None:
    validation = {**VALID_CONFIG["validation"], field: 1.5}

    with pytest.raises(ValueError, match=field):
        load_config(write_config(tmp_path, validation=validation))


def test_load_config_rejects_a_non_positive_minimum_row_count(tmp_path: Path) -> None:
    validation = {**VALID_CONFIG["validation"], "min_rows": 0}

    with pytest.raises(ValueError, match="min_rows"):
        load_config(write_config(tmp_path, validation=validation))


def test_load_config_rejects_a_value_range_with_min_above_max(tmp_path: Path) -> None:
    validation = {**VALID_CONFIG["validation"], "value_ranges": {"age": {"min": 10, "max": 5}}}

    with pytest.raises(ValueError, match="age"):
        load_config(write_config(tmp_path, validation=validation))


def test_load_config_rejects_a_value_range_that_is_not_a_mapping(tmp_path: Path) -> None:
    validation = {**VALID_CONFIG["validation"], "value_ranges": {"age": [0, 120]}}

    with pytest.raises(ValueError, match="age"):
        load_config(write_config(tmp_path, validation=validation))


def test_load_config_accepts_omitted_value_ranges(tmp_path: Path) -> None:
    validation = dict(VALID_CONFIG["validation"])
    del validation["value_ranges"]

    config = load_config(write_config(tmp_path, validation=validation))

    assert config.validation.value_ranges == {}


def test_config_path_from_the_environment_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_config(tmp_path)
    monkeypatch.setenv("ML_CONFIG_PATH", str(path))

    assert load_config().seed == 7


def test_shipped_config_matches_the_shipped_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The template must train on a fresh clone using the documented command."""
    monkeypatch.chdir(REPO_ROOT)
    config = load_config(REPO_ROOT / "config" / "config.yaml")
    assert config.data.raw_path.is_file()

    frame = pd.read_csv(config.data.raw_path)
    for column in [*config.features.all_features, config.data.target]:
        assert column in frame.columns

    assert set(config.validation.value_ranges).issubset(config.features.numerical)
    assert len(frame) >= config.validation.min_rows
