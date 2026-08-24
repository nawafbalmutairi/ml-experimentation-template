from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_template.config import Config
from scripts.lift_analysis import DECILES, held_out_scores, render_lift_table
from tests.test_train import train_on


def ranked(actual: list[int]) -> pd.DataFrame:
    """A ranking already sorted by score, which is what render_lift_table is handed."""
    return pd.DataFrame({"score": np.linspace(1.0, 0.0, len(actual)), "actual": actual})


def test_lift_table_reports_the_lift_of_a_perfect_ranking() -> None:
    table = render_lift_table(ranked([1] * 10 + [0] * 90))

    assert "base rate: 10.00%" in table
    assert "| 1 | 10 | 10 | 100.0% | 10.00x | 100.0% |" in table


def test_lift_table_rejects_a_held_out_set_smaller_than_the_deciles() -> None:
    """Fewer rows than deciles leaves empty chunks, which used to divide by zero."""
    with pytest.raises(SystemExit, match=str(DECILES)):
        render_lift_table(ranked([1, 0] * 4 + [1]))


def test_lift_table_rejects_a_held_out_set_without_positives() -> None:
    with pytest.raises(SystemExit, match="no positives"):
        render_lift_table(ranked([0] * 40))


def test_held_out_scores_rank_the_positive_class_the_model_was_fitted_on(
    config: Config, training_frame: pd.DataFrame
) -> None:
    train_on(config, training_frame)

    scored = held_out_scores(config)

    assert set(scored["actual"]) <= {0, 1}
    assert scored["actual"].sum() > 0
    assert scored["score"].is_monotonic_decreasing
    assert render_lift_table(scored).startswith(f"Held-out rows: {len(scored)},")
