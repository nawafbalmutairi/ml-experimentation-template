from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml_template.feature_engineer import build_preprocessor
from ml_template.group_encoder import GroupTargetEncoder


def households(outcomes: dict[str, list[int]]) -> tuple[pd.DataFrame, pd.Series]:
    keys = [name for name, members in outcomes.items() for _ in members]
    values = [value for members in outcomes.values() for value in members]
    return pd.DataFrame({"household": keys}), pd.Series(values)


def test_a_row_is_never_encoded_with_its_own_label() -> None:
    """The whole point: otherwise every row carries its own answer and the model memorises it."""
    frame, target = households({"a": [1, 0, 0]})

    encoded = GroupTargetEncoder("household").fit_transform(frame, target)

    # Each row sees the mean of the other two, never itself.
    assert encoded[:, 0].tolist() == [0.0, 0.5, 0.5]


def test_a_group_of_one_reports_nothing_rather_than_its_own_label() -> None:
    frame, target = households({"alone": [1]})

    encoded = GroupTargetEncoder("household").fit_transform(frame, target)

    assert np.isnan(encoded[0, 0])
    assert encoded[0, 1] == 0.0


def test_unseen_groups_are_reported_as_unknown() -> None:
    frame, target = households({"a": [1, 1]})
    encoder = GroupTargetEncoder("household").fit(frame, target)

    encoded = encoder.transform(pd.DataFrame({"household": ["a", "b"]}))

    assert encoded[0].tolist() == [1.0, 1.0]
    assert np.isnan(encoded[1, 0])
    assert encoded[1, 1] == 0.0


def test_new_rows_are_scored_from_training_rows_only() -> None:
    """A held-out row must be encoded from what training knew, not from its own outcome."""
    frame, target = households({"a": [1, 1, 1]})
    encoder = GroupTargetEncoder("household").fit(frame, target)

    encoded = encoder.transform(pd.DataFrame({"household": ["a"]}))

    assert encoded[0, 0] == 1.0


def test_transform_uses_only_the_rows_the_encoder_was_fitted_on() -> None:
    """The precise guarantee: a held-out row's encoding is computable from training rows alone.

    Asserted by construction rather than by a score, so it fails if a single validation label
    reaches the transform.
    """
    training = pd.DataFrame({"household": ["a", "a", "b", "b"]})
    training_target = pd.Series([1, 1, 0, 0])
    encoder = GroupTargetEncoder("household").fit(training, training_target)

    # Held-out rows from the same households, with the opposite outcomes.
    held_out = pd.DataFrame({"household": ["a", "b"]})
    encoded = encoder.transform(held_out)

    # Rates reflect the training rows only; the held-out labels do not exist as far as this knows.
    assert encoded[:, 0].tolist() == [1.0, 0.0]


def test_fitting_per_fold_inflates_less_than_encoding_the_whole_dataset_first() -> None:
    """Rates computed before splitting let a fold read labels from its own validation rows."""
    generator = np.random.default_rng(0)
    groups, size = 60, 4
    outcomes = generator.integers(0, 2, size=groups)
    keys, labels = [], []
    for group in range(groups):
        for _ in range(size):
            keys.append(f"h{group}")
            labels.append(outcomes[group] if generator.random() > 0.15 else 1 - outcomes[group])

    frame, target = pd.DataFrame({"household": keys}), pd.Series(labels)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)

    sums = target.groupby(frame["household"]).transform("sum")
    counts = target.groupby(frame["household"]).transform("size")
    precomputed = ((sums - target) / (counts - 1)).to_frame("rate")
    leaky = cross_val_score(
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        precomputed,
        target,
        cv=folds,
        scoring="accuracy",
    )
    fitted = cross_val_score(
        Pipeline(
            [
                ("pre", build_preprocessor([], [], ["household"])),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        frame,
        target,
        cv=folds,
        scoring="accuracy",
    )

    assert fitted.mean() < leaky.mean(), (
        f"fitting per fold should report a lower, honester number: "
        f"fitted {fitted.mean():.4f} vs precomputed {leaky.mean():.4f}"
    )


def test_fit_requires_the_target() -> None:
    frame, _ = households({"a": [1, 1]})

    with pytest.raises(ValueError, match="target"):
        GroupTargetEncoder("household").fit(frame)


def test_preprocessor_emits_a_rate_and_a_known_flag_per_key() -> None:
    frame = pd.DataFrame({"household": ["a", "a", "b"], "age": [1.0, 2.0, 3.0]})
    target = pd.Series([1, 0, 1])

    preprocessor = build_preprocessor(["age"], [], ["household"])
    encoded = preprocessor.fit_transform(frame, target)

    assert encoded.shape == (3, 3)
    assert list(preprocessor.get_feature_names_out())[-2:] == [
        "group:household__household",
        "group:household__household_known",
    ]
