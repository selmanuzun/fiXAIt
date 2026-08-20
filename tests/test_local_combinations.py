from __future__ import annotations

import numpy as np
import pandas as pd

from fixait.combinations import (
    canonicalize_combinations,
    generate_cyclic_ecfc_combinations,
    generate_exhaustive_combinations,
    generate_local_combinations,
)
from fixait.local import explain_local


class _LinearProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        values = np.asarray(X, dtype=float)
        coefficients = np.linspace(0.2, 1.0, values.shape[1])
        logits = values @ coefficients
        probability = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probability, probability])


def test_exhaustive_generation_contains_every_non_empty_proper_subset():
    features = ["a", "b", "c", "d"]
    combinations = generate_exhaustive_combinations(features)

    assert len(combinations) == 14
    assert ["a", "c"] in combinations
    assert ["b", "d"] in combinations
    assert [] not in combinations
    assert features not in combinations
    assert len({tuple(combination) for combination in combinations}) == 14


def test_local_strategy_switches_at_seven_candidate_features():
    six_features = [f"f{index}" for index in range(6)]
    seven_features = [f"f{index}" for index in range(7)]

    six_combinations, six_strategy = generate_local_combinations(six_features)
    seven_combinations, seven_strategy = generate_local_combinations(seven_features)

    assert six_strategy == "exhaustive"
    assert len(six_combinations) == (2**6) - 2
    assert seven_strategy == "ecfc"
    assert len(seven_combinations) == 7 * 6
    assert seven_combinations == generate_cyclic_ecfc_combinations(seven_features)


def test_local_strategy_can_force_ecfc_below_the_automatic_threshold():
    below_threshold = [f"f{index}" for index in range(6)]
    at_threshold = [f"f{index}" for index in range(7)]

    forced_ecfc, ecfc_strategy = generate_local_combinations(
        below_threshold, strategy="ecfc"
    )
    forced_exhaustive, exhaustive_strategy = generate_local_combinations(
        at_threshold, strategy="exhaustive"
    )

    assert ecfc_strategy == "ecfc"
    assert len(forced_ecfc) == 6 * 5
    assert exhaustive_strategy == "exhaustive"
    assert len(forced_exhaustive) == (2**7) - 2


def test_seven_feature_local_explanation_uses_the_ecfc_branch_end_to_end():
    features = [f"f{index}" for index in range(7)]
    rng = np.random.RandomState(42)
    reference = pd.DataFrame(rng.normal(size=(40, 7)), columns=features)
    combinations, strategy = generate_local_combinations(features)

    result = explain_local(
        model=_LinearProbabilityModel(),
        X_reference=reference,
        x_instance=reference.iloc[0],
        selected_features=features,
        combinations=combinations,
        alphas=[0.01, 0.1, 1.0],
        baseline_method="median",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        target_class=1,
        combination_strategy=strategy,
        random_state=42,
    )

    assert result.metadata["combination_strategy"] == "ecfc"
    assert result.metadata["n_combinations"] == 42
    assert result.metadata["n_surrogate_rows"] == 43
    assert result.metadata["combination_space_complete"] is False
    assert result.metadata["empty_coalition_included"] is False
    assert result.metadata["empty_coalition_score"] is None
    assert result.metadata["full_coalition_included"] is True


def test_exhaustive_surrogate_includes_empty_and_full_coalitions():
    features = [f"f{index}" for index in range(4)]
    rng = np.random.RandomState(7)
    reference = pd.DataFrame(rng.normal(size=(30, 4)), columns=features)
    combinations, strategy = generate_local_combinations(features)
    model = _LinearProbabilityModel()

    result = explain_local(
        model=model,
        X_reference=reference,
        x_instance=reference.iloc[0],
        selected_features=features,
        combinations=combinations,
        alphas=[0.01, 0.1, 1.0],
        baseline_method="median",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        target_class=1,
        combination_strategy=strategy,
        random_state=42,
    )
    baseline_frame = pd.DataFrame([reference.median()], columns=features)
    expected_empty_score = float(model.predict_proba(baseline_frame)[0, 1])

    assert result.metadata["combination_strategy"] == "exhaustive"
    assert result.metadata["n_combinations"] == 14
    assert result.metadata["n_surrogate_rows"] == 16
    assert result.metadata["empty_coalition_included"] is True
    assert result.metadata["full_coalition_included"] is True
    assert result.metadata["sc_includes_empty_coalition"] is False
    assert np.isclose(result.metadata["empty_coalition_score"], expected_empty_score)
    assert result.metadata["ridge_cv_strategy"] == "canonical_shuffled_kfold"
    assert result.metadata["ridge_cv_shuffle"] is True
    assert result.metadata["ridge_cv_random_state"] == 42


def test_local_fei_and_alpha_are_invariant_to_coalition_row_order():
    features = [f"f{index}" for index in range(5)]
    rng = np.random.RandomState(11)
    reference = pd.DataFrame(rng.normal(size=(35, 5)), columns=features)
    combinations, strategy = generate_local_combinations(features)
    shuffled = [list(reversed(coalition)) for coalition in combinations]
    rng.shuffle(shuffled)

    common = dict(
        model=_LinearProbabilityModel(),
        X_reference=reference,
        x_instance=reference.iloc[0],
        selected_features=features,
        alphas=[0.001, 0.01, 0.1, 1.0],
        baseline_method="median",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        target_class=1,
        combination_strategy=strategy,
        random_state=73,
    )
    ordered_result = explain_local(combinations=combinations, **common)
    shuffled_result = explain_local(combinations=shuffled, **common)

    assert canonicalize_combinations(features, shuffled) == combinations
    assert ordered_result.metadata["alpha_fei"] == shuffled_result.metadata["alpha_fei"]
    assert (
        ordered_result.metadata["alpha_legacy_fvi"]
        == shuffled_result.metadata["alpha_legacy_fvi"]
    )
    assert ordered_result.local_fei.keys() == shuffled_result.local_fei.keys()
    assert ordered_result.legacy_local_fvi.keys() == shuffled_result.legacy_local_fvi.keys()
    np.testing.assert_allclose(
        list(ordered_result.local_fei.values()),
        list(shuffled_result.local_fei.values()),
    )
    np.testing.assert_allclose(
        list(ordered_result.legacy_local_fvi.values()),
        list(shuffled_result.legacy_local_fvi.values()),
    )
