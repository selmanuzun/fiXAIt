from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig
from fixait.optimization import (
    optimize_fei_rank_gradient,
    summarize_fei_weight_change,
)
from fixait.selection import select_global_impacts


def _data():
    values, labels = make_classification(
        n_samples=240,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=23,
    )
    return (
        pd.DataFrame(values, columns=[f"f{index}" for index in range(7)]),
        pd.Series(labels, name="decision"),
    )


def _model():
    return RandomForestClassifier(
        n_estimators=12,
        max_depth=4,
        random_state=42,
        n_jobs=1,
    )


def test_notebook_threshold_filters_fei_and_fvi_together():
    fei, fvi, dropped = select_global_impacts(
        {"a": 6.0, "b": 3.0, "c": 0.1, "d": 0.0, "e": -1.0},
        {"a": 1.2349, "b": -2.3451, "c": 3.0, "d": 4.0, "e": 5.0},
        threshold_pct=3.0,
        inclusive=True,
        drop_non_positive=True,
    )

    assert fei == {"a": 6.0, "b": 3.0}
    assert fvi == {"a": 1.235, "b": -2.345}
    assert dropped == ["c", "d", "e"]


def test_rank_gradient_preserves_signs_and_follows_permutation_ranking():
    optimized = optimize_fei_rank_gradient(
        {"a": 0.20, "b": -0.80, "c": 0.50},
        {"a": 0.90, "b": 0.10, "c": 0.50},
        n_steps=40,
        learning_rate=0.05,
        random_state=42,
        reg_lambda=0.05,
    )

    assert optimized["a"] > optimized["c"] > abs(optimized["b"])
    assert optimized["b"] <= 0.0


def test_rank_gradient_respects_the_per_feature_percentage_limit():
    original = {"a": 2.0, "b": -1.0, "c": 0.1, "zero": 0.0}
    optimized = optimize_fei_rank_gradient(
        original,
        {"a": 0.1, "b": 0.5, "c": 0.9, "zero": 1.0},
        n_steps=80,
        learning_rate=0.05,
        random_state=42,
        reg_lambda=0.0,
        max_weight_change_pct=20.0,
    )
    changes = summarize_fei_weight_change(original, optimized)

    assert changes["max_weight_change_pct"] <= 20.0001
    assert optimized["b"] <= 0.0
    assert optimized["zero"] == 0.0
    for feature in ["a", "b", "c"]:
        ratio = abs(optimized[feature]) / abs(original[feature])
        assert 0.799999 <= ratio <= 1.200001


def test_true_false_selects_one_final_global_output_mode():
    X, y = _data()
    common = dict(
        group_size=5,
        faithfulness_runs_per_feature=2,
        n_jobs=1,
        model_n_jobs=1,
        random_state=42,
    )
    base_explainer = FiXAIt(
        _model(),
        config=FiXAItConfig(**common, optimize_faithfulness=False),
    ).fit(X, y)
    optimized_explainer = FiXAIt(
        _model(),
        config=FiXAItConfig(
            **common,
            optimize_faithfulness=True,
            faithfulness_optimizer_steps=40,
            faithfulness_accept_only_if_improved=False,
        ),
    ).fit(X, y)

    base_result = base_explainer.explain_global()
    optimized_result = optimized_explainer.explain_global()

    assert not base_result.optimization_applied
    assert optimized_result.optimization_applied
    assert base_result.metadata["optimization"]["method"] is None
    assert optimized_result.metadata["optimization"]["method"] == "rank_gradient"
    assert optimized_result.metadata["optimization"]["requested"] is True
    assert optimized_result.metadata["optimization"]["accepted"] is True
    assert (
        optimized_result.metadata["optimization"]["reason"]
        == "acceptance_guard_disabled"
    )
    assert (
        optimized_result.metadata["optimization"]["max_weight_change_pct"]
        <= 20.0001
    )
    assert set(base_result.global_fei) == set(base_result.global_fvi)
    assert set(optimized_result.global_fei) == set(optimized_result.global_fvi)
    assert not hasattr(base_result, "raw_global_fei")
    assert not hasattr(optimized_result, "optimized_global_fei")
    assert optimized_result.global_fei != base_result.global_fei
    assert optimized_result.global_sc == base_result.global_sc

    assert base_explainer.core_ is not None
    notebook_fvi = base_explainer.core_.compute_value_impact(normalize=False)
    expected_fei, expected_fvi, _ = select_global_impacts(
        base_explainer.core_.new_weight_format or {},
        notebook_fvi,
        threshold_pct=3.0,
        inclusive=True,
        drop_non_positive=True,
    )
    assert base_result.global_fei == expected_fei
    assert base_result.global_fvi == expected_fvi
    for feature, value in base_result.global_fvi.items():
        assert value == round(notebook_fvi[feature], 3)


def test_optimization_requires_a_validation_split():
    try:
        FiXAItConfig(
            optimize_faithfulness=True,
            validation_size=0.0,
        )
    except ValueError as error:
        assert "validation_size" in str(error)
    else:
        raise AssertionError("Expected validation-size validation to fail.")


def test_guard_rejects_candidate_below_the_required_improvement():
    X, y = _data()
    common = dict(
        group_size=5,
        faithfulness_runs_per_feature=2,
        n_jobs=1,
        model_n_jobs=1,
        random_state=42,
    )
    base_result = FiXAIt(
        _model(),
        config=FiXAItConfig(**common, optimize_faithfulness=False),
    ).fit(X, y).explain_global()
    guarded_result = FiXAIt(
        _model(),
        config=FiXAItConfig(
            **common,
            optimize_faithfulness=True,
            faithfulness_optimizer_steps=20,
            faithfulness_min_improvement=3.0,
        ),
    ).fit(X, y).explain_global()

    metadata = guarded_result.metadata["optimization"]
    assert metadata["requested"] is True
    assert metadata["accepted"] is False
    assert metadata["applied"] is False
    assert metadata["reason"] == "validation_improvement_below_threshold"
    assert metadata["mean_weight_change_pct"] == 0.0
    assert metadata["max_weight_change_pct"] == 0.0
    assert metadata["candidate_max_weight_change_pct"] <= 20.0001
    assert guarded_result.optimization_applied is False
    assert guarded_result.global_fei == base_result.global_fei
    assert guarded_result.global_fvi == base_result.global_fvi


def test_guard_configuration_validates_shift_and_improvement_bounds():
    for kwargs, expected in [
        ({"faithfulness_min_improvement": -0.01}, "min_improvement"),
        ({"faithfulness_max_weight_change_pct": -0.01}, "max_weight_change_pct"),
        ({"faithfulness_max_weight_change_pct": 100.01}, "max_weight_change_pct"),
    ]:
        try:
            FiXAItConfig(**kwargs)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"Expected validation failure for {kwargs}.")
