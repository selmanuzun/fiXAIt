from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig
from fixait.local import explain_local
from fixait.selection import select_local_impacts


class _DominantFeatureModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        frame = pd.DataFrame(X, columns=["dominant", "weak_a", "weak_b"])
        logit = (
            8.0 * frame["dominant"].to_numpy()
            + 0.01 * frame["weak_a"].to_numpy()
            - 0.01 * frame["weak_b"].to_numpy()
        )
        probability = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - probability, probability])


def test_local_threshold_uses_absolute_share_and_preserves_signs():
    fei = {"positive": 0.78, "negative": -0.20, "weak": 0.02}
    fvi = {"positive": 4.0, "negative": -2.0, "weak": 1.0}

    selected_fei, selected_fvi, dropped = select_local_impacts(
        fei,
        fvi,
        threshold_pct=3.0,
    )

    assert list(selected_fei) == ["positive", "negative"]
    assert selected_fei["negative"] == -0.20
    assert selected_fvi == {"positive": 4.0, "negative": -2.0}
    assert dropped == ["weak"]


def test_local_threshold_is_inclusive_and_none_disables_it():
    fei = {"large": 0.70, "negative": -0.27, "exact": 0.03}
    fvi = {feature: index + 1.0 for index, feature in enumerate(fei)}

    inclusive_fei, _, inclusive_dropped = select_local_impacts(
        fei,
        fvi,
        threshold_pct=3.0,
        inclusive=True,
    )
    all_fei, all_fvi, no_dropped = select_local_impacts(
        fei,
        fvi,
        threshold_pct=None,
    )

    assert "exact" not in inclusive_fei
    assert inclusive_dropped == ["exact"]
    assert all_fei == fei
    assert all_fvi == fvi
    assert no_dropped == []


def test_local_output_filters_all_fvi_views_and_renormalizes():
    values, target = make_classification(
        n_samples=220,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        random_state=17,
    )
    X = pd.DataFrame(values, columns=[f"x{index}" for index in range(7)])
    explainer = FiXAIt(
        RandomForestClassifier(
            n_estimators=12,
            max_depth=4,
            random_state=42,
            n_jobs=1,
        ),
        config=FiXAItConfig(
            group_size=5,
            fei_threshold_pct=20.0,
            n_jobs=1,
            model_n_jobs=1,
        ),
    ).fit(X, target)

    result = explainer.explain_local(X.iloc[0])
    selected = set(result.selected_features)

    assert selected == set(result.local_fei)
    assert selected == set(result.local_fvi)
    assert selected == set(result.raw_local_fvi)
    assert selected == set(result.legacy_local_fvi)
    assert selected.isdisjoint(result.dropped_features)
    assert selected | set(result.dropped_features) == set(
        result.metadata["candidate_features"]
    )
    normalized_sum = sum(abs(value) for value in result.local_fvi.values())
    assert np.isclose(normalized_sum, 0.0) or np.isclose(normalized_sum, 1.0)


def test_one_selected_local_feature_preserves_its_negative_direction():
    fei = {"dominant": -0.98, "weak_a": 0.01, "weak_b": -0.01}
    fvi = {"dominant": -3.0, "weak_a": 1.0, "weak_b": -1.0}

    selected_fei, selected_fvi, dropped = select_local_impacts(
        fei,
        fvi,
        threshold_pct=5.0,
    )

    assert selected_fei == {"dominant": -0.98}
    assert selected_fvi == {"dominant": -3.0}
    assert dropped == ["weak_a", "weak_b"]


def test_one_feature_local_result_marks_rank_metrics_non_informative():
    features = ["dominant", "weak_a", "weak_b"]
    reference = pd.DataFrame(
        {
            "dominant": [-1.0, 0.0, 1.0, -0.5, 0.5],
            "weak_a": [-1.0, 0.0, 1.0, 0.5, -0.5],
            "weak_b": [1.0, 0.0, -1.0, 0.5, -0.5],
        }
    )
    combinations = [
        ["dominant"],
        ["weak_a"],
        ["weak_b"],
        ["dominant", "weak_a"],
        ["dominant", "weak_b"],
        ["weak_a", "weak_b"],
    ]

    result = explain_local(
        model=_DominantFeatureModel(),
        X_reference=reference,
        x_instance=pd.Series({feature: 1.0 for feature in features}),
        selected_features=features,
        combinations=combinations,
        alphas=[0.001, 0.01, 0.1],
        baseline_method="zero",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        fei_threshold_pct=5.0,
        random_state=42,
        local_faithfulness_runs_per_feature=5,
    )

    assert result.selected_features == ["dominant"]
    assert result.dropped_features == ["weak_a", "weak_b"]
    assert result.fei_fvi_agreement_spearman == 0.0
    assert result.local_faithfulness_spearman == 0.0
    assert result.metadata["faithfulness_informative"] is False
    assert result.metadata["fei_fvi_agreement_informative"] is False
    assert result.metadata["local_faithfulness_informative"] is False
    assert result.metadata["fidelity_informative"] is True
    assert set(result.local_fei) == set(result.local_fvi) == {"dominant"}


def test_local_faithfulness_uses_repeated_reference_perturbations():
    features = ["dominant", "weak_a", "weak_b"]
    reference = pd.DataFrame(
        {
            "dominant": [0.1, 0.2, 0.3, 0.4, 0.5],
            "weak_a": [-0.5, -0.25, 0.25, 0.5, 0.75],
            "weak_b": [0.75, 0.5, 0.25, -0.25, -0.5],
        }
    )
    combinations = [
        ["dominant"],
        ["weak_a"],
        ["weak_b"],
        ["dominant", "weak_a"],
        ["dominant", "weak_b"],
        ["weak_a", "weak_b"],
    ]
    common = dict(
        model=_DominantFeatureModel(),
        X_reference=reference,
        x_instance=pd.Series({feature: 1.0 for feature in features}),
        selected_features=features,
        combinations=combinations,
        alphas=[0.001, 0.01, 0.1],
        baseline_method="zero",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        fei_threshold_pct=None,
        random_state=17,
        local_faithfulness_runs_per_feature=len(reference),
    )

    first = explain_local(**common)
    second = explain_local(**common)

    impacts = first.metadata["local_faithfulness_impacts"]
    assert set(impacts) == set(features)
    assert first.metadata["local_faithfulness_runs_per_feature"] == len(reference)
    assert sorted(first.metadata["local_faithfulness_reference_indices"]) == list(
        range(len(reference))
    )
    assert all(value >= 0.0 and np.isfinite(value) for value in impacts.values())
    assert first.local_faithfulness_spearman == second.local_faithfulness_spearman
    assert impacts == second.metadata["local_faithfulness_impacts"]
    assert any(
        not np.isclose(impacts[feature], abs(first.raw_local_fvi[feature]))
        for feature in features
    )


def test_optional_local_faithfulness_optimization_is_bounded_and_keeps_sc():
    pytest.importorskip("torch")
    features = ["dominant", "weak_a", "weak_b"]
    reference = pd.DataFrame(
        {
            "dominant": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            "weak_a": [-0.8, -0.4, 0.1, 0.4, 0.8, 1.2],
            "weak_b": [1.1, 0.7, 0.3, -0.2, -0.6, -1.0],
        }
    )
    combinations = [
        ["dominant"],
        ["weak_a"],
        ["weak_b"],
        ["dominant", "weak_a"],
        ["dominant", "weak_b"],
        ["weak_a", "weak_b"],
    ]
    common = dict(
        model=_DominantFeatureModel(),
        X_reference=reference,
        x_instance=pd.Series({feature: 0.9 for feature in features}),
        selected_features=features,
        combinations=combinations,
        alphas=[0.001, 0.01, 0.1],
        baseline_method="zero",
        score_mode="proba",
        sc_metric="legacy_overlap",
        fvi_method="finite_difference",
        fei_threshold_pct=None,
        random_state=23,
        local_faithfulness_runs_per_feature=2,
    )

    base = explain_local(**common)
    optimized = explain_local(
        **common,
        optimize_faithfulness=True,
        local_faithfulness_calibration_runs_per_feature=2,
        local_faithfulness_optimizer_steps=25,
        local_faithfulness_accept_only_if_improved=False,
        local_faithfulness_max_weight_change_pct=20.0,
    )

    assert base.optimization_applied is False
    assert optimized.optimization_applied is True
    assert optimized.metadata["optimization"]["requested"] is True
    assert optimized.metadata["optimization"]["accepted"] is True
    assert optimized.metadata["local_sc_uses_optimized_fei"] is False
    assert optimized.local_sc == base.local_sc
    assert (
        optimized.metadata["optimization"]["max_weight_change_pct"]
        <= 20.0 + 1e-6
    )
    assert (
        optimized.metadata["local_faithfulness_evaluation_overlap_count"] == 0
    )
    for feature in features:
        assert np.sign(optimized.local_fei[feature]) == np.sign(base.local_fei[feature])
    assert any(
        not np.isclose(optimized.local_fei[feature], base.local_fei[feature])
        for feature in features
    )
