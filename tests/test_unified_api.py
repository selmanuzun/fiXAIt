from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning

from fixait import FiXAIt, FiXAItConfig


def _dataset():
    X, y = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    labels = pd.Series(np.asarray(["low", "medium", "high"])[y], name="target")
    return frame, labels


def test_unified_global_and_local_api_with_small_group_is_warning_free():
    X, y = _dataset()
    config = FiXAItConfig(
        group_size=4,
        test_size=0.20,
        validation_size=0.20,
        random_state=42,
        faithfulness_runs_per_feature=2,
        local_faithfulness_runs_per_feature=5,
        n_jobs=1,
        model_n_jobs=1,
    )
    model = RandomForestClassifier(
        n_estimators=20,
        max_depth=4,
        random_state=42,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        warnings.simplefilter("error", UserWarning)
        explainer = FiXAIt(model, config=config).fit(X, y)
        global_result = explainer.explain_global()
        local_result = explainer.explain_local(X.iloc[0], target_class="medium")

    assert 0 < len(global_result.selected_features) <= 4
    assert global_result.metadata["n_combinations"] == 12
    assert global_result.global_sc.metric == "legacy_overlap"
    assert set(global_result.global_fei) == set(global_result.global_fvi)
    assert global_result.metadata["fvi_normalized"] is False
    assert all(value == round(value, 3) for value in global_result.global_fvi.values())
    assert np.isfinite(global_result.faithfulness)
    assert np.isfinite(global_result.fidelity)

    assert local_result.target_class == "medium"
    assert 0 < len(local_result.selected_features) <= 4
    assert set(local_result.selected_features) == set(local_result.local_fei)
    assert set(local_result.local_fei) == set(local_result.local_fvi)
    assert set(local_result.local_fei) == set(local_result.raw_local_fvi)
    assert set(local_result.local_fei) == set(local_result.legacy_local_fvi)
    assert set(local_result.selected_features).isdisjoint(local_result.dropped_features)
    assert set(local_result.selected_features) | set(local_result.dropped_features) == set(
        local_result.metadata["candidate_features"]
    )
    assert local_result.metadata["n_combinations"] == 14
    assert local_result.metadata["n_surrogate_rows"] == 16
    assert local_result.metadata["combination_strategy"] == "exhaustive"
    assert local_result.metadata["combination_space_complete"] is True
    assert local_result.metadata["empty_coalition_included"] is True
    assert local_result.metadata["full_coalition_included"] is True
    assert local_result.metadata["sc_includes_empty_coalition"] is False
    assert np.isfinite(local_result.metadata["empty_coalition_score"])
    assert local_result.metadata["fvi_method"] == "finite_difference"
    assert local_result.metadata["threshold_mode"] == "absolute_local_fei_share"
    assert local_result.metadata["drop_non_positive_fei_applied"] is False
    assert local_result.metadata["local_sc_scope"] == "pre_threshold_candidate_features"
    assert local_result.metadata["fei_fvi_agreement_scope"] == (
        "post_threshold_selected_features"
    )
    assert local_result.metadata["local_faithfulness_scope"] == (
        "post_threshold_selected_features"
    )
    assert local_result.metadata["local_faithfulness_sampling"] == (
        "shared_marginal_reference_rows"
    )
    assert local_result.metadata["local_faithfulness_runs_per_feature"] == 5
    assert set(local_result.metadata["local_faithfulness_impacts"]) == set(
        local_result.selected_features
    )
    assert local_result.metadata["fidelity_scope"] == "post_threshold_selected_features"
    assert local_result.legacy_local_fvi is not None
    assert np.isfinite(local_result.fidelity_r2)
    assert np.isfinite(local_result.fei_fvi_agreement_spearman)
    assert np.isfinite(local_result.local_faithfulness_spearman)
    assert (
        local_result.faithfulness_spearman
        == local_result.fei_fvi_agreement_spearman
    )
    assert local_result.to_dict()["faithfulness_spearman"] == (
        local_result.fei_fvi_agreement_spearman
    )
    local_fvi_sum = sum(abs(v) for v in local_result.local_fvi.values())
    assert np.isclose(local_fvi_sum, 0.0) or np.isclose(local_fvi_sum, 1.0)


def test_local_legacy_fvi_is_explicitly_selectable():
    X, y = _dataset()
    explainer = FiXAIt(
        RandomForestClassifier(n_estimators=12, max_depth=3, random_state=42, n_jobs=1),
        config=FiXAItConfig(
            group_size=4,
            local_faithfulness_runs_per_feature=3,
            n_jobs=1,
            model_n_jobs=1,
        ),
    ).fit(X, y)

    result = explainer.explain_local(
        X.iloc[1],
        target_class="high",
        fvi_method="legacy_ridge",
    )

    assert result.metadata["fvi_method"] == "legacy_ridge"
    assert result.legacy_local_fvi is not None
    assert result.raw_local_fvi == result.legacy_local_fvi


def test_fit_accepts_exact_precomputed_split_indices():
    X, y = _dataset()
    supplied = {
        "train": np.arange(0, 156),
        "validation": np.arange(156, 208),
        "test": np.arange(208, 260),
    }
    explainer = FiXAIt(
        RandomForestClassifier(n_estimators=8, max_depth=3, random_state=5, n_jobs=1),
        config=FiXAItConfig(group_size=4, n_jobs=1, model_n_jobs=1),
    ).fit(X, y, split_indices=supplied)

    split = explainer.core_.get_splits()
    assert np.array_equal(split.train_idx, supplied["train"])
    assert np.array_equal(split.opt_idx, supplied["validation"])
    assert np.array_equal(split.test_idx, supplied["test"])
    assert len(explainer.X_train_) == 156


def test_precomputed_split_indices_must_be_disjoint_and_complete():
    X, y = _dataset()
    with pytest.raises(ValueError, match="overlap"):
        FiXAIt(
            RandomForestClassifier(n_estimators=4, random_state=5, n_jobs=1),
            config=FiXAItConfig(group_size=4, n_jobs=1, model_n_jobs=1),
        ).fit(
            X,
            y,
            split_indices={
                "train": np.arange(0, 180),
                "validation": np.arange(170, 210),
                "test": np.arange(210, 260),
            },
        )
