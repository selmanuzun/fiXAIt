from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig
from fixait.evaluation import (
    evaluate_global_faithfulness,
    evaluate_global_fidelity,
)


def _fitted_explainer():
    X_values, y_values = make_classification(
        n_samples=260,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        random_state=19,
    )
    X = pd.DataFrame(
        X_values,
        columns=[f"feature_{index}" for index in range(X_values.shape[1])],
    )
    y = pd.Series(np.asarray(["low", "medium", "high"])[y_values], name="outcome")
    explainer = FiXAIt(
        RandomForestClassifier(
            n_estimators=16,
            max_depth=4,
            random_state=42,
            n_jobs=1,
        ),
        config=FiXAItConfig(
            group_size=4,
            faithfulness_runs_per_feature=2,
            n_jobs=1,
            model_n_jobs=1,
        ),
    ).fit(
        X,
        y,
        categorical_features=["feature_0"],
        ordinal_features=["feature_1"],
    )
    return explainer, X


def test_global_faithfulness_and_fidelity_are_attached_to_the_selected_mode():
    explainer, _ = _fitted_explainer()
    global_result = explainer.explain_global()
    faithfulness = explainer.evaluate_global_faithfulness(
        split="validation",
        runs_per_feature=2,
        absolute_drop=True,
        n_jobs=1,
    )
    fidelity = explainer.evaluate_global_fidelity()

    assert set(faithfulness.drop_impacts) == set(global_result.global_fei)
    assert np.isfinite(faithfulness.score)
    assert np.isfinite(global_result.faithfulness)
    assert np.isfinite(global_result.fidelity)
    assert np.isfinite(fidelity.score)
    assert not global_result.optimization_applied
    assert not hasattr(explainer, "calibrate_global_fei")


def test_local_stability_uses_a_fixed_target_and_semantic_feature_types():
    explainer, X = _fitted_explainer()
    result = explainer.evaluate_local_stability(
        X.iloc[0],
        target_class="medium",
        n_perturbations=3,
        numeric_scale=0.05,
        categorical_flip_probability=0.50,
        ordinal_step_probability=1.0,
        target_mean_absolute_behavior_change=0.0,
        max_budget_tries=1,
    )

    assert result.informative
    assert result.n_perturbations == 3
    assert result.spearman_abs_mean is not None
    assert result.cosine_mean is not None
    assert np.isfinite(result.mean_absolute_behavior_change)


def test_probability_faithfulness_maps_labels_and_conditional_permutation():
    values, encoded = make_classification(
        n_samples=180,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=31,
    )
    X = pd.DataFrame(values, columns=[f"x{index}" for index in range(5)])
    y = pd.Series(np.where(encoded == 1, "yes", "no"))
    X_train, X_test = X.iloc[:120], X.iloc[120:]
    y_train, y_test = y.iloc[:120], y.iloc[120:]
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        n_jobs=1,
    ).fit(X_train, y_train)
    importance = {column: float(index + 1) for index, column in enumerate(X.columns)}

    faithfulness = evaluate_global_faithfulness(
        model=model,
        X_eval=X_test,
        y_eval=y_test,
        importance_scores=importance,
        drop_mode="probability",
        target_class="yes",
        runs_per_feature=2,
        conditional_permutation=True,
        compute_pd_variance=True,
        top_k=3,
        n_jobs=1,
    )
    fidelity, surrogate = evaluate_global_fidelity(
        model=model,
        X_train=X_train,
        X_test=X_test,
        importance_scores=importance,
        top_k=3,
        max_depth="auto",
    )

    assert np.isfinite(faithfulness.score)
    assert len(faithfulness.drop_impacts) == 3
    assert set(faithfulness.pd_variance) == set(faithfulness.drop_impacts)
    assert fidelity.selected_features == ["x4", "x3", "x2"]
    assert np.isfinite(fidelity.score)
    assert surrogate is not None
