from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .results import LocalExplanation, StabilityResult


def _spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    left = np.abs(a)
    right = np.abs(b)
    if len(np.unique(left)) <= 1 or len(np.unique(right)) <= 1:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def _perturb_instance(
    x: pd.Series,
    X_train: pd.DataFrame,
    selected_features: Sequence[str],
    *,
    categorical_features: set[str],
    ordinal_features: set[str],
    rng: np.random.RandomState,
    numeric_scale: float,
    categorical_flip_probability: float,
    ordinal_step_probability: float,
) -> pd.Series:
    perturbed = x.copy()
    for feature in selected_features:
        column = X_train[feature]
        if feature in categorical_features:
            if rng.rand() < categorical_flip_probability:
                perturbed[feature] = column.iloc[int(rng.randint(0, len(column)))]
            continue
        if feature in ordinal_features:
            if rng.rand() < ordinal_step_probability:
                unique = np.sort(column.dropna().unique())
                if len(unique) > 1:
                    current = float(perturbed[feature])
                    position = int(np.argmin(np.abs(unique.astype(float) - current)))
                    direction = int(rng.choice([-1, 1]))
                    perturbed[feature] = unique[int(np.clip(position + direction, 0, len(unique) - 1))]
            continue

        std = float(column.std(ddof=0))
        if std > 0 and np.isfinite(std):
            value = float(perturbed[feature]) + float(rng.normal(0.0, numeric_scale * std))
            perturbed[feature] = float(np.clip(value, column.min(), column.max()))
    return perturbed


def evaluate_local_stability(
    *,
    x_instance: pd.Series,
    X_train_raw: pd.DataFrame,
    base_explanation: LocalExplanation,
    explain: Callable[[pd.Series], LocalExplanation],
    categorical_features: Optional[Iterable[str]] = None,
    ordinal_features: Optional[Iterable[str]] = None,
    n_perturbations: int = 20,
    random_state: int = 42,
    numeric_scale: float = 0.02,
    categorical_flip_probability: float = 0.15,
    ordinal_step_probability: float = 0.60,
    target_mean_absolute_behavior_change: float = 1e-3,
    max_budget_tries: int = 4,
    budget_growth: float = 2.0,
) -> StabilityResult:
    if n_perturbations < 1:
        raise ValueError("n_perturbations must be at least 1.")
    if max_budget_tries < 1:
        raise ValueError("max_budget_tries must be at least 1.")
    if numeric_scale < 0:
        raise ValueError("numeric_scale cannot be negative.")
    if not 0.0 <= categorical_flip_probability <= 1.0:
        raise ValueError("categorical_flip_probability must be between 0 and 1.")
    if not 0.0 <= ordinal_step_probability <= 1.0:
        raise ValueError("ordinal_step_probability must be between 0 and 1.")
    if budget_growth <= 1.0:
        raise ValueError("budget_growth must be greater than 1.")

    selected = list(base_explanation.selected_features)
    if not selected:
        return StabilityResult(
            informative=False,
            n_perturbations=n_perturbations,
            numeric_scale=float(numeric_scale),
            categorical_flip_probability=float(categorical_flip_probability),
            ordinal_step_probability=float(ordinal_step_probability),
            mean_absolute_behavior_change=0.0,
            max_absolute_behavior_change=0.0,
            spearman_abs_mean=None,
            cosine_mean=None,
            mean_absolute_fei_change=None,
            l2_fei_change=None,
            metadata={
                "attempt": 0,
                "random_state": random_state,
                "note": "No local features remained after FEI thresholding.",
            },
        )
    base_vector = np.asarray(
        [float(base_explanation.local_fei[feature]) for feature in selected],
        dtype=float,
    )
    base_behavior = np.asarray(
        base_explanation.metadata.get(
            "combination_behavior_scores",
            base_explanation.metadata.get(
                "ecfc_behavior_scores",
                [base_explanation.metadata["full_score"]],
            ),
        ),
        dtype=float,
    )
    categorical = set(categorical_features or [])
    ordinal = set(ordinal_features or [])
    rng = np.random.RandomState(random_state)
    current_scale = float(numeric_scale)
    current_flip = float(categorical_flip_probability)

    last_behavior_changes: list[float] = []
    last_max_behavior_changes: list[float] = []
    last_scale = current_scale
    last_flip = current_flip
    for attempt in range(max_budget_tries):
        last_scale = current_scale
        last_flip = current_flip
        spearman_values = []
        cosine_values = []
        mean_abs_fei_changes = []
        l2_changes = []
        mean_behavior_changes = []
        max_behavior_changes = []

        for _ in range(n_perturbations):
            perturbed = _perturb_instance(
                x_instance,
                X_train_raw,
                selected,
                categorical_features=categorical,
                ordinal_features=ordinal,
                rng=rng,
                numeric_scale=current_scale,
                categorical_flip_probability=current_flip,
                ordinal_step_probability=ordinal_step_probability,
            )
            explanation = explain(perturbed)
            vector = np.asarray(
                [float(explanation.local_fei.get(feature, 0.0)) for feature in selected],
                dtype=float,
            )
            behavior = np.asarray(
                explanation.metadata.get(
                    "combination_behavior_scores",
                    explanation.metadata.get(
                        "ecfc_behavior_scores",
                        [explanation.metadata["full_score"]],
                    ),
                ),
                dtype=float,
            )
            delta_behavior = np.abs(base_behavior - behavior)
            mean_behavior_changes.append(float(np.mean(delta_behavior)))
            max_behavior_changes.append(float(np.max(delta_behavior)))
            spearman_values.append(_spearman_abs(base_vector, vector))
            cosine_values.append(_cosine(base_vector, vector))
            mean_abs_fei_changes.append(float(np.mean(np.abs(base_vector - vector))))
            l2_changes.append(float(np.linalg.norm(base_vector - vector)))

        last_behavior_changes = mean_behavior_changes
        last_max_behavior_changes = max_behavior_changes
        mean_behavior_change = float(np.mean(mean_behavior_changes))
        if mean_behavior_change >= target_mean_absolute_behavior_change:
            return StabilityResult(
                informative=True,
                n_perturbations=n_perturbations,
                numeric_scale=current_scale,
                categorical_flip_probability=current_flip,
                ordinal_step_probability=ordinal_step_probability,
                mean_absolute_behavior_change=mean_behavior_change,
                max_absolute_behavior_change=float(np.mean(max_behavior_changes)),
                spearman_abs_mean=float(np.mean(spearman_values)),
                cosine_mean=float(np.mean(cosine_values)),
                mean_absolute_fei_change=float(np.mean(mean_abs_fei_changes)),
                l2_fei_change=float(np.mean(l2_changes)),
                metadata={"attempt": attempt + 1, "random_state": random_state},
            )

        current_scale *= budget_growth
        current_flip = min(1.0, current_flip * budget_growth)

    return StabilityResult(
        informative=False,
        n_perturbations=n_perturbations,
        numeric_scale=last_scale,
        categorical_flip_probability=last_flip,
        ordinal_step_probability=ordinal_step_probability,
        mean_absolute_behavior_change=float(np.mean(last_behavior_changes)) if last_behavior_changes else 0.0,
        max_absolute_behavior_change=(
            float(np.mean(last_max_behavior_changes))
            if last_max_behavior_changes
            else 0.0
        ),
        spearman_abs_mean=None,
        cosine_mean=None,
        mean_absolute_fei_change=None,
        l2_fei_change=None,
        metadata={
            "attempt": max_budget_tries,
            "random_state": random_state,
            "note": "Model behavior did not move enough for an informative stability score.",
        },
    )
