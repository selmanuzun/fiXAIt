from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .combinations import canonicalize_combinations
from .consistency import calculate_self_consistency
from .masking import resolve_baseline
from .optimization import optimize_fei_rank_gradient, summarize_fei_weight_change
from .results import LocalExplanation
from .scoring import ModelScorer
from .selection import select_local_impacts


def _normalize_signed(values: Mapping[str, float]) -> dict[str, float]:
    denominator = sum(abs(float(value)) for value in values.values())
    if denominator <= 0:
        return {feature: 0.0 for feature in values}
    return {feature: float(value) / denominator for feature, value in values.items()}


def _safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    if len(a_array) < 2 or len(np.unique(a_array)) <= 1 or len(np.unique(b_array)) <= 1:
        return 0.0
    value = spearmanr(a_array, b_array).statistic
    return float(value) if np.isfinite(value) else 0.0


def _spearman_is_informative(a: Sequence[float], b: Sequence[float]) -> bool:
    a_array = np.asarray(a, dtype=float)
    b_array = np.asarray(b, dtype=float)
    return bool(
        len(a_array) >= 2
        and len(np.unique(a_array)) > 1
        and len(np.unique(b_array)) > 1
    )


def _local_perturbation_impacts(
    *,
    scorer: ModelScorer,
    target: Any,
    full_score: float,
    X_reference: pd.DataFrame,
    x_instance: pd.Series,
    features: Sequence[str],
    all_columns: Sequence[str],
    runs_per_feature: int,
    random_state: int,
    excluded_reference_indices: Optional[Sequence[int]] = None,
) -> tuple[dict[str, float], list[int]]:
    """Measure per-feature local effects using repeated reference-value draws."""

    if runs_per_feature < 1:
        raise ValueError("runs_per_feature must be at least 1.")
    if X_reference.empty:
        raise ValueError("X_reference cannot be empty.")
    if not features:
        return {}, []

    excluded = np.asarray(
        [] if excluded_reference_indices is None else excluded_reference_indices,
        dtype=int,
    )
    if len(excluded) and (np.any(excluded < 0) or np.any(excluded >= len(X_reference))):
        raise ValueError("excluded_reference_indices contains an invalid row index.")
    available_indices = np.setdiff1d(
        np.arange(len(X_reference), dtype=int),
        np.unique(excluded),
        assume_unique=False,
    )
    # Very small reference sets may leave no held-out row. In that edge case,
    # fall back to the full reference set and report the overlap in metadata.
    if len(available_indices) == 0:
        available_indices = np.arange(len(X_reference), dtype=int)

    rng = np.random.RandomState(random_state)
    reference_indices = rng.choice(
        available_indices,
        size=runs_per_feature,
        replace=runs_per_feature > len(available_indices),
    )
    column_positions = {column: index for index, column in enumerate(all_columns)}
    base_row = x_instance[list(all_columns)].to_numpy(copy=True)
    perturbation_rows: list[np.ndarray] = []

    for feature in features:
        sampled_values = X_reference.iloc[reference_indices][feature].to_numpy()
        feature_rows = np.tile(base_row, (runs_per_feature, 1))
        feature_rows[:, column_positions[feature]] = sampled_values
        perturbation_rows.extend(feature_rows)

    perturbation_frame = pd.DataFrame(perturbation_rows, columns=all_columns)
    perturbation_scores = np.asarray(
        scorer.score(perturbation_frame, target),
        dtype=float,
    )

    impacts: dict[str, float] = {}
    start = 0
    for feature in features:
        stop = start + runs_per_feature
        impacts[feature] = float(
            np.mean(np.abs(full_score - perturbation_scores[start:stop]))
        )
        start = stop
    return impacts, [int(index) for index in reference_indices]


def explain_local(
    *,
    model: Any,
    X_reference: pd.DataFrame,
    x_instance: pd.Series,
    selected_features: Sequence[str],
    combinations: Iterable[Sequence[str]],
    alphas: Sequence[float],
    baseline_method: str,
    score_mode: str,
    sc_metric: str,
    fvi_method: str,
    target_class: Optional[Any] = None,
    reported_target_class: Optional[Any] = None,
    categorical_features: Optional[Iterable[str]] = None,
    random_state: int = 42,
    local_faithfulness_runs_per_feature: int = 30,
    optimize_faithfulness: bool = False,
    local_faithfulness_calibration_runs_per_feature: int = 30,
    local_faithfulness_optimizer_steps: int = 500,
    local_faithfulness_optimizer_lr: float = 0.05,
    local_faithfulness_optimizer_tau: float = 25.0,
    local_faithfulness_optimizer_pair_batch: int = 4096,
    local_faithfulness_reg_lambda: float = 0.10,
    local_faithfulness_accept_only_if_improved: bool = True,
    local_faithfulness_min_improvement: float = 0.01,
    local_faithfulness_max_weight_change_pct: Optional[float] = 20.0,
    fei_threshold_pct: Optional[float] = 3.0,
    threshold_inclusive: bool = True,
    combination_strategy: str = "ecfc",
) -> LocalExplanation:
    if isinstance(x_instance, pd.DataFrame):
        if len(x_instance) != 1:
            raise ValueError("x_instance DataFrame must contain exactly one row.")
        x_instance = x_instance.iloc[0]
    if not isinstance(x_instance, pd.Series):
        raise TypeError("x_instance must be a pandas Series or one-row DataFrame.")

    selected = list(selected_features)
    all_columns = list(X_reference.columns)
    missing = [column for column in all_columns if column not in x_instance.index]
    if missing:
        raise ValueError(f"x_instance is missing columns: {missing[:10]}")
    if len(selected) < 2:
        raise ValueError("At least two selected features are required for local ECFC.")

    combo_list = canonicalize_combinations(selected, combinations)
    if combination_strategy == "exhaustive":
        expected = (2 ** len(selected)) - 2
    elif combination_strategy == "ecfc":
        expected = len(selected) * (len(selected) - 1)
    else:
        raise ValueError("combination_strategy must be 'exhaustive' or 'ecfc'.")
    if len(combo_list) != expected:
        raise ValueError(
            f"{combination_strategy} combination count mismatch: "
            f"received {len(combo_list)}, expected {expected}."
        )

    baseline = resolve_baseline(
        X_reference,
        selected,
        method=baseline_method,
        categorical_features=categorical_features,
    )
    scorer = ModelScorer(model, score_mode=score_mode)
    full_frame = pd.DataFrame([x_instance[all_columns].to_numpy()], columns=all_columns)
    target = scorer.resolve_target(full_frame, target_class)
    full_score = float(scorer.score(full_frame, target)[0])

    presence_rows: List[List[float]] = []
    value_rows: List[List[float]] = []
    masked_rows: list[np.ndarray] = []

    for combination in combo_list:
        present = set(combination)
        presence_rows.append([1.0 if feature in present else 0.0 for feature in selected])

        masked = x_instance.copy()
        for feature in selected:
            if feature not in present:
                masked[feature] = baseline[feature]
        value_rows.append([float(masked[feature]) for feature in selected])
        masked_rows.append(masked[all_columns].to_numpy())

    empty_coalition_score: Optional[float] = None
    scoring_rows = list(masked_rows)
    if combination_strategy == "exhaustive":
        empty_coalition = x_instance.copy()
        for feature in selected:
            empty_coalition[feature] = baseline[feature]
        scoring_rows.insert(0, empty_coalition[all_columns].to_numpy())

    masked_frame = pd.DataFrame(scoring_rows, columns=all_columns)
    scored_rows = np.asarray(scorer.score(masked_frame, target), dtype=float)
    if combination_strategy == "exhaustive":
        empty_coalition_score = float(scored_rows[0])
        scored_rows = scored_rows[1:]
    behavior_values = [float(value) for value in scored_rows]
    behavior_by_combination = {
        tuple(combination): score
        for combination, score in zip(combo_list, behavior_values)
    }

    surrogate_presence_rows = list(presence_rows)
    surrogate_value_rows = list(value_rows)
    surrogate_scores = list(behavior_values)
    if combination_strategy == "exhaustive":
        assert empty_coalition_score is not None
        surrogate_presence_rows.insert(0, [0.0] * len(selected))
        surrogate_value_rows.insert(
            0,
            [float(baseline[feature]) for feature in selected],
        )
        surrogate_scores.insert(0, empty_coalition_score)
    surrogate_presence_rows.append([1.0] * len(selected))
    surrogate_value_rows.append(
        [float(x_instance[feature]) for feature in selected]
    )
    surrogate_scores.append(full_score)

    B = np.asarray(surrogate_presence_rows, dtype=float)
    Z = np.asarray(surrogate_value_rows, dtype=float)
    y = np.asarray(surrogate_scores, dtype=float)

    ridge_cv_splits = min(5, max(2, len(y) // 3))
    ridge_cv = KFold(
        n_splits=ridge_cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    ridge_fei = RidgeCV(
        alphas=list(alphas),
        cv=ridge_cv,
        scoring="neg_mean_squared_error",
    ).fit(B, y)
    candidate_local_fei = {
        feature: float(value)
        for feature, value in zip(selected, np.asarray(ridge_fei.coef_).reshape(-1))
    }

    ridge_legacy_fvi = RidgeCV(
        alphas=list(alphas),
        cv=ridge_cv,
        scoring="neg_mean_squared_error",
    ).fit(Z, y)
    legacy_local_fvi = {
        feature: float(value)
        for feature, value in zip(selected, np.asarray(ridge_legacy_fvi.coef_).reshape(-1))
    }

    replacement_rows: list[np.ndarray] = []
    for feature in selected:
        replaced = x_instance.copy()
        replaced[feature] = baseline[feature]
        replacement_rows.append(replaced[all_columns].to_numpy())
    replacement_frame = pd.DataFrame(replacement_rows, columns=all_columns)
    replacement_scores = scorer.score(replacement_frame, target)
    finite_difference = {
        feature: full_score - float(replaced_score)
        for feature, replaced_score in zip(selected, replacement_scores)
    }

    if fvi_method == "finite_difference":
        candidate_raw_local_fvi = finite_difference
    elif fvi_method == "legacy_ridge":
        candidate_raw_local_fvi = legacy_local_fvi
    else:
        raise ValueError("fvi_method must be 'finite_difference' or 'legacy_ridge'.")

    local_sc = calculate_self_consistency(
        combo_list,
        behavior_by_combination,
        candidate_local_fei,
        metric=sc_metric,
        behavior_measure="target_class_score",
    )

    optimization_applied = False
    fei_for_selection = candidate_local_fei
    calibration_impacts: dict[str, float] = {}
    calibration_reference_indices: list[int] = []
    evaluation_impacts_all: dict[str, float] = {}
    evaluation_reference_indices: list[int] = []
    evaluation_overlap_count = 0
    optimization_metadata: dict[str, Any] = {
        "requested": bool(optimize_faithfulness),
        "accepted": False,
        "applied": False,
        "method": "rank_gradient" if optimize_faithfulness else None,
        "reason": "not_requested",
    }

    if optimize_faithfulness:
        calibration_impacts, calibration_reference_indices = (
            _local_perturbation_impacts(
                scorer=scorer,
                target=target,
                full_score=full_score,
                X_reference=X_reference,
                x_instance=x_instance,
                features=selected,
                all_columns=all_columns,
                runs_per_feature=(
                    local_faithfulness_calibration_runs_per_feature
                ),
                random_state=random_state,
            )
        )
        optimized_fei_candidate = optimize_fei_rank_gradient(
            candidate_local_fei,
            calibration_impacts,
            n_steps=local_faithfulness_optimizer_steps,
            learning_rate=local_faithfulness_optimizer_lr,
            tau=local_faithfulness_optimizer_tau,
            pair_batch=local_faithfulness_optimizer_pair_batch,
            random_state=random_state,
            reg_lambda=local_faithfulness_reg_lambda,
            max_weight_change_pct=local_faithfulness_max_weight_change_pct,
        )
        evaluation_impacts_all, evaluation_reference_indices = (
            _local_perturbation_impacts(
                scorer=scorer,
                target=target,
                full_score=full_score,
                X_reference=X_reference,
                x_instance=x_instance,
                features=selected,
                all_columns=all_columns,
                runs_per_feature=local_faithfulness_runs_per_feature,
                random_state=random_state + 1,
                excluded_reference_indices=calibration_reference_indices,
            )
        )
        evaluation_overlap_count = len(
            set(calibration_reference_indices) & set(evaluation_reference_indices)
        )
        evaluation_values = [evaluation_impacts_all[feature] for feature in selected]
        faithfulness_before = _safe_spearman(
            [abs(candidate_local_fei[feature]) for feature in selected],
            evaluation_values,
        )
        faithfulness_after = _safe_spearman(
            [abs(optimized_fei_candidate[feature]) for feature in selected],
            evaluation_values,
        )
        evaluation_informative = _spearman_is_informative(
            [abs(candidate_local_fei[feature]) for feature in selected],
            evaluation_values,
        )
        faithfulness_improvement = float(faithfulness_after - faithfulness_before)

        if local_faithfulness_accept_only_if_improved:
            optimization_applied = bool(
                evaluation_informative
                and faithfulness_improvement >= local_faithfulness_min_improvement
            )
            if not evaluation_informative:
                reason = "evaluation_faithfulness_not_informative"
            elif optimization_applied:
                reason = "evaluation_improvement_met_threshold"
            else:
                reason = "evaluation_improvement_below_threshold"
        else:
            optimization_applied = True
            reason = "acceptance_guard_disabled"

        if optimization_applied:
            fei_for_selection = optimized_fei_candidate

        change_summary = summarize_fei_weight_change(
            candidate_local_fei,
            optimized_fei_candidate,
        )
        actual_change_summary = {
            key: value if optimization_applied else 0.0
            for key, value in change_summary.items()
        }
        optimization_metadata.update(
            {
                "accepted": optimization_applied,
                "applied": optimization_applied,
                "reason": reason,
                "evaluation_faithfulness_before": faithfulness_before,
                "evaluation_faithfulness_after": faithfulness_after,
                "evaluation_faithfulness_improvement": faithfulness_improvement,
                "evaluation_faithfulness_informative": evaluation_informative,
                "accept_only_if_improved": (
                    local_faithfulness_accept_only_if_improved
                ),
                "min_improvement": local_faithfulness_min_improvement,
                "max_weight_change_pct_allowed": (
                    local_faithfulness_max_weight_change_pct
                ),
                "candidate_mean_weight_change_pct": change_summary[
                    "mean_weight_change_pct"
                ],
                "candidate_max_weight_change_pct": change_summary[
                    "max_weight_change_pct"
                ],
                **actual_change_summary,
                "steps": local_faithfulness_optimizer_steps,
                "learning_rate": local_faithfulness_optimizer_lr,
                "tau": local_faithfulness_optimizer_tau,
                "pair_batch": local_faithfulness_optimizer_pair_batch,
                "reg_lambda": local_faithfulness_reg_lambda,
            }
        )

    local_fei, selected_raw_local_fvi, dropped_features = select_local_impacts(
        fei_for_selection,
        candidate_raw_local_fvi,
        threshold_pct=fei_threshold_pct,
        inclusive=threshold_inclusive,
    )
    final_features = list(local_fei)
    kept_indices = [selected.index(feature) for feature in final_features]
    raw_local_fvi = {
        feature: float(selected_raw_local_fvi[feature]) for feature in final_features
    }
    legacy_local_fvi = {
        feature: float(legacy_local_fvi[feature]) for feature in final_features
    }
    local_fvi = _normalize_signed(raw_local_fvi)

    if final_features:
        B_final = B[:, kept_indices]
        cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=random_state)
        cv_prediction = cross_val_predict(
            Ridge(alpha=float(ridge_fei.alpha_)),
            B_final,
            y,
            cv=cv,
        )
        fidelity_r2 = float(r2_score(y, cv_prediction))
        fidelity_informative = True
    else:
        fidelity_r2 = 0.0
        fidelity_informative = False

    agreement_fei = [abs(local_fei[feature]) for feature in final_features]
    agreement_fvi = [abs(finite_difference[feature]) for feature in final_features]
    fei_fvi_agreement = _safe_spearman(
        agreement_fei,
        agreement_fvi,
    )
    fei_fvi_agreement_informative = _spearman_is_informative(
        agreement_fei,
        agreement_fvi,
    )

    if optimize_faithfulness:
        local_perturbation_impacts = {
            feature: evaluation_impacts_all[feature] for feature in final_features
        }
        local_faithfulness_reference_indices = evaluation_reference_indices
    else:
        local_perturbation_impacts, local_faithfulness_reference_indices = (
            _local_perturbation_impacts(
                scorer=scorer,
                target=target,
                full_score=full_score,
                X_reference=X_reference,
                x_instance=x_instance,
                features=final_features,
                all_columns=all_columns,
                runs_per_feature=local_faithfulness_runs_per_feature,
                random_state=random_state,
            )
        )
    local_faithfulness_fei = [
        abs(local_fei[feature]) for feature in final_features
    ]
    local_faithfulness_impacts = [
        local_perturbation_impacts[feature] for feature in final_features
    ]
    local_faithfulness = _safe_spearman(
        local_faithfulness_fei,
        local_faithfulness_impacts,
    )
    local_faithfulness_informative = _spearman_is_informative(
        local_faithfulness_fei,
        local_faithfulness_impacts,
    )

    display_target = target.label if reported_target_class is None else reported_target_class
    return LocalExplanation(
        local_fei=local_fei,
        local_fvi=local_fvi,
        local_sc=local_sc,
        selected_features=final_features,
        dropped_features=dropped_features,
        target_class=display_target,
        target_class_index=target.index,
        baseline=baseline_method,
        score_mode=score_mode,
        raw_local_fvi=raw_local_fvi,
        legacy_local_fvi=legacy_local_fvi,
        fidelity_r2=fidelity_r2,
        fei_fvi_agreement_spearman=fei_fvi_agreement,
        local_faithfulness_spearman=local_faithfulness,
        optimization_applied=optimization_applied,
        metadata={
            "full_score": full_score,
            "n_combinations": len(combo_list),
            "n_surrogate_rows": len(y),
            "n_candidate_features": len(selected),
            "combination_strategy": combination_strategy,
            "combination_space_complete": combination_strategy == "exhaustive",
            "combination_rule": "explicit_combinations_from_caller",
            "empty_coalition_included": combination_strategy == "exhaustive",
            "empty_coalition_score": empty_coalition_score,
            "full_coalition_included": True,
            "sc_includes_empty_coalition": False,
            "candidate_features": selected,
            "alpha_fei": float(ridge_fei.alpha_),
            "alpha_legacy_fvi": float(ridge_legacy_fvi.alpha_),
            "ridge_cv_strategy": "canonical_shuffled_kfold",
            "ridge_cv_splits": ridge_cv_splits,
            "ridge_cv_shuffle": True,
            "ridge_cv_random_state": random_state,
            "coalition_order": "subset_size_then_selected_feature_index",
            "in_sample_r2_fei": float(r2_score(y, ridge_fei.predict(B))),
            "in_sample_r2_legacy_fvi": float(r2_score(y, ridge_legacy_fvi.predict(Z))),
            "fvi_method": fvi_method,
            "threshold_pct": fei_threshold_pct,
            "threshold_inclusive": threshold_inclusive,
            "threshold_mode": "absolute_local_fei_share",
            "drop_non_positive_fei_applied": False,
            "local_sc_scope": "pre_threshold_candidate_features",
            "local_sc_uses_optimized_fei": False,
            "optimization": optimization_metadata,
            "fei_fvi_agreement_scope": "post_threshold_selected_features",
            "fei_fvi_agreement_reference": "absolute_finite_difference_fvi",
            "fei_fvi_agreement_informative": fei_fvi_agreement_informative,
            "local_faithfulness_scope": "post_threshold_selected_features",
            "local_faithfulness_sampling": (
                "held_out_shared_marginal_reference_rows"
                if optimize_faithfulness
                else "shared_marginal_reference_rows"
            ),
            "local_faithfulness_runs_per_feature": (
                local_faithfulness_runs_per_feature
            ),
            "local_faithfulness_calibration_runs_per_feature": (
                local_faithfulness_calibration_runs_per_feature
                if optimize_faithfulness
                else 0
            ),
            "local_faithfulness_calibration_impacts": calibration_impacts,
            "local_faithfulness_calibration_reference_indices": (
                calibration_reference_indices
            ),
            "local_faithfulness_evaluation_overlap_count": (
                evaluation_overlap_count
            ),
            "local_faithfulness_impacts": local_perturbation_impacts,
            "local_faithfulness_reference_indices": (
                local_faithfulness_reference_indices
            ),
            "local_faithfulness_informative": local_faithfulness_informative,
            "faithfulness_scope": "post_threshold_selected_features",
            "faithfulness_informative": fei_fvi_agreement_informative,
            "faithfulness_spearman_deprecated_alias_for": (
                "fei_fvi_agreement_spearman"
            ),
            "fidelity_scope": "post_threshold_selected_features",
            "fidelity_informative": fidelity_informative,
            "baseline_values": {feature: baseline[feature] for feature in selected},
            "combination_behavior_scores": behavior_values,
            "ecfc_behavior_scores": behavior_values,
        },
    )
