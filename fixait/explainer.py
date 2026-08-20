from __future__ import annotations

from dataclasses import replace
from os import PathLike
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .combinations import generate_local_combinations
from .config import FiXAItConfig
from .consistency import calculate_self_consistency
from .core import CalcFeatureWeight
from .evaluation import (
    evaluate_global_faithfulness as calculate_global_faithfulness,
    evaluate_global_fidelity as calculate_global_fidelity,
)
from .local import explain_local as calculate_local_explanation
from .optimization import optimize_fei_rank_gradient, summarize_fei_weight_change
from .preprocessing import TabularPreprocessor
from .results import (
    FaithfulnessResult,
    FidelityResult,
    GlobalExplanation,
    LocalExplanation,
    StabilityResult,
)
from .selection import select_global_impacts
from .stability import evaluate_local_stability as calculate_local_stability
from .splitting import make_split_indices, normalize_split_indices


class FiXAIt:
    """Unified global and local interface for the fiXAIt algorithm."""

    def __init__(self, model: Any, *, config: Optional[FiXAItConfig] = None) -> None:
        self.model = model
        self.config = config or FiXAItConfig()
        self.core_: Optional[CalcFeatureWeight] = None
        self.model_: Optional[Any] = None
        self.label_encoder_: Optional[LabelEncoder] = None
        self.preprocessor_: Optional[TabularPreprocessor] = None
        self.feature_names_: Optional[list[str]] = None
        self.categorical_features_: set[str] = set()
        self.ordinal_features_: set[str] = set()
        self.X_train_: Optional[pd.DataFrame] = None
        self.X_train_raw_: Optional[pd.DataFrame] = None
        self.target_column_: Optional[str] = None
        self.global_surrogate_: Optional[Any] = None
        self._global_result: Optional[GlobalExplanation] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray, list]] = None,
        *,
        target_column: Optional[str] = None,
        categorical_features: Optional[Iterable[str]] = None,
        ordinal_features: Optional[Iterable[str]] = None,
        ordinal_categories: Optional[Mapping[str, Sequence[Any]]] = None,
        split_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ) -> "FiXAIt":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named columns.")
        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate feature names.")

        if target_column is not None:
            if y is not None:
                raise ValueError("Pass either y or target_column, not both.")
            if target_column not in X.columns:
                raise ValueError(
                    f"Target column {target_column!r} was not found. "
                    f"Available columns: {list(X.columns)}"
                )
            y_values = X[target_column].copy()
            features = X.drop(columns=[target_column]).copy()
            resolved_target_column = target_column
        else:
            if y is None:
                raise ValueError("Provide y or specify target_column.")
            y_values = y
            features = X.copy()
            resolved_target_column = getattr(y, "name", None)

        if "class" in features.columns:
            raise ValueError(
                "The feature name 'class' is reserved internally. Rename that feature "
                "or use it as target_column."
            )
        if len(features) != len(y_values):
            raise ValueError("X and y must contain the same number of rows.")
        if features.empty:
            raise ValueError("X cannot be empty.")
        categorical = set(categorical_features or [])
        ordinal = set(ordinal_features or [])
        unknown_categorical = sorted(categorical - set(features.columns))
        if unknown_categorical:
            raise ValueError(
                f"categorical_features contains unknown columns: {unknown_categorical}"
            )
        unknown_ordinal = sorted(ordinal - set(features.columns))
        if unknown_ordinal:
            raise ValueError(f"ordinal_features contains unknown columns: {unknown_ordinal}")
        overlap = sorted(categorical & ordinal)
        if overlap:
            raise ValueError(
                f"Features cannot be both categorical and ordinal: {overlap}"
            )

        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(np.asarray(y_values))
        config = self.config
        if split_indices is None:
            train_idx, validation_idx, test_idx = make_split_indices(
                encoded_y,
                test_size=config.test_size,
                validation_size=config.validation_size,
                random_state=config.random_state,
                stratify=config.stratify,
            )
        else:
            train_idx, validation_idx, test_idx = normalize_split_indices(
                len(features), split_indices
            )
        preprocessor = TabularPreprocessor(
            categorical_features=categorical,
            ordinal_features=ordinal,
            ordinal_categories=ordinal_categories,
        ).fit(features.iloc[train_idx])
        processed_features = preprocessor.transform(features)

        data = processed_features.copy()
        data["class"] = encoded_y

        core = CalcFeatureWeight(
            df=data,
            model=self.model,
            group_size=config.group_size,
            step=config.step,
            alphas=list(config.alphas),
            test_size=config.test_size,
            opt_size=config.validation_size,
            random_state=config.random_state,
            stratify=config.stratify,
            feature_selection_scope=config.feature_selection_scope,
            top_k_groups=config.top_k_groups,
            compatibility_mode=config.compatibility_mode,
            n_jobs=config.n_jobs,
            prefer=config.prefer,
            model_n_jobs=config.model_n_jobs,
            split_indices=(train_idx, validation_idx, test_idx),
            auto_run=True,
            plot=False,
            verbose=config.verbose,
        )
        split = core.get_splits()
        if not (
            np.array_equal(split.train_idx, train_idx)
            and np.array_equal(split.opt_idx, validation_idx)
            and np.array_equal(split.test_idx, test_idx)
        ):
            raise RuntimeError(
                "The preprocessing and fiXAIt data splits diverged unexpectedly."
            )
        X_train = pd.DataFrame(split.X_train, columns=split.feature_names)
        fitted_model = core._clone_model()
        fitted_model.fit(X_train, split.y_train)

        self.core_ = core
        self.model_ = fitted_model
        self.label_encoder_ = encoder
        self.preprocessor_ = preprocessor
        self.feature_names_ = list(features.columns)
        self.categorical_features_ = set(preprocessor.categorical_features_)
        self.ordinal_features_ = set(preprocessor.ordinal_features_)
        self.X_train_ = X_train
        self.X_train_raw_ = features.iloc[split.train_idx].copy().reset_index(drop=True)
        self.target_column_ = resolved_target_column
        self.global_surrogate_ = None
        self._global_result = None
        return self

    def fit_csv(
        self,
        path: Union[str, PathLike[str]],
        *,
        target_column: str,
        usecols: Optional[Iterable[str]] = None,
        categorical_features: Optional[Iterable[str]] = None,
        ordinal_features: Optional[Iterable[str]] = None,
        ordinal_categories: Optional[Mapping[str, Sequence[Any]]] = None,
        split_indices: Optional[Mapping[str, Sequence[int]]] = None,
        read_csv_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> "FiXAIt":
        columns = None if usecols is None else list(usecols)
        if columns is not None and target_column not in columns:
            columns.append(target_column)
        kwargs = dict(read_csv_kwargs or {})
        if "usecols" in kwargs:
            raise ValueError("Pass usecols through the dedicated usecols parameter.")
        data = pd.read_csv(path, usecols=columns, **kwargs)
        return self.fit(
            data,
            target_column=target_column,
            categorical_features=categorical_features,
            ordinal_features=ordinal_features,
            ordinal_categories=ordinal_categories,
            split_indices=split_indices,
        )

    def _require_fitted(self) -> CalcFeatureWeight:
        if self.core_ is None or self.model_ is None or self.label_encoder_ is None:
            raise RuntimeError("Call fit(X, y) before requesting an explanation.")
        return self.core_

    def _preprocessing_metadata(self) -> Mapping[str, Any]:
        if self.preprocessor_ is None:
            return {}
        summary = self.preprocessor_.summary()
        return {
            "numeric_features": list(summary.numeric_features),
            "categorical_features": list(summary.categorical_features),
            "ordinal_features": list(summary.ordinal_features),
            "categories": {
                feature: list(categories)
                for feature, categories in summary.categories.items()
            },
            "fit_scope": "train",
            "unknown_category_code": -1.0,
            "output_feature_names": list(self.preprocessor_.feature_names_),
        }

    def explain_global(self) -> GlobalExplanation:
        core = self._require_fitted()
        if self._global_result is not None:
            return self._global_result
        if core.new_weight_format is None or core.features is None:
            raise RuntimeError("Global fiXAIt outputs are unavailable after fitting.")

        candidate_features = [feature for feature in core.features if feature != "class"]
        base_fei = dict(core.new_weight_format)
        # The published/legacy SC stage regenerates ECFC after sorting features
        # by the final FEI weights. Preserve that order for comparable SC values.
        sc_feature_order = list(base_fei.keys())
        combinations = core.generate_combinations(sc_feature_order)
        behavior_scores = {
            tuple(combination): float(core.Acc_(list(combination) + ["class"]))
            for combination in combinations
        }
        global_sc = calculate_self_consistency(
            combinations,
            behavior_scores,
            base_fei,
            metric=self.config.sc_metric,
            behavior_measure="validation_accuracy",
        )
        split = core.get_splits()
        feature_names = split.feature_names
        X_train = pd.DataFrame(split.X_train, columns=feature_names)
        X_validation = pd.DataFrame(split.X_opt, columns=feature_names)
        X_test = pd.DataFrame(split.X_test, columns=feature_names)
        y_validation = pd.Series(split.y_opt)
        y_test = pd.Series(split.y_test)

        optimization_metadata: dict[str, Any] = {
            "requested": bool(self.config.optimize_faithfulness),
            "accepted": False,
            "applied": False,
            "method": "rank_gradient" if self.config.optimize_faithfulness else None,
            "reason": "not_requested",
        }
        fei_for_selection = base_fei
        optimization_applied = False
        if self.config.optimize_faithfulness:
            validation_faithfulness_before = calculate_global_faithfulness(
                model=self.model_,
                X_eval=X_validation,
                y_eval=y_validation,
                importance_scores=base_fei,
                metric=self.config.faithfulness_metric,
                split="validation",
                runs_per_feature=self.config.faithfulness_runs_per_feature,
                random_state=self.config.random_state,
                absolute_drop=False,
                drop_mode=self.config.faithfulness_drop_mode,
                probability_abs_drop=self.config.faithfulness_probability_abs_drop,
                conditional_permutation=(
                    self.config.faithfulness_conditional_permutation
                ),
                top_k=None,
                compute_pd_variance=False,
                n_jobs=self.config.n_jobs,
                prefer=self.config.prefer,
            )
            optimized_fei_candidate = optimize_fei_rank_gradient(
                base_fei,
                validation_faithfulness_before.drop_impacts,
                n_steps=self.config.faithfulness_optimizer_steps,
                learning_rate=self.config.faithfulness_optimizer_lr,
                tau=self.config.faithfulness_optimizer_tau,
                pair_batch=self.config.faithfulness_optimizer_pair_batch,
                random_state=self.config.random_state,
                reg_lambda=self.config.faithfulness_reg_lambda,
                max_weight_change_pct=(
                    self.config.faithfulness_max_weight_change_pct
                ),
            )
            validation_faithfulness_after = calculate_global_faithfulness(
                model=self.model_,
                X_eval=X_validation,
                y_eval=y_validation,
                importance_scores=optimized_fei_candidate,
                metric=self.config.faithfulness_metric,
                split="validation",
                runs_per_feature=self.config.faithfulness_runs_per_feature,
                random_state=self.config.random_state,
                absolute_drop=False,
                drop_mode=self.config.faithfulness_drop_mode,
                probability_abs_drop=self.config.faithfulness_probability_abs_drop,
                conditional_permutation=(
                    self.config.faithfulness_conditional_permutation
                ),
                top_k=None,
                compute_pd_variance=False,
                n_jobs=self.config.n_jobs,
                prefer=self.config.prefer,
            )
            validation_improvement = float(
                validation_faithfulness_after.score
                - validation_faithfulness_before.score
            )
            if self.config.faithfulness_accept_only_if_improved:
                optimization_applied = bool(
                    validation_improvement
                    >= self.config.faithfulness_min_improvement
                )
                reason = (
                    "validation_improvement_met_threshold"
                    if optimization_applied
                    else "validation_improvement_below_threshold"
                )
            else:
                optimization_applied = True
                reason = "acceptance_guard_disabled"
            if optimization_applied:
                fei_for_selection = optimized_fei_candidate

            change_summary = summarize_fei_weight_change(
                base_fei,
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
                    "validation_faithfulness_before": (
                        validation_faithfulness_before.score
                    ),
                    "validation_faithfulness_after": (
                        validation_faithfulness_after.score
                    ),
                    "validation_faithfulness_improvement": validation_improvement,
                    "accept_only_if_improved": (
                        self.config.faithfulness_accept_only_if_improved
                    ),
                    "min_improvement": self.config.faithfulness_min_improvement,
                    "max_weight_change_pct_allowed": (
                        self.config.faithfulness_max_weight_change_pct
                    ),
                    "candidate_mean_weight_change_pct": change_summary[
                        "mean_weight_change_pct"
                    ],
                    "candidate_max_weight_change_pct": change_summary[
                        "max_weight_change_pct"
                    ],
                    **actual_change_summary,
                    "steps": self.config.faithfulness_optimizer_steps,
                    "learning_rate": self.config.faithfulness_optimizer_lr,
                    "tau": self.config.faithfulness_optimizer_tau,
                    "pair_batch": self.config.faithfulness_optimizer_pair_batch,
                    "reg_lambda": self.config.faithfulness_reg_lambda,
                }
            )

        notebook_fvi = core.compute_value_impact(normalize=False)
        final_fei, final_fvi, dropped_features = select_global_impacts(
            fei_for_selection,
            notebook_fvi,
            threshold_pct=self.config.fei_threshold_pct,
            inclusive=self.config.threshold_inclusive,
            drop_non_positive=self.config.drop_non_positive_fei,
        )

        if final_fei:
            global_faithfulness = calculate_global_faithfulness(
                model=self.model_,
                X_eval=X_test,
                y_eval=y_test,
                importance_scores=final_fei,
                metric=self.config.faithfulness_metric,
                split="test",
                runs_per_feature=self.config.faithfulness_runs_per_feature,
                random_state=self.config.random_state,
                absolute_drop=False,
                drop_mode=self.config.faithfulness_drop_mode,
                probability_abs_drop=self.config.faithfulness_probability_abs_drop,
                conditional_permutation=(
                    self.config.faithfulness_conditional_permutation
                ),
                top_k=self.config.faithfulness_top_k,
                compute_pd_variance=(
                    self.config.faithfulness_compute_pd_variance
                ),
                n_jobs=self.config.n_jobs,
                prefer=self.config.prefer,
            )
        else:
            global_faithfulness = FaithfulnessResult(
                score=0.0,
                drop_impacts={},
                metric=self.config.faithfulness_metric,
                split="test",
                runs_per_feature=self.config.faithfulness_runs_per_feature,
                metadata={
                    "informative": False,
                    "reason": "all features were removed by FEI selection",
                },
            )

        global_fidelity, surrogate = calculate_global_fidelity(
            model=self.model_,
            X_train=X_train,
            X_test=X_test,
            importance_scores=final_fei,
            top_k=self.config.fidelity_top_k,
            metric=self.config.fidelity_metric,
            max_depth=self.config.fidelity_max_depth,
            random_state=self.config.random_state,
        )
        self.global_surrogate_ = surrogate

        result = GlobalExplanation(
            global_fei=final_fei,
            global_fvi=final_fvi,
            global_sc=global_sc,
            global_faithfulness=global_faithfulness,
            global_fidelity=global_fidelity,
            selected_features=list(final_fei),
            dropped_features=dropped_features,
            selected_feature_accuracy=float(core.acc_select or 0.0),
            optimization_applied=optimization_applied,
            metadata={
                "legacy_algorithm_consistency": float(core.alg_consistency),
                "feature_selection_scope": core.feature_selection_scope,
                "n_train": int(len(split.X_train)),
                "n_validation": int(len(split.X_opt)),
                "n_test": int(len(split.X_test)),
                "n_combinations": len(combinations),
                "sc_feature_order": sc_feature_order,
                "candidate_features": candidate_features,
                "optimization": optimization_metadata,
                "threshold_pct": self.config.fei_threshold_pct,
                "threshold_inclusive": self.config.threshold_inclusive,
                "drop_non_positive_fei": self.config.drop_non_positive_fei,
                "fvi_normalized": False,
                "target_column": self.target_column_,
                "preprocessing": self._preprocessing_metadata(),
            },
        )
        self._global_result = result
        return result

    def explain_local(
        self,
        x_instance: Union[pd.Series, pd.DataFrame],
        *,
        target_class: Optional[Any] = None,
        baseline: Optional[str] = None,
        score_mode: Optional[str] = None,
        fvi_method: Optional[str] = None,
        optimize_faithfulness: Optional[bool] = None,
        local_faithfulness_runs_per_feature: Optional[int] = None,
        local_faithfulness_calibration_runs_per_feature: Optional[int] = None,
    ) -> LocalExplanation:
        core = self._require_fitted()
        assert self.feature_names_ is not None
        assert self.X_train_ is not None
        assert self.label_encoder_ is not None
        assert self.preprocessor_ is not None

        if isinstance(x_instance, pd.DataFrame):
            if len(x_instance) != 1:
                raise ValueError("x_instance DataFrame must contain exactly one row.")
            raw_instance = x_instance.iloc[0]
        elif isinstance(x_instance, pd.Series):
            raw_instance = x_instance
        else:
            raise TypeError("x_instance must be a pandas Series or one-row DataFrame.")

        missing = [feature for feature in self.feature_names_ if feature not in raw_instance.index]
        if missing:
            raise ValueError(f"x_instance is missing columns: {missing[:10]}")
        if core.scaler_ is None:
            raise RuntimeError("The fitted feature scaler is unavailable.")

        raw_frame = pd.DataFrame(
            [raw_instance[self.feature_names_].to_numpy()],
            columns=self.feature_names_,
        )
        processed_frame = self.preprocessor_.transform(raw_frame)
        transformed = core.scaler_.transform(
            processed_frame.to_numpy(dtype=float, copy=False)
        )
        transformed_instance = pd.Series(transformed[0], index=self.feature_names_)

        encoded_target = None
        reported_target = target_class
        if target_class is not None:
            try:
                encoded_target = int(self.label_encoder_.transform([target_class])[0])
            except ValueError as exc:
                raise ValueError(
                    f"target_class={target_class!r} was not observed during fit."
                ) from exc

        selected = [feature for feature in list(core.features or []) if feature != "class"]
        combinations, combination_strategy = generate_local_combinations(
            selected,
            strategy=self.config.local_combination_strategy,
        )
        result = calculate_local_explanation(
            model=self.model_,
            X_reference=self.X_train_,
            x_instance=transformed_instance,
            selected_features=selected,
            combinations=combinations,
            alphas=list(self.config.alphas),
            baseline_method=baseline or self.config.local_baseline,
            score_mode=score_mode or self.config.local_score_mode,
            sc_metric=self.config.sc_metric,
            fvi_method=fvi_method or self.config.local_fvi_method,
            target_class=encoded_target,
            reported_target_class=reported_target,
            categorical_features=self.categorical_features_,
            random_state=self.config.random_state,
            local_faithfulness_runs_per_feature=(
                self.config.local_faithfulness_runs_per_feature
                if local_faithfulness_runs_per_feature is None
                else local_faithfulness_runs_per_feature
            ),
            optimize_faithfulness=(
                self.config.optimize_local_faithfulness
                if optimize_faithfulness is None
                else bool(optimize_faithfulness)
            ),
            local_faithfulness_calibration_runs_per_feature=(
                self.config.local_faithfulness_calibration_runs_per_feature
                if local_faithfulness_calibration_runs_per_feature is None
                else local_faithfulness_calibration_runs_per_feature
            ),
            local_faithfulness_optimizer_steps=(
                self.config.local_faithfulness_optimizer_steps
            ),
            local_faithfulness_optimizer_lr=(
                self.config.local_faithfulness_optimizer_lr
            ),
            local_faithfulness_optimizer_tau=(
                self.config.local_faithfulness_optimizer_tau
            ),
            local_faithfulness_optimizer_pair_batch=(
                self.config.local_faithfulness_optimizer_pair_batch
            ),
            local_faithfulness_reg_lambda=(
                self.config.local_faithfulness_reg_lambda
            ),
            local_faithfulness_accept_only_if_improved=(
                self.config.local_faithfulness_accept_only_if_improved
            ),
            local_faithfulness_min_improvement=(
                self.config.local_faithfulness_min_improvement
            ),
            local_faithfulness_max_weight_change_pct=(
                self.config.local_faithfulness_max_weight_change_pct
            ),
            fei_threshold_pct=self.config.fei_threshold_pct,
            threshold_inclusive=self.config.threshold_inclusive,
            combination_strategy=combination_strategy,
        )

        result = replace(
            result,
            metadata={
                **result.metadata,
                "combination_rule": (
                    "auto:n<7 exhaustive; n>=7 ecfc"
                    if self.config.local_combination_strategy == "auto"
                    else f"forced:{self.config.local_combination_strategy}"
                ),
                "local_combination_strategy_config": (
                    self.config.local_combination_strategy
                ),
                "target_column": self.target_column_,
                "preprocessing": self._preprocessing_metadata(),
            },
        )

        if target_class is None:
            encoded_label = result.target_class
            try:
                original_label = self.label_encoder_.inverse_transform([int(encoded_label)])[0]
            except (ValueError, TypeError):
                original_label = encoded_label
            result = replace(result, target_class=original_label)
        return result

    def _evaluation_split(self, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
        core = self._require_fitted()
        split = core.get_splits()
        if split_name == "train":
            X_values, y_values = split.X_train, split.y_train
        elif split_name in {"validation", "opt"}:
            X_values, y_values = split.X_opt, split.y_opt
        elif split_name == "test":
            X_values, y_values = split.X_test, split.y_test
        else:
            raise ValueError("split must be 'train', 'validation', or 'test'.")
        if len(X_values) == 0:
            raise ValueError(f"The requested {split_name!r} split is empty.")
        frame = pd.DataFrame(X_values, columns=split.feature_names)
        return frame, pd.Series(y_values)

    def evaluate_global_faithfulness(
        self,
        *,
        split: str = "test",
        importance_scores: Optional[Mapping[str, float]] = None,
        metric: Optional[str] = None,
        runs_per_feature: Optional[int] = None,
        absolute_drop: bool = False,
        drop_mode: Optional[str] = None,
        target_class: Optional[Any] = None,
        probability_abs_drop: Optional[bool] = None,
        conditional_permutation: Optional[bool] = None,
        top_k: Optional[int] = None,
        compute_pd_variance: Optional[bool] = None,
        n_jobs: Optional[int] = None,
    ) -> FaithfulnessResult:
        self._require_fitted()
        assert self.model_ is not None
        X_eval, y_eval = self._evaluation_split(split)
        scores = (
            dict(importance_scores)
            if importance_scores is not None
            else dict(self.explain_global().global_fei)
        )
        return calculate_global_faithfulness(
            model=self.model_,
            X_eval=X_eval,
            y_eval=y_eval,
            importance_scores=scores,
            metric=metric or self.config.faithfulness_metric,
            split="validation" if split == "opt" else split,
            runs_per_feature=(
                self.config.faithfulness_runs_per_feature
                if runs_per_feature is None
                else runs_per_feature
            ),
            random_state=self.config.random_state,
            absolute_drop=absolute_drop,
            drop_mode=drop_mode or self.config.faithfulness_drop_mode,
            target_class=target_class,
            probability_abs_drop=(
                self.config.faithfulness_probability_abs_drop
                if probability_abs_drop is None
                else probability_abs_drop
            ),
            conditional_permutation=(
                self.config.faithfulness_conditional_permutation
                if conditional_permutation is None
                else conditional_permutation
            ),
            top_k=self.config.faithfulness_top_k if top_k is None else top_k,
            compute_pd_variance=(
                self.config.faithfulness_compute_pd_variance
                if compute_pd_variance is None
                else compute_pd_variance
            ),
            n_jobs=self.config.n_jobs if n_jobs is None else n_jobs,
            prefer=self.config.prefer,
        )

    def evaluate_global_fidelity(
        self,
        *,
        importance_scores: Optional[Mapping[str, float]] = None,
        top_k: Optional[int] = None,
        metric: Optional[str] = None,
        max_depth: Optional[object] = None,
    ) -> FidelityResult:
        self._require_fitted()
        assert self.model_ is not None
        X_train, _ = self._evaluation_split("train")
        X_test, _ = self._evaluation_split("test")
        scores = (
            dict(importance_scores)
            if importance_scores is not None
            else dict(self.explain_global().global_fei)
        )
        result, surrogate = calculate_global_fidelity(
            model=self.model_,
            X_train=X_train,
            X_test=X_test,
            importance_scores=scores,
            top_k=self.config.fidelity_top_k if top_k is None else top_k,
            metric=metric or self.config.fidelity_metric,
            max_depth=(
                self.config.fidelity_max_depth if max_depth is None else max_depth
            ),
            random_state=self.config.random_state,
        )
        self.global_surrogate_ = surrogate
        return result

    def evaluate_local_stability(
        self,
        x_instance: Union[pd.Series, pd.DataFrame],
        *,
        target_class: Optional[Any] = None,
        baseline: Optional[str] = None,
        score_mode: Optional[str] = None,
        fvi_method: Optional[str] = None,
        n_perturbations: int = 20,
        numeric_scale: float = 0.02,
        categorical_flip_probability: float = 0.15,
        ordinal_step_probability: float = 0.60,
        target_mean_absolute_behavior_change: float = 1e-3,
        max_budget_tries: int = 4,
        budget_growth: float = 2.0,
    ) -> StabilityResult:
        self._require_fitted()
        if self.X_train_raw_ is None:
            raise RuntimeError("Raw training rows are unavailable.")
        if isinstance(x_instance, pd.DataFrame):
            if len(x_instance) != 1:
                raise ValueError("x_instance DataFrame must contain exactly one row.")
            raw_instance = x_instance.iloc[0]
        elif isinstance(x_instance, pd.Series):
            raw_instance = x_instance
        else:
            raise TypeError("x_instance must be a pandas Series or one-row DataFrame.")

        base = self.explain_local(
            raw_instance,
            target_class=target_class,
            baseline=baseline,
            score_mode=score_mode,
            fvi_method=fvi_method,
        )

        def explain(perturbed: pd.Series) -> LocalExplanation:
            return self.explain_local(
                perturbed,
                target_class=base.target_class,
                baseline=base.baseline,
                score_mode=base.score_mode,
                fvi_method=str(base.metadata["fvi_method"]),
            )

        return calculate_local_stability(
            x_instance=raw_instance,
            X_train_raw=self.X_train_raw_,
            base_explanation=base,
            explain=explain,
            categorical_features=self.categorical_features_,
            ordinal_features=self.ordinal_features_,
            n_perturbations=n_perturbations,
            random_state=self.config.random_state,
            numeric_scale=numeric_scale,
            categorical_flip_probability=categorical_flip_probability,
            ordinal_step_probability=ordinal_step_probability,
            target_mean_absolute_behavior_change=target_mean_absolute_behavior_change,
            max_budget_tries=max_budget_tries,
            budget_growth=budget_growth,
        )
