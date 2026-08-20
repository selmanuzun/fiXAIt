from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .results import FaithfulnessResult, FidelityResult


def _safe_spearman(a: Sequence[float], b: Sequence[float]) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or len(np.unique(left)) <= 1 or len(np.unique(right)) <= 1:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _model_performance(model: Any, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y, model.predict(X)))
    if metric == "f1_weighted":
        return float(f1_score(y, model.predict(X), average="weighted"))
    if metric == "neg_log_loss":
        if not hasattr(model, "predict_proba"):
            raise ValueError("metric='neg_log_loss' requires model.predict_proba().")
        return float(-log_loss(y, model.predict_proba(X), labels=model.classes_))
    raise ValueError("metric must be 'accuracy', 'f1_weighted', or 'neg_log_loss'.")


def _select_importance_features(
    importance_scores: Mapping[str, float],
    available_columns: Sequence[str],
    *,
    top_k: Optional[int],
    epsilon: float,
) -> list[str]:
    available = set(available_columns)
    importance = pd.Series(
        {
            feature: float(value)
            for feature, value in importance_scores.items()
            if feature in available
        },
        dtype=float,
    )
    if importance.empty:
        return []
    active = importance[importance.abs() > epsilon]
    if active.empty:
        active = importance
    if top_k is not None and top_k < len(active):
        return active.abs().nlargest(top_k).index.tolist()
    return active.index.tolist()


def _class_positions(model: Any, labels: Sequence[Any]) -> np.ndarray:
    if not hasattr(model, "classes_"):
        raise ValueError("Probability scoring requires a fitted model.classes_.")
    mapping = {label: index for index, label in enumerate(model.classes_)}
    try:
        return np.asarray([mapping[label] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Class label {exc.args[0]!r} is not present in model.classes_."
        ) from exc


def evaluate_global_faithfulness(
    *,
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    importance_scores: Mapping[str, float],
    metric: str = "accuracy",
    split: str = "test",
    runs_per_feature: int = 30,
    random_state: int = 42,
    absolute_drop: bool = False,
    drop_mode: str = "metric",
    target_class: Optional[Any] = None,
    probability_abs_drop: bool = True,
    conditional_permutation: bool = False,
    top_k: Optional[int] = None,
    epsilon: float = 1e-8,
    compute_pd_variance: bool = False,
    pd_grid_resolution: int = 30,
    pd_percentiles: tuple[float, float] = (0.05, 0.95),
    n_jobs: int = 1,
    prefer: str = "threads",
) -> FaithfulnessResult:
    """Notebook-compatible permutation faithfulness with optional diagnostics."""

    if X_eval.empty or len(y_eval) == 0:
        raise ValueError("The evaluation split cannot be empty.")
    if len(X_eval) != len(y_eval):
        raise ValueError("X_eval and y_eval must have the same number of rows.")
    if runs_per_feature < 1:
        raise ValueError("runs_per_feature must be at least 1.")
    if drop_mode not in {"metric", "probability"}:
        raise ValueError("drop_mode must be 'metric' or 'probability'.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive or None.")

    features = _select_importance_features(
        importance_scores,
        X_eval.columns,
        top_k=top_k,
        epsilon=epsilon,
    )
    if not features:
        raise ValueError("None of the importance-score features exist in X_eval.")

    baseline_score: Optional[float] = None
    baseline_probabilities: Optional[np.ndarray] = None
    probability_positions: Optional[np.ndarray] = None
    if drop_mode == "metric":
        baseline_score = _model_performance(model, X_eval, y_eval, metric)
    else:
        if not hasattr(model, "predict_proba"):
            raise ValueError("drop_mode='probability' requires model.predict_proba().")
        probabilities = np.asarray(model.predict_proba(X_eval), dtype=float)
        if target_class is None:
            probability_positions = _class_positions(model, y_eval.to_numpy())
        else:
            position = int(_class_positions(model, [target_class])[0])
            probability_positions = np.full(len(X_eval), position, dtype=int)
        baseline_probabilities = probabilities[
            np.arange(len(X_eval)), probability_positions
        ]

    master_rng = np.random.RandomState(random_state)
    seeds = {
        feature: master_rng.randint(
            0,
            np.iinfo(np.int32).max,
            size=runs_per_feature,
        )
        for feature in features
    }

    permutation_groups: dict[str, list[np.ndarray]] = {}
    if conditional_permutation:
        for feature in features:
            try:
                bins = pd.qcut(X_eval[feature], q=10, duplicates="drop")
                groups = [
                    np.flatnonzero((bins == category).to_numpy())
                    for category in bins.dropna().unique()
                ]
                permutation_groups[feature] = [
                    group for group in groups if len(group) > 1
                ]
            except (TypeError, ValueError):
                permutation_groups[feature] = []

    def one_feature(feature: str) -> tuple[str, float]:
        impacts = []
        original = X_eval[feature].to_numpy(copy=True)
        for seed in seeds[feature]:
            rng = np.random.RandomState(int(seed))
            shuffled = original.copy()
            groups = permutation_groups.get(feature, [])
            if conditional_permutation and groups:
                for indices in groups:
                    values = shuffled[indices].copy()
                    rng.shuffle(values)
                    shuffled[indices] = values
            else:
                rng.shuffle(shuffled)

            permuted = X_eval.copy()
            permuted[feature] = shuffled
            if drop_mode == "metric":
                assert baseline_score is not None
                drop = baseline_score - _model_performance(
                    model,
                    permuted,
                    y_eval,
                    metric,
                )
                impacts.append(abs(drop) if absolute_drop else drop)
            else:
                assert baseline_probabilities is not None
                assert probability_positions is not None
                probabilities = np.asarray(model.predict_proba(permuted), dtype=float)
                perturbed = probabilities[
                    np.arange(len(permuted)), probability_positions
                ]
                difference = baseline_probabilities - perturbed
                if probability_abs_drop:
                    difference = np.abs(difference)
                else:
                    difference = np.maximum(difference, 0.0)
                impacts.append(float(np.mean(difference)))
        finite = np.asarray(impacts, dtype=float)
        finite = finite[np.isfinite(finite)]
        return feature, float(np.mean(finite)) if len(finite) else 0.0

    if n_jobs == 1 or len(features) == 1:
        drop_impacts = dict(one_feature(feature) for feature in features)
    else:
        drop_impacts = dict(
            Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(one_feature)(feature) for feature in features
            )
        )

    pd_variance: dict[str, float] = {}
    if compute_pd_variance:
        low_percentile, high_percentile = pd_percentiles

        def one_pd_variance(feature: str) -> tuple[str, float]:
            low = float(X_eval[feature].quantile(low_percentile))
            high = float(X_eval[feature].quantile(high_percentile))
            grid = np.linspace(low, high, pd_grid_resolution)
            predictions = np.zeros((len(X_eval), len(grid)), dtype=float)
            for index, value in enumerate(grid):
                replaced = X_eval.copy()
                replaced[feature] = value
                if hasattr(model, "predict_proba"):
                    probabilities = np.asarray(model.predict_proba(replaced), dtype=float)
                    if target_class is None:
                        positions = _class_positions(model, y_eval.to_numpy())
                    else:
                        target_position = int(
                            _class_positions(model, [target_class])[0]
                        )
                        positions = np.full(len(replaced), target_position, dtype=int)
                    predictions[:, index] = probabilities[
                        np.arange(len(replaced)), positions
                    ]
                else:
                    predictions[:, index] = np.asarray(model.predict(replaced), dtype=float)
            value = float(np.mean(np.var(predictions, axis=1)))
            return feature, value if np.isfinite(value) else 0.0

        if n_jobs == 1 or len(features) == 1:
            pd_variance = dict(one_pd_variance(feature) for feature in features)
        else:
            pd_variance = dict(
                Parallel(n_jobs=n_jobs, prefer=prefer)(
                    delayed(one_pd_variance)(feature) for feature in features
                )
            )

    faithfulness = _safe_spearman(
        [abs(float(importance_scores[feature])) for feature in features],
        [float(drop_impacts[feature]) for feature in features],
    )
    if drop_mode == "metric":
        baseline_metadata = baseline_score
    else:
        assert baseline_probabilities is not None
        baseline_metadata = float(np.mean(baseline_probabilities))
    return FaithfulnessResult(
        score=faithfulness,
        drop_impacts=drop_impacts,
        metric=metric if drop_mode == "metric" else "target_probability",
        split=split,
        runs_per_feature=runs_per_feature,
        pd_variance=pd_variance,
        metadata={
            "baseline_score": baseline_metadata,
            "absolute_drop": absolute_drop,
            "probability_abs_drop": probability_abs_drop,
            "drop_mode": drop_mode,
            "conditional_permutation": conditional_permutation,
            "n_features": len(features),
            "evaluated_features": features,
            "random_state": random_state,
        },
    )


def evaluate_global_fidelity(
    *,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    importance_scores: Mapping[str, float],
    top_k: int = 7,
    metric: str = "accuracy",
    max_depth: object = "auto",
    random_state: int = 42,
    epsilon: float = 1e-3,
) -> tuple[FidelityResult, Optional[DecisionTreeClassifier]]:
    """Train the notebook decision-tree surrogate and measure black-box fidelity."""

    if X_train.empty or X_test.empty:
        raise ValueError("X_train and X_test must be non-empty.")
    if metric not in {"accuracy", "f1_weighted"}:
        raise ValueError("metric must be 'accuracy' or 'f1_weighted'.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    features = _select_importance_features(
        importance_scores,
        X_train.columns,
        top_k=top_k,
        epsilon=epsilon,
    )
    features = [feature for feature in features if feature in X_test.columns]
    if not features:
        return (
            FidelityResult(
                score=0.0,
                metric=metric,
                selected_features=[],
                top_k=top_k,
                best_max_depth=None,
                metadata={"informative": False, "reason": "no selected features"},
            ),
            None,
        )

    black_box_train = np.asarray(model.predict(X_train))
    black_box_test = np.asarray(model.predict(X_test))
    best_depth: Optional[int]
    if max_depth == "auto":
        counts = pd.Series(black_box_train).value_counts()
        minimum_class_count = int(counts.min()) if not counts.empty else 0
        folds = min(5, minimum_class_count)
        if folds >= 2 and len(counts) >= 2:
            scoring = "accuracy" if metric == "accuracy" else "f1_weighted"
            search = GridSearchCV(
                DecisionTreeClassifier(random_state=random_state),
                {"max_depth": list(range(2, 11))},
                scoring=scoring,
                cv=StratifiedKFold(
                    n_splits=folds,
                    shuffle=True,
                    random_state=random_state,
                ),
                n_jobs=1,
            )
            search.fit(X_train[features], black_box_train)
            best_depth = int(search.best_params_["max_depth"])
        else:
            best_depth = None
    else:
        best_depth = None if max_depth is None else int(max_depth)

    surrogate = DecisionTreeClassifier(
        max_depth=best_depth,
        random_state=random_state,
    )
    surrogate.fit(X_train[features], black_box_train)
    surrogate_predictions = surrogate.predict(X_test[features])
    if metric == "accuracy":
        score = float(accuracy_score(black_box_test, surrogate_predictions))
    else:
        score = float(
            f1_score(
                black_box_test,
                surrogate_predictions,
                average="weighted",
            )
        )
    return (
        FidelityResult(
            score=score,
            metric=metric,
            selected_features=features,
            top_k=top_k,
            best_max_depth=best_depth,
            metadata={
                "informative": True,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "target": "black_box_predictions",
            },
        ),
        surrogate,
    )
