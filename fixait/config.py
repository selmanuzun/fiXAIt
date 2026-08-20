from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class FiXAItConfig:
    """Configuration shared by global and local fiXAIt explanations."""

    group_size: int = 7
    step: int = 1
    alphas: Sequence[float] = field(
        default_factory=lambda: (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    )
    test_size: float = 0.20
    validation_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    n_jobs: int = -1
    prefer: str = "threads"
    model_n_jobs: Optional[int] = 1
    feature_selection_scope: str = "train"
    top_k_groups: Optional[int] = 12
    sc_metric: str = "legacy_overlap"
    optimize_faithfulness: bool = False
    fei_threshold_pct: Optional[float] = 3.0
    threshold_inclusive: bool = True
    drop_non_positive_fei: bool = True
    faithfulness_metric: str = "accuracy"
    faithfulness_drop_mode: str = "metric"
    faithfulness_runs_per_feature: int = 30
    faithfulness_top_k: Optional[int] = 7
    faithfulness_probability_abs_drop: bool = True
    faithfulness_conditional_permutation: bool = False
    faithfulness_compute_pd_variance: bool = False
    faithfulness_optimizer_steps: int = 500
    faithfulness_optimizer_lr: float = 0.05
    faithfulness_optimizer_tau: float = 25.0
    faithfulness_optimizer_pair_batch: int = 4096
    faithfulness_reg_lambda: float = 0.10
    faithfulness_accept_only_if_improved: bool = True
    faithfulness_min_improvement: float = 0.01
    faithfulness_max_weight_change_pct: Optional[float] = 20.0
    fidelity_metric: str = "accuracy"
    fidelity_top_k: int = 7
    fidelity_max_depth: Optional[Union[str, int]] = "auto"
    local_baseline: str = "median"
    local_score_mode: str = "proba"
    local_fvi_method: str = "finite_difference"
    local_combination_strategy: str = "auto"
    optimize_local_faithfulness: bool = False
    local_faithfulness_runs_per_feature: int = 30
    local_faithfulness_calibration_runs_per_feature: int = 30
    local_faithfulness_optimizer_steps: int = 500
    local_faithfulness_optimizer_lr: float = 0.05
    local_faithfulness_optimizer_tau: float = 25.0
    local_faithfulness_optimizer_pair_batch: int = 4096
    local_faithfulness_reg_lambda: float = 0.10
    local_faithfulness_accept_only_if_improved: bool = True
    local_faithfulness_min_improvement: float = 0.01
    local_faithfulness_max_weight_change_pct: Optional[float] = 20.0
    compatibility_mode: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2.")
        if self.step < 1:
            raise ValueError("step must be at least 1.")
        if self.feature_selection_scope not in {"train", "full"}:
            raise ValueError("feature_selection_scope must be 'train' or 'full'.")
        if self.sc_metric not in {"legacy_overlap", "jaccard", "exact"}:
            raise ValueError("sc_metric must be 'legacy_overlap', 'jaccard', or 'exact'.")
        if self.optimize_faithfulness and self.validation_size <= 0:
            raise ValueError(
                "validation_size must be positive when optimize_faithfulness=True."
            )
        if self.fei_threshold_pct is not None and self.fei_threshold_pct < 0:
            raise ValueError("fei_threshold_pct must be non-negative or None.")
        if self.faithfulness_metric not in {
            "accuracy",
            "f1_weighted",
            "neg_log_loss",
        }:
            raise ValueError(
                "faithfulness_metric must be 'accuracy', 'f1_weighted', "
                "or 'neg_log_loss'."
            )
        if self.faithfulness_drop_mode not in {"metric", "probability"}:
            raise ValueError(
                "faithfulness_drop_mode must be 'metric' or 'probability'."
            )
        if self.faithfulness_runs_per_feature < 1:
            raise ValueError("faithfulness_runs_per_feature must be at least 1.")
        if self.faithfulness_top_k is not None and self.faithfulness_top_k < 1:
            raise ValueError("faithfulness_top_k must be positive or None.")
        if self.faithfulness_optimizer_steps < 1:
            raise ValueError("faithfulness_optimizer_steps must be at least 1.")
        if self.faithfulness_optimizer_lr <= 0:
            raise ValueError("faithfulness_optimizer_lr must be positive.")
        if self.faithfulness_optimizer_tau <= 0:
            raise ValueError("faithfulness_optimizer_tau must be positive.")
        if self.faithfulness_optimizer_pair_batch < 1:
            raise ValueError("faithfulness_optimizer_pair_batch must be at least 1.")
        if self.faithfulness_reg_lambda < 0:
            raise ValueError("faithfulness_reg_lambda must be non-negative.")
        if self.faithfulness_min_improvement < 0:
            raise ValueError("faithfulness_min_improvement must be non-negative.")
        if self.faithfulness_max_weight_change_pct is not None and not (
            0.0 <= self.faithfulness_max_weight_change_pct <= 100.0
        ):
            raise ValueError(
                "faithfulness_max_weight_change_pct must be between 0 and 100 "
                "or None."
            )
        if self.fidelity_metric not in {"accuracy", "f1_weighted"}:
            raise ValueError(
                "fidelity_metric must be 'accuracy' or 'f1_weighted'."
            )
        if self.fidelity_top_k < 1:
            raise ValueError("fidelity_top_k must be at least 1.")
        if not (
            self.fidelity_max_depth == "auto"
            or self.fidelity_max_depth is None
            or (
                isinstance(self.fidelity_max_depth, int)
                and self.fidelity_max_depth >= 1
            )
        ):
            raise ValueError(
                "fidelity_max_depth must be 'auto', None, or a positive integer."
            )
        if self.local_baseline not in {"median", "mean", "zero"}:
            raise ValueError("local_baseline must be 'median', 'mean', or 'zero'.")
        if self.local_score_mode not in {"proba", "margin", "logit"}:
            raise ValueError("local_score_mode must be 'proba', 'margin', or 'logit'.")
        if self.local_fvi_method not in {"finite_difference", "legacy_ridge"}:
            raise ValueError(
                "local_fvi_method must be 'finite_difference' or 'legacy_ridge'."
            )
        if self.local_combination_strategy not in {"auto", "exhaustive", "ecfc"}:
            raise ValueError(
                "local_combination_strategy must be 'auto', 'exhaustive', or 'ecfc'."
            )
        if self.local_faithfulness_runs_per_feature < 1:
            raise ValueError(
                "local_faithfulness_runs_per_feature must be at least 1."
            )
        if self.local_faithfulness_calibration_runs_per_feature < 1:
            raise ValueError(
                "local_faithfulness_calibration_runs_per_feature must be at least 1."
            )
        if self.local_faithfulness_optimizer_steps < 1:
            raise ValueError("local_faithfulness_optimizer_steps must be at least 1.")
        if self.local_faithfulness_optimizer_lr <= 0:
            raise ValueError("local_faithfulness_optimizer_lr must be positive.")
        if self.local_faithfulness_optimizer_tau <= 0:
            raise ValueError("local_faithfulness_optimizer_tau must be positive.")
        if self.local_faithfulness_optimizer_pair_batch < 1:
            raise ValueError(
                "local_faithfulness_optimizer_pair_batch must be at least 1."
            )
        if self.local_faithfulness_reg_lambda < 0:
            raise ValueError("local_faithfulness_reg_lambda must be non-negative.")
        if self.local_faithfulness_min_improvement < 0:
            raise ValueError(
                "local_faithfulness_min_improvement must be non-negative."
            )
        if self.local_faithfulness_max_weight_change_pct is not None and not (
            0.0 <= self.local_faithfulness_max_weight_change_pct <= 100.0
        ):
            raise ValueError(
                "local_faithfulness_max_weight_change_pct must be between 0 and "
                "100 or None."
            )
