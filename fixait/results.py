from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class SelfConsistencyResult:
    overall: float
    by_subset_size: Mapping[int, float]
    metric: str
    behavior_measure: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    drop_impacts: Mapping[str, float]
    metric: str
    split: str
    runs_per_feature: int
    pd_variance: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FidelityResult:
    score: float
    metric: str
    selected_features: List[str]
    top_k: int
    best_max_depth: Optional[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalExplanation:
    global_fei: Mapping[str, float]
    global_fvi: Mapping[str, float]
    global_sc: SelfConsistencyResult
    global_faithfulness: FaithfulnessResult
    global_fidelity: FidelityResult
    selected_features: List[str]
    dropped_features: List[str]
    selected_feature_accuracy: float
    optimization_applied: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fei(self) -> Mapping[str, float]:
        return self.global_fei

    @property
    def fvi(self) -> Mapping[str, float]:
        return self.global_fvi

    @property
    def sc(self) -> SelfConsistencyResult:
        return self.global_sc

    @property
    def faithfulness(self) -> float:
        return self.global_faithfulness.score

    @property
    def fidelity(self) -> float:
        return self.global_fidelity.score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalExplanation:
    local_fei: Mapping[str, float]
    local_fvi: Mapping[str, float]
    local_sc: SelfConsistencyResult
    selected_features: List[str]
    dropped_features: List[str]
    target_class: Any
    target_class_index: int
    baseline: str
    score_mode: str
    raw_local_fvi: Mapping[str, float]
    legacy_local_fvi: Optional[Mapping[str, float]]
    fidelity_r2: float
    fei_fvi_agreement_spearman: float
    local_faithfulness_spearman: float
    optimization_applied: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fei(self) -> Mapping[str, float]:
        return self.local_fei

    @property
    def fvi(self) -> Mapping[str, float]:
        return self.local_fvi

    @property
    def sc(self) -> SelfConsistencyResult:
        return self.local_sc

    @property
    def faithfulness_spearman(self) -> float:
        """Deprecated alias for the former FEI--FVI agreement field."""

        return self.fei_fvi_agreement_spearman

    def to_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        values["faithfulness_spearman"] = self.fei_fvi_agreement_spearman
        return values


@dataclass(frozen=True)
class StabilityResult:
    informative: bool
    n_perturbations: int
    numeric_scale: float
    categorical_flip_probability: float
    ordinal_step_probability: float
    mean_absolute_behavior_change: float
    max_absolute_behavior_change: float
    spearman_abs_mean: Optional[float]
    cosine_mean: Optional[float]
    mean_absolute_fei_change: Optional[float]
    l2_fei_change: Optional[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
