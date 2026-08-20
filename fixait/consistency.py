from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .results import SelfConsistencyResult


Combination = Tuple[str, ...]


def _pair_similarity(a: Combination, b: Combination, metric: str) -> float:
    if metric == "exact":
        return float(a == b)
    intersection = len(set(a) & set(b))
    if metric == "legacy_overlap":
        return float(intersection / max(1, len(a)))
    if metric == "jaccard":
        union = len(set(a) | set(b))
        return float(intersection / max(1, union))
    raise ValueError("metric must be 'legacy_overlap', 'jaccard', or 'exact'.")


def calculate_self_consistency(
    combinations: Iterable[Sequence[str]],
    behavior_scores: Mapping[Combination, float],
    explanation_scores: Mapping[str, float],
    *,
    metric: str = "legacy_overlap",
    behavior_measure: str,
) -> SelfConsistencyResult:
    groups: Dict[int, List[Combination]] = defaultdict(list)
    for combination in combinations:
        groups[len(combination)].append(tuple(combination))

    by_size: Dict[int, float] = {}
    for subset_size, group in sorted(groups.items()):
        behavior_rank = sorted(
            group,
            key=lambda comb: float(behavior_scores[comb]),
            reverse=True,
        )
        explanation_rank = sorted(
            group,
            key=lambda comb: sum(
                float(explanation_scores.get(feature, 0.0)) for feature in comb
            ),
            reverse=True,
        )
        similarities = [
            _pair_similarity(a, b, metric)
            for a, b in zip(behavior_rank, explanation_rank)
        ]
        by_size[subset_size] = float(np.mean(similarities)) if similarities else 0.0

    overall = float(np.mean(list(by_size.values()))) if by_size else 0.0
    return SelfConsistencyResult(
        overall=overall,
        by_subset_size=by_size,
        metric=metric,
        behavior_measure=behavior_measure,
    )
