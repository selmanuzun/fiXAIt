from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence


LOCAL_EXHAUSTIVE_FEATURE_LIMIT = 7


def generate_exhaustive_combinations(
    features: Sequence[str],
) -> list[list[str]]:
    """Return every non-empty proper subset in deterministic size order."""

    ordered = list(features)
    if len(ordered) != len(set(ordered)):
        raise ValueError("features contains duplicate names.")
    return [
        list(subset)
        for subset_size in range(1, len(ordered))
        for subset in combinations(ordered, subset_size)
    ]


def generate_cyclic_ecfc_combinations(
    features: Sequence[str],
) -> list[list[str]]:
    """Return the original cyclic ECFC subsets in deterministic order."""

    ordered = list(features)
    if len(ordered) != len(set(ordered)):
        raise ValueError("features contains duplicate names.")
    combination_set: set[tuple[str, ...]] = set()
    for index in range(len(ordered)):
        rotated = ordered[index:] + ordered[:index]
        for subset_size in range(1, len(ordered)):
            combination_set.add(tuple(sorted(rotated[:subset_size])))
    return [list(subset) for subset in sorted(combination_set)]


def generate_local_combinations(
    features: Sequence[str],
    strategy: str = "auto",
) -> tuple[list[list[str]], str]:
    """Generate local coalitions using the automatic or an explicit strategy."""

    if strategy not in {"auto", "exhaustive", "ecfc"}:
        raise ValueError("strategy must be 'auto', 'exhaustive', or 'ecfc'.")
    resolved = strategy
    if strategy == "auto":
        resolved = (
            "exhaustive"
            if len(features) < LOCAL_EXHAUSTIVE_FEATURE_LIMIT
            else "ecfc"
        )
    if resolved == "exhaustive":
        return generate_exhaustive_combinations(features), "exhaustive"
    return generate_cyclic_ecfc_combinations(features), "ecfc"


def canonicalize_combinations(
    features: Sequence[str],
    coalitions: Iterable[Sequence[str]],
) -> list[list[str]]:
    """Canonicalize coalition members and rows independently of input order."""

    ordered_features = list(features)
    feature_positions = {
        feature: index for index, feature in enumerate(ordered_features)
    }
    if len(feature_positions) != len(ordered_features):
        raise ValueError("features contains duplicate names.")

    canonical: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for coalition in coalitions:
        members = list(coalition)
        if len(members) != len(set(members)):
            raise ValueError(f"Coalition contains duplicate features: {members}")
        unknown = [feature for feature in members if feature not in feature_positions]
        if unknown:
            raise ValueError(f"Coalition contains unknown features: {unknown}")
        normalized = tuple(
            sorted(members, key=lambda feature: feature_positions[feature])
        )
        if not normalized or len(normalized) == len(ordered_features):
            raise ValueError(
                "Coalitions must be non-empty proper subsets of selected features."
            )
        if normalized in seen:
            raise ValueError(f"Duplicate coalition received: {list(normalized)}")
        seen.add(normalized)
        canonical.append(normalized)

    canonical.sort(
        key=lambda coalition: (
            len(coalition),
            tuple(feature_positions[feature] for feature in coalition),
        )
    )
    return [list(coalition) for coalition in canonical]
