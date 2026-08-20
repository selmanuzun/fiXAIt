from __future__ import annotations

from typing import Mapping, Optional


def select_global_impacts(
    global_fei: Mapping[str, float],
    global_fvi: Mapping[str, float],
    *,
    threshold_pct: Optional[float] = 3.0,
    inclusive: bool = True,
    drop_non_positive: bool = True,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Apply the notebook FEI threshold and keep FVI keys aligned."""

    fei = {feature: float(value) for feature, value in global_fei.items()}
    fvi = {feature: float(value) for feature, value in global_fvi.items()}

    if drop_non_positive:
        dropped = {feature for feature, value in fei.items() if value <= 0.0}
    else:
        dropped = {feature for feature, value in fei.items() if value < 0.0}

    positive = {
        feature: value
        for feature, value in fei.items()
        if feature not in dropped and value > 0.0
    }
    positive_sum = sum(positive.values())
    if threshold_pct is not None and positive_sum > 0:
        threshold = float(threshold_pct) / 100.0
        for feature, value in positive.items():
            share = value / positive_sum
            if (inclusive and share <= threshold) or (
                not inclusive and share < threshold
            ):
                dropped.add(feature)

    common = set(fei) & set(fvi)
    kept = sorted(common - dropped)
    selected_fei = {feature: round(fei[feature], 3) for feature in kept}
    selected_fvi = {feature: round(fvi[feature], 3) for feature in kept}
    all_dropped = sorted((set(fei) | set(fvi)) - set(kept))
    return selected_fei, selected_fvi, all_dropped


def select_local_impacts(
    local_fei: Mapping[str, float],
    local_fvi: Mapping[str, float],
    *,
    threshold_pct: Optional[float] = 3.0,
    inclusive: bool = True,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    """Threshold local FEI by absolute share and keep signed impacts aligned.

    Unlike global selection, negative local FEI values are meaningful: they show
    that a feature moves the explained instance away from the target score.  The
    threshold is therefore calculated from absolute FEI magnitudes while the
    original signs and candidate-feature order are preserved.
    """

    fei = {feature: float(value) for feature, value in local_fei.items()}
    fvi = {feature: float(value) for feature, value in local_fvi.items()}
    common = set(fei) & set(fvi)
    dropped: set[str] = (set(fei) | set(fvi)) - common

    magnitude_sum = sum(abs(fei[feature]) for feature in fei if feature in common)
    if threshold_pct is not None and magnitude_sum > 0.0:
        threshold = float(threshold_pct) / 100.0
        for feature in fei:
            if feature not in common:
                continue
            share = abs(fei[feature]) / magnitude_sum
            if (inclusive and share <= threshold) or (
                not inclusive and share < threshold
            ):
                dropped.add(feature)

    kept = [feature for feature in fei if feature in common and feature not in dropped]
    selected_fei = {feature: fei[feature] for feature in kept}
    selected_fvi = {feature: fvi[feature] for feature in kept}
    all_features = list(dict.fromkeys([*fei, *fvi]))
    all_dropped = [feature for feature in all_features if feature not in kept]
    return selected_fei, selected_fvi, all_dropped
