from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetClass:
    label: Any
    index: int


class ModelScorer:
    """Produce a scalar model score while keeping class labels and columns aligned."""

    def __init__(self, model: Any, score_mode: str = "proba") -> None:
        if score_mode not in {"proba", "margin", "logit"}:
            raise ValueError("score_mode must be 'proba', 'margin', or 'logit'.")
        self.model = model
        self.score_mode = score_mode

    def resolve_target(self, X: pd.DataFrame, target_class: Optional[Any] = None) -> TargetClass:
        classes = np.asarray(getattr(self.model, "classes_", []))
        if classes.size:
            if target_class is None:
                if hasattr(self.model, "predict_proba"):
                    idx = int(np.argmax(np.asarray(self.model.predict_proba(X), dtype=float)[0]))
                    return TargetClass(classes[idx].item() if hasattr(classes[idx], "item") else classes[idx], idx)
                predicted = np.asarray(self.model.predict(X)).reshape(-1)[0]
                matches = np.flatnonzero(classes == predicted)
                if not len(matches):
                    raise ValueError(f"Predicted class {predicted!r} is missing from model.classes_.")
                return TargetClass(predicted, int(matches[0]))

            matches = np.flatnonzero(classes == target_class)
            if not len(matches):
                raise ValueError(
                    f"target_class={target_class!r} is not present in model.classes_={classes.tolist()}."
                )
            return TargetClass(target_class, int(matches[0]))

        if target_class is None:
            predicted = np.asarray(self.model.predict(X)).reshape(-1)[0]
            return TargetClass(predicted, 0)
        return TargetClass(target_class, int(target_class) if isinstance(target_class, int) else 0)

    def score(self, X: pd.DataFrame, target: TargetClass) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(X), dtype=float)
            p_target = proba[:, target.index]
            if self.score_mode == "proba":
                return p_target
            if self.score_mode == "margin":
                if proba.shape[1] < 2:
                    return p_target
                other = np.delete(proba, target.index, axis=1)
                return p_target - np.max(other, axis=1)
            eps = 1e-9
            p_target = np.clip(p_target, eps, 1.0 - eps)
            return np.log(p_target / (1.0 - p_target))

        if hasattr(self.model, "decision_function"):
            decision = np.asarray(self.model.decision_function(X), dtype=float)
            if decision.ndim == 1:
                sign = 1.0 if target.index == 1 else -1.0
                return sign * decision
            return decision[:, target.index]

        predicted = np.asarray(self.model.predict(X)).reshape(-1)
        return (predicted == target.label).astype(float)

