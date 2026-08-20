from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fixait.consistency import calculate_self_consistency
from fixait.scoring import ModelScorer


def test_model_scorer_maps_non_numeric_class_labels_to_probability_columns():
    X = pd.DataFrame({"x1": [-2.0, -1.0, 1.0, 2.0], "x2": [0.0, 0.2, 0.8, 1.0]})
    y = np.array(["decline", "decline", "approve", "approve"])
    model = LogisticRegression(random_state=42).fit(X, y)
    scorer = ModelScorer(model, score_mode="proba")
    target = scorer.resolve_target(X.iloc[[0]], "approve")

    expected_index = int(np.flatnonzero(model.classes_ == "approve")[0])
    assert target.index == expected_index
    np.testing.assert_allclose(
        scorer.score(X.iloc[[0]], target)[0],
        model.predict_proba(X.iloc[[0]])[0, expected_index],
    )


def test_self_consistency_metrics_are_named_and_distinct():
    combinations = [["a"], ["b"], ["a", "b"], ["b", "c"]]
    behavior = {
        ("a",): 0.9,
        ("b",): 0.8,
        ("a", "b"): 0.7,
        ("b", "c"): 0.6,
    }
    explanation = {"a": 0.1, "b": 0.8, "c": 0.0}

    legacy = calculate_self_consistency(
        combinations,
        behavior,
        explanation,
        metric="legacy_overlap",
        behavior_measure="test",
    )
    exact = calculate_self_consistency(
        combinations,
        behavior,
        explanation,
        metric="exact",
        behavior_measure="test",
    )

    assert legacy.metric == "legacy_overlap"
    assert exact.metric == "exact"
    assert 0.0 <= exact.overall <= legacy.overall <= 1.0

