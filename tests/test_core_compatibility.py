from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from fixait.core import CalcFeatureWeight


ROOT = pathlib.Path(__file__).resolve().parents[1]

def test_compatibility_mode_matches_published_global_reference():
    X, y = make_classification(
        n_samples=220,
        n_features=7,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    data = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    data["class"] = y
    model = RandomForestClassifier(
        n_estimators=16,
        max_depth=4,
        random_state=42,
        n_jobs=1,
    )
    common = dict(
        df=data,
        model=model,
        group_size=5,
        test_size=0.20,
        opt_size=0.0,
        random_state=42,
        stratify=True,
        n_jobs=1,
        model_n_jobs=1,
        plot=False,
        verbose=False,
    )

    reference = json.loads(
        (ROOT / "tests" / "fixtures" / "legacy_global_reference.json").read_text(
            encoding="utf-8"
        )
    )
    current = CalcFeatureWeight(
        **common,
        feature_selection_scope="full",
        compatibility_mode=True,
    )

    assert current.features == reference["features"]
    assert current.alg_consistency == reference["algorithm_consistency"]
    assert current.new_weight_format is not None
    assert current.new_weight_format.keys() == reference["global_fei"].keys()
    np.testing.assert_allclose(
        list(current.new_weight_format.values()),
        list(reference["global_fei"].values()),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        list(current.compute_value_impact().values()),
        list(reference["global_fvi"].values()),
        rtol=0.0,
        atol=1e-12,
    )
