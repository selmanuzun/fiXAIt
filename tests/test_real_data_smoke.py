from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fixait import FiXAIt, FiXAItConfig


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_student_dataset_global_and_local_smoke():
    dataset_path = ROOT / "benchmarks" / "data" / "student.csv"
    data = pd.read_csv(dataset_path)
    X = data.drop(columns="class")
    y = data["class"]
    explainer = FiXAIt(
        RandomForestClassifier(
            n_estimators=12,
            max_depth=4,
            random_state=42,
            n_jobs=1,
        ),
        config=FiXAItConfig(
            group_size=7,
            faithfulness_runs_per_feature=2,
            local_faithfulness_runs_per_feature=3,
            n_jobs=1,
            model_n_jobs=1,
            random_state=42,
        ),
    ).fit_csv(
        dataset_path,
        target_column="class",
    )

    global_result = explainer.explain_global()
    local_result = explainer.explain_local(X.iloc[0])

    assert 0 < len(global_result.selected_features) <= 7
    assert global_result.metadata["n_combinations"] == 42
    assert 0.0 <= global_result.global_sc.overall <= 1.0
    assert round(global_result.global_sc.overall, 3) == global_result.metadata[
        "legacy_algorithm_consistency"
    ], (
        global_result.global_sc.overall,
        global_result.metadata["legacy_algorithm_consistency"],
    )
    assert np.isfinite(list(global_result.global_fei.values())).all()
    assert np.isfinite(list(global_result.global_fvi.values())).all()
    assert np.isfinite(global_result.faithfulness)
    assert np.isfinite(global_result.fidelity)

    assert 0 < len(local_result.selected_features) <= 7
    assert set(local_result.local_fei) == set(local_result.local_fvi)
    assert set(local_result.selected_features).isdisjoint(local_result.dropped_features)
    assert set(local_result.selected_features) | set(local_result.dropped_features) == set(
        local_result.metadata["candidate_features"]
    )
    assert local_result.metadata["n_combinations"] == 42
    assert local_result.metadata["n_surrogate_rows"] == 43
    assert local_result.metadata["combination_strategy"] == "ecfc"
    assert local_result.metadata["combination_space_complete"] is False
    assert local_result.metadata["empty_coalition_included"] is False
    assert local_result.metadata["sc_includes_empty_coalition"] is False
    assert 0.0 <= local_result.local_sc.overall <= 1.0
    assert np.isfinite(list(local_result.local_fei.values())).all()
    assert np.isfinite(list(local_result.local_fvi.values())).all()
    assert np.isfinite(local_result.fei_fvi_agreement_spearman)
    assert np.isfinite(local_result.local_faithfulness_spearman)


def test_student_dataset_can_force_seven_feature_local_ecfc():
    dataset_path = ROOT / "benchmarks" / "data" / "student.csv"
    data = pd.read_csv(dataset_path)
    X = data.drop(columns="class")
    explainer = FiXAIt(
        RandomForestClassifier(
            n_estimators=8,
            max_depth=4,
            random_state=17,
            n_jobs=1,
        ),
        config=FiXAItConfig(
            group_size=7,
            local_combination_strategy="ecfc",
            faithfulness_runs_per_feature=2,
            local_faithfulness_runs_per_feature=3,
            n_jobs=1,
            model_n_jobs=1,
            random_state=17,
        ),
    ).fit_csv(dataset_path, target_column="class")

    local_result = explainer.explain_local(X.iloc[0])

    assert local_result.metadata["combination_strategy"] == "ecfc"
    assert local_result.metadata["local_combination_strategy_config"] == "ecfc"
    assert local_result.metadata["combination_rule"] == "forced:ecfc"
    assert local_result.metadata["n_candidate_features"] == 7
    assert local_result.metadata["n_combinations"] == 42
    assert local_result.metadata["n_surrogate_rows"] == 43
    assert local_result.metadata["empty_coalition_included"] is False
