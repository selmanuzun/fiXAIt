from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixait import FiXAIt, FiXAItConfig  # noqa: E402


DATASETS = {
    "adult": ("benchmarks/data/adult.csv", "income_label"),
    "boston_housing": ("benchmarks/data/boston_housing.csv", "price_band"),
    "compas": ("benchmarks/data/compas.csv", "risk_level"),
    "german_credit": ("benchmarks/data/german_credit.csv", "creditability"),
    "student": ("benchmarks/data/student.csv", "grade"),
}


def run(
    max_rows: int,
    n_estimators: int,
    *,
    optimize_faithfulness: bool = False,
    optimize_local_faithfulness: bool = False,
    optimizer_steps: int = 100,
) -> list[dict]:
    reports = []
    for name, (relative_path, target_name) in DATASETS.items():
        started = time.perf_counter()
        data = pd.read_csv(ROOT / relative_path)
        if len(data) > max_rows:
            data = data.sample(max_rows, random_state=42).reset_index(drop=True)
        data = data.rename(columns={"class": target_name})
        feature_count = data.shape[1] - 1
        group_size = min(7, feature_count)
        explainer = FiXAIt(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=42,
                n_jobs=1,
            ),
            config=FiXAItConfig(
                group_size=group_size,
                optimize_faithfulness=optimize_faithfulness,
                optimize_local_faithfulness=optimize_local_faithfulness,
                faithfulness_runs_per_feature=3,
                local_faithfulness_runs_per_feature=5,
                local_faithfulness_calibration_runs_per_feature=5,
                faithfulness_optimizer_steps=optimizer_steps,
                local_faithfulness_optimizer_steps=optimizer_steps,
                n_jobs=1,
                model_n_jobs=1,
                random_state=42,
            ),
        ).fit(data, target_column=target_name)

        global_result = explainer.explain_global()
        local_result = explainer.explain_local(data.drop(columns=target_name).iloc[0])
        optimization = global_result.metadata["optimization"]
        reports.append(
            {
                "dataset": name,
                "target_column": target_name,
                "rows": len(data),
                "features": feature_count,
                "selected": global_result.selected_features,
                "optimization_requested": optimization["requested"],
                "optimization_accepted": optimization["accepted"],
                "optimization_applied": global_result.optimization_applied,
                "global_sc": global_result.global_sc.overall,
                "local_sc": local_result.local_sc.overall,
                "local_combination_strategy": local_result.metadata[
                    "combination_strategy"
                ],
                "local_combinations": local_result.metadata["n_combinations"],
                "local_surrogate_rows": local_result.metadata["n_surrogate_rows"],
                "empty_coalition_included": local_result.metadata[
                    "empty_coalition_included"
                ],
                "global_faithfulness": global_result.faithfulness,
                "global_fidelity": global_result.fidelity,
                "local_fidelity_r2": local_result.fidelity_r2,
                "local_fei_fvi_agreement_spearman": (
                    local_result.fei_fvi_agreement_spearman
                ),
                "local_faithfulness_spearman": (
                    local_result.local_faithfulness_spearman
                ),
                "local_optimization_requested": local_result.metadata[
                    "optimization"
                ]["requested"],
                "local_optimization_accepted": local_result.metadata[
                    "optimization"
                ]["accepted"],
                "local_optimization_applied": local_result.optimization_applied,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick fiXAIt validation suite.")
    parser.add_argument("--max-rows", type=int, default=1500)
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--optimize-faithfulness", action="store_true")
    parser.add_argument("--optimize-local-faithfulness", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=100)
    arguments = parser.parse_args()
    report = run(
        arguments.max_rows,
        arguments.n_estimators,
        optimize_faithfulness=arguments.optimize_faithfulness,
        optimize_local_faithfulness=arguments.optimize_local_faithfulness,
        optimizer_steps=arguments.optimizer_steps,
    )
    if not all(np.isfinite(item["global_sc"]) for item in report):
        raise SystemExit("A non-finite global SC value was produced.")
    print(json.dumps(report, indent=2))
