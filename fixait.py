
"""
calc_feature_weight_optimized.py

Refactor + performance optimizations:
- CSV read / train_test_split / scaling are executed once (cached) → no repeated split+scaler work
- Accuracy for feature subsets is cached → prevents repeated fit/predict on the same subset
- Plots are disabled by default (plot=False)
- Single-layer parallelism: outer joblib.Parallel only; RidgeCV is used instead of GridSearchCV to avoid nested parallelism
- The model is cloned for each job (deepcopy/clone). Optionally force estimator n_jobs=1 to prevent CPU oversubscription

Notes:
- Data format: the label column must be named "class" (compatible with the original code)
- group_size: length of the feature group to select (the original code used a global group_size; it is now a parameter)
"""


from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Iterable

import threading

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeCV
from sklearn.base import clone as sk_clone


@dataclass
class SplitData:
    """One-time split + scaling output (train/opt/test)."""
    feature_names: List[str]
    X_train: np.ndarray
    X_opt: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_opt: np.ndarray
    y_test: np.ndarray
    train_idx: np.ndarray
    opt_idx: np.ndarray
    test_idx: np.ndarray


class CalcFeatureWeight:
    """
    A faster and more controllable version of the original CalcFeatureWeight class.

    MAIN OUTPUTS (kept compatible with the original code):
      - self.features                  : selected feature list + ["class"]
      - self.new_weight_format         : Existence Importance (p) dictionary
      - self.grouped_weight_data_valimp: intermediate data for Value Impact computation
      - self.alg_consistency           : best consistency score
    """


    def __init__(
        self,
        *,
        data_path: Optional[str] = None,
        usecols: Optional[List[str]] = None,
        df: Optional[pd.DataFrame] = None,
        model: Any,
        group_size: int = 7,
        step: int = 1,
        alphas: Optional[List[float]] = None,
        test_size: float = 0.20,
        opt_size: Optional[float] = 0.0,
        random_state: int = 42,
        stratify: bool = False,
        n_jobs: int = -1,
        prefer: str = "threads",
        model_n_jobs: Optional[int] = 1,
        auto_run: bool = True,
        plot: bool = False,
        verbose: bool = True,
    ):
        if (df is None) == (data_path is None):
            raise ValueError("Provide exactly ONE of df or data_path.")

        self.verbose = verbose
        self.plot = plot

        self.group_size = int(group_size)
        self.step = int(step)

        if df is not None:
            self.data_f = df.copy()
        else:
            _usecols = None
            if usecols is not None:
                _usecols = list(usecols)
                if "class" not in _usecols:
                    _usecols.append("class")
            self.data_f = pd.read_csv(data_path, usecols=_usecols)
        self.num_of_data = len(self.data_f)

        # Model template (should be unfitted)
        self.pick_model = model
        self.model_n_jobs = model_n_jobs

        self.alphas = alphas if alphas is not None else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.opt_size = 0.0 if opt_size is None else float(opt_size)
        self.do_stratify = bool(stratify)

        self.n_jobs = int(n_jobs)
        self.prefer = str(prefer)

        # Original fields (kept for backward compatibility)
        self.df2: Optional[pd.DataFrame] = None
        self.df_norm: Optional[pd.DataFrame] = None
        self.avg_list: Optional[List[List[float]]] = None
        self.cols: Optional[List[str]] = None
        self.features: Optional[List[str]] = None
        self.acc_select: Optional[float] = None

        self.grouped_weight_data = None
        self.grouped_weight_data_valimp = None
        self.grouped_weight_data_nonzero = None
        self.grouped_weight_data_nonzero_valimp = None

        self.data_acc_weight_sorted = None
        self.new_weight_format: Optional[Dict[str, float]] = None
        self.alg_consistency = 0.0

        self.num_of_class = 0
        self.num_of_feature = 0

        # Split + scaling cache (this is where the big speed-up comes from)
        self.split_: Optional[SplitData] = None

        # Accuracy cache: prevents repeated fit/predict for the same feature subset.
        # Key: tuple(sorted(feature_names_without_class))
        self._acc_cache: Dict[Tuple[str, ...], float] = {}
        self._acc_cache_lock = threading.Lock()

        # Fast feature-name to column-index mapping for repeated subset selection.
        self._feat_idx_map: Optional[Dict[str, int]] = None

        if auto_run:
            self.run(plot=plot)

    # ----------------------------
    # Utilities
    # ----------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _clone_model(self):
        """Safely clone the sklearn model and optionally reduce n_jobs."""
        try:
            m = sk_clone(self.pick_model)
        except Exception:
            m = deepcopy(self.pick_model)

        if self.model_n_jobs is not None and hasattr(m, "n_jobs"):
            try:
                setattr(m, "n_jobs", self.model_n_jobs)
            except Exception:
                pass
        return m

    def _pred_to_1d_label(self, y_pred):
        y_pred = np.asarray(y_pred)

        # Already (n,)
        if y_pred.ndim == 1:
            return y_pred

        # If (n, 1)
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            return y_pred.ravel()

        # If (n, k): probabilities/scores -> class label
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            return np.argmax(y_pred, axis=1)

        # Safe fallback
        return y_pred.ravel()
    def _selection_split(self, split: SplitData) -> Tuple[np.ndarray, np.ndarray]:
        """Split to use for *selection/tuning* computations.

        Prefer OPT (validation) if available; otherwise fall back to TEST.
        This keeps the final test set untouched by repeated selection loops.
        """
        try:
            X_opt = getattr(split, "X_opt", None)
            y_opt = getattr(split, "y_opt", None)
            if X_opt is not None and y_opt is not None and len(X_opt) > 0:
                return X_opt, y_opt
        except Exception:
            pass
        return split.X_test, split.y_test




    def _ensure_split(self) -> SplitData:
        if self.split_ is not None:
            return self.split_

        if "class" not in self.data_f.columns:
            raise ValueError('Label column "class" was not found. The label column must be named "class".')

        X = self.data_f.drop(columns=["class"])
        y = self.data_f["class"].values

        # If stratify is enabled (more stable when classes are imbalanced)
        strat = y if (self.do_stratify and len(np.unique(y)) > 1) else None

        train_idx, opt_idx, test_idx = self._split_indices(len(X), y, strat)

        X_train = X.iloc[train_idx].to_numpy(dtype=float, copy=True)
        X_opt = X.iloc[opt_idx].to_numpy(dtype=float, copy=True)
        X_test = X.iloc[test_idx].to_numpy(dtype=float, copy=True)

        # Single scaler: fit on train, apply to train/opt/test
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        if len(opt_idx) > 0:
            X_opt_s = scaler.transform(X_opt)
        else:
            X_opt_s = np.empty((0, X_train.shape[1]), dtype=float)
        X_test_s = scaler.transform(X_test)

        self.split_ = SplitData(
            feature_names=list(X.columns),
            X_train=X_train_s,
            X_opt=X_opt_s,
            X_test=X_test_s,
            y_train=y[train_idx],
            y_opt=y[opt_idx],
            y_test=y[test_idx],
            train_idx=train_idx,
            opt_idx=opt_idx,
            test_idx=test_idx,
        )

        # Fast name->index lookup for repeated subset selections
        self._feat_idx_map = {name: i for i, name in enumerate(self.split_.feature_names)}
        return self.split_

    def _split_indices(self, n: int, y: np.ndarray, stratify: Optional[np.ndarray]):
        """Index-based split.

        - If opt_size is None or <= 0: perform a 2-way split (train/test) and return an
          empty opt split.
        - If opt_size > 0: perform the original 3-way split (train/opt/test).
        """
        idx = np.arange(n)

        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be between 0 and 1.")

        # 1) train_full vs test
        train_full_idx, test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        # 2) If validation/opt is disabled, return train/test only.
        if self.opt_size is None or self.opt_size <= 0.0:
            return (
                np.asarray(train_full_idx),
                np.empty(0, dtype=int),
                np.asarray(test_idx),
            )

        if not (0.0 < self.opt_size < 1.0):
            raise ValueError("opt_size must be between 0 and 1 when validation is enabled.")
        if self.test_size + self.opt_size >= 1.0:
            raise ValueError("test_size + opt_size must be less than 1.")

        # 3) train vs opt (so that opt_size is a fraction of the full dataset)
        opt_rel = self.opt_size / max(1e-12, (1.0 - self.test_size))
        opt_rel = float(np.clip(opt_rel, 1e-6, 0.999999))
        strat2 = None
        if stratify is not None:
            strat2 = y[np.asarray(train_full_idx)]
        train_idx, opt_idx = train_test_split(
            np.asarray(train_full_idx),
            test_size=opt_rel,
            random_state=self.random_state + 1,
            stratify=strat2,
        )
        return np.asarray(train_idx), np.asarray(opt_idx), np.asarray(test_idx)

    def _acc_for_feature_set(self, feat_names_with_class: List[str]) -> float:
        """Compute test accuracy for a given feature subset (cached).

        This function can be called many times across the pipeline (feature selection,
        combination scoring, consistency checks). Caching prevents repeated model
        fits for the same subset, which can dramatically speed up the run.
        """
        split = self._ensure_split()

        # Exclude the label column and canonicalize to maximize cache hits.
        feat_names = [f for f in feat_names_with_class if f != "class"]
        if not feat_names:
            return 0.0
        key = tuple(sorted(feat_names))

        with self._acc_cache_lock:
            cached = self._acc_cache.get(key)
        if cached is not None:
            return cached

        # Build feature index map if needed.
        if self._feat_idx_map is None:
            self._feat_idx_map = {name: i for i, name in enumerate(split.feature_names)}

        try:
            idxs = [self._feat_idx_map[f] for f in key]
        except KeyError as e:
            raise ValueError(f"Unknown feature name: {e.args[0]}") from e

        X_train = split.X_train[:, idxs]
        X_sel, y_sel = self._selection_split(split)
        X_sel = X_sel[:, idxs]

        model = self._clone_model()
        model.fit(X_train, split.y_train)
        y_pred = model.predict(X_sel)
        y_pred = self._pred_to_1d_label(y_pred)
        acc = float(accuracy_score(y_sel, y_pred))


        # Store in cache (do not overwrite if another worker already stored it).
        with self._acc_cache_lock:
            self._acc_cache.setdefault(key, acc)

        return acc


    # ----------------------------
    # Stage 1: Prepare data (normalize + class means)
    # ----------------------------
    def prepare_data(self) -> None:
        self._log("prepare_data: normalize + class means ...")

        class_element = self.data_f["class"].copy()
        feature_elements = self.data_f.drop(columns=["class"]).copy()

        # Like the original code: normalize on the full dataset (for feature selection)
        scaler = MinMaxScaler()
        feature_elements_normalize = pd.DataFrame(
            scaler.fit_transform(feature_elements),
            columns=feature_elements.columns,
            index=feature_elements.index,
        )
        self.df_norm = pd.concat([feature_elements_normalize, class_element], axis=1)

        self.num_of_class = int(np.unique(self.df_norm["class"]).shape[0])
        averages = self.df_norm.groupby("class").mean(numeric_only=True)

        self.avg_list = averages.values.tolist()
        self.cols = self.df_norm.columns.tolist()
        self.num_of_feature = len(feature_elements.columns)

        # Also build the split cache (model scoring is used many times)
        self._ensure_split()

    # ----------------------------
    # Original comb_diff_total
    # ----------------------------
    @staticmethod
    def comb_diff_total(feature_maps: List[List[float]]) -> np.ndarray:
        feature_maps = np.array(feature_maps, dtype=float).T
        diffs = np.abs(feature_maps[:, :, np.newaxis] - feature_maps[:, np.newaxis, :])
        triu_indices = np.triu_indices(diffs.shape[1], k=1)
        result = np.sum(diffs[:, triu_indices[0], triu_indices[1]], axis=1)
        return result

    # ----------------------------
    # Grouping helpers
    # ----------------------------
    def grouping(self, items: List[str], n: int, step: int) -> List[List[str]]:
        k = [items[i:i + n] for i in range(0, len(items) - n + 1, step)]
        for j in k:
            j.append("class")
        return k

    # ----------------------------
    # Choose important features by testing candidate groups
    # ----------------------------
    def important_feature(self, group_features: List[List[str]], top_k_groups: int = 12) -> List[str]:
        self._log("important_feature: picking the candidate group with the best accuracy ...")
        gs = min(top_k_groups, len(group_features))
        params = group_features[:gs]

        # Outer parallelism is here
        accs = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(self._acc_for_feature_set)(feat_list) for feat_list in params
        )

        bo_oz = list(zip(accs, params))
        selected = sorted(bo_oz, key=lambda x: x[0], reverse=True)
        return selected[0][1]

    # ----------------------------
    # Select features
    # ----------------------------
    def select_features(self) -> None:
        if self.df_norm is None or self.avg_list is None or self.cols is None:
            raise RuntimeError("select_features() cannot be called before prepare_data().")

        self._log("select_features: selecting features ...")
        feature_select_alg = self.comb_diff_total(self.avg_list)

        df2 = pd.DataFrame([feature_select_alg], columns=self.cols[:-1])
        self.df2 = df2.sort_values(by=0, axis=1, ascending=False)

        ordered_feature_names = self.df2.columns.tolist()
        group_features = self.grouping(ordered_feature_names, self.group_size, self.step)

        list_features = self.important_feature(group_features)
        # In the original code: sorted(list_features[:-1]) + ["class"]
        self.features = sorted(list_features[:-1]) + ["class"]

    # ----------------------------
    # Combinations helpers
    # ----------------------------
    @staticmethod
    def eliminated_by_group_n(lst: List[List[str]]) -> List[List[List[str]]]:
        groups: Dict[int, List[List[str]]] = {}
        for l in lst:
            key = len(l)
            groups.setdefault(key, []).append(l)
        return list(groups.values())

    @staticmethod
    def generate_combinations(properties: List[str]) -> List[List[str]]:
        # Original logic is preserved, but duplicates are reduced
        n = len(properties)
        combos_set = set()
        for i in range(n):
            rotated = properties[i:] + properties[:i]
            for j in range(1, n):
                combo = tuple(sorted(rotated[:j]))
                if list(combo) != properties:
                    combos_set.add(combo)
        return [list(c) for c in sorted(combos_set)]

    @staticmethod
    def convert_to_values(properties: List[str], values: np.ndarray, combination: List[str]) -> List[float]:
        value_dict = dict(zip(properties, values))
        return [value_dict[item] if item in combination else 0.0 for item in properties]

    # ----------------------------
    # Ridge (fast): use RidgeCV instead of GridSearchCV
    # ----------------------------
    @staticmethod
    def regression_ridge_weights(X: np.ndarray, y: np.ndarray, feature_names: List[str], alphas: List[float]) -> Dict[str, float]:
        """
        Select alpha with RidgeCV and return coefficients.
        feature_names: feature names excluding the target

        Patch:
        - make cv dynamic so small sample groups do not fail
        - return zero weights if there are not enough samples to fit safely
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = len(X)

        if n_samples < 2:
            return dict(zip(feature_names, np.zeros(len(feature_names), dtype=float)))

        cv_splits = min(5, n_samples)

        # RidgeCV is MSE-based; it also works with class labels (as a regression proxy)
        ridge = RidgeCV(alphas=alphas, cv=cv_splits)
        ridge.fit(X, y)
        coef = np.asarray(ridge.coef_)
        if coef.ndim == 2:
            # If (1, n_features): flatten to (n_features,)
            if coef.shape[0] == 1:
                coef = coef.ravel()
            else:
                # If (n_classes, n_features): reduce to one value per feature
                coef = np.mean(np.abs(coef), axis=0)

        coef = np.round(coef, 9)
        return dict(zip(feature_names, coef))


    # ----------------------------
    # Compute weights+acc for a combination (single cached split/scaler)
    # ----------------------------
    def _feature_accuracy_weight_fast(
        self,
        comb_select_feature_with_class: List[str],
        selected_all_features_with_class: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """
        A faster version of the original feature_accuracy_weight logic:
        - split+scaler cache
        - comb ile model fit/predict
        - on the test set, set non-combination columns to 0 and extract weights via RidgeCV
        """
        split = self._ensure_split()

        # features without class
        selected_feats = [f for f in selected_all_features_with_class if f != "class"]
        comb_feats = [f for f in comb_select_feature_with_class if f != "class"]

        if len(comb_feats) == 0:
            return {f: 0.0 for f in selected_feats}, 0.0

        # Map indices efficiently.
        if self._feat_idx_map is None:
            self._feat_idx_map = {name: i for i, name in enumerate(split.feature_names)}
        sel_pos = {f: i for i, f in enumerate(selected_feats)}
        comb_idxs_in_sel = [sel_pos[f] for f in comb_feats]

        # Train/test matrices for combination features only
        comb_global_idxs = [self._feat_idx_map[f] for f in comb_feats]
        X_train = split.X_train[:, comb_global_idxs]
        X_sel, y_sel = self._selection_split(split)
        X_sel = X_sel[:, comb_global_idxs]

        model = self._clone_model()
        model.fit(X_train, split.y_train)
        y_pred = model.predict(X_sel)
        y_pred = self._pred_to_1d_label(y_pred)
        acc = float(accuracy_score(y_sel, y_pred))

        # Ridge input matrix:
        # Populate the accuracy cache for this subset so downstream stages (e.g.,
        # algorithm_consistency) do not refit the same subset again.
        key = tuple(sorted(comb_feats))
        with self._acc_cache_lock:
            self._acc_cache.setdefault(key, acc)

        # Ridge input matrix: all selected feats (comb feats filled, others 0) on test rows
        X_ridge = np.zeros((X_sel.shape[0], len(selected_feats)), dtype=float)
        # fill comb columns from corresponding columns in X_test (comb order)
        for j, sel_pos in enumerate(comb_idxs_in_sel):
            X_ridge[:, sel_pos] = X_sel[:, j]

        weights = self.regression_ridge_weights(
            X=X_ridge,
            y=y_pred.astype(float),
            feature_names=selected_feats,
            alphas=self.alphas,
        )
        return weights, acc

    # ----------------------------
    # Main: feature_combination_acc
    # ----------------------------
    def feature_combination_acc(self) -> Tuple[List[List[Any]], List[List[Any]]]:
        """
        Original feature_combination_acc function:
        - takes values from df2 (feature selection scores)
        - generates combinations
        - computes (weights, acc) for each combination (in parallel)
        """
        if self.df2 is None or self.features is None:
            raise RuntimeError("feature_combination_acc() cannot be called before select_features() finishes.")

        self._log("feature_combination_acc: generating combinations + scoring ...")

        selected_all_features_with_class = self.features[:]  # includes class
        selected_feats = selected_all_features_with_class[:-1]

        df_feature_val = self.df2[selected_feats]
        values = df_feature_val.iloc[0].values

        combinations = self.generate_combinations(selected_feats)
        values_combinations = [self.convert_to_values(selected_feats, values, combo) for combo in combinations]
        result_df = pd.DataFrame(values_combinations, columns=selected_feats)

        # non-zero columns => combination features
        imp_feature_comb = [row[row != 0].index.tolist() for _, row in result_df.iterrows()]
        imp_feature_comb = [sorted(c) for c in imp_feature_comb if len(c) > 0]
        # Patch: de-duplicate combinations to avoid repeated work ----------------
        seen = set()
        imp_feature_comb_uniq = []
        for c in imp_feature_comb:
            t = tuple(c)
            if t not in seen:
                seen.add(t)
                imp_feature_comb_uniq.append(c)
        imp_feature_comb = imp_feature_comb_uniq
        # ------------------------------------------------------------------

        # Add "class"
        imp_feature_comb_w_class = [c + ["class"] for c in imp_feature_comb]

        # Parallel weights+acc
        def one_job(feat_list_with_class: List[str]):
            return self._feature_accuracy_weight_fast(
                comb_select_feature_with_class=feat_list_with_class,
                selected_all_features_with_class=selected_all_features_with_class,
            )

        results = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(one_job)(features) for features in imp_feature_comb_w_class
        )

        imp_feat_comb_acc_weight, on_oz_komb_acc = zip(*results)

        # Output format (original): [[comb_features, weights_dict], ...]
        result1 = [[sublist[:-1], value] for sublist, value in zip(imp_feature_comb_w_class, imp_feat_comb_acc_weight)]
        # Output format (original): [[comb_features, acc], ...]
        result2 = [sublist[:-1] + [value] for sublist, value in zip(imp_feature_comb_w_class, on_oz_komb_acc)]
        return result1, result2

    # ----------------------------
    # Regression compute per group (RidgeCV)
    # ----------------------------
    @staticmethod
    def regression_compute_ridgecv(zero_count: int, group: List[List[float]], features: List[str], acc: float, alphas: List[float]) -> Tuple[int, Dict[str, float]]:
        """
        Original regression_compute_Ridge: use RidgeCV instead of GridSearchCV.

        Patch:
        - make cv dynamic so small groups do not fail when n_samples < 3
        - keep the existing anchor-row logic
        """
        X = [item[:-1] for item in group]
        X.append([1.0] * len(features))
        y = [item[-1] for item in group]
        y.append(float(acc))

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = len(X)

        if n_samples < 2:
            return zero_count, dict(zip(features, np.zeros(len(features), dtype=float)))

        cv_splits = min(3, n_samples)

        ridge = RidgeCV(alphas=alphas, cv=cv_splits)
        ridge.fit(X, y)

        weights = dict(zip(features, np.round(ridge.coef_, 9)))
        return zero_count, weights

    # ----------------------------
    # grouped weights (valimp)
    # ----------------------------
    @staticmethod
    def grouped_weights_valimp(eliminated_element_weight: List[List[Any]], selected_all_feature_weights: Dict[str, float]) -> Dict[int, Dict[str, float]]:
        grouped_data = defaultdict(list)
        for item in eliminated_element_weight:
            length = len(item[0])
            grouped_data[length].append(item)

        result = defaultdict(lambda: defaultdict(float))
        for group_size, items in grouped_data.items():
            for _, weights in items:
                for feature, value in weights.items():
                    result[group_size][feature] += float(value)

        #result_dict = {(group_size - key): dict(value) for key, value in result.items()}
        total_selected = len(selected_all_feature_weights)  # total number of selected features (excluding 'class')
        result_dict = {(total_selected - key): dict(value) for key, value in result.items()}
        result_dict[0] = selected_all_feature_weights
        return result_dict

    # ----------------------------
    # grouped weights (existence)
    # ----------------------------
    def grouped_weights(self, lst: List[List[Any]], features_with_class: List[str], acc: float) -> List[Dict[int, Dict[str, float]]]:
        features = [f for f in features_with_class if f != "class"]

        data = [[1 if element in sub_list[:-1] else 0 for element in features] + [sub_list[-1]] for sub_list in lst]
        groups: Dict[int, List[List[float]]] = {}
        for item in data:
            zero_count = item.count(0)
            groups.setdefault(zero_count, []).append(item)

        if 0 not in groups:
            groups[0] = data

        def process(zero_count: int, group: List[List[float]]):
            return self.regression_compute_ridgecv(zero_count, group, features, acc, self.alphas)

        results = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(process)(zero_count, group) for zero_count, group in groups.items()
        )
        grid_results = {zero_count: result for zero_count, result in results}
        return [grid_results]

    # ----------------------------
    # grouped list helper
    # ----------------------------
    @staticmethod
    def grouped_list(lst: List[List[str]]) -> Dict[int, List[List[str]]]:
        result = defaultdict(list)
        for sub_lst in lst:
            result[len(sub_lst)].append(sub_lst)
        return dict(result)

    # ----------------------------
    # Consistency calculation
    # ----------------------------
    def Acc_(self, features_with_class: List[str]) -> float:
        return self._acc_for_feature_set(features_with_class)

    @staticmethod
    def avg_pos(a: List[Tuple[str, ...]], b: List[Tuple[str, ...]]) -> float:
        def pos(t1, t2):
            set_t1, set_t2 = set(t1), set(t2)
            return len(set_t1 & set_t2)
        return sum(pos(a[i], b[i]) / len(a[i]) for i in range(len(a))) / len(a)

    def algorithm_consistency(self, comb_grp: Dict[int, List[List[str]]], graph_v: Dict[str, float]) -> float:
        l_comb = []
        g_comb = []
        for _, d in comb_grp.items():
            comb_acc = {}
            graph_l = {}
            for x in d:
                x_w = x + ["class"]
                acc = self.Acc_(x_w)
                comb_acc[tuple(x)] = acc

                s = 0.0
                for a in x:
                    s += float(graph_v.get(a, 0.0))
                graph_l[tuple(x)] = s

            g_comb.append(graph_l)
            l_comb.append(comb_acc)

        comb_l = []
        graph_l2 = []
        for g in g_comb:
            g_sorted = dict(sorted(g.items(), key=lambda item: item[1], reverse=True))
            graph_l2.append(list(g_sorted.keys()))
        for cmb in l_comb:
            k_sorted = dict(sorted(cmb.items(), key=lambda item: item[1], reverse=True))
            comb_l.append(list(k_sorted.keys()))

        es = len(comb_grp)
        ssum = 0.0
        for i in range(es):
            ssum += self.avg_pos(comb_l[i], graph_l2[i])
        return ssum / es if es > 0 else 0.0

    def process_i(self, i: int, grouped_weight_data: List[Dict[int, Dict[str, float]]], grouped_weight_data_nonzero: List[Dict[int, Dict[str, float]]], num_group_elem: int):
        # ks: average over the nonzero groups
        ks = {
            key: sum([grouped_weight_data_nonzero[i][k][key] for k in grouped_weight_data_nonzero[i].keys()]) / len(grouped_weight_data_nonzero[i])
            for key in next(iter(grouped_weight_data_nonzero[i].values())).keys()
        }

        data = grouped_weight_data[i]
        min_key = min(data.keys())
        all_comb_ks = data[min_key]
        all_comb_keys = list(all_comb_ks.keys())
        all_comb_vals = list(all_comb_ks.values())

        weight1 = (num_group_elem - 1) / num_group_elem
        weight2 = 1 / num_group_elem

        new_ks = {
            key: ks.get(key, 0.0) * weight1 + all_comb_ks.get(key, 0.0) * weight2
            for key in set(ks) | set(all_comb_ks)
        }
        new_ks_sorted = dict(sorted(new_ks.items(), key=lambda item: item[1], reverse=True))

        comb1 = self.generate_combinations(list(new_ks_sorted.keys()))
        comb1_grp = self.grouped_list(comb1)

        t_method = self.algorithm_consistency(comb1_grp, new_ks_sorted)
        return [t_method, new_ks_sorted, [all_comb_keys, all_comb_vals], i]

    # ----------------------------
    # Main pipeline steps
    # ----------------------------
    def combFeatures(self) -> None:
        if self.features is None:
            raise RuntimeError("combFeatures() cannot be called before select_features().")

        self._log("combFeatures: computing combination weights ...")

        feature_comb_weight, oz_komb = self.feature_combination_acc()

        # all selected features accuracy
        self.acc_select = round(self._acc_for_feature_set(self.features), 3)

        # All-selected weights (for value impact) -> combination is all features (all ones)
        # Here we create "selected_all_feature_weights" via a single model fit + ridge:
        weights_all, _ = self._feature_accuracy_weight_fast(self.features, self.features)

        self.grouped_weight_data_valimp = self.grouped_weights_valimp(feature_comb_weight, weights_all)
        self.grouped_weight_data = self.grouped_weights(oz_komb, self.features, self.acc_select)

        self.grouped_weight_data_nonzero = deepcopy(self.grouped_weight_data)
        if 0 in self.grouped_weight_data_nonzero[0]:
            del self.grouped_weight_data_nonzero[0][0]

        self.grouped_weight_data_nonzero_valimp = self.grouped_weight_data_valimp.copy()
        if 0 in self.grouped_weight_data_nonzero_valimp:
            del self.grouped_weight_data_nonzero_valimp[0]

        self.grouped_weight_data_valimp = [self.grouped_weight_data_valimp]
        self.grouped_weight_data_nonzero_valimp = [self.grouped_weight_data_nonzero_valimp]

        self._log("combFeatures: done. Groups were created.")

    def algorithmConsistency(self) -> None:
        if self.grouped_weight_data is None or self.grouped_weight_data_nonzero is None:
            raise RuntimeError("algorithmConsistency() cannot be called before combFeatures().")

        self._log("algorithmConsistency: searching for the best consistency ...")
        num_group_elem = self.group_size

        results = Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(self.process_i)(
                i,
                self.grouped_weight_data,
                self.grouped_weight_data_nonzero,
                num_group_elem
            ) for i in range(len(self.grouped_weight_data))
        )

        self.data_acc_weight_sorted = sorted(results, key=lambda x: x[0], reverse=True)
        self.alg_consistency = round(float(self.data_acc_weight_sorted[0][0]), 3)
        self.new_weight_format = self.data_acc_weight_sorted[0][1]

    # ----------------------------
    # Public API
    # ----------------------------
    def run(self, plot: Optional[bool] = None) -> "CalcFeatureWeight":
        if plot is not None:
            self.plot = bool(plot)

        self.prepare_data()
        self.select_features()
        self.combFeatures()
        self.algorithmConsistency()

        # Plots are disabled by default
        if self.plot:
            self.showGraphs()

        return self

    def compute_value_impact(self) -> Dict[str, float]:
        """
        Expose the original val_impact_dict computation as a method.
        """
        if self.grouped_weight_data_valimp is None:
            raise RuntimeError("You must run run() first.")

        main_dict_valimp = self.grouped_weight_data_valimp[0]
        auto_keys_valimp = list(next(iter(main_dict_valimp.values())).keys())

        val_impact_dict = {
            key: round(sum(float(sub_dict.get(key, 0.0)) for sub_dict in main_dict_valimp.values()) / len(main_dict_valimp), 3)
            for key in auto_keys_valimp
        }
        return val_impact_dict

    def get_splits(self) -> SplitData:
        """Return the one-time split+scaled arrays (useful for the tuner)."""
        return self._ensure_split()

    # ----------------------------
    # Optional plotting (can be disabled)
    # ----------------------------
    def showGraphs(self) -> None:
        """
        Original showGraphs. Disabled by default for performance.
        Enable by passing plot=True if you want visuals.
        """
        import matplotlib.pyplot as plt

        if self.data_acc_weight_sorted is None or self.grouped_weight_data_nonzero is None:
            raise RuntimeError("You must complete run() first.")

        # weights from all combinations
        tum_komb_keys, tum_komb_vals = self.data_acc_weight_sorted[0][2][0], self.data_acc_weight_sorted[0][2][1]
        tum_komb_sorted_vals, tum_komb_sorted_keys = zip(*sorted(zip(tum_komb_vals, tum_komb_keys), reverse=True))

        new_weight = self.data_acc_weight_sorted[0][1]
        sorted_keys = sorted(new_weight.keys(), key=lambda x: new_weight[x], reverse=True)
        sorted_values = [new_weight[key] for key in sorted_keys]
        best_index = self.data_acc_weight_sorted[0][3]

        plt.figure(figsize=(5, 3))
        plt.barh(tum_komb_sorted_keys, tum_komb_sorted_vals)
        plt.xlabel("Values")
        plt.ylabel("Keys")
        plt.title("weights Obtained from All Selected Features")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.show()

        # Nonzero grouped weights (may create many subplots; use a simple bar plot)
        data = self.grouped_weight_data_nonzero[best_index]
        for k, v in data.items():
            plt.figure(figsize=(max(4, len(v) * 0.6), 3))
            plt.bar(list(v.keys()), list(v.values()))
            plt.xticks(rotation=45, ha="right")
            plt.title(f"{k} Feature Eliminated")
            plt.tight_layout()
            plt.show()

        plt.figure(figsize=(max(6, len(sorted_keys) * 0.7), 3))
        plt.barh(sorted_keys, sorted_values)
        plt.title("Final Relational Result Graph of Features")
        plt.tight_layout()
        plt.show()
