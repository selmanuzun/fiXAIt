# xai_faithfulness_tuner.py
from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils import check_random_state

from copy import deepcopy
from sklearn.model_selection import train_test_split

from joblib import Parallel, delayed


# -----------------------------
# 1) Faithfulness evaluator
# -----------------------------
def evaluate_faithfulness(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance_scores: dict,
    *,
    # --- legacy arguments ---
    top_k: int = 7,
    metric: str = "accuracy",
    random_state: int = 42,
    runs_per_feature: int = 30,
    epsilon: float = 1e-8,
    normalize: bool = False,
    verbose: bool = False,
    grid_resolution: int = 30,
    pdp_percentiles: tuple = (0.05, 0.95),
    # --- new arguments ---
    drop_mode: str = "metric",       # "metric" | "prob"
    class_id: int | None = None,     # for drop_mode="prob" (probability column index)
    abs_drop: bool = True,           # take |.| of the probability drop
    conditional_permutation: bool = False,  # more robust to correlation
    # --- practical: optional PD-variance computation ---
    compute_pd_var: bool = False,
    # --- parallel ---
    n_jobs: int = 1,
    prefer: str = "threads",
):
    """
Compute feature faithfulness using either:
  - drop in a classic performance metric (accuracy/f1/log_loss), or
  - drop in the predicted class probability.

Returns
-------
faithfulness : float
drop_impacts : dict[str, float]
(optional) pd_var : dict[str, float]
    Only returned when compute_pd_var=True.
"""

    # ------- 1) Importance scores + filtering ---------------------------- #
    imp = pd.Series(importance_scores, dtype=float)
    if normalize:
        m = imp.abs().max()
        if m > 0:
            imp /= m

    active = imp.abs() > epsilon
    feats = (
        imp.loc[active].abs().nlargest(top_k).index.tolist()
        if active.any() and 0 < top_k < active.sum()
        else imp.loc[active].index.tolist() if active.any()
        else imp.index.tolist()
    )
    if verbose:
        print(f"Selected {len(feats)} features:", ", ".join(feats))

    # ------- 2) Baseline score / probability ---------------------------- #
    metric = metric.lower()
    if drop_mode == "metric":
        if metric == "accuracy":
            scorer = accuracy_score
        elif metric == "f1":
            scorer = lambda y, yp: f1_score(y, yp, average="weighted")
        elif metric == "log_loss":
            if not hasattr(model, "predict_proba"):
                raise ValueError("metric='log_loss' requires model.predict_proba.")
            base_score = -log_loss(y_test, model.predict_proba(X_test))  # higher = better
            scorer = None
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if metric != "log_loss":
            y_pred_base = model.predict(X_test)
            base_score = scorer(y_test, y_pred_base)

    elif drop_mode == "prob":
        if not hasattr(model, "predict_proba"):
            raise ValueError("drop_mode='prob' requires model.predict_proba.")
        if class_id is None:
            class_id_vec = y_test.to_numpy()
            proba = model.predict_proba(X_test)
            base_scores = proba[np.arange(len(X_test)), class_id_vec]
        else:
            base_scores = model.predict_proba(X_test)[:, class_id]
    else:
        raise ValueError("drop_mode must be 'metric' or 'prob'.")

    if verbose and drop_mode == "metric":
        print(f"Base {metric}: {base_score:.4f}")

    # ------- 3) Impact via permutation ---------------------------------- #
    rng_master = check_random_state(random_state)

    # For conditional_permutation: precompute bin indices once per feature
    groups_map: Dict[str, List[np.ndarray]] = {}
    if conditional_permutation:
        for f in feats:
            q = pd.qcut(X_test[f], q=10, duplicates="drop")
            groups: List[np.ndarray] = []
            for cat in q.unique():
                idx = np.where(q == cat)[0]
                if len(idx) > 1:
                    groups.append(idx)
            groups_map[f] = groups

    # Pre-generate per-run seeds for deterministic parallel execution
    seeds_map: Dict[str, np.ndarray] = {
        f: rng_master.randint(0, np.iinfo(np.int32).max, size=runs_per_feature)
        for f in feats
    }

    def _impact_one_run(f: str, seed: int) -> float:
        rng = np.random.RandomState(int(seed))
        Xp = X_test.copy()
        col = Xp[f].to_numpy().copy()

        if conditional_permutation:
            for idx in groups_map.get(f, []):
                rng.shuffle(col[idx])
        else:
            rng.shuffle(col)

        Xp[f] = col

        if drop_mode == "metric":
            if metric == "log_loss":
                score = -log_loss(y_test, model.predict_proba(Xp))
            else:
                score = scorer(y_test, model.predict(Xp))
            return float(base_score - score)

        # "prob"
        proba = model.predict_proba(Xp)
        if class_id is None:
            p = proba[np.arange(len(Xp)), class_id_vec]
        else:
            p = proba[:, class_id]

        diff = base_scores - p
        if abs_drop:
            diff = np.abs(diff)
        else:
            diff = np.maximum(diff, 0)
        return float(np.mean(diff))

    def _impact_for_feature(f: str) -> Tuple[str, float]:
        impacts = [_impact_one_run(f, s) for s in seeds_map[f]]
        return f, float(np.mean(impacts))

    if (n_jobs is None) or (int(n_jobs) == 1) or (len(feats) <= 1):
        drop_impacts = dict(_impact_for_feature(f) for f in feats)
    else:
        results = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_impact_for_feature)(f) for f in feats
        )
        drop_impacts = dict(results)

    if verbose:
        for f in feats:
            print(f"Impact {f}: {drop_impacts[f]:.4f}")

    # ------- 4) (Optional) PD variance ---------------------------------- #
    pd_var: Dict[str, float] = {}
    if compute_pd_var:
        # Note: PD variance is NOT included in faithfulness. It is only diagnostic.
        target_col = None
        # Caution: predict_proba(...).mean(axis=1) can become uninformative in multiclass.
        # Here we track class_id if provided; otherwise we track the true-class column.
        if hasattr(model, "predict_proba"):
            if (drop_mode == "prob") and (class_id is not None):
                target_col = class_id
            else:
                # true-class column (assumes label == column index) – consistent with the previous setup
                target_col = None

        low_p, high_p = pdp_percentiles

        def _pd_var_for_feature(f: str) -> Tuple[str, float]:
            low, high = X_train[f].quantile(low_p), X_train[f].quantile(high_p)
            grid = np.linspace(low, high, grid_resolution)

            pd_mat = np.zeros((len(X_test), len(grid)))
            for i, v in enumerate(grid):
                Xt = X_test.copy()
                Xt[f] = v
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xt)
                    if target_col is not None:
                        pr = proba[:, target_col]
                    else:
                        cls = y_test.to_numpy()
                        pr = proba[np.arange(len(Xt)), cls]
                else:
                    pr = model.predict(Xt)
                pd_mat[:, i] = pr

            return f, float(np.mean(np.var(pd_mat, axis=1)))

        if (n_jobs is None) or (int(n_jobs) == 1) or (len(feats) <= 1):
            pd_var = dict(_pd_var_for_feature(f) for f in feats)
        else:
            results = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(_pd_var_for_feature)(f) for f in feats
            )
            pd_var = dict(results)

        if verbose:
            for f in feats:
                print(f"PD-var {f}: {pd_var[f]:.4f}")

        # ------- 5) Spearman correlation ------------------------------------ #
    imp_vals = [imp.abs()[f] for f in feats]
    impact_vals = [drop_impacts[f] for f in feats]

    if len(set(imp_vals)) <= 1 or len(set(impact_vals)) <= 1:
        faithfulness = 0.0
        if verbose:
            print("No variance → faithfulness = 0")
    else:
        faithfulness, _ = spearmanr(imp_vals, impact_vals)
        if verbose:
            print(f"Faithfulness (Spearman): {faithfulness:.4f}")

    if compute_pd_var:
        return faithfulness, drop_impacts, pd_var
    return faithfulness, drop_impacts


# -----------------------------
# 2) Optimizer (rank-grad)
# -----------------------------
def _safe_spearman(a, b) -> float:
    r = spearmanr(a, b)[0]
    if (r is None) or np.isnan(r):
        return 0.0
    return float(r)


def optimize_weights_with_rank_grad(
    features: List[str],
    drop_impacts: Dict[str, float],
    *,
    positive_only: bool = True,
    n_steps: int = 400,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    seed: int = 42,
    log_every: Optional[int] = 40,
    w_init: Optional[Dict[str, float]] = None,
    reg_lambda: float = 0.10,
) -> Dict[str, float]:
    """
    Moves the ranking of w closer to the ranking of drop_impacts (pairwise rank loss).
    - If w_init is given, start from it instead of random.
    - If reg_lambda > 0, keep w close to w_init.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    def _norm01_from_map(feat_list, m, eps=1e-12):
        v = np.array([float(m.get(f, 0.0)) for f in feat_list], dtype=float)
        v = np.maximum(0.0, v)
        mn, mx = float(v.min()), float(v.max())
        if mx - mn < eps:
            return np.ones_like(v, dtype=float)
        return (v - mn) / (mx - mn + eps)

    def _norm_signed_from_map(feat_list, m, eps=1e-12):
        v = np.array([float(m.get(f, 0.0)) for f in feat_list], dtype=float)
        den = float(np.max(np.abs(v)) + eps)
        return v / den

    d_map = {f: float(drop_impacts[f]) for f in features}
    d_tgt = _norm01_from_map(features, d_map) if positive_only else _norm_signed_from_map(features, d_map)
    d_t = torch.tensor(d_tgt, dtype=torch.float32)

    clamp_lo, clamp_hi = (0.0, 1.0) if positive_only else (-1.0, 1.0)

    if w_init is None:
        if positive_only:
            w0 = torch.rand(len(features))
        else:
            w0 = torch.rand(len(features)) * 2.0 - 1.0
        w_ref_t = None
    else:
        if positive_only:
            w_init_vec = _norm01_from_map(features, {k: abs(v) for k, v in w_init.items()})
        else:
            w_init_vec = _norm_signed_from_map(features, w_init)
        w0 = torch.tensor(w_init_vec, dtype=torch.float32)
        w_ref_t = torch.tensor(w_init_vec, dtype=torch.float32)

    w = torch.nn.Parameter(w0.clone())
    opt = torch.optim.Adam([w], lr=lr)

    def sample_pairs(n, k=None):
        if (k is None) or (k >= (n * (n - 1)) // 2):
            idx_i, idx_j = np.triu_indices(n, k=1)
        else:
            idx_i = rng.randint(0, n, size=k)
            idx_j = rng.randint(0, n, size=k)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
        return torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long)

    for step in range(1, n_steps + 1):
        opt.zero_grad()
        with torch.no_grad():
            w.clamp_(clamp_lo, clamp_hi)

        i_idx, j_idx = sample_pairs(len(features), pair_batch)
        d_diff = d_t[i_idx] - d_t[j_idx]
        sgn = torch.sign(d_diff)
        mask = sgn != 0
        if torch.any(mask):
            i_idx, j_idx, sgn = i_idx[mask], j_idx[mask], sgn[mask]
        else:
            break

        w_diff = w[i_idx] - w[j_idx]
        logits = tau * sgn * w_diff
        rank_loss = torch.nn.functional.softplus(-logits).mean()

        if (reg_lambda > 0) and (w_ref_t is not None):
            reg = torch.mean((w - w_ref_t) ** 2)
            loss = rank_loss + reg_lambda * reg
        else:
            loss = rank_loss

        loss.backward()
        opt.step()

        if (log_every is not None) and (step % log_every == 0):
            with torch.no_grad():
                w_eval = w.clamp(clamp_lo, clamp_hi).detach().cpu().numpy()
                faith_sur = _safe_spearman(w_eval, d_tgt)
                print(f"[RankOpt] step {step:04d} | SurrogateFaith={faith_sur:.4f} | Loss={loss.item():.4f}")

    with torch.no_grad():
        w_out = w.clamp(clamp_lo, clamp_hi).detach().cpu().numpy()
        if positive_only:
            v = np.maximum(0.0, w_out)
            mn, mx = float(v.min()), float(v.max())
            w_out = np.ones_like(v) if (mx - mn) < 1e-12 else (v - mn) / (mx - mn + 1e-12)
        else:
            m = float(np.max(np.abs(w_out)) + 1e-12)
            w_out = w_out / m

    return dict(zip(features, w_out.tolist()))


def optimize_importances_with_faithfulness(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    importance_scores: Dict[str, float],
    *,
    positive_only: bool = True,
    metric: str = "accuracy",
    drop_mode: str = "metric",
    class_id: int | None = None,
    runs_per_feature: int = 30,
    n_steps: int = 800,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    random_state: int = 42,
    verbose: bool = True,
    reg_lambda: float = 0.10,
    X_opt=None, y_opt=None,
    X_eval=None, y_eval=None,
    n_jobs: int = 1,
    prefer: str = "threads",
) -> Tuple[Dict[str, float], float, float, Dict[str, float], Dict[str, float]]:
    """
    Optimize importance_scores to increase Spearman faithfulness based on drop_impacts computed by evaluate_faithfulness (rank-based).
    """
    if X_opt is None:  X_opt = X_test
    if y_opt is None:  y_opt = y_test
    if X_eval is None: X_eval = X_test
    if y_eval is None: y_eval = y_test

    if verbose and (X_opt is X_eval) and (y_opt is y_eval):
        print("[WARN] X_opt and X_eval are the same: optimizing and reporting on the same set can inflate faithfulness. "
              "If possible, optimize on a validation set and report on a test set.")

    feats_all = list(importance_scores.keys())

    # 1) Compute drop_impacts on the optimization set
    faith_init, drop_impacts_init = evaluate_faithfulness(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_opt,
        y_test=y_opt,
        importance_scores=importance_scores,
        top_k=len(feats_all),
        metric=metric,
        random_state=random_state,
        runs_per_feature=runs_per_feature,
        drop_mode=drop_mode,
        class_id=class_id,
        abs_drop=True,
        conditional_permutation=False,
        verbose=verbose,
        compute_pd_var=False,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    if verbose:
        print(f"\n[FAITH] Initial faithfulness (on opt-set): {faith_init:.4f}")

    features = list(drop_impacts_init.keys())
    if verbose:
        print(f"Optimization will use {len(features)} features (after epsilon/top_k logic).")

    # 2) Rank-grad optimize
    optimized_partial = optimize_weights_with_rank_grad(
        features=features,
        drop_impacts=drop_impacts_init,
        positive_only=positive_only,
        n_steps=n_steps,
        lr=lr,
        tau=tau,
        pair_batch=pair_batch,
        seed=random_state,
        log_every=40 if verbose else None,
        w_init=importance_scores,
        reg_lambda=reg_lambda,
    )

    optimized_full = dict(importance_scores)
    optimized_full.update(optimized_partial)

    # 3) Report on the evaluation set
    faith_final, drop_impacts_final = evaluate_faithfulness(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_eval,
        y_test=y_eval,
        importance_scores=optimized_full,
        top_k=len(feats_all),
        metric=metric,
        random_state=random_state,
        runs_per_feature=runs_per_feature,
        drop_mode=drop_mode,
        class_id=class_id,
        abs_drop=True,
        conditional_permutation=False,
        verbose=verbose,
        compute_pd_var=False,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    if verbose:
        print(f"\n[FAITH] Final   faithfulness (on eval-set): {faith_final:.4f}")

    return optimized_full, faith_init, faith_final, drop_impacts_init, drop_impacts_final


# -----------------------------
# 3) Convenience API for p (Existence Impact)
# -----------------------------
def tune_existence_impact(
    *,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    fit_model: bool = False,
    model_fit_params: Optional[dict] = None,
    p_exist: Dict[str, float],          # senin p (signed olabilir)
    X_opt: Optional[pd.DataFrame] = None,
    y_opt: Optional[pd.Series] = None,
    metric: str = "accuracy",
    drop_mode: str = "metric",
    runs_per_feature: int = 80,
    n_steps: int = 500,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    random_state: int = 42,
    reg_lambda: float = 0.10,
    verbose: bool = True,
    n_jobs: int = 1,
    prefer: str = "threads",
) -> Tuple[Dict[str, float], float, float, Dict[str, float], Dict[str, float]]:
    """
    Calibrate p (Existence Impact) scores for faithfulness:
      - optimization is performed on |p| (positive-only)
      - the original sign is restored afterwards

    Returns
    -------
    p_opt_signed, faith_init, faith_final, drop_init, drop_final
    """
    # Optional: fit the model inside this function (clone first to avoid mutating the original object)
    if fit_model:
        mfit = deepcopy(model)
        params = model_fit_params or {}
        if not hasattr(mfit, "fit"):
            raise TypeError("The model object does not have a fit() method.")
        mfit.fit(X_train, y_train, **params)
        model = mfit

    feats = [f for f in p_exist.keys() if f != "class" and f in X_train.columns and f in X_eval.columns]
    if len(feats) == 0:
        raise ValueError("Features in p_exist do not match the columns in X_train/X_eval.")

    sign = {f: (1.0 if float(p_exist[f]) >= 0 else -1.0) for f in feats}
    p_mag = {f: float(abs(p_exist[f])) for f in feats}

    opt_mag, faith_init, faith_final, drop_init, drop_final = optimize_importances_with_faithfulness(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_eval,   y_test=y_eval,
        importance_scores=p_mag,
        positive_only=True,
        metric=metric,
        drop_mode=drop_mode,
        class_id=None,
        runs_per_feature=runs_per_feature,
        n_steps=n_steps,
        lr=lr,
        tau=tau,
        pair_batch=pair_batch,
        random_state=random_state,
        verbose=verbose,
        reg_lambda=reg_lambda,
        X_opt=X_opt, y_opt=y_opt,
        X_eval=X_eval, y_eval=y_eval,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    p_opt_signed = dict(p_exist)
    for f in feats:
        p_opt_signed[f] = sign[f] * float(opt_mag.get(f, 0.0))

    return p_opt_signed, faith_init, faith_final, drop_init, drop_final


def tune_existence_impact_from_path(
    *,
    path: str,
    label_col: str,
    base_model,                 # e.g., pick_mdl (unfitted estimator)
    p_exist: Dict[str, float],  # senin p (existence impact)
    test_size: float = 0.20,
    val_size_in_trainfull: float = 0.25,   # validation fraction inside train_full
    random_state: int = 42,
    stratify: bool = True,
    # tune parametreleri
    metric: str = "accuracy",
    drop_mode: str = "metric",
    runs_per_feature: int = 100,
    n_steps: int = 500,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    reg_lambda: float = 0.10,
    verbose: bool = True,
    n_jobs: int = 1,
    prefer: str = "threads",
):
    """
    Read data from CSV -> split into train/val/test -> fit the model on train
    -> run tune_existence_impact.

    Returns
    -------
    p_opt, report, splits, fitted_model
    """
    # 1) Load data
    data = pd.read_csv(path)
    if label_col not in data.columns:
        raise ValueError(f"label_col='{label_col}' not found in columns: {list(data.columns)}")

    y = data[label_col]
    X = data.drop(columns=[label_col])

    # 2) Split
    strat = y if stratify else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    strat2 = y_train_full if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size_in_trainfull,
        random_state=random_state,
        stratify=strat2
    )

    # 3) Fit the model on TRAIN only
    bb_model = deepcopy(base_model)
    bb_model.fit(X_train, y_train)


    p_opt, f0, f1, drop0, drop1 = tune_existence_impact(
        model=bb_model,
        X_train=X_train, y_train=y_train,
        X_opt=X_val, y_opt=y_val,
        X_eval=X_test, y_eval=y_test,
        p_exist=p_exist,
        metric=metric,
        drop_mode=drop_mode,
        runs_per_feature=runs_per_feature,
        n_steps=n_steps,
        lr=lr,
        tau=tau,
        pair_batch=pair_batch,
        random_state=random_state,
        reg_lambda=reg_lambda,
        verbose=verbose,
        n_jobs=n_jobs,
        prefer=prefer,
    )

    report = {
        "faith_init": f0,
        "faith_final": f1,
        "drop_init": drop0,
        "drop_final": drop1,
    }
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,     "y_val": y_val,
        "X_test": X_test,   "y_test": y_test,
    }

    return p_opt, report, splits, bb_model




__all__ = [
    "evaluate_faithfulness",
    "optimize_weights_with_rank_grad",
    "optimize_importances_with_faithfulness",
    "tune_existence_impact",
    "tune_existence_impact_from_path",
]


# -----------------------------
# 4) Wrappers for prepared data (no path read / no re-splitting)
# -----------------------------
def _as_dataframe(X, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Accept a pd.DataFrame or np.ndarray; return a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("np.ndarray was provided; feature_names is required.")
        return pd.DataFrame(X, columns=list(feature_names))
    raise TypeError(f"Unsupported X type: {type(X)}")


def _as_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, (np.ndarray, list, tuple)):
        return pd.Series(y)
    raise TypeError(f"Unsupported y type: {type(y)}")


def tune_existence_impact_from_splitdata(
    *,
    model,
    split,
    p_exist: Dict[str, float],
    feature_subset: Optional[List[str]] = None,
    # Opt set: you can override externally; otherwise split.X_opt/y_opt is used
    X_opt=None,
    y_opt=None,
    # --- tune_existence_impact parameters (explicitly exposed) ---
    fit_model: bool = False,
    model_fit_params: Optional[dict] = None,
    metric: str = "accuracy",
    drop_mode: str = "metric",
    runs_per_feature: int = 80,
    n_steps: int = 500,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    random_state: int = 42,
    reg_lambda: float = 0.10,
    verbose: bool = True,
    n_jobs: int = 1,
    prefer: str = "threads",
    # Accept extra future parameters as well
    **kwargs,
):
    """
    Call the tuner from a prepared split object (e.g., CalcFeatureWeight.get_splits()).

    Parametreler
    -----------
    split : any object with the required attributes
        Expected fields:
          - feature_names
          - X_train, y_train
          - X_test/y_test veya X_eval/y_eval
          - (opsiyonel) X_opt/y_opt

    feature_subset : list[str] | None
        Must match the columns used to fit the model (e.g., only the selected 7 features).
        If the model was fitted on the full feature set, leave this as None.

    runs_per_feature : int
        How many permutation trials to run per feature when computing faithfulness impacts.

    Note
    ----
    Earlier versions passed runs_per_feature implicitly via **kwargs; now it is explicitly in the signature and forwarded to tune_existence_impact.
    """
    feature_names = getattr(split, "feature_names")
    X_train = _as_dataframe(getattr(split, "X_train"), feature_names)
    y_train = _as_series(getattr(split, "y_train"))

    # Eval/Test set (priority: X_eval/y_eval -> X_test/y_test)
    if hasattr(split, "X_eval"):
        X_eval = _as_dataframe(getattr(split, "X_eval"), feature_names)
        y_eval = _as_series(getattr(split, "y_eval"))
    else:
        X_eval = _as_dataframe(getattr(split, "X_test"), feature_names)
        y_eval = _as_series(getattr(split, "y_test"))

    # Opt/Validation set (priority: user override -> split.X_opt/y_opt)
    if X_opt is not None:
        X_opt_df = _as_dataframe(X_opt, feature_names)
    elif hasattr(split, "X_opt"):
        X_opt_df = _as_dataframe(getattr(split, "X_opt"), feature_names)
    else:
        X_opt_df = None

    y_opt_sr = _as_series(y_opt) if y_opt is not None else (_as_series(getattr(split, "y_opt")) if hasattr(split, "y_opt") else None)

    # Optional: subset X_* to specific columns (must match how the model was fitted)
    if feature_subset is not None:
        feature_subset = [c for c in feature_subset if c in X_train.columns]
        X_train = X_train[feature_subset]
        X_eval = X_eval[feature_subset]
        if X_opt_df is not None:
            X_opt_df = X_opt_df[feature_subset]

    return tune_existence_impact(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        fit_model=fit_model,
        model_fit_params=model_fit_params,
        p_exist=p_exist,
        X_opt=X_opt_df,
        y_opt=y_opt_sr,
        metric=metric,
        drop_mode=drop_mode,
        runs_per_feature=runs_per_feature,
        n_steps=n_steps,
        lr=lr,
        tau=tau,
        pair_batch=pair_batch,
        random_state=random_state,
        reg_lambda=reg_lambda,
        verbose=verbose,
        n_jobs=n_jobs,
        prefer=prefer,
        **kwargs,
    )


def fiXAItImportanceTuner(
    *,
    cfw,
    model,
    p_exist: Optional[Dict[str, float]] = None,
    feature_subset: Optional[List[str]] = None,
    X_opt=None,
    y_opt=None,
    # --- tune_existence_impact parameters (explicit) ---
    fit_model: bool = False,
    model_fit_params: Optional[dict] = None,
    metric: str = "accuracy",
    drop_mode: str = "metric",
    runs_per_feature: int = 80,
    n_steps: int = 500,
    lr: float = 0.05,
    tau: float = 25.0,
    pair_batch: int = 4096,
    random_state: int = 42,
    reg_lambda: float = 0.10,
    verbose: bool = True,
    n_jobs: int = 1,
    prefer: str = "threads",
    **kwargs,
):
    """
    Wire a CalcFeatureWeight instance (cfw) directly into the tuner.

    What is cfw?
    -----------
    An instance of your CalcFeatureWeight class (including the optimized version).
    Beklenenler:
      - cfw.get_splits()  -> SplitData (X_train/X_opt/X_test + feature_names)
      - cfw.new_weight_format -> p (existence impact) (used if p_exist is not provided)

    Why is feature_subset needed?
    ---------------------------
    If you fitted the base model using only a subset of features, the tuner must predict using the same column set. Otherwise:
      - sklearn can raise a "feature mismatch" error (DataFrame columns differ)
      - or you can get incorrect results due to column order/misalignment.

    Why did runs_per_feature look missing?
    -------------------------------------
    In earlier versions the wrapper forwarded it via **kwargs; here it is explicitly in the signature.
    """
    if p_exist is None:
        p_exist = getattr(cfw, "new_weight_format", None)
        if p_exist is None:
            raise ValueError("p_exist was not provided and cfw.new_weight_format is empty.")

    split = cfw.get_splits() if hasattr(cfw, "get_splits") else getattr(cfw, "split_")
    return tune_existence_impact_from_splitdata(
        model=model,
        split=split,
        p_exist=p_exist,
        feature_subset=feature_subset,
        X_opt=X_opt,
        y_opt=y_opt,
        fit_model=fit_model,
        model_fit_params=model_fit_params,
        metric=metric,
        drop_mode=drop_mode,
        runs_per_feature=runs_per_feature,
        n_steps=n_steps,
        lr=lr,
        tau=tau,
        pair_batch=pair_batch,
        random_state=random_state,
        reg_lambda=reg_lambda,
        verbose=verbose,
        n_jobs=n_jobs,
        prefer=prefer,
        **kwargs,
    )

