"""Tabular models with the protocol shared by all experiments: random hyper-parameter search selected on validation AUC
(GBDT), then retraining with several seeds.

Models: lgbm / lgbm_cwtr (LightGBM), xgb, cat (CatBoost), rf, lr, mlp.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .train import train_nn

GBDT = ("lgbm", "lgbm_cwtr", "xgb", "cat")


def sample(kind, rng):
    """One random hyper-parameter configuration (the draw order is part of the protocol: same rng seed -> same configurations)."""
    log_uniform = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    lr = float(rng.choice([0.02, 0.05, 0.1]))
    if kind.startswith("lgbm"):
        return {"learning_rate": lr, "num_leaves": int(rng.integers(15, 255)), "min_child_samples": int(rng.integers(10, 300)),
                "subsample": float(rng.uniform(0.6, 1)), "colsample_bytree": float(rng.uniform(0.4, 1)), "reg_lambda": log_uniform(0.1, 30)}
    if kind == "xgb":
        return {"learning_rate": lr, "max_depth": int(rng.integers(3, 11)), "min_child_weight": log_uniform(1, 50),
                "subsample": float(rng.uniform(0.6, 1)), "colsample_bytree": float(rng.uniform(0.4, 1)), "reg_lambda": log_uniform(0.1, 30)}
    if kind == "cat":
        return {"learning_rate": lr, "depth": int(rng.integers(4, 11)), "l2_leaf_reg": log_uniform(1, 30), "random_strength": log_uniform(0.1, 10)}
    raise ValueError(kind)


def fit_gbdt(kind, params, seed, Xtr, ytr, Xva, yva, a):
    """Fit one gradient-boosted model with early stopping (100 rounds) on the validation set. Returns (model, best iteration)."""
    n_estimators = getattr(a, "n_estimators", 5000)
    if kind.startswith("lgbm"):
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=n_estimators, subsample_freq=1, random_state=seed, n_jobs=a.n_jobs, verbose=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration_
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=n_estimators, early_stopping_rounds=100, eval_metric="auc", tree_method="hist",
                              device=a.xgb_device, random_state=seed, n_jobs=a.n_jobs, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m, m.best_iteration
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=n_estimators, od_type="Iter", od_wait=100, eval_metric="AUC", random_seed=seed, verbose=0,
                           thread_count=a.n_jobs, task_type=a.cat_task, allow_writing_files=False, **params)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m, m.get_best_iteration()


def run_tab(kind, X, y, idx, a, n_seeds=None):
    """Yield (seed, validation prob., test prob., info dict, fitted model or None) for one model kind.

    GBDT: a.n_trials random configurations are scored on validation AUC (seed 0) and the best is retrained for n_seeds seeds.
    rf: a.n_seeds_rf seeds.  lr: C chosen on validation (single fit).  mlp: n_seeds seeds."""
    n_seeds = a.n_seeds if n_seeds is None else n_seeds
    tr, va, te = idx["train"], idx["val"], idx["test"]
    if kind in GBDT:
        rng = np.random.default_rng(0)
        best = (-1, None)
        for _ in range(a.n_trials):
            p = sample(kind, rng)
            m, _ = fit_gbdt(kind, p, 0, X[tr], y[tr], X[va], y[va], a)
            score = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
            if score > best[0]:
                best = (score, p)
        print(f"    {kind}: best validation AUC {best[0]:.4f} {best[1]}", flush=True)
        for seed in range(n_seeds):
            m, it = fit_gbdt(kind, best[1], seed, X[tr], y[tr], X[va], y[va], a)
            yield seed, m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1], {"iters": it}, m
    elif kind == "rf":
        for seed in range(a.n_seeds_rf):
            m = RandomForestClassifier(500, min_samples_leaf=5, max_features="sqrt", n_jobs=a.n_jobs, random_state=seed).fit(X[tr], y[tr])
            yield seed, m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1], {}, m
    elif kind == "lr":
        best = (-1, None, None)
        for C in (0.01, 0.1, 1.0, 10.0):
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(X[tr], y[tr])
            score = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
            if score > best[0]:
                best = (score, m, C)
        yield 0, best[1].predict_proba(X[va])[:, 1], best[1].predict_proba(X[te])[:, 1], {"C": best[2]}, best[1]
    elif kind == "mlp":
        prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X[tr])
        Z = np.clip(prep.transform(X), -10, 10).astype(np.float32)
        for seed in range(n_seeds):
            pv, pt, ep = train_nn("mlp", {"X": Z}, y, idx, seed, a)
            yield seed, pv, pt, {"epochs": ep}, None
    else:
        raise ValueError(kind)
