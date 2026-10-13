#!/usr/bin/env python3
"""E8: leakage audit for all six models (LightGBM, XGBoost, CatBoost, random forest, logistic regression, MLP).

On cut50/60/70 every learner / course statistic of the base features is computed only from the training pool (train + validation).
Three inputs per model: base_ind (144) | cwtr_ind (+50 CWTR) | cwtrN_ind (+57 course-normalised). Protocol as in E4/E5 (own hyper-parameter
search per model x input; GBDT 12 configurations, 5 seeds; RF 3 seeds; LR chooses C).

  python -m experiments.e8_leakage_all_models run       --data D --out runs_e8 --device cuda --xgb_device cuda --cat_task GPU
  python -m experiments.e8_leakage_all_models summarize --out runs_e8 --e4 runs_e4 --e5 runs_e5
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cwtr.build_helpers import train_only_base
from cwtr.cli import add_data_args, add_train_args
from cwtr.norm import tab_norm
from cwtr.splits import make_split
from cwtr.stats import boot, mean_pred, wins
from cwtr.tab import run_tab
from cwtr.temporal import load_all

warnings.filterwarnings("ignore")
MODELS = ("lgbm", "xgb", "cat", "rf", "lr", "mlp")
INPUTS = ("base_ind", "cwtr_ind", "cwtrN_ind")
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random forest", "lr": "Logistic regression", "mlp": "MLP"}
TAGS = ("cut50", "cut60", "cut70")


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e8.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["input"], d["model"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y = D["y"].astype(int)
    code = pd.factorize(D["course"])[0]
    T = D["T"]
    Tn = tab_norm(T, code)
    for sp in a.splits.split(","):
        idx, info = make_split(D, a.data, float(sp))
        tag = f"cut{int(round(float(sp) * 100))}"
        tr, va, te = idx["train"], idx["val"], idx["test"]
        print(f"[{tag}] {info} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{tag}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{tag}_course_test.npy"), code[te])
        ref = np.zeros(len(y), bool)
        ref[np.concatenate([tr, va])] = True
        flat = train_only_base(a.data, a.cluster, ref)
        flat.index = T.index
        inputs = {"base_ind": flat, "cwtr_ind": pd.concat([flat, T], axis=1), "cwtrN_ind": pd.concat([flat, T, Tn], axis=1)}
        for inp in a.inputs.split(","):
            X = inputs[inp].values.astype(np.float32)
            for model in a.models.split(","):
                n_exp = 1 if model == "lr" else (a.n_seeds_rf if model == "rf" else a.n_seeds)
                if all((tag, inp, model, s) in done for s in range(n_exp)):
                    continue
                print(f"  {inp} / {model}: {X.shape[1]} features", flush=True)
                t0 = time.time()
                for seed, pv, pt, inf, _ in run_tab(model, X, y, idx, a):
                    stem = os.path.join(a.out, "preds", f"{tag}_{inp}_{model}_s{seed}")
                    np.save(stem + "_val.npy", pv.astype(np.float32))
                    np.save(stem + "_test.npy", pt.astype(np.float32))
                    row = {"split": tag, "input": inp, "model": model, "seed": seed, "n_feat": X.shape[1], "auc_val": roc_auc_score(y[va], pv),
                           "auc_test": roc_auc_score(y[te], pt), "sec": time.time() - t0, "info": ";".join(f"{k}={v}" for k, v in inf.items())}
                    pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                    print(f"    seed={seed} val={row['auc_val']:.4f} test={row['auc_test']:.4f}", flush=True)
                    t0 = time.time()
                if a.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()


def summarize(a):
    R = pd.read_csv(os.path.join(a.out, "results_e8.csv"))
    tags = [t for t in TAGS if t in set(R.split)]
    A, B, C = {}, {}, {}
    for t in tags:
        y = np.load(os.path.join(a.out, "preds", f"{t}_y_test.npy"))
        c = np.load(os.path.join(a.out, "preds", f"{t}_course_test.npy"))
        for m, nm in NAMES.items():
            P = {inp: mean_pred(a.out, t, f"{inp}_{m}") for inp in INPUTS}
            P = {k: v for k, v in P.items() if v is not None}
            for inp, p in P.items():
                A.setdefault((nm, inp), {})[t] = round(roc_auc_score(y, p), 4)
            for X, Y, lab in (("cwtrN_ind", "base_ind", "+CWTR+N - Base"), ("cwtr_ind", "base_ind", "+CWTR - Base"), ("cwtrN_ind", "cwtr_ind", "normalisation (+CWTR+N - +CWTR)")):
                if X in P and Y in P:
                    d = roc_auc_score(y, P[X]) - roc_auc_score(y, P[Y])
                    lo, hi = boot(y, P[X], P[Y], c, a.n_boot)
                    B.setdefault((nm, lab), {})[t] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]; {wins(y, c, P[X], P[Y])}"
            # comparison with E5 / E4 (statistics computed on all data -> training pool only), same test labels
            for lab, d_, key, inp in (("Base", a.e5, f"{m}_base", "base_ind"), ("+CWTR+N", a.e4, m, "cwtrN_ind")):
                old = mean_pred(d_, t, key)
                if old is not None and len(old) == len(y) and inp in P:
                    new = roc_auc_score(y, P[inp])
                    C.setdefault((nm, lab), {})[t] = f"{roc_auc_score(y, old):.4f} -> {new:.4f} ({new - roc_auc_score(y, old):+.4f})"
    ix = lambda d: pd.DataFrame(d).T.reindex(columns=tags)
    L = ["# E8: leakage audit for six models\n", "Learner / course statistics of the base features use only the training pool (train + validation).\n",
         "## A. Test AUC\n", ix(A).to_markdown(), "",
         f"## B. Gains [95% CI, bootstrap over courses, {a.n_boot} resamples]; courses won (courses with >= 200 enrollments and both classes)\n", ix(B).to_markdown(), "",
         "## C. Versus statistics computed on all data (E5 / E4)\n", ix(C).to_markdown(), "",
         "## Seeds run\n", R.groupby(["split", "input", "model"]).seed.nunique().unstack("model").to_markdown(), ""]
    open(os.path.join(a.out, "summary_e8.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--splits", default="0.5,0.6,0.7")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--inputs", default=",".join(INPUTS))
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p.add_argument("--e4", default="runs_e4")
    p.add_argument("--e5", default="runs_e5")
    p.add_argument("--n_boot", type=int, default=500)
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
