#!/usr/bin/env python3
"""E7: leakage audit for LightGBM and uncertainty over learners.

  inductive  On cut50/60/70 every learner / course statistic of the base features is computed ONLY from the training pool (train + validation
             courses); nothing from later courses is used. Variants: base_ind, cwtr_ind, cwtrN_ind (course normalisation is computed on the
             course itself and uses no labels), and *_nocl without the user-cluster label. Parameters come from E6 (params_<split>.json), 3 seeds.
  stats      No training. (a) bootstrap over LEARNERS (not enrollments) on the official split for the gain of CWTR of the six models of E4/E5;
             (b) seed-to-seed variation of that gain; (c) threshold chosen by macro-F1 on validation.

  python -m experiments.e7_leakage_lgbm run       --data D --out runs_e7 --e4 runs_e4 --e5 runs_e5 --e6 runs_e6
  python -m experiments.e7_leakage_lgbm summarize --out runs_e7 --e6 runs_e6
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from cwtr.build_helpers import train_only_base
from cwtr.cli import add_data_args
from cwtr.norm import tab_norm
from cwtr.splits import make_split
from cwtr.stats import boot, load_preds, mean_pred, wins
from cwtr.tab import fit_gbdt
from cwtr.temporal import load_all

warnings.filterwarnings("ignore")
MODELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random forest", "lr": "Logistic regression", "mlp": "MLP"}


def part_stats(a, D):
    rows = []
    y_te = np.load(os.path.join(a.e4, "preds", "official_y_test.npy"))
    y_va = np.load(os.path.join(a.e4, "preds", "official_y_val.npy"))
    users = D["user"][D["idx"]["test"]]
    assert len(users) == len(y_te)
    learners = [np.where(users == u)[0] for u in np.unique(users)]
    rng = np.random.default_rng(0)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    for m, nm in MODELS.items():
        B, C = load_preds(a.e5, "official", f"{m}_base"), load_preds(a.e4, "official", m)
        Bv, Cv = load_preds(a.e5, "official", f"{m}_base", "val"), load_preds(a.e4, "official", m, "val")
        if not B or not C:
            continue
        pb, pc = np.mean(B, 0), np.mean(C, 0)
        gain = roc_auc_score(y_te, pc) - roc_auc_score(y_te, pb)
        boots = []
        for _ in range(a.n_boot):
            i = np.concatenate([learners[j] for j in rng.integers(0, len(learners), len(learners))])
            boots.append(roc_auc_score(y_te[i], pc[i]) - roc_auc_score(y_te[i], pb[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        k = min(len(B), len(C))
        seed_gain = [roc_auc_score(y_te, C[s]) - roc_auc_score(y_te, B[s]) for s in range(k)]
        row = {"model": nm, "gain": gain, "ci_learner_lo": lo, "ci_learner_hi": hi, "n_seed_pairs": k, "seed_gain_min": min(seed_gain),
               "seed_gain_max": max(seed_gain), "seed_gain_sd": float(np.std(seed_gain, ddof=1)) if k > 1 else np.nan}
        for inp, pv, pt in (("base", np.mean(Bv, 0), pb), ("cwtr", np.mean(Cv, 0), pc)):
            thr = grid[np.argmax([f1_score(y_va, (pv >= g).astype(int), average="macro") for g in grid])]
            row[f"thr_macro_{inp}"] = thr
            row[f"macroF1_{inp}"] = f1_score(y_te, (pt >= thr).astype(int), average="macro")
        rows.append(row)
        print(f"  {nm}: gain {gain:+.4f} CI(learner) [{lo:+.4f}, {hi:+.4f}] seeds {min(seed_gain):+.4f}..{max(seed_gain):+.4f} "
              f"macro-F1 {row['macroF1_base']:.4f}->{row['macroF1_cwtr']:.4f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.out, "stats_e7.csv"), index=False)


def part_inductive(a, D):
    y = D["y"].astype(int)
    code = pd.factorize(D["course"])[0]
    T = D["T"]
    Tn = tab_norm(T, code)
    res = os.path.join(a.out, "results_e7.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["variant"], d["seed"]))
    for sp in a.splits.split(","):
        idx, info = make_split(D, a.data, float(sp))
        tag = f"cut{int(round(float(sp) * 100))}"
        tr, va, te = idx["train"], idx["val"], idx["test"]
        params = json.load(open(os.path.join(a.e6, f"params_{tag}.json")))
        ref = np.zeros(len(y), bool)
        ref[np.concatenate([tr, va])] = True
        flat = train_only_base(a.data, a.cluster, ref)
        flat.index = T.index
        no_cluster = flat[[c for c in flat.columns if not c.startswith("cluster_label")]]
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{tag}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{tag}_course_test.npy"), code[te])
        variants = {"base_ind": flat, "cwtr_ind": pd.concat([flat, T], axis=1), "cwtrN_ind": pd.concat([flat, T, Tn], axis=1),
                    "base_ind_nocl": no_cluster, "cwtrN_ind_nocl": pd.concat([no_cluster, T, Tn], axis=1)}
        print(f"[{tag}] {info}", flush=True)
        for name, Xdf in variants.items():
            X = Xdf.values.astype(np.float32)
            for seed in range(a.n_seeds):
                if (tag, name, seed) in done:
                    continue
                t0 = time.time()
                m, _ = fit_gbdt("lgbm", params, seed, X[tr], y[tr], X[va], y[va], a)
                pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
                np.save(os.path.join(a.out, "preds", f"{tag}_{name}_s{seed}_val.npy"), pv.astype(np.float32))
                np.save(os.path.join(a.out, "preds", f"{tag}_{name}_s{seed}_test.npy"), pt.astype(np.float32))
                row = {"split": tag, "variant": name, "seed": seed, "n_feat": X.shape[1], "auc_val": roc_auc_score(y[va], pv),
                       "auc_test": roc_auc_score(y[te], pt), "sec": time.time() - t0}
                pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                print(f"  {name:<16} seed={seed} test={row['auc_test']:.4f}", flush=True)


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    D = load_all(a.data, a.cluster, "official")
    if "stats" in a.parts:
        part_stats(a, D)
    if "inductive" in a.parts:
        part_inductive(a, D)


def summarize(a):
    L = ["# E7\n"]
    s = os.path.join(a.out, "stats_e7.csv")
    if os.path.exists(s):
        L += ["## Official split: bootstrap over LEARNERS, seed variation, macro-F1 with the macro-F1-optimal threshold\n", pd.read_csv(s).to_markdown(index=False), ""]
    r = os.path.join(a.out, "results_e7.csv")
    if os.path.exists(r):
        R = pd.read_csv(r)
        tags = [t for t in ("cut50", "cut60", "cut70") if t in set(R.split)]
        table, diffs = {}, {}
        for t in tags:
            y = np.load(os.path.join(a.out, "preds", f"{t}_y_test.npy"))
            c = np.load(os.path.join(a.out, "preds", f"{t}_course_test.npy"))
            P = {v: mean_pred(a.out, t, v) for v in R.loc[R.split == t, "variant"].unique()}
            for k in ("base", "cwtr", "cwtr_norm"):          # counterparts with statistics computed on all data (E6, full window)
                p = load_preds(a.e6, t, f"w36_{k}")
                if p and len(p[0]) == len(y):
                    P[f"e6_{k}"] = np.mean(p, 0)
            for v, p in P.items():
                table.setdefault(v, {})[t] = round(roc_auc_score(y, p), 4)
            for A, B in (("cwtr_ind", "base_ind"), ("cwtrN_ind", "base_ind"), ("cwtrN_ind", "cwtr_ind"), ("cwtrN_ind_nocl", "base_ind_nocl"),
                         ("base_ind", "e6_base"), ("cwtrN_ind", "e6_cwtr_norm")):
                if A in P and B in P:
                    d = roc_auc_score(y, P[A]) - roc_auc_score(y, P[B])
                    lo, hi = boot(y, P[A], P[B], c, 500)
                    diffs.setdefault(f"{A} - {B}", {})[t] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] {wins(y, c, P[A], P[B])}"
        L += ["## Unseen courses: test AUC (base statistics computed from the training pool only; e6_* = statistics on all data)\n", pd.DataFrame(table).T[tags].to_markdown(), "",
              "## Differences [95% CI over courses], courses won\n", pd.DataFrame(diffs).T.reindex(columns=tags).to_markdown(), ""]
    open(os.path.join(a.out, "summary_e7.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    for d in ("e4", "e5", "e6"):
        p.add_argument(f"--{d}", default=f"runs_{d}")
    p.add_argument("--splits", default="0.5,0.6,0.7")
    p.add_argument("--parts", default="stats,inductive")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--n_jobs", type=int, default=16)
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p.add_argument("--e6", default="runs_e6")
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
