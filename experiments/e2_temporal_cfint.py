#!/usr/bin/env python3
"""E2: does temporal information help? CFIN-T versus CFIN and LightGBM, on the official split and on a split by course start date.

Models: lgbm_base (round-10 features) | lgbm_temp (+ CWTR) | orig (CFIN) | temp (temporal branch + context only) |
        both (CFIN-T) | both_mean (CFIN-T without day-level attention).
Splits: official (random, as in the CFIN paper) | temporal (early courses train, late courses test).

  python -m experiments.e2_temporal_cfint run       --data D --out runs_e2 --device cuda
  python -m experiments.e2_temporal_cfint summarize --out runs_e2
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cwtr.cli import add_data_args, add_train_args
from cwtr.stats import boot, load_preds, logit, mean_pred
from cwtr.tab import run_tab
from cwtr.temporal import load_all
from cwtr.train import CFINT_KEYS, CFINT_VARIANTS, fit_cfint

warnings.filterwarnings("ignore")


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e2.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["model"], d["seed"]))
    for split in a.splits.split(","):
        D = load_all(a.data, a.cluster, split)
        idx, y = D["idx"], D["y"]
        print(f"[{split}] train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])} "
              f"dropout train/test={y[idx['train']].mean():.3f}/{y[idx['test']].mean():.3f} "
              f"courses train/test={len(set(D['course'][idx['train']]))}/{len(set(D['course'][idx['test']]))} temporal features={D['T'].shape[1]}", flush=True)
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{split}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{split}_recency_test.npy"), D["T"]["t_recency"].values[idx["test"]])

        def save(model, seed, pv, pt, extra):
            np.save(os.path.join(a.out, "preds", f"{split}_{model}_s{seed}_val.npy"), pv.astype(np.float32))
            np.save(os.path.join(a.out, "preds", f"{split}_{model}_s{seed}_test.npy"), pt.astype(np.float32))
            row = {"split": split, "model": model, "seed": seed, "auc_val": roc_auc_score(y[idx["val"]], pv), "auc_test": roc_auc_score(y[idx["test"]], pt)}
            row.update(extra)
            pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
            print(f"  {model:<10} seed={seed} val={row['auc_val']:.4f} test={row['auc_test']:.4f} {extra}", flush=True)

        for model in a.models.split(","):
            if model.startswith("lgbm"):
                if all((split, model, s) in done for s in range(a.n_seeds_lgbm)):
                    continue
                X = D["flat"] if model == "lgbm_base" else pd.concat([D["flat"], D["T"]], axis=1)
                t0 = time.time()
                for seed, pv, pt, info, m in run_tab("lgbm", X.values.astype(np.float32), y.astype(int), idx, a, n_seeds=a.n_seeds_lgbm):
                    save(model, seed, pv, pt, {"epochs": info["iters"], "sec": time.time() - t0})
                    if seed == 0 and model == "lgbm_temp":
                        imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
                        (imp / imp.sum()).head(30).round(4).to_csv(os.path.join(a.out, f"importance_{split}.csv"))
            else:
                T = {k: torch.as_tensor(D[k], device=a.device) for k in CFINT_KEYS + ("y",)}
                for seed in range(a.n_seeds):
                    if (split, model, seed) in done:
                        continue
                    t0 = time.time()
                    pv, pt, ep = fit_cfint(D, T, model, seed, a)
                    save(model, seed, pv, pt, {"epochs": ep, "sec": time.time() - t0})


def summarize(a):
    r = pd.read_csv(os.path.join(a.out, "results_e2.csv"))
    L = ["# E2: CFIN-T (temporal information)\n",
         "Reference (E1, official split): CFIN 0.8571-0.8580; LightGBM 0.8634. Criterion: CFIN-T beats the best LightGBM by >= +0.005 with a CI excluding 0.\n"]
    for split, g in r.groupby("split"):
        t = g.groupby("model").agg(n=("seed", "size"), val_mean=("auc_val", "mean"), test_mean=("auc_test", "mean"), test_sd=("auc_test", "std"), epochs=("epochs", "mean")).round(4)
        y, yv = np.load(os.path.join(a.out, "preds", f"{split}_y_test.npy")), np.load(os.path.join(a.out, "preds", f"{split}_y_val.npy"))
        P = {m: mean_pred(a.out, split, m, "test") for m in t.index}
        V = {m: mean_pred(a.out, split, m, "val") for m in t.index}
        t["test_seed_avg"] = [round(roc_auc_score(y, P[m]), 4) for m in t.index]
        L += [f"\n## Split: {split}\n", t.sort_values("test_seed_avg", ascending=False).to_markdown(), ""]
        if "both" in P and "lgbm_temp" in P:    # logit blend, weight chosen on validation
            ws = np.linspace(0, 1, 21)
            w = float(ws[int(np.argmax([roc_auc_score(yv, x * logit(V["both"]) + (1 - x) * logit(V["lgbm_temp"])) for x in ws]))])
            P["blend"] = 1 / (1 + np.exp(-(w * logit(P["both"]) + (1 - w) * logit(P["lgbm_temp"]))))
            L.append(f"Blend both + lgbm_temp: weight of both = {w:.2f} (chosen on validation), test AUC = {roc_auc_score(y, P['blend']):.4f}\n")
        L += ["| A | B | AUC A | AUC B | diff | 95% CI |", "|---|---|---|---|---|---|"]
        for m1, m2 in (("lgbm_temp", "lgbm_base"), ("both", "orig"), ("both", "temp"), ("both", "both_mean"), ("temp", "orig"),
                       ("both", "lgbm_base"), ("both", "lgbm_temp"), ("blend", "lgbm_temp"), ("blend", "both")):
            if P.get(m1) is not None and P.get(m2) is not None:
                lo, hi = boot(y, P[m1], P[m2])
                L.append(f"| {m1} | {m2} | {roc_auc_score(y, P[m1]):.4f} | {roc_auc_score(y, P[m2]):.4f} | {roc_auc_score(y, P[m1]) - roc_auc_score(y, P[m2]):+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        rp = os.path.join(a.out, "preds", f"{split}_recency_test.npy")
        if os.path.exists(rp):
            rec = np.load(rp)
            L += ["", "AUC by days between the last activity and the end of the window (recency):", "",
                  "| group | n | dropout rate | " + " | ".join(P) + " |", "|---|---|---|" + "---|" * len(P)]
            for name, mk in (("0-2 days", rec <= 2), ("3-7", (rec > 2) & (rec <= 7)), ("8-14", (rec > 7) & (rec <= 14)), ("15+", rec > 14)):
                if mk.sum() > 50 and 0 < y[mk].mean() < 1:
                    L.append(f"| {name} | {int(mk.sum())} | {y[mk].mean():.3f} | " + " | ".join(f"{roc_auc_score(y[mk], P[m][mk]):.4f}" for m in P) + " |")
        ip = os.path.join(a.out, f"importance_{split}.csv")
        if os.path.exists(ip):
            L += ["", "Ten most important features of lgbm_temp (share of gain):", "", pd.read_csv(ip, index_col=0).head(10).to_markdown()]
    open(os.path.join(a.out, "summary_e2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    add_train_args(p, n_seeds=10, n_trials=20, max_epochs=40, patience=5, n_estimators=4000)
    p.add_argument("--splits", default="official,temporal")
    p.add_argument("--models", default="lgbm_base,lgbm_temp,orig,temp,both,both_mean")
    p.add_argument("--n_seeds_lgbm", type=int, default=5)
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
