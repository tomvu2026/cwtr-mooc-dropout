#!/usr/bin/env python3
"""E3: unseen courses, with and without course-level normalisation.

Splits: courses sorted by start date; the first 50% / 60% / 70% of enrollments train, the later courses test (cut50 / cut60 / cut70).
Validation = randomly chosen courses of the training pool (~15% of its enrollments).
Models: lgbm_temp (base + CWTR) | lgbm_norm (+ course-normalised features) | cfint (CFIN-T, mean over days) | cfint_norm (+ normalised daily channels).

  python -m experiments.e3_unseen_courses run       --data D --out runs_e3 --device cuda
  python -m experiments.e3_unseen_courses summarize --out runs_e3
"""
import argparse
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cwtr.cli import add_data_args, add_train_args
from cwtr.norm import course_norm
from cwtr.splits import make_split
from cwtr.stats import boot, mean_pred, wins
from cwtr.tab import run_tab
from cwtr.temporal import load_all
from cwtr.train import CFINT_KEYS, fit_cfint

warnings.filterwarnings("ignore")
PAIRS = (("lgbm_norm", "lgbm_temp"), ("cfint_norm", "cfint"), ("cfint", "lgbm_temp"), ("cfint_norm", "lgbm_norm"), ("cfint_norm", "lgbm_temp"))


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e3.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["cut"].round(2), d["model"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y = D["y"]
    S_raw = np.load(os.path.join(a.data, "seq.npy"))
    code = pd.factorize(D["course"])[0]
    extra, Tn = course_norm(S_raw, code, D["T"])
    L = np.log1p(S_raw.astype(np.float32))
    del S_raw
    print(f"daily channels: raw {L.shape[2]} + normalised {extra.shape[2]}; tabular: CWTR input {D['flat'].shape[1] + D['T'].shape[1]} + normalised {Tn.shape[1]}", flush=True)
    X_temp = pd.concat([D["flat"], D["T"]], axis=1)
    X_norm = pd.concat([X_temp, Tn], axis=1)
    for cut in [float(c) for c in a.cuts.split(",")]:
        idx, info = make_split(D, a.data, cut)
        D["idx"] = idx
        tag = f"cut{int(round(cut * 100))}"
        sd = L[idx["train"]].reshape(-1, L.shape[2]).std(0)
        sd[sd == 0] = 1
        S_base = (L / sd).astype(np.float32)
        print(f"[{tag}] {info} train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])} "
              f"dropout train/val/test={y[idx['train']].mean():.3f}/{y[idx['val']].mean():.3f}/{y[idx['test']].mean():.3f}", flush=True)
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{tag}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{tag}_course_test.npy"), code[idx["test"]])
        np.save(os.path.join(a.out, "preds", f"{tag}_recency_test.npy"), D["T"]["t_recency"].values[idx["test"]])

        def save(model, seed, pv, pt, extra_info):
            np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_val.npy"), pv.astype(np.float32))
            np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_test.npy"), pt.astype(np.float32))
            row = {"cut": cut, "model": model, "seed": seed, "auc_val": roc_auc_score(y[idx["val"]], pv), "auc_test": roc_auc_score(y[idx["test"]], pt)}
            row.update(extra_info)
            pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
            print(f"  {model:<10} seed={seed} val={row['auc_val']:.4f} test={row['auc_test']:.4f} {extra_info}", flush=True)

        for model in a.models.split(","):
            if model.startswith("lgbm"):
                if all((round(cut, 2), model, s) in done for s in range(a.n_seeds_lgbm)):
                    continue
                X = X_temp if model == "lgbm_temp" else X_norm
                t0 = time.time()
                for seed, pv, pt, info_, m in run_tab("lgbm", X.values.astype(np.float32), y.astype(int), idx, a, n_seeds=a.n_seeds_lgbm):
                    save(model, seed, pv, pt, {"epochs": info_["iters"], "sec": time.time() - t0})
                    if seed == 0:
                        imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
                        (imp / imp.sum()).head(20).round(4).to_csv(os.path.join(a.out, f"importance_{tag}_{model}.csv"))
            else:
                D["S"] = S_base if model == "cfint" else np.concatenate([S_base, extra], 2)
                T = {k: torch.as_tensor(D[k], device=a.device) for k in CFINT_KEYS + ("y",)}
                for seed in range(a.n_seeds):
                    if (round(cut, 2), model, seed) in done:
                        continue
                    t0 = time.time()
                    pv, pt, ep = fit_cfint(D, T, "both_mean", seed, a)
                    save(model, seed, pv, pt, {"epochs": ep, "sec": time.time() - t0})
                del T
                if a.device.startswith("cuda"):
                    torch.cuda.empty_cache()


def summarize(a):
    r = pd.read_csv(os.path.join(a.out, "results_e3.csv"))
    r["tag"] = "cut" + (r["cut"] * 100).round().astype(int).astype(str)
    L = ["# E3: unseen courses, with / without course-level normalisation\n",
         "Criterion: CFIN-T beats LightGBM on the same input by >= +0.005, CI (bootstrap over courses) excluding 0, at all three cuts.\n"]
    rows = []
    for tag, g in r.groupby("tag"):
        y = np.load(os.path.join(a.out, "preds", f"{tag}_y_test.npy"))
        c = np.load(os.path.join(a.out, "preds", f"{tag}_course_test.npy"))
        t = g.groupby("model").agg(n=("seed", "size"), val_mean=("auc_val", "mean"), test_mean=("auc_test", "mean"), test_sd=("auc_test", "std"), epochs=("epochs", "mean")).round(4)
        P = {m: mean_pred(a.out, tag, m) for m in t.index}
        t["test_seed_avg"] = [round(roc_auc_score(y, P[m]), 4) for m in t.index]
        L += [f"\n## {tag}: test = {len(y)} enrollments, {len(np.unique(c))} courses\n", t.sort_values("test_seed_avg", ascending=False).to_markdown(), "",
              "| A | B | AUC A | AUC B | diff | 95% CI (over courses) | A wins on courses |", "|---|---|---|---|---|---|---|"]
        for m1, m2 in PAIRS:
            if P.get(m1) is None or P.get(m2) is None:
                continue
            d = roc_auc_score(y, P[m1]) - roc_auc_score(y, P[m2])
            lo, hi = boot(y, P[m1], P[m2], c, 500)
            L.append(f"| {m1} | {m2} | {roc_auc_score(y, P[m1]):.4f} | {roc_auc_score(y, P[m2]):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {wins(y, c, P[m1], P[m2])} |")
            rows.append((f"{m1} - {m2}", tag, d, lo, hi))
        rec = np.load(os.path.join(a.out, "preds", f"{tag}_recency_test.npy"))
        L += ["", "| recency | n | dropout rate | " + " | ".join(P) + " |", "|---|---|---|" + "---|" * len(P)]
        for name, mk in (("0-2 days", rec <= 2), ("3-7", (rec > 2) & (rec <= 7)), ("8-14", (rec > 7) & (rec <= 14)), ("15+", rec > 14)):
            if mk.sum() > 50 and 0 < y[mk].mean() < 1:
                L.append(f"| {name} | {int(mk.sum())} | {y[mk].mean():.3f} | " + " | ".join(f"{roc_auc_score(y[mk], P[m][mk]):.4f}" for m in P) + " |")
    if rows:
        s = pd.DataFrame(rows, columns=["comparison", "cut", "diff", "lo", "hi"])
        s["cell"] = s.apply(lambda x: f"{x['diff']:+.4f} [{x.lo:+.4f}, {x.hi:+.4f}]", axis=1)
        L += ["\n## Summary over the three cuts\n", s.pivot(index="comparison", columns="cut", values="cell").to_markdown()]
    for f in sorted(glob.glob(os.path.join(a.out, "importance_*_lgbm_norm.csv")))[:1]:
        L += ["", f"Ten most important features ({os.path.basename(f)}):", "", pd.read_csv(f, index_col=0).head(10).to_markdown()]
    open(os.path.join(a.out, "summary_e3.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    add_train_args(p, n_seeds=10, n_trials=20, max_epochs=40, patience=5, n_estimators=4000)
    p.add_argument("--cuts", default="0.5,0.6,0.7")
    p.add_argument("--models", default="lgbm_temp,lgbm_norm,cfint,cfint_norm")
    p.add_argument("--n_seeds_lgbm", type=int, default=5)
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
