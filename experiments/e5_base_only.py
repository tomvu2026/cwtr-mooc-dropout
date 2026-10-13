#!/usr/bin/env python3
"""E5: the same models as E4 on the base features only (115 activity features + context = 144 columns), to measure the gain of CWTR per model.

Same protocol as E4: splits, 12-configuration search, seeds, choice of C for LR, MLP. Models are stored as <model>_base.
`summarize` pairs these runs with E4 (the "+CWTR+N" input) on the same test labels.

  python -m experiments.e5_base_only run       --data D --out runs_e5 --device cuda --xgb_device cuda --cat_task GPU
  python -m experiments.e5_base_only summarize --out runs_e5 --e4 runs_e4
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cwtr.cli import add_data_args, add_train_args
from cwtr.splits import make_split
from cwtr.stats import boot, mean_pred, wins
from cwtr.tab import run_tab
from cwtr.temporal import load_all

warnings.filterwarnings("ignore")
MODELS = ("lgbm", "xgb", "cat", "rf", "lr", "mlp")
NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random forest", "lr": "Logistic regression", "mlp": "MLP"}
TAGS = ("official", "cut50", "cut60", "cut70")


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e5.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["model"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y, official_idx = D["y"].astype(int), D["idx"]
    code = pd.factorize(D["course"])[0]
    X_base = D["flat"].values.astype(np.float32)
    print(f"base features: {X_base.shape[1]} columns (no CWTR, no course normalisation)", flush=True)
    for sp in a.splits.split(","):
        if sp == "official":
            idx, tag, info = official_idx, "official", {}
        else:
            idx, info = make_split(D, a.data, float(sp))
            tag = f"cut{int(round(float(sp) * 100))}"
        print(f"[{tag}] {info} train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}", flush=True)
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{tag}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{tag}_course_test.npy"), code[idx["test"]])
        for kind in a.models.split(","):
            model = f"{kind}_base"
            n_exp = 1 if kind == "lr" else (a.n_seeds_rf if kind == "rf" else a.n_seeds)
            if all((tag, model, s) in done for s in range(n_exp)):
                continue
            t_prev = time.time()
            for seed, pv, pt, inf, _ in run_tab(kind, X_base, y, idx, a):
                now = time.time()
                np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_val.npy"), pv.astype(np.float32))
                np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_test.npy"), pt.astype(np.float32))
                row = {"split": tag, "model": model, "seed": seed, "auc_val": roc_auc_score(y[idx["val"]], pv),
                       "auc_test": roc_auc_score(y[idx["test"]], pt), "sec": now - t_prev}
                row.update(inf)
                pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                print(f"  {model:<10} seed={seed} val={row['auc_val']:.4f} test={row['auc_test']:.4f} {inf} ({now - t_prev:.0f}s)", flush=True)
                t_prev = now
            if a.device.startswith("cuda"):
                import torch
                torch.cuda.empty_cache()


def summarize(a):
    r = pd.read_csv(os.path.join(a.out, "results_e5.csv"))
    tags = [t for t in TAGS if t in set(r["split"])]
    base, cw, gain, vs_lgbm = {}, {}, {}, {}
    for tag in tags:
        y = np.load(os.path.join(a.out, "preds", f"{tag}_y_test.npy"))
        c = np.load(os.path.join(a.out, "preds", f"{tag}_course_test.npy"))
        e4_labels = os.path.join(a.e4, "preds", f"{tag}_y_test.npy")
        same = os.path.exists(e4_labels) and np.array_equal(np.load(e4_labels), y)
        if not same:
            print(f"WARNING [{tag}]: test labels of E4 and E5 differ -> no paired comparison")
        Pb = {}
        for m, nm in NAMES.items():
            pb = mean_pred(a.out, tag, f"{m}_base")
            pc = mean_pred(a.e4, tag, m) if same else None
            if pb is not None:
                base.setdefault(nm, {})[tag] = round(roc_auc_score(y, pb), 4)
                Pb[m] = pb
            if pc is not None:
                cw.setdefault(nm, {})[tag] = round(roc_auc_score(y, pc), 4)
            if pb is not None and pc is not None:
                d = roc_auc_score(y, pc) - roc_auc_score(y, pb)
                official = tag == "official"
                lo, hi = boot(y, pc, pb, None if official else c, 1000 if official else 500)
                gain.setdefault(nm, {})[tag] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if official else f" wins {wins(y, c, pc, pb)} courses")
        if "lgbm" in Pb:
            for m, p in Pb.items():
                if m == "lgbm":
                    continue
                d = roc_auc_score(y, p) - roc_auc_score(y, Pb["lgbm"])
                official = tag == "official"
                lo, hi = boot(y, p, Pb["lgbm"], None if official else c, 1000 if official else 500)
                vs_lgbm.setdefault(NAMES[m], {})[tag] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
    L = ["# E5: models x {base, base + CWTR + normalisation}\n",
         "## Test AUC, base features only\n", pd.DataFrame(base).T[tags].sort_values(tags[0], ascending=False).to_markdown(), "",
         "## Test AUC, base + CWTR + normalisation (from E4)\n", pd.DataFrame(cw).T.reindex(columns=tags).to_markdown(), "",
         "## Gain from CWTR + normalisation per model [95% CI]\n", pd.DataFrame(gain).T.reindex(columns=tags).to_markdown(), ""]
    if vs_lgbm:
        L += ["## Base features only: difference to base LightGBM [95% CI]\n", pd.DataFrame(vs_lgbm).T.reindex(columns=tags).to_markdown(), ""]
    g = r.groupby(["split", "model"]).agg(n=("seed", "size"), test_sd=("auc_test", "std"), val=("auc_val", "mean"), sec=("sec", "mean")).round(4)
    L += ["## Per-seed details\n", g.to_markdown()]
    open(os.path.join(a.out, "summary_e5.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--splits", default="official,0.5,0.6,0.7")
    p.add_argument("--models", default=",".join(MODELS))
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p.add_argument("--e4", default="runs_e4")
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
