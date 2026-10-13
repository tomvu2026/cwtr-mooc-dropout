#!/usr/bin/env python3
"""E4: CWTR-LightGBM against other models on the SAME input (base + CWTR + course-normalised features: 251 columns).

Models: lgbm (CWTR-LightGBM) | lgbm_cwtr (ablation: without the 115 CFIN activity features) | xgb, cat (same 12-configuration search budget) |
        rf, lr, mlp | lstm_att (LSTM + multi-head attention + attentive pooling on the daily matrix, no contrastive loss) | cnn1d (multi-width 1-D CNN).
Sequence models read the 36-day matrix with 23 raw + 25 course-normalised channels and the context vector.
Splits: official | cut50 | cut60 | cut70 (unseen courses, validation = random courses of the training pool).

  python -m experiments.e4_model_comparison run       --data D --out runs_e4 --device cuda --xgb_device cuda --cat_task GPU
  python -m experiments.e4_model_comparison summarize --out runs_e4
Do not run several GPU processes at once with CatBoost on the GPU (it reserves most GPU memory).
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cwtr.cli import add_data_args, add_train_args
from cwtr.norm import course_norm
from cwtr.splits import make_split
from cwtr.stats import boot, mean_pred, wins
from cwtr.tab import run_tab
from cwtr.temporal import load_all
from cwtr.train import train_nn

warnings.filterwarnings("ignore")
TAB = ("lgbm", "lgbm_cwtr", "xgb", "cat", "rf", "lr", "mlp")
SEQ = ("lstm_att", "cnn1d")
NAMES = {"lgbm": "CWTR-LightGBM", "lgbm_cwtr": "LightGBM, without the 115 CFIN features", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random forest",
         "lr": "Logistic regression", "mlp": "MLP", "lstm_att": "LSTM + attention", "cnn1d": "CNN 1D"}
TAGS = ("official", "cut50", "cut60", "cut70")


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e4.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["model"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y, official_idx = D["y"].astype(int), D["idx"]
    S_raw = np.load(os.path.join(a.data, "seq.npy"))
    code = pd.factorize(D["course"])[0]
    extra, Tn = course_norm(S_raw, code, D["T"])
    L = np.log1p(S_raw.astype(np.float32))
    del S_raw
    flat = D["flat"]
    activity = [c for c in flat.columns if c.startswith("a") and c[1:].isdigit()]
    context = [c for c in flat.columns if c not in activity]
    X_full = pd.concat([flat, D["T"], Tn], axis=1).values.astype(np.float32)
    X_cwtr = pd.concat([flat[context], D["T"], Tn], axis=1).values.astype(np.float32)
    X_static = flat[context].values.astype(np.float32)
    print(f"features: full {X_full.shape[1]} | without the 115 CFIN features {X_cwtr.shape[1]} | context for sequence models {X_static.shape[1]} | "
          f"daily channels {L.shape[2] + extra.shape[2]}", flush=True)
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

        def save(model, seed, pv, pt, info_, t0):
            np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_val.npy"), pv.astype(np.float32))
            np.save(os.path.join(a.out, "preds", f"{tag}_{model}_s{seed}_test.npy"), pt.astype(np.float32))
            row = {"split": tag, "model": model, "seed": seed, "auc_val": roc_auc_score(y[idx["val"]], pv), "auc_test": roc_auc_score(y[idx["test"]], pt),
                   "sec": time.time() - t0}
            row.update(info_)
            pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
            print(f"  {model:<9} seed={seed} val={row['auc_val']:.4f} test={row['auc_test']:.4f} {info_}", flush=True)

        for model in a.models.split(","):
            n_exp = 1 if model == "lr" else (a.n_seeds_rf if model == "rf" else a.n_seeds)
            if all((tag, model, s) in done for s in range(n_exp)):
                continue
            t0 = time.time()
            if model in TAB:
                kind = "lgbm" if model == "lgbm_cwtr" else model
                for seed, pv, pt, info_, _ in run_tab(kind, X_cwtr if model == "lgbm_cwtr" else X_full, y, idx, a):
                    save(model, seed, pv, pt, info_, t0)
            else:
                sd = L[idx["train"]].reshape(-1, L.shape[2]).std(0)
                sd[sd == 0] = 1
                S = np.concatenate([L / sd, extra], 2).astype(np.float32)
                static = np.clip(StandardScaler().fit(X_static[idx["train"]]).transform(X_static), -10, 10).astype(np.float32)
                for seed in range(a.n_seeds):
                    if (tag, model, seed) in done:
                        continue
                    t1 = time.time()
                    pv, pt, ep = train_nn(model, {"S": S, "X": static}, y, idx, seed, a)
                    save(model, seed, pv, pt, {"epochs": ep}, t1)
                del S
                if a.device.startswith("cuda"):
                    import torch
                    torch.cuda.empty_cache()


def summarize(a):
    r = pd.read_csv(os.path.join(a.out, "results_e4.csv"))
    tags = [t for t in TAGS if t in set(r["split"])]
    L = ["# E4: CWTR-LightGBM against other models on the same input\n"]
    table, rows = {}, []
    for tag in tags:
        y = np.load(os.path.join(a.out, "preds", f"{tag}_y_test.npy"))
        c = np.load(os.path.join(a.out, "preds", f"{tag}_course_test.npy"))
        P = {m: mean_pred(a.out, tag, m) for m in NAMES}
        P = {m: p for m, p in P.items() if p is not None}
        for m, p in P.items():
            table.setdefault(NAMES[m], {})[tag] = round(roc_auc_score(y, p), 4)
        if "lgbm" not in P:
            continue
        for m, p in P.items():
            if m == "lgbm":
                continue
            d = roc_auc_score(y, p) - roc_auc_score(y, P["lgbm"])
            official = tag == "official"
            lo, hi = boot(y, p, P["lgbm"], None if official else c, 1000 if official else 500)
            rows.append({"model": NAMES[m], "split": tag, "cell": f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if official else f" wins {wins(y, c, p, P['lgbm'])} courses")})
    t = pd.DataFrame(table).T[tags].sort_values(tags[0], ascending=False)
    L += ["## Test AUC (seed-averaged predictions)\n", t.to_markdown(), ""]
    if rows:
        s = pd.DataFrame(rows).pivot(index="model", columns="split", values="cell")[tags]
        L += ["## AUC difference to CWTR-LightGBM [95% CI] (negative = worse than CWTR-LightGBM)\n", s.to_markdown(), ""]
    g = r.groupby(["split", "model"]).agg(n=("seed", "size"), test_sd=("auc_test", "std"), val=("auc_val", "mean"), sec=("sec", "mean")).round(4)
    L += ["## Per-seed details\n", g.to_markdown()]
    open(os.path.join(a.out, "summary_e4.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    add_train_args(p)
    p.add_argument("--splits", default="official,0.5,0.6,0.7")
    p.add_argument("--models", default=",".join(TAB + SEQ))
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
