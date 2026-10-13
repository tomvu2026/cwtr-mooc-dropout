#!/usr/bin/env python3
"""E6: ablations and robustness checks for CWTR-LightGBM (LightGBM only).

Hyper-parameters are searched once per split (12 configurations, base + CWTR + normalisation, full window), saved to params_<split>.json,
and reused by every variant (3 seeds each), so variants are directly comparable.

  ablation   drop one group of CWTR features at a time; control "base + weekly counts" (full window, no normalisation)
  window     short observation windows: only the first W in {7,14,21,28,36} days are used to predict the SAME label;
             every feature (base features too) is recomputed from those W days
  inductive  (official split only) course normalisation using only the training enrollments of each course
  metrics    no training: accuracy / precision / recall / F1 / macro-F1 / confusion matrix from the saved predictions of E4 and E5

  python -m experiments.e6_ablation_windows run       --data D --out runs_e6 [--parts window,inductive,ablation]
  python -m experiments.e6_ablation_windows summarize --out runs_e6
  python -m experiments.e6_ablation_windows metrics   --e4 runs_e4 --e5 runs_e5 --out runs_e6
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from cwtr.cli import add_data_args
from cwtr.data import build
from cwtr.norm import KEY_T, tab_norm, tab_norm_inductive
from cwtr.splits import make_split
from cwtr.stats import boot, load_preds, mean_pred, wins
from cwtr.tab import fit_gbdt, sample
from cwtr.temporal import WEEKLY, load_all, temporal_table
from cwtr.build_helpers import base_from_seq

warnings.filterwarnings("ignore")
# CWTR feature groups used for the leave-one-group-out ablation (names without the 't_' prefix)
FEATURE_GROUPS = {
    "presence": ["n_active", "first_day", "last_day", "recency", "span"],
    "gaps": ["max_gap", "mean_gap"],
    "recent": ["ev_last7", "active_last7", "ev_last14", "active_last14", "ev_first7", "active_first7", "share_last7", "share_last14"],
    "dynamics": ["slope", "daily_std", "daily_max"],
    "sessions": ["sessions", "sessions_last7", "ev_per_session"],
    "action": [f"{g}_{s}" for g in ("video", "problem", "forum", "click") for s in ("w0", "w1", "w2", "w3", "w4", "lastday", "days")],
    "correct": ["correct_ratio"],
}
WINDOWS = (7, 14, 21, 28, 36)
TAGS = ("official", "cut50", "cut60", "cut70")


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    parts = a.parts.split(",")
    res = os.path.join(a.out, "results_e6.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["variant"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y = D["y"].astype(int)
    code = pd.factorize(D["course"])[0]
    flat, T = D["flat"], D["T"]
    activity = [c for c in flat.columns if c.startswith("a") and c[1:].isdigit()]
    context = flat[[c for c in flat.columns if c not in activity]]
    Tn = tab_norm(T, code)

    variants = {}       # name -> function building the feature matrix (built lazily to save memory)
    if "ablation" in parts:
        variants["base"] = lambda: flat
        variants["base_weekly"] = lambda: pd.concat([flat, T[WEEKLY]], axis=1)
        variants["cwtr"] = lambda: pd.concat([flat, T], axis=1)
        for g, cols in FEATURE_GROUPS.items():
            variants[f"cwtr_minus_{g}"] = (lambda cols=cols: pd.concat([flat, T.drop(columns=[f"t_{c}" for c in cols])], axis=1))
        variants["cwtr_norm"] = lambda: pd.concat([flat, T, Tn], axis=1)
    if "window" in parts:
        S_raw = np.load(os.path.join(a.data, "seq.npy"))
        user = D["user"]
        a_orig = D["A"][:, ::5]       # own value of the 23 count features, from the original pipeline
        a_new = base_from_seq(S_raw, user, D["course"], context)[[f"a{i}" for i in range(0, 115, 5)]].values
        r = [np.corrcoef(a_orig[:, k], a_new[:, k])[0, 1] for k in range(23)]
        print(f"check: base features recomputed from the daily matrix (W=36) vs original: correlation min={np.nanmin(r):.4f} median={np.nanmedian(r):.4f}", flush=True)

        def window_variant(W, kind):
            def make():
                Sw = S_raw[:, :W, :]
                Bw = base_from_seq(Sw, user, D["course"], context)
                if kind == "base":
                    return Bw
                Tw = temporal_table(Sw)
                Tw.index = flat.index
                return pd.concat([Bw, Tw], axis=1) if kind == "cwtr" else pd.concat([Bw, Tw, tab_norm(Tw, code)], axis=1)
            return make
        for W in WINDOWS:
            for kind in ("base", "cwtr", "cwtr_norm"):
                variants[f"w{W}_{kind}"] = window_variant(W, kind)

    for sp in a.splits.split(","):
        if sp == "official":
            idx, tag, info = D["idx"], "official", {}
        else:
            idx, info = make_split(D, a.data, float(sp))
            tag = f"cut{int(round(float(sp) * 100))}"
        tr, va, te = idx["train"], idx["val"], idx["test"]
        print(f"[{tag}] {info} train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        for k in ("val", "test"):
            np.save(os.path.join(a.out, "preds", f"{tag}_y_{k}.npy"), y[idx[k]])
        np.save(os.path.join(a.out, "preds", f"{tag}_course_test.npy"), code[te])

        params_file = os.path.join(a.out, f"params_{tag}.json")
        if os.path.exists(params_file):
            params = json.load(open(params_file))
        else:
            X = pd.concat([flat, T, Tn], axis=1).values.astype(np.float32)
            rng = np.random.default_rng(0)
            best = (-1, None)
            for _ in range(a.n_trials):
                q = sample("lgbm", rng)
                m, _ = fit_gbdt("lgbm", q, 0, X[tr], y[tr], X[va], y[va], a)
                score = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
                if score > best[0]:
                    best = (score, q)
            params = best[1]
            json.dump(params, open(params_file, "w"))
            print(f"  parameters {tag}: validation AUC {best[0]:.4f} {params}", flush=True)
            del X

        todo = dict(variants)
        if "inductive" in parts and tag == "official":
            ref = np.zeros(len(y), bool)
            ref[np.concatenate([tr, va])] = True
            Tn_ind = tab_norm_inductive(T, code, ref)
            todo["ind_norm"] = lambda: pd.concat([flat, T, Tn_ind], axis=1)
            todo["ind_all"] = lambda: pd.concat([build(a.data, a.cluster, "train", True)["flat"], T, Tn_ind], axis=1)
            todo.setdefault("cwtr_norm", lambda: pd.concat([flat, T, Tn], axis=1))
        if a.only:
            todo = {k: v for k, v in todo.items() if k in a.only.split(",")}
        for name, make in todo.items():
            if all((tag, name, s) in done for s in range(a.n_seeds)):
                continue
            Xdf = make()
            X = Xdf.values.astype(np.float32)
            for seed in range(a.n_seeds):
                if (tag, name, seed) in done:
                    continue
                t0 = time.time()
                m, it = fit_gbdt("lgbm", params, seed, X[tr], y[tr], X[va], y[va], a)
                pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
                np.save(os.path.join(a.out, "preds", f"{tag}_{name}_s{seed}_val.npy"), pv.astype(np.float32))
                np.save(os.path.join(a.out, "preds", f"{tag}_{name}_s{seed}_test.npy"), pt.astype(np.float32))
                row = {"split": tag, "variant": name, "seed": seed, "n_feat": X.shape[1], "auc_val": roc_auc_score(y[va], pv),
                       "auc_test": roc_auc_score(y[te], pt), "iters": it, "sec": time.time() - t0}
                pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                print(f"  {name:<22} seed={seed} n={X.shape[1]} val={row['auc_val']:.4f} test={row['auc_test']:.4f} ({row['sec']:.0f}s)", flush=True)
            del X, Xdf


def summarize(a):
    r = pd.read_csv(os.path.join(a.out, "results_e6.csv"))
    tags = [t for t in TAGS if t in set(r["split"])]
    Y, C, P = {}, {}, {}
    for t in tags:
        Y[t] = np.load(os.path.join(a.out, "preds", f"{t}_y_test.npy"))
        C[t] = np.load(os.path.join(a.out, "preds", f"{t}_course_test.npy"))
        P[t] = {v: mean_pred(a.out, t, v) for v in r.loc[r.split == t, "variant"].unique()}
    auc = lambda t, v: roc_auc_score(Y[t], P[t][v])

    def compare(A, B):
        row = {}
        for t in tags:
            if A not in P[t] or B not in P[t]:
                continue
            d = auc(t, A) - auc(t, B)
            official = t == "official"
            lo, hi = boot(Y[t], P[t][A], P[t][B], None if official else C[t], 1000 if official else 500)
            row[t] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if official else f" {wins(Y[t], C[t], P[t][A], P[t][B])}")
        return row

    L = ["# E6: ablations and robustness (LightGBM, fixed parameters per split, mean of 3 seeds)\n"]
    tab = pd.DataFrame({t: {v: round(auc(t, v), 4) for v in P[t]} for t in tags})
    L += ["## Test AUC per variant\n", tab.to_markdown(), ""]
    if "cwtr" in tab.index:
        rows = {"base + weekly counts vs base": compare("base_weekly", "base"),
                "full CWTR vs base + weekly counts": compare("cwtr", "base_weekly"),
                "full CWTR vs base": compare("cwtr", "base")}
        for g in FEATURE_GROUPS:
            rows[f"without group {g} vs full CWTR"] = compare(f"cwtr_minus_{g}", "cwtr")
        rows["CWTR + normalisation vs CWTR"] = compare("cwtr_norm", "cwtr")
        L += ["## Ablation by feature group (AUC difference [95% CI], courses won)\n", pd.DataFrame(rows).T.reindex(columns=tags).to_markdown(), ""]
    if any(f"w{W}_{k}" in tab.index for W in WINDOWS for k in ("base", "cwtr", "cwtr_norm")):
        for t in tags:
            m = pd.DataFrame({k: {W: tab.loc[f"w{W}_{k}", t] if f"w{W}_{k}" in tab.index else np.nan for W in WINDOWS} for k in ("base", "cwtr", "cwtr_norm")})
            L += [f"## Short windows [{t}]: test AUC by number of observed days\n", m.to_markdown(), ""]
        rows = {}
        for W in WINDOWS:
            rows[f"W={W}: CWTR vs base"] = compare(f"w{W}_cwtr", f"w{W}_base")
            rows[f"W={W}: + normalisation vs CWTR"] = compare(f"w{W}_cwtr_norm", f"w{W}_cwtr")
        L += ["## Short windows: AUC difference [95% CI]\n", pd.DataFrame(rows).T.reindex(columns=tags).to_markdown(), ""]
    if "ind_norm" in P.get("official", {}):
        rows = {"normalisation from training rows only vs normalisation using test rows too": compare("ind_norm", "cwtr_norm"),
                "normalisation + base statistics from training rows only vs full-data version": compare("ind_all", "cwtr_norm")}
        L += ["## Normalisation restricted to the training set (official split)\n", pd.DataFrame(rows).T.reindex(columns=["official"]).to_markdown(), ""]
    g = r.groupby(["split", "variant"]).agg(n=("seed", "size"), n_feat=("n_feat", "first"), test_sd=("auc_test", "std"), val=("auc_val", "mean"), sec=("sec", "mean")).round(4)
    L += ["## Details\n", g.to_markdown()]
    open(os.path.join(a.out, "summary_e6.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


MODELS = {"lgbm": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "rf": "Random forest", "lr": "Logistic regression", "mlp": "MLP"}


def metrics(a):
    """Classification metrics from saved predictions (positive class = dropout). Threshold maximises F1 on validation; threshold 0.5 also reported."""
    os.makedirs(a.out, exist_ok=True)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows, confusion = [], []
    for tag in TAGS:
        f = os.path.join(a.e4, "preds", f"{tag}_y_test.npy")
        if not os.path.exists(f):
            continue
        yv, yt = np.load(os.path.join(a.e4, "preds", f"{tag}_y_val.npy")), np.load(f)
        for m, nm in MODELS.items():
            for inp, d, key in (("Base", a.e5, f"{m}_base"), ("+CWTR", a.e4, m)):
                pv, pt = mean_pred(d, tag, key, "val"), mean_pred(d, tag, key, "test")
                if pv is None or pt is None:
                    continue
                thr = grid[np.argmax([f1_score(yv, (pv >= g).astype(int)) for g in grid])]
                for mode, t in (("val_threshold", thr), ("threshold_0.5", 0.5)):
                    h = (pt >= t).astype(int)
                    rows.append({"split": tag, "model": nm, "input": inp, "mode": mode, "thr": round(t, 2), "acc": accuracy_score(yt, h), "prec": precision_score(yt, h),
                                 "rec": recall_score(yt, h), "f1": f1_score(yt, h), "f1_macro": f1_score(yt, h, average="macro"), "auc": roc_auc_score(yt, pt)})
                if tag == "official" and m == "lgbm":
                    tn, fp, fn, tp = confusion_matrix(yt, (pt >= thr).astype(int)).ravel()
                    confusion.append({"input": inp, "thr": thr, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    R = pd.DataFrame(rows)
    R.round(4).to_csv(os.path.join(a.out, "metrics_e6.csv"), index=False)
    L = ["# E6: classification metrics (positive class = dropout; threshold chosen on validation to maximise F1)\n"]
    for tag in TAGS:
        s = R[(R.split == tag) & (R["mode"] == "val_threshold")].drop(columns=["split", "mode"]).round(4)
        if len(s):
            L += [f"## {tag}\n", s.to_markdown(index=False), ""]
    s = R[R["mode"] == "threshold_0.5"].pivot_table(index=["model", "input"], columns="split", values="f1_macro").round(4)
    L += ["## Macro-F1 at threshold 0.5 (reference)\n", s.reindex(columns=[t for t in TAGS if t in s.columns]).to_markdown(), ""]
    if confusion:
        L += ["## Confusion matrix, LightGBM, official split (validation threshold)\n", pd.DataFrame(confusion).to_markdown(index=False)]
    open(os.path.join(a.out, "metrics_e6.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    p.add_argument("--splits", default="official,0.5,0.6,0.7")
    p.add_argument("--parts", default="ablation,window,inductive")
    p.add_argument("--only", default="", help="comma-separated variant names to run (default: all variants of the selected parts)")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--n_jobs", type=int, default=16)
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p = sub.add_parser("metrics")
    p.add_argument("--e4", default="runs_e4")
    p.add_argument("--e5", default="runs_e5")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "summarize": summarize, "metrics": metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
