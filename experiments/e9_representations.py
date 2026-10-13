#!/usr/bin/env python3
"""E9: CWTR against other temporal representations.

Every variant = base features (learner / course statistics from the training pool only, as in E7/E8) + one temporal representation.
No course-level normalisation, so that only the representations are compared.

  base              base features only (144)
  weekly_course     weekly counts aligned to the COURSE START (4 action groups x 5 weeks = 20)        ~ weekly features in the style of Kloft et al.
  weekly_learner    the same 20 features with weeks counted from the learner's FIRST ACTIVE DAY
  weekly_both       both (40)
  cwtr              full CWTR, course-aligned (50)
  cwtr_learner      the 50 CWTR features computed on the daily matrix shifted so that the first active day is day 0
  cumulative        cumulative windows in the spirit of Zhang & Duan (NOT a faithful re-implementation of their definition): 5 cumulative windows
                    (days 0-6, 0-13, 0-20, 0-27, 0-35) x (log events of the 4 action groups + active days) = 25, plus 6 weekly trend features = 31
  cwtr_cumulative   cwtr + cumulative (81)

Models: LightGBM (parameters from E6 params_<split>.json, 3 seeds) and logistic regression (C chosen on validation). Splits: official, cut50/60/70.

  python -m experiments.e9_representations run       --data D --out runs_e9 --e6 runs_e6
  python -m experiments.e9_representations summarize --out runs_e9
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cwtr.build_helpers import train_only_base
from cwtr.cli import add_data_args
from cwtr.splits import make_split
from cwtr.stats import boot, mean_pred, wins
from cwtr.tab import fit_gbdt, run_tab
from cwtr.temporal import CHANNEL_GROUPS, WEEKLY, load_all, temporal_table

warnings.filterwarnings("ignore")
VARIANTS = ("base", "weekly_course", "weekly_learner", "weekly_both", "cwtr", "cwtr_learner", "cumulative", "cwtr_cumulative")
WEEKS = [slice(0, 7), slice(7, 14), slice(14, 21), slice(21, 28), slice(28, 36)]
NAMES = {"base": "Base", "weekly_course": "Weekly, course-aligned (Kloft-style)", "weekly_learner": "Weekly, learner-aligned",
         "weekly_both": "Weekly, both alignments", "cwtr": "CWTR (course-aligned)", "cwtr_learner": "CWTR features, learner-aligned",
         "cumulative": "Cumulative windows (Zhang&Duan-style)", "cwtr_cumulative": "CWTR + cumulative"}
PAIRS = [("weekly_course", "weekly_learner", "Course-aligned vs learner-aligned (weekly counts, 20 vs 20)"),
         ("cwtr", "cwtr_learner", "Course-aligned vs learner-aligned (CWTR, 50 vs 50)"),
         ("weekly_both", "weekly_course", "Adding learner-aligned weekly counts to course-aligned ones"),
         ("cwtr", "cumulative", "CWTR vs cumulative windows"),
         ("cwtr", "weekly_course", "CWTR vs course-aligned weekly counts"),
         ("cwtr_cumulative", "cwtr", "Adding cumulative windows to CWTR"),
         ("weekly_course", "base", "Course-aligned weekly counts vs Base"),
         ("weekly_learner", "base", "Learner-aligned weekly counts vs Base"),
         ("cumulative", "base", "Cumulative windows vs Base")]
TAGS = ("official", "cut50", "cut60", "cut70")


def shift_to_first_activity(S):
    """Shift every daily matrix so that the enrollment's first active day becomes day 0 (the tail is zero-filled).
    Enrollments without any activity are left unchanged (all zeros). Returns (shifted matrices, first active day)."""
    n, D, C = S.shape
    active = S[:, :, :C - 1].sum(2) > 0
    first = np.where(active.any(1), np.argmax(active, 1), 0)
    j = np.arange(D)[None, :] + first[:, None]
    out = S[np.arange(n)[:, None], np.minimum(j, D - 1)]
    out[~(j < D)] = 0
    return out, first


def weekly_counts(S, prefix):
    ev = S[:, :, :S.shape[2] - 1].astype(np.float32)
    f = {}
    for g, idx in CHANNEL_GROUPS.items():
        x = ev[:, :, [i for i in idx if i < ev.shape[2]]].sum(2)
        for w, sl in enumerate(WEEKS):
            f[f"{prefix}_{g}_w{w}"] = np.log1p(x[:, sl].sum(1))
    return pd.DataFrame(f).astype(np.float32)


def cumulative_table(S):
    ev = S[:, :, :S.shape[2] - 1].astype(np.float32)
    total = ev.sum(2)
    active = total > 0
    grp = {g: ev[:, :, [i for i in idx if i < ev.shape[2]]].sum(2) for g, idx in CHANNEL_GROUPS.items()}
    f = {}
    for k, end in enumerate((7, 14, 21, 28, 36)):
        for g, x in grp.items():
            f[f"cum{k}_{g}"] = np.log1p(x[:, :end].sum(1))
        f[f"cum{k}_active"] = active[:, :end].sum(1)
    week_events = np.stack([np.log1p(total[:, sl].sum(1)) for sl in WEEKS], 1)
    week_active = np.stack([active[:, sl].sum(1) for sl in WEEKS], 1).astype(np.float32)
    t = np.arange(5) - 2.0
    f["trend_slope_events"] = (week_events * t).sum(1) / (t ** 2).sum()
    f["trend_slope_active"] = (week_active * t).sum(1) / (t ** 2).sum()
    f["trend_ratio_events"] = week_events[:, 4] / (week_events[:, :4].mean(1) + 1)
    f["trend_ratio_active"] = week_active[:, 4] / (week_active[:, :4].mean(1) + 1)
    on = week_active > 0
    f["trend_weeks_active"] = on.sum(1)
    run, longest = np.zeros(len(on)), np.zeros(len(on))
    for w in range(5):
        run = np.where(on[:, w], 0, run + 1)
        longest = np.maximum(longest, run)
    f["trend_max_inactive_weeks"] = longest
    return pd.DataFrame(f).astype(np.float32)


def save(a, res, y, idx, tag, variant, model, n_feat, seed, pv, pt, info, t0):
    stem = os.path.join(a.out, "preds", f"{tag}_{variant}_{model}_s{seed}")
    np.save(stem + "_val.npy", pv.astype(np.float32))
    np.save(stem + "_test.npy", pt.astype(np.float32))
    row = {"split": tag, "variant": variant, "model": model, "seed": seed, "n_feat": n_feat, "auc_val": roc_auc_score(y[idx["val"]], pv),
           "auc_test": roc_auc_score(y[idx["test"]], pt), "info": info, "sec": time.time() - t0}
    pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
    print(f"  {variant:<16} {model:<4} seed={seed} n={n_feat} val={row['auc_val']:.4f} test={row['auc_test']:.4f}", flush=True)


def run(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_e9.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["split"], d["variant"], d["model"], d["seed"]))
    D = load_all(a.data, a.cluster, "official")
    y = D["y"].astype(int)
    code = pd.factorize(D["course"])[0]
    T = D["T"]
    index = T.index
    S = np.load(os.path.join(a.data, "seq.npy"))
    shifted, first = shift_to_first_activity(S)
    print(f"first active day: median {np.median(first):.0f}, share starting after day 6: {(first > 6).mean():.3f}", flush=True)
    R = {"weekly_course": T[WEEKLY].copy(), "weekly_learner": weekly_counts(shifted, "lw"),
         "cwtr_learner": temporal_table(shifted).add_prefix("l"), "cumulative": cumulative_table(S)}
    for v in R.values():
        v.index = index
    corr = np.corrcoef(weekly_counts(S, "cw").values.ravel(), R["weekly_course"].values.ravel())[0, 1]
    print(f"check: weekly counts recomputed vs CWTR weekly features, correlation = {corr:.6f}", flush=True)
    del S, shifted
    for k, v in R.items():
        print(f"  {k}: {v.shape[1]} features", flush=True)

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
        params = json.load(open(os.path.join(a.e6, f"params_{tag}.json")))
        ref = np.zeros(len(y), bool)
        ref[np.concatenate([tr, va])] = True
        base = train_only_base(a.data, a.cluster, ref)
        base.index = index
        make = {"base": lambda: base,
                "weekly_course": lambda: pd.concat([base, R["weekly_course"]], axis=1),
                "weekly_learner": lambda: pd.concat([base, R["weekly_learner"]], axis=1),
                "weekly_both": lambda: pd.concat([base, R["weekly_course"], R["weekly_learner"]], axis=1),
                "cwtr": lambda: pd.concat([base, T], axis=1),
                "cwtr_learner": lambda: pd.concat([base, R["cwtr_learner"]], axis=1),
                "cumulative": lambda: pd.concat([base, R["cumulative"]], axis=1),
                "cwtr_cumulative": lambda: pd.concat([base, T, R["cumulative"]], axis=1)}
        for v in a.variants.split(","):
            X = None
            for model in a.models.split(","):
                n_exp = 1 if model == "lr" else a.n_seeds
                if all((tag, v, model, s) in done for s in range(n_exp)):
                    continue
                if X is None:
                    X = make[v]().values.astype(np.float32)
                t0 = time.time()
                if model == "lgbm":
                    for seed in range(a.n_seeds):
                        if (tag, v, model, seed) in done:
                            continue
                        m, it = fit_gbdt("lgbm", params, seed, X[tr], y[tr], X[va], y[va], a)
                        save(a, res, y, idx, tag, v, model, X.shape[1], seed, m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1], f"iters={it}", t0)
                        t0 = time.time()
                else:
                    for seed, pv, pt, inf, _ in run_tab(model, X, y, idx, a):
                        save(a, res, y, idx, tag, v, model, X.shape[1], seed, pv, pt, ";".join(f"{k}={x}" for k, x in inf.items()), t0)


def summarize(a):
    R = pd.read_csv(os.path.join(a.out, "results_e9.csv"))
    tags = [t for t in TAGS if t in set(R.split)]
    L = ["# E9: CWTR against other temporal representations\n",
         "Each variant = base features (statistics from the training pool only) + one temporal representation; no course normalisation.\n"]
    for model in sorted(R.model.unique()):
        table, diffs = {}, {}
        for t in tags:
            y = np.load(os.path.join(a.out, "preds", f"{t}_y_test.npy"))
            c = np.load(os.path.join(a.out, "preds", f"{t}_course_test.npy"))
            P = {v: mean_pred(a.out, t, f"{v}_{model}") for v in NAMES}
            P = {k: v for k, v in P.items() if v is not None}
            n_feat = R[(R.split == t) & (R.model == model)].groupby("variant").n_feat.first()
            for v, p in P.items():
                table.setdefault(f"{NAMES[v]} ({n_feat.get(v, '?')})", {})[t] = round(roc_auc_score(y, p), 4)
            for X, Y, label in PAIRS:
                if X in P and Y in P:
                    d = roc_auc_score(y, P[X]) - roc_auc_score(y, P[Y])
                    official = t == "official"
                    lo, hi = boot(y, P[X], P[Y], None if official else c, 1000 if official else 500)
                    diffs.setdefault(label, {})[t] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if official else f"; {wins(y, c, P[X], P[Y])}")
        ix = lambda d: pd.DataFrame(d).T.reindex(columns=tags)
        L += [f"## {model}: test AUC (number of features in brackets)\n", ix(table).to_markdown(), "",
              f"## {model}: AUC difference [95% CI]; courses won\n", ix(diffs).to_markdown(), ""]
    L += ["## Seeds run\n", R.groupby(["split", "variant", "model"]).seed.nunique().unstack("model").to_markdown(), ""]
    open(os.path.join(a.out, "summary_e9.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    add_data_args(p)
    p.add_argument("--e6", default="runs_e6", help="E6 output directory (params_<split>.json)")
    p.add_argument("--splits", default="official,0.5,0.6,0.7")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--models", default="lgbm,lr")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--n_seeds_rf", type=int, default=3)
    p.add_argument("--n_trials", type=int, default=12)
    p.add_argument("--n_jobs", type=int, default=16)
    p.add_argument("--n_estimators", type=int, default=5000)
    p.add_argument("--xgb_device", default="cpu")
    p.add_argument("--cat_task", default="CPU")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
