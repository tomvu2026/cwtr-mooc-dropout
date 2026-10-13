#!/usr/bin/env python3
"""E1: replication of CFIN (AAAI 2019) on XuetangX, with classical baselines.

Branch A (as in the authors' code): train on the full training set for a fixed 27 epochs, learner/course statistics from train+test.
Branch B (strict): 20% of train is validation, early stopping, test touched once; 10 seeds; statistics on train+test ("all") or train only.
fix=0 keeps two bugs of the original code (gender and course category always 0); fix=1 repairs them.

  python -m experiments.e1_cfin_replication cfin       --data D --out runs_e1 [--device cuda]
  python -m experiments.e1_cfin_replication baselines  --data D --out runs_e1
  python -m experiments.e1_cfin_replication summarize  --out runs_e1
Re-running a command skips the configurations already in the results CSV.
"""
import argparse
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from cwtr.data import PAPER_AUC, build, val_split
from cwtr.nets import CFIN
from cwtr.stats import boot

warnings.filterwarnings("ignore")
KEYS = ("A", "UI", "UV", "CI", "CV")


@torch.no_grad()
def predict(model, T, idx, bs=8192):
    model.eval()
    return torch.cat([torch.sigmoid(model(*(T[k][idx[i:i + bs]] for k in KEYS))).float().cpu() for i in range(0, len(idx), bs)]).numpy()


def train_cfin(D, T, train_idx, eval_sets, a, seed, epochs, patience=None):
    """eval_sets: name -> row indices scored every epoch. patience=None: fixed epochs; else early stopping on eval_sets['val']."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CFIN(D["A"].shape[1], D["u_dim"], D["c_dim"], D["UI"].shape[1], D["CI"].shape[1], use_attn=not a.no_attn).to(a.device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    y, history, best = T["y"], [], (-1, None, 0)
    tr = torch.as_tensor(train_idx, device=a.device)
    for ep in range(1, epochs + 1):
        m.train()
        t0 = time.time()
        perm = tr[torch.randperm(len(tr), device=a.device)]
        for i in range(len(perm) // a.batch):                # the original code drops the last incomplete batch
            j = perm[i * a.batch:(i + 1) * a.batch]
            loss = F.binary_cross_entropy_with_logits(m(*(T[k][j] for k in KEYS)), y[j]) + a.l2 * m.l2()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scores = {k: float(roc_auc_score(D["y"][v], predict(m, T, torch.as_tensor(v, device=a.device)))) for k, v in eval_sets.items()}
        history.append(scores)
        print(f"    epoch {ep:>2} " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        if patience is not None:
            if scores["val"] > best[0]:
                best = (scores["val"], {k: v.detach().clone() for k, v in m.state_dict().items()}, ep)
            elif ep - best[2] >= patience:
                break
    if patience is not None:
        m.load_state_dict(best[1])
    return m, history, (best[2] if patience is not None else epochs)


def run_cfin(a):
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_cfin.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["mode"], d["agg"], d["fix"], d["seed"]))
    print("device:", a.device, flush=True)
    for agg in a.aggs.split(","):
        for fix in [int(x) for x in a.fixes.split(",")]:
            runs = [("A", int(s)) for s in a.seeds_A.split(",") if "A" in a.modes and agg == "all"] + \
                   [("B", s) for s in range(a.n_seeds_B) if "B" in a.modes]        # branch A is only meaningful with agg=all (the original code)
            runs = [r for r in runs if (r[0], agg, fix, r[1]) not in done]
            if not runs:
                continue
            D = build(a.data, a.cluster, agg, bool(fix))
            tr, te = np.where(D["train"])[0], np.where(~D["train"])[0]
            T = {k: torch.as_tensor(D[k], device=a.device) for k in KEYS + ("y",)}
            print(f"[agg={agg} fix={fix}] {D['info']}", flush=True)
            np.save(os.path.join(a.out, "preds", "test_ids.npy"), D["ids"][te])
            np.save(os.path.join(a.out, "preds", "test_y.npy"), D["y"][te])
            for mode, seed in runs:
                print(f"  -- branch {mode} seed {seed}", flush=True)
                t0 = time.time()
                if mode == "A":
                    m, hist, ep = train_cfin(D, T, tr, {"test": te}, a, seed, a.epochs)
                    val_auc = np.nan
                    best_ep = int(np.argmax([h["test"] for h in hist])) + 1
                    best_test = max(h["test"] for h in hist)          # diagnostic only: selecting the epoch on test is leakage
                else:
                    i, j = val_split(len(tr), seed)
                    m, hist, ep = train_cfin(D, T, tr[i], {"val": tr[j]}, a, seed, a.max_epochs_B, a.patience)
                    val_auc, best_ep, best_test = hist[ep - 1]["val"], np.nan, np.nan
                p = predict(m, T, torch.as_tensor(te, device=a.device))
                y = D["y"][te]
                row = {"mode": mode, "agg": agg, "fix": fix, "seed": seed, "epochs": ep, "auc_test": roc_auc_score(y, p), "f1_test": f1_score(y, p >= 0.5),
                       "auc_val": val_auc, "A_best_epoch_on_test": best_ep, "A_auc_at_best_epoch": best_test, "n_params": m.n_params(), "sec": time.time() - t0}
                np.save(os.path.join(a.out, "preds", f"cfin_{mode}_{agg}_fix{fix}_s{seed}.npy"), p.astype(np.float32))
                pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                print(f"  => test AUC={row['auc_test']:.4f} F1={row['f1_test']:.4f} epochs={ep} params={row['n_params']} ({row['sec']:.0f}s)", flush=True)


def run_baselines(a):
    """LR, RF, GBDT (XGBoost), DNN (MLP) as in the CFIN paper's table, plus a tuned LightGBM, on the same features as CFIN."""
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    res = os.path.join(a.out, "results_base.csv")
    done = set()
    if os.path.exists(res):
        d = pd.read_csv(res)
        done = set(zip(d["model"], d["agg"], d["fix"], d["seed"]))
    for agg in a.aggs.split(","):
        for fix in [int(x) for x in a.fixes.split(",")]:
            D = build(a.data, a.cluster, agg, bool(fix))
            X, y = D["flat"].values.astype(np.float32), D["y"].astype(int)
            Xtr, ytr, Xte, yte = X[D["train"]], y[D["train"]], X[~D["train"]], y[~D["train"]]
            print(f"[agg={agg} fix={fix}] train={len(ytr)} test={len(yte)} features={X.shape[1]}", flush=True)
            for model in a.models.split(","):
                for seed in range(a.n_seeds if model in ("dnn", "lgbm_tuned", "rf") else 1):
                    if (model, agg, fix, seed) in done:
                        continue
                    t0, val_auc = time.time(), np.nan
                    if model == "lr":
                        p = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                    elif model == "rf":
                        p = RandomForestClassifier(500, min_samples_leaf=5, n_jobs=a.n_jobs, random_state=seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                    elif model == "gbdt":
                        import xgboost as xgb
                        p = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                              device=a.xgb_device, n_jobs=a.n_jobs, random_state=seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                    elif model == "dnn":
                        p = MLPClassifier((256, 256), early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, max_iter=100,
                                          random_state=seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                    elif model == "lgbm_tuned":
                        i, j = val_split(len(ytr), seed)
                        p, val_auc = tuned_lightgbm(Xtr[i], ytr[i], Xtr[j], ytr[j], Xte, seed, a.n_trials, a.n_jobs)
                    row = {"model": model, "agg": agg, "fix": fix, "seed": seed, "auc_test": roc_auc_score(yte, p), "f1_test": f1_score(yte, p >= 0.5),
                           "auc_val": val_auc, "sec": time.time() - t0}
                    np.save(os.path.join(a.out, "preds", f"{model}_{agg}_fix{fix}_s{seed}.npy"), p.astype(np.float32))
                    pd.DataFrame([row]).to_csv(res, mode="a", header=not os.path.exists(res), index=False)
                    print(f"  {model:<11} seed={seed} AUC={row['auc_test']:.4f} F1={row['f1_test']:.4f} ({row['sec']:.0f}s)", flush=True)


def tuned_lightgbm(Xtr, ytr, Xva, yva, Xte, seed, n_trials, n_jobs):
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    best = (-1, None)
    for _ in range(n_trials):
        p = {"learning_rate": float(rng.choice([0.02, 0.05, 0.1])), "num_leaves": int(rng.integers(15, 255)), "min_child_samples": int(rng.integers(10, 300)),
             "subsample": float(rng.uniform(0.6, 1)), "colsample_bytree": float(rng.uniform(0.4, 1)), "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(30))))}
        m = lgb.LGBMClassifier(n_estimators=4000, subsample_freq=1, random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        score = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        if score > best[0]:
            best = (score, m)
    return best[1].predict_proba(Xte)[:, 1], best[0]


def summarize(a):
    mean_pred = lambda pattern: (lambda f: (np.mean([np.load(x) for x in f], 0), len(f)) if f else (None, 0))(sorted(glob.glob(os.path.join(a.out, "preds", pattern))))
    c = pd.read_csv(os.path.join(a.out, "results_cfin.csv"))
    L = ["# E1: CFIN replication on XuetangX\n", f"Parameters (batch-norm excluded): {int(c.n_params.iloc[0])}\n", "## CFIN\n",
         "Paper: CFIN 0.8640. Pre-registered criterion: branch A, agg=all within 0.864 +/- 0.005 and above GBDT.\n"]
    g = c.groupby(["mode", "agg", "fix"]).agg(n=("seed", "size"), auc_mean=("auc_test", "mean"), auc_sd=("auc_test", "std"), auc_min=("auc_test", "min"),
                                               auc_max=("auc_test", "max"), f1_mean=("f1_test", "mean"), epochs=("epochs", "mean"),
                                               A_best_epoch_on_test=("A_best_epoch_on_test", "mean"), A_auc_if_epoch_chosen_on_test=("A_auc_at_best_epoch", "mean")).round(4)
    L += [g.to_markdown(), ""]
    s = c[(c["mode"] == "A") & (c.seed == 12345)]
    if len(s):
        L += ["Original seed 12345, branch A: " + "; ".join(f"fix={int(r.fix)}: {r.auc_test:.4f}" for r in s.itertuples()), ""]
    base_path = os.path.join(a.out, "results_base.csv")
    if os.path.exists(base_path):
        b = pd.read_csv(base_path)
        paper_name = {"lr": "LR", "rf": "RF", "gbdt": "GBDT", "dnn": "DNN"}
        t = b.groupby(["agg", "fix", "model"]).agg(n=("seed", "size"), auc_mean=("auc_test", "mean"), auc_sd=("auc_test", "std"), f1_mean=("f1_test", "mean")).round(4).reset_index()
        t["paper"] = t["model"].map(lambda m: PAPER_AUC.get(paper_name.get(m, ""), np.nan))
        L += ["## Baselines\n", t.to_markdown(index=False), ""]
        y = np.load(os.path.join(a.out, "preds", "test_y.npy"))
        L += ["## Paired comparison of seed-averaged predictions (bootstrap, 1000 resamples)\n", "| setting | A | B | AUC A | AUC B | diff | 95% CI |", "|---|---|---|---|---|---|---|"]
        for agg in sorted(c["agg"].unique()):
            for fix in sorted(c["fix"].unique()):
                pc, n1 = mean_pred(f"cfin_B_{agg}_fix{fix}_s*.npy")
                for other in ("lgbm_tuned", "gbdt", "dnn"):
                    po, n2 = mean_pred(f"{other}_{agg}_fix{fix}_s*.npy")
                    if pc is None or po is None:
                        continue
                    lo, hi = boot(y, pc, po)
                    d = roc_auc_score(y, pc) - roc_auc_score(y, po)
                    L.append(f"| agg={agg} fix={fix} | CFIN-B ({n1} seeds) | {other} ({n2}) | {roc_auc_score(y, pc):.4f} | {roc_auc_score(y, po):.4f} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    open(os.path.join(a.out, "summary_e1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True)
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--data", required=True)
    data.add_argument("--cluster", default="cfin_authors/cluster")
    data.add_argument("--aggs", default="all,train")
    data.add_argument("--fixes", default="0,1")
    p = sub.add_parser("cfin", parents=[common, data])
    p.add_argument("--modes", default="A,B")
    p.add_argument("--seeds_A", default="12345,1,2,3,4,5")
    p.add_argument("--n_seeds_B", type=int, default=10)
    p.add_argument("--epochs", type=int, default=27)
    p.add_argument("--max_epochs_B", type=int, default=60)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--no_attn", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p = sub.add_parser("baselines", parents=[common, data])
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--models", default="lr,rf,gbdt,dnn,lgbm_tuned")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_jobs", type=int, default=16)
    p.add_argument("--xgb_device", default="cpu")
    sub.add_parser("summarize", parents=[common])
    a = ap.parse_args()
    {"cfin": run_cfin, "baselines": run_baselines, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
