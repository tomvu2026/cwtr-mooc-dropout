"""CWTR: Course-aligned Weekly/daily Temporal Representation features (tabular summaries of the daily matrix) and the
loader that assembles base features + temporal features + splits."""
import os

import numpy as np
import pandas as pd

from .data import build, val_split

# Channel groups of the daily matrix. Channels: 0-4 video, 5-10 problems, 11-14 forum, 15-19 clicks,
# 20 close_courseware, 21 close_forum, 22 number of sessions.
# The 20 weekly count features (4 action groups x 5 weeks, course-aligned) inside the CWTR table.
WEEKLY = [f"t_{g}_w{w}" for g in ("video", "problem", "forum", "click") for w in range(5)]
CHANNEL_GROUPS = {"video": list(range(0, 5)), "problem": list(range(5, 11)), "forum": list(range(11, 15)), "click": list(range(15, 20)) + [20, 21]}


def temporal_table(S):
    """50 CWTR features from a daily matrix S (N x days x channels, last channel = sessions). Columns are prefixed 't_'."""
    S = S.astype(np.float32)
    n, D, C = S.shape
    ev, ses = S[:, :, :C - 1], S[:, :, C - 1]
    tot = ev.sum(2)
    act = tot > 0
    days = np.arange(D)[None, :]
    has = act.any(1)
    f = {}
    # presence: how many days, when first / last, how long ago
    f["n_active"] = act.sum(1)
    f["first_day"] = np.where(has, np.argmax(act, 1), D)
    last = np.where(has, D - 1 - np.argmax(act[:, ::-1], 1), -1)
    f["last_day"] = last
    f["recency"] = D - 1 - last
    f["span"] = last - f["first_day"]
    # gaps between active days
    prev, max_gap, sum_gap = np.full(n, -1.0), np.zeros(n), np.zeros(n)
    for d in range(D):
        a = act[:, d]
        gap = np.where(a & (prev >= 0), d - prev, 0)
        max_gap, sum_gap = np.maximum(max_gap, gap), sum_gap + gap
        prev = np.where(a, d, prev)
    f["max_gap"] = max_gap
    f["mean_gap"] = sum_gap / np.maximum(f["n_active"] - 1, 1)
    # recent vs early activity
    L = np.log1p(tot)
    for name, sl in (("last7", slice(D - 7, D)), ("last14", slice(D - 14, D)), ("first7", slice(0, 7))):
        f[f"ev_{name}"] = np.log1p(tot[:, sl].sum(1))
        f[f"active_{name}"] = act[:, sl].sum(1)
    T = tot.sum(1)
    f["share_last7"] = tot[:, D - 7:].sum(1) / (T + 1)
    f["share_last14"] = tot[:, D - 14:].sum(1) / (T + 1)
    # dynamics
    dc = days - days.mean()
    f["slope"] = (L * dc).sum(1) / (dc ** 2).sum()
    f["daily_std"] = L.std(1)
    f["daily_max"] = L.max(1)
    # sessions
    f["sessions"] = np.log1p(ses.sum(1))
    f["sessions_last7"] = np.log1p(ses[:, D - 7:].sum(1))
    f["ev_per_session"] = T / (ses.sum(1) + 1)
    # per-group weekly counts (5 weeks), last active day, active days
    weeks = [slice(0, 7), slice(7, 14), slice(14, 21), slice(21, 28), slice(28, D)]
    for g, idx in CHANNEL_GROUPS.items():
        x = ev[:, :, [i for i in idx if i < C - 1]].sum(2)
        for w, sl in enumerate(weeks):
            f[f"{g}_w{w}"] = np.log1p(x[:, sl].sum(1))
        f[f"{g}_lastday"] = np.log1p(x[np.arange(n), np.maximum(last, 0)])
        f[f"{g}_days"] = (x > 0).sum(1)
    correct, incorrect = ev[:, :, 9].sum(1), ev[:, :, 10].sum(1)
    f["correct_ratio"] = correct / (correct + incorrect + 1)
    return pd.DataFrame(f).astype(np.float32).add_prefix("t_")


def load_all(data, cluster, split="official", fix=True):
    """Base features (build(agg='all')) + daily matrices + CWTR table + split indices.

    split='official': the official train / test split; 20% of train is held out as validation (seed 0).
    split='temporal': courses ordered by start date, first 60% of enrollments = train, next 10% = validation, rest = test.
    Returns the dict of ``build`` plus S (log-scaled daily matrices), T (CWTR table), idx, course and user arrays.
    """
    D = build(data, cluster, "all", fix)
    n = len(D["y"])
    S = np.load(os.path.join(data, "seq.npy"))
    assert len(S) == n
    T = temporal_table(S)
    T.index = D["flat"].index
    Sn = np.log1p(S.astype(np.float32))
    base = pd.read_pickle(os.path.join(data, "base_counts.pkl"))
    if split == "official":
        tr, te = np.where(D["train"])[0], np.where(~D["train"])[0]
        i, j = val_split(len(tr), 0)
        tr, va = tr[i], tr[j]
    elif split == "temporal":
        ci = pd.read_csv(os.path.join(data, "course_info.csv")).drop_duplicates("course_id").set_index("course_id")
        start = pd.to_datetime(ci["start"].reindex(base["course_id"]).values, errors="coerce")
        g = pd.DataFrame({"c": base["course_id"].values, "s": start}).groupby("c").agg(s=("s", "first"), n=("s", "size")).sort_values("s")
        cum = g["n"].cumsum() / g["n"].sum()
        part = np.where(cum <= 0.6, "train", np.where(cum <= 0.7, "val", "test"))
        p = pd.Series(part, index=g.index).reindex(base["course_id"]).values
        tr, va, te = (np.where(p == k)[0] for k in ("train", "val", "test"))
    else:
        raise ValueError(split)
    sd = Sn[tr].reshape(-1, Sn.shape[2]).std(0)
    sd[sd == 0] = 1
    Sn = Sn / sd
    D.update(S=Sn.astype(np.float32), T=T, idx={"train": tr, "val": va, "test": te},
             course=base["course_id"].values, user=base["username"].values)
    return D
