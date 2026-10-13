"""Saved predictions and paired bootstrap comparisons shared by the summary scripts."""
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def load_preds(directory, tag, key, part="test"):
    """List of per-seed prediction arrays saved as <directory>/preds/<tag>_<key>_s<seed>_<part>.npy."""
    files = sorted(glob.glob(os.path.join(directory, "preds", f"{tag}_{key}_s*_{part}.npy")))
    return [np.load(f) for f in files]


def mean_pred(directory, tag, key, part="test"):
    """Seed-averaged prediction or None when nothing was saved."""
    p = load_preds(directory, tag, key, part)
    return np.mean(p, 0) if p else None


def boot(y, p1, p2, groups=None, n=1000):
    """95% bootstrap interval of AUC(p1) - AUC(p2). groups=None resamples enrollments; otherwise resamples whole groups
    (e.g. courses, learners) with replacement."""
    rng = np.random.default_rng(0)
    diffs = []
    if groups is None:
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    else:
        members = [np.where(groups == k)[0] for k in np.unique(groups)]
        for _ in range(n):
            i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
            diffs.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.percentile(diffs, [2.5, 97.5])


def wins(y, courses, p1, p2, min_n=200):
    """'w/t': number of courses (with >= min_n enrollments and both classes) where p1 has the higher AUC, over the number of such courses."""
    w = t = 0
    for k in np.unique(courses):
        m = courses == k
        if m.sum() >= min_n and 0 < y[m].mean() < 1:
            t += 1
            w += roc_auc_score(y[m], p1[m]) > roc_auc_score(y[m], p2[m])
    return f"{w}/{t}"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
