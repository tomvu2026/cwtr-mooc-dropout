"""Base features recomputed from a (possibly truncated) daily matrix, and train-only base features."""
import numpy as np
import pandas as pd


def base_from_seq(Sw, user, course, context):
    """115 base activity features + context, recomputed from the first W days of the daily matrix Sw (N x W x channels).

    Same construction as ``cwtr.data.build`` (standardised own value, learner mean/max, course mean/max over all enrollments).
    The session count is approximated by the number of events (the original code counts log rows that carry a session id).
    ``context`` is the context part of the base feature table (columns not named a<digits>)."""
    ev = Sw[:, :, :22].sum(1).astype(np.float64)
    total = ev.sum(1, keepdims=True)
    C = np.concatenate([total, total, ev[:, :21]], 1)
    C = (C - C.mean(0)) / np.where(C.std(0) == 0, 1, C.std(0))
    Cd = pd.DataFrame(C)
    by_user, by_course = Cd.groupby(user), Cd.groupby(course)
    A = np.stack([C, by_user.transform("mean").values, by_user.transform("max").values,
                  by_course.transform("mean").values, by_course.transform("max").values], 2).reshape(len(C), -1)
    return pd.concat([pd.DataFrame(A, index=context.index, columns=[f"a{i}" for i in range(A.shape[1])]), context], axis=1).astype(np.float32)


def train_only_base(data, cluster, ref_mask):
    """Base feature table whose learner / course statistics (mean, max, enrollment counts, standardisation) use only the rows with
    ref_mask == True (plus the row itself for the other rows). Used by the leakage audits on the splits by course start date."""
    from .data import build
    return build(data, cluster, "train", True, train_mask=ref_mask)["flat"]
