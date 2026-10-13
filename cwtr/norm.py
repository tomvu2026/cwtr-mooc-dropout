"""Course-level normalisation of the temporal features ("+N" inputs).

Only activity inside the observation window is used (never labels). Three flavours:
  course_norm           per-course z-scored daily channels + 57 tabular features (rank inside the course, course means)
  tab_norm              only the 57 tabular features
  tab_norm_inductive    like tab_norm, but the reference distribution of a course uses only the rows flagged in ``ref`` (training rows)
"""
import numpy as np
import pandas as pd

# Seven key CWTR features whose course-level mean is added to the tabular normalised block.
KEY_T = ["t_n_active", "t_recency", "t_active_last14", "t_share_last14", "t_sessions", "t_ev_last7", "t_span"]


def course_norm(S_raw, code, T):
    """S_raw: raw daily matrices (N x 36 x 23); code: integer course id per row; T: CWTR table.
    Returns (extra daily channels, 57 normalised tabular features).

    The extra channels (25 = 23 z-scored channels + share of the course's enrollments active that day + own activity flag)
    are concatenated to the log-scaled daily matrix for the sequence models."""
    L = np.log1p(S_raw.astype(np.float32))
    n, D, C = L.shape
    K = code.max() + 1
    cnt = np.bincount(code, minlength=K).astype(np.float32)
    s1 = np.zeros((K, D, C), np.float32)
    np.add.at(s1, code, L)
    mean_cd = s1 / cnt[:, None, None]
    s2 = np.zeros((K, C), np.float32)
    np.add.at(s2, code, (L ** 2).sum(1))
    m_c = s1.sum(1) / (cnt[:, None] * D)
    std_c = np.sqrt(np.maximum(s2 / (cnt[:, None] * D) - m_c ** 2, 0))
    z = np.clip((L - mean_cd[code]) / (std_c[code][:, None, :] + 0.05), -10, 10)
    act = (S_raw[:, :, :C - 1].sum(2) > 0).astype(np.float32)
    a1 = np.zeros((K, D), np.float32)
    np.add.at(a1, code, act)
    frac = (a1 / cnt[:, None])[code]
    extra = np.concatenate([z, frac[:, :, None], act[:, :, None]], 2)
    g = T.groupby(code)
    ranks = g.rank(pct=True).add_suffix("_rk").astype(np.float32)
    means = g[KEY_T].transform("mean").add_suffix("_cmean").astype(np.float32)
    return extra, pd.concat([ranks, means], axis=1)


def tab_norm(T, code):
    """57 course-normalised tabular features: percentile rank inside the course for all 50 CWTR features + course mean of 7 key features."""
    g = T.groupby(code)
    return pd.concat([g.rank(pct=True).add_suffix("_rk"), g[KEY_T].transform("mean").add_suffix("_cmean")], axis=1).astype(np.float32)


def tab_norm_inductive(T, code, ref):
    """Same output as ``tab_norm`` but each course's reference distribution is built only from the rows with ref[row] == True."""
    V = T.values.astype(np.float64)
    R = np.full(V.shape, 0.5)
    M = np.zeros((len(T), len(KEY_T)))
    key_cols = [T.columns.get_loc(k) for k in KEY_T]
    for c in np.unique(code):
        rows = np.where(code == c)[0]
        base = V[rows[ref[rows]]]
        if len(base) == 0:
            continue
        for j in range(V.shape[1]):
            s = np.sort(base[:, j])
            x = V[rows, j]
            R[rows, j] = (np.searchsorted(s, x, "left") + np.searchsorted(s, x, "right")) / (2 * len(s))
        M[rows] = base[:, key_cols].mean(0)
    return pd.concat([pd.DataFrame(R, index=T.index, columns=[f"{c}_rk" for c in T.columns]),
                      pd.DataFrame(M, index=T.index, columns=[f"{k}_cmean" for k in KEY_T])], axis=1).astype(np.float32)
