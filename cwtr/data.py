"""Base features for the XuetangX dropout task (follows the feature pipeline of the public CFIN code).

``build`` returns, for every enrollment, the 115 activity features (23 count features x 5 views: the enrollment itself and
the mean / max over the learner's and over the course's enrollments), the context features (age, gender, user cluster,
course category, enrollment counts) and the label.

fix=False keeps two bugs of the original code (gender and course category are always 0); fix=True repairs them.
agg="all"   learner / course statistics are computed on train + test (as in the original code).
agg="train" statistics use only the rows flagged by ``train_mask`` (default: the official train split); a test row sees
            those rows plus itself.
"""
import os
import pickle

import numpy as np
import pandas as pd

VIDEO = ["seek_video", "play_video", "pause_video", "stop_video", "load_video"]
PROBLEM = ["problem_get", "problem_check", "problem_save", "reset_problem", "problem_check_correct", "problem_check_incorrect"]
FORUM = ["create_thread", "create_comment", "delete_thread", "delete_comment"]
CLICK = ["click_info", "click_courseware", "click_about", "click_forum", "click_progress"]
ACTIONS = VIDEO + PROBLEM + FORUM + CLICK + ["close_courseware"]      # 21 actions counted by the original code
ACT = ["all#count", "session#count"] + [a + "#num" for a in ACTIONS]  # 23 count features per enrollment
PAPER_AUC = {"LR": 0.8223, "SVM": 0.8286, "RF": 0.8311, "GBDT": 0.8518, "DNN": 0.8564, "CFIN": 0.8640, "CFIN-en": 0.8671}


def _group_stats(df, key, train, mode):
    """Mean and max of the 23 count features grouped by ``key`` (same index as ``df``)."""
    X = df[ACT]
    if mode == "all":
        g = X.groupby(df[key])
        return g.transform("mean"), g.transform("max")
    g = X[train].groupby(df.loc[train, key])
    k = df[key]
    s = g.sum().reindex(k).fillna(0).values
    n = g.size().reindex(k).fillna(0).values
    m = g.max().reindex(k).fillna(-np.inf).values
    test = (~train)[:, None]
    own = X.values
    mean = np.where(test, (s + own) / (n[:, None] + 1), s / np.maximum(n[:, None], 1))
    mx = np.where(test, np.maximum(m, own), m)
    return pd.DataFrame(mean, index=df.index, columns=ACT), pd.DataFrame(mx, index=df.index, columns=ACT)


def _group_sizes(df, key, train, mode):
    """Number of enrollments per ``key`` group (train-only mode: train rows of the group, plus the row itself for test rows)."""
    if mode == "all":
        return df.groupby(key)[ACT[0]].transform("size").values
    n = df.loc[train, key].value_counts().reindex(df[key]).fillna(0).values
    return n + (~train)


def _encode_fields(df, categorical, numeric):
    """Sparse-field encoding used by the CFIN context branch: (feature index, feature value, total dimension)."""
    idx, val, dim = [], [], 0
    for f in numeric:
        idx.append(np.full(len(df), dim))
        val.append(df[f].values.astype(float))
        dim += 1
    for f in categorical:
        codes = pd.factorize(df[f].values)[0]
        idx.append(codes + dim)
        val.append(np.ones(len(df)))
        dim += codes.max() + 1
    return np.stack(idx, 1).astype(np.int64), np.stack(val, 1).astype(np.float32), int(dim)


def build(data_dir, cluster_dir, agg="all", fix=False, train_mask=None):
    df = pd.read_pickle(os.path.join(data_dir, "base_counts.pkl")).copy()
    train = (df["split"] == "train").values if train_mask is None else np.asarray(train_mask, dtype=bool)

    ui = pd.read_csv(os.path.join(data_dir, "user_info.csv")).drop_duplicates("user_id").set_index("user_id")
    age = 2018 - ui["birth"].reindex(df["username"]).values
    df["age"] = np.where(np.isnan(age) | (age > 70) | (age < 10), 0, age)
    gender = ui["gender"].reindex(df["username"])
    # The original code compares with 'm'/'f' while the file stores 'male'/'female', so gender is always 0.
    gmap = {"male": 1, "female": 2, "m": 1, "f": 2} if fix else {"m": 1, "f": 2}
    df["gender"] = gender.map(gmap).fillna(0).astype(int).values

    user_dict = pickle.load(open(os.path.join(cluster_dir, "user_dict"), "rb"), encoding="latin1")
    labels = np.load(os.path.join(cluster_dir, "label_5_10time.npy"))
    row = df["username"].map(user_dict)
    df["cluster_label"] = np.where(row.isna(), -1, labels[row.fillna(0).astype(int).values])
    miss_cluster = float(row.isna().mean())

    if fix:   # the original code indexes by the numeric 'id' column with a string course_id, so the category is always 0
        ci = pd.read_csv(os.path.join(data_dir, "course_info.csv")).drop_duplicates("course_id").set_index("course_id")
        df["course_category"] = pd.factorize(ci["category"].reindex(df["course_id"]).fillna("__none__").values)[0]
    else:
        df["course_category"] = 0

    df["user_enroll_num"] = _group_sizes(df, "username", train, agg)
    df["course_enroll_num"] = _group_sizes(df, "course_id", train, agg)
    numeric = ACT + ["age", "course_enroll_num", "user_enroll_num"]
    # Standardise first, then compute the learner / course mean and max (order of the original code).
    ref = df[numeric] if agg == "all" else df.loc[train, numeric]
    mu, sd = ref.mean(), ref.std(ddof=0).replace(0, 1)
    df[numeric] = (df[numeric] - mu) / sd
    um, ux = _group_stats(df, "username", train, agg)
    cm, cx = _group_stats(df, "course_id", train, agg)
    A = np.stack([df[ACT].values, um.values, ux.values, cm.values, cx.values], axis=2).reshape(len(df), -1)   # 23 groups x 5 views

    UI, UV, u_dim = _encode_fields(df, ["gender", "cluster_label"], ["age", "user_enroll_num"])
    CI, CV, c_dim = _encode_fields(df, ["course_category"], ["course_enroll_num"])
    flat = pd.concat([pd.DataFrame(A, index=df.index, columns=[f"a{i}" for i in range(A.shape[1])]),
                      df[["age", "user_enroll_num", "course_enroll_num"]],
                      pd.get_dummies(df[["gender", "cluster_label", "course_category"]].astype(str), dtype=float)], axis=1)
    info = {"n_train": int(train.sum()), "n_test": int((~train).sum()), "dropout_train": float(df.loc[train, "truth"].mean()),
            "dropout_test": float(df.loc[~train, "truth"].mean()), "miss_cluster": miss_cluster,
            "gender_nonzero": float((df["gender"] > 0).mean()), "category_nonzero": float((df["course_category"] > 0).mean()),
            "age_nonzero": float((age == age).mean())}
    return {"A": A.astype(np.float32), "UI": UI, "UV": UV, "CI": CI, "CV": CV, "u_dim": u_dim, "c_dim": c_dim,
            "y": df["truth"].values.astype(np.float32), "train": train, "ids": df.index.values, "flat": flat, "info": info}


def val_split(n_train, seed, frac=0.2):
    """Random split of the training rows into (fit, validation) index arrays."""
    p = np.random.default_rng(seed).permutation(n_train)
    k = int(n_train * (1 - frac))
    return np.sort(p[:k]), np.sort(p[k:])
