"""Splits by course start date ("unseen courses"): the courses that start last are the test set."""
import os

import numpy as np
import pandas as pd


def make_split(D, data, cut, val_frac=0.15):
    """Courses sorted by start date; the first ``cut`` fraction of enrollments is the training pool, the later courses are the test set.
    Validation = randomly chosen courses of the training pool (~``val_frac`` of its enrollments), so it never overlaps training courses.

    Returns ({'train','val','test'} -> row indices, info dict)."""
    base = pd.read_pickle(os.path.join(data, "base_counts.pkl"))
    ci = pd.read_csv(os.path.join(data, "course_info.csv")).drop_duplicates("course_id").set_index("course_id")
    start = pd.to_datetime(ci["start"].reindex(base["course_id"]).values, errors="coerce")
    g = (pd.DataFrame({"c": base["course_id"].values, "s": start}).groupby("c").agg(s=("s", "first"), n=("s", "size"))
         .sort_values("s", kind="stable"))
    cum = g["n"].cumsum() / g["n"].sum()
    train_courses, test_courses = g.index[cum <= cut].values, g.index[cum > cut].values
    perm = np.random.default_rng(0).permutation(train_courses)
    share = g.loc[perm, "n"].cumsum() / g.loc[train_courses, "n"].sum()
    val_courses = set(perm[(share.shift(fill_value=0) < val_frac).values])      # add courses until ~val_frac is reached
    c = base["course_id"].values
    is_test, is_val = np.isin(c, test_courses), np.isin(c, list(val_courses))
    info = {"cut_date": str(g.loc[train_courses, "s"].max()), "courses_train": len(train_courses) - len(val_courses),
            "courses_val": len(val_courses), "courses_test": len(test_courses)}
    return {"train": np.where(~is_test & ~is_val)[0], "val": np.where(is_val)[0], "test": np.where(is_test)[0]}, info
