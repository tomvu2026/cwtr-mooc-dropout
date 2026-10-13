"""Step 1: per-enrollment activity counts from the raw click logs (same definition as the original CFIN feature extraction).

    python -m cwtr.counts --data <xuetangx>

<xuetangx> must contain prediction_log/ (or prediction_data.tar), user_info.csv and course_info.csv.
Writes <xuetangx>/base_counts.pkl and <xuetangx>/base_report.json. The logs are read in chunks to bound memory.
"""
import argparse
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd

from .data import ACT, ACTIONS


def count_log(path, chunk):
    parts, enrolls, seen, n, t0 = [], [], {}, 0, time.time()
    for c in pd.read_csv(path, usecols=["enroll_id", "username", "course_id", "session_id", "action", "time"], chunksize=chunk):
        n += len(c)
        g = c.groupby("enroll_id")
        d = pd.DataFrame({"all#count": g["action"].count(), "session#count": g["session_id"].count()})   # rows with a session id
        ct = pd.crosstab(c["enroll_id"], c["action"]).reindex(columns=ACTIONS, fill_value=0)
        ct.columns = [a + "#num" for a in ACTIONS]
        d = d.join(ct, how="left").fillna(0)
        d["t_min"], d["t_max"] = g["time"].min(), g["time"].max()
        parts.append(d)
        enrolls.append(c[["enroll_id", "username", "course_id"]].drop_duplicates("enroll_id"))
        for a, v in c["action"].value_counts().items():
            seen[a] = seen.get(a, 0) + int(v)
        print(f"  {os.path.basename(path)}: {n / 1e6:.0f}M rows ({time.time() - t0:.0f}s)", flush=True)
    d = pd.concat(parts)
    agg = {c: "sum" for c in ACT}
    agg.update(t_min="min", t_max="max")
    d = d.groupby(level=0).agg(agg)
    e = pd.concat(enrolls).drop_duplicates("enroll_id").set_index("enroll_id")
    return d.join(e), seen, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    a = ap.parse_args()
    log_dir = os.path.join(a.data, "prediction_log")
    if not os.path.isdir(log_dir):
        tars = [f for f in ("prediction_data.tar", "prediction_data.tar.gz") if os.path.exists(os.path.join(a.data, f))]
        assert tars, f"neither prediction_log/ nor prediction_data.tar(.gz) found in {a.data}"
        print("extracting", tars[0], flush=True)
        tarfile.open(os.path.join(a.data, tars[0])).extractall(a.data)

    parts, report = [], {}
    for split in ("train", "test"):
        d, seen, n = count_log(os.path.join(log_dir, f"{split}_log.csv"), a.chunk)
        truth = pd.read_csv(os.path.join(log_dir, f"{split}_truth.csv"), index_col="enroll_id")
        report[split] = {"log_rows": n, "enroll_in_log": len(d), "enroll_in_truth": len(truth), "actions": seen,
                         "unknown_actions": sorted(set(seen) - set(ACTIONS))}
        d = d.join(truth, how="inner")
        d["split"] = split
        parts.append(d)
    df = pd.concat(parts)
    assert df.index.is_unique
    tr, te = df[df.split == "train"], df[df.split == "test"]
    span = (pd.to_datetime(df["t_max"]) - pd.to_datetime(df["t_min"])).dt.days
    ci = pd.read_csv(os.path.join(a.data, "course_info.csv")).drop_duplicates("course_id").set_index("course_id")
    start = pd.to_datetime(ci["start"].reindex(df["course_id"]).values, errors="coerce")
    report["summary"] = {
        "n_train": len(tr), "n_test": len(te), "dropout_train": float(tr.truth.mean()), "dropout_test": float(te.truth.mean()),
        "users": int(df.username.nunique()), "courses": int(df.course_id.nunique()),
        "courses_train_only": int(len(set(tr.course_id) - set(te.course_id))), "courses_test_only": int(len(set(te.course_id) - set(tr.course_id))),
        "test_users_seen_in_train": float(te.username.isin(set(tr.username)).mean()),
        "enroll_per_user": float(len(df) / df.username.nunique()), "max_span_days": int(span.max()),
        "course_start_known": float(pd.notna(start).mean()),
        "course_start_median_train": str(pd.Series(start[(df.split == "train").values]).median()),
        "course_start_median_test": str(pd.Series(start[(df.split == "test").values]).median()),
        "first_log_median_train": str(pd.to_datetime(tr.t_min).median()), "first_log_median_test": str(pd.to_datetime(te.t_min).median())}
    df.to_pickle(os.path.join(a.data, "base_counts.pkl"))
    json.dump(report, open(os.path.join(a.data, "base_report.json"), "w"), indent=1)
    print(json.dumps(report["summary"], indent=1))
    print("actions not counted by the original code:", report["train"]["unknown_actions"], report["test"]["unknown_actions"])


if __name__ == "__main__":
    main()
