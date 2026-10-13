"""Step 2: daily activity matrices. Each enrollment becomes a 36-day x 23-channel matrix
(22 actions + number of sessions); day 0 is the course start date (course_info.start), and the observation window of the
dataset is 35 days from it.

    python -m cwtr.daily --data <xuetangx>      # needs base_counts.pkl from cwtr.counts

Writes <xuetangx>/seq.npy (N x 36 x 23, same row order as base_counts.pkl) and <xuetangx>/seq_report.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .data import ACTIONS

DAY_ACTIONS = ACTIONS + ["close_forum"]    # 22 action channels; channel 22 counts distinct sessions per day
N_DAYS = 36


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    a = ap.parse_args()
    base = pd.read_pickle(os.path.join(a.data, "base_counts.pkl"))
    row_of = pd.Index(base.index)
    ci = pd.read_csv(os.path.join(a.data, "course_info.csv")).drop_duplicates("course_id").set_index("course_id")
    start = pd.to_datetime(ci["start"], errors="coerce")
    S = np.zeros((len(base), N_DAYS, len(DAY_ACTIONS) + 1), dtype=np.int32)
    channel = {x: i for i, x in enumerate(DAY_ACTIONS)}
    rep = {"rows": 0, "day_lt0": 0, "day_gt35": 0, "no_start": 0, "unknown_action": 0}
    t0 = time.time()
    for split in ("train", "test"):
        path = os.path.join(a.data, "prediction_log", f"{split}_log.csv")
        for c in pd.read_csv(path, usecols=["enroll_id", "course_id", "session_id", "action", "time"], chunksize=a.chunk):
            t = pd.to_datetime(c["time"], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
            day = np.floor((t.values - start.reindex(c["course_id"]).values) / np.timedelta64(1, "D"))
            rep["rows"] += len(c)
            rep["no_start"] += int(np.isnan(day).sum())
            rep["day_lt0"] += int((day < 0).sum())
            rep["day_gt35"] += int((day > 35).sum())
            day = np.clip(np.nan_to_num(day, nan=0), 0, N_DAYS - 1).astype(int)
            r = row_of.get_indexer(c["enroll_id"])
            k = c["action"].map(channel)
            ok = (r >= 0) & k.notna().values
            rep["unknown_action"] += int(k.isna().sum())
            np.add.at(S, (r[ok], day[ok], k[ok].astype(int).values), 1)
            sessions = pd.DataFrame({"r": r, "d": day, "s": c["session_id"].values})[r >= 0].dropna().drop_duplicates()
            np.add.at(S, (sessions["r"].values, sessions["d"].values, len(DAY_ACTIONS)), 1)
            print(f"  {split}: {rep['rows'] / 1e6:.0f}M rows ({time.time() - t0:.0f}s)", flush=True)
    total = S[:, :, :len(DAY_ACTIONS)].sum(2)
    last_active = np.where(total.sum(1) > 0, N_DAYS - 1 - np.argmax((total > 0)[:, ::-1], 1), 0)
    rep.update(n=len(base), empty_enroll=int((total.sum(1) == 0).sum()), mean_active_days=float((total > 0).sum(1).mean()),
               check_total_vs_base_counts=float(np.mean(total.sum(1) == base["all#count"].values)),
               last_day_hist=np.bincount(last_active, minlength=N_DAYS).tolist())
    np.save(os.path.join(a.data, "seq.npy"), np.minimum(S, 65535).astype(np.uint16))
    json.dump(rep, open(os.path.join(a.data, "seq_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "last_day_hist"}, indent=1))


if __name__ == "__main__":
    main()
