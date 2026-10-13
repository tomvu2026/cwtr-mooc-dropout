# CWTR: source code for dropout prediction on XuetangX

This repository contains the code behind all experiments of the manuscript: the data pipeline, the CWTR
(course-aligned temporal representation) features, CFIN, the baselines, and one script per experiment (E1–E9).
It is meant to let a reviewer (a) check that the pipeline does what the manuscript says and (b) re-run any experiment.

The data are **not** included (see [Data](#data)). Per-seed results of our own runs are in [`reference_results/`](reference_results).

## Contents

```
cwtr/                        shared library
  data.py                    base features (115 activity features + context), as in the public CFIN code; val_split
  counts.py                  step 1  raw logs -> per-enrollment counts (base_counts.pkl)        python -m cwtr.counts
  daily.py                   step 2  raw logs -> 36-day x 23-channel matrix per enrollment (seq.npy)   python -m cwtr.daily
  temporal.py                the 50 CWTR features (temporal_table) and the data loader (load_all)
  norm.py                    course-level normalisation (uses activity only, never labels)
  splits.py                  split by course start date (unseen courses), validation = held-out courses
  build_helpers.py           base features from a truncated daily matrix; base features using only the training pool
  nets.py                    CFIN (PyTorch port of the authors' TensorFlow model), CFIN-T, MLP, LSTM+attention, CNN-1D
  train.py                   neural training loops (early stopping on validation AUC)
  tab.py                     GBDT / RF / LR / MLP with the shared hyper-parameter-search protocol
  stats.py                   saved predictions, paired bootstrap (over enrollments or over groups), per-course win counts
  cli.py                     command-line arguments shared by the experiments
experiments/                 one script per experiment, each with `run` and `summarize` sub-commands
reference_results/           per-seed results (CSV) of our runs, same columns as the output of the scripts
setup_env.sh  requirements.txt
```

## Environment

Results were produced with Python 3.12.3, numpy 2.5.1, pandas 3.0.5, scipy 1.18.0, scikit-learn 1.9.1, torch 2.14.0 (CUDA 13),
lightgbm 4.7.0, xgboost 3.4.1, catboost 1.2.10, on 10 CPU cores, 32 GB RAM and one 24 GB NVIDIA RTX A5000.
Any recent versions should work. A GPU is optional: the neural models (E2, E3, E4) run on CPU, but slowly.

```
python -m venv .venv && . .venv/bin/activate        # optional
pip install -r requirements.txt
bash setup_env.sh                                   # checks the packages, clones the CFIN authors' user-cluster files into cfin_authors/
```

`setup_env.sh` downloads `github.com/wzfhaha/dropout_prediction` (the CFIN authors' code, third party, not redistributed here) only to obtain
`cfin_authors/cluster/{user_dict,label_5_10time.npy}`, the user-cluster labels used by the base features. All commands below are run from this
directory; `--cluster` defaults to `cfin_authors/cluster`.

## Data

The XuetangX dropout data`http://moocdata.cn/data/user-activity` released with the CFIN paper (Feng et al., AAAI 2019), in one directory (`$DATA` below):

```
$DATA/prediction_log/{train,test}_log.csv      raw click logs (or prediction_data.tar, extracted automatically by step 1)
$DATA/prediction_log/{train,test}_truth.csv    dropout labels (1 = dropout), fixed official split
$DATA/user_info.csv   $DATA/course_info.csv
```

## Pipeline

```
python -m cwtr.counts --data $DATA        # ~5 min on our machine. Writes base_counts.pkl, base_report.json
python -m cwtr.daily  --data $DATA        # ~5 min. Writes seq.npy (225,642 x 36 x 23, uint16), seq_report.json
```

Checks to read in the printed reports: `base_report.json` should show ~157.9k train / ~67.7k test enrollments and a dropout rate ~0.758;
`seq_report.json` should show `check_total_vs_base_counts` = 1.0 and `day_lt0`, `day_gt35`, `no_start`, `unknown_action` = 0.

All experiment scripts are run as modules from the repository root. Every `run` sub-command appends one row per (configuration, seed) to a
results CSV and **skips configurations already present**, so an interrupted run can simply be started again with the same command.
Predictions of every run are stored in `<out>/preds/` and are what `summarize` reads (confidence intervals are computed from them).

### Quick sanity check (about 3 minutes, CPU)

```
python -m experiments.e5_base_only run --data $DATA --out /tmp/check_e5 --splits 0.6 --models lgbm --n_trials 2 --n_seeds 1
```

must print `lgbm_base seed=0 val=0.8620 test=0.8247 {'iters': 339}`, and

```
python -m experiments.e4_model_comparison run --data $DATA --out /tmp/check_e4 --splits 0.6 --models lgbm --n_trials 2 --n_seeds 1
```

must print `lgbm seed=0 val=0.8897 test=0.8755 {'iters': 409}`. Tree-model results are exactly reproducible (see [Reproducibility](#reproducibility)).

## Experiments

| ID | Script | Question it answers | Needs |
|---|---|---|---|
| E1 | `e1_cfin_replication` | Can the published CFIN result (AUC 0.864) be reproduced with the authors' protocol? Strict re-evaluation (validation, early stopping, 10 seeds), classical baselines. | data |
| E2 | `e2_temporal_cfint` | Does temporal information help, and does CFIN-T (CFIN + day-level temporal branch) beat LightGBM? Official split and a split by course start date. | data |
| E3 | `e3_unseen_courses` | Same question on **unseen courses** (cuts at 50/60/70% of enrollments), with and without course-level normalisation. | data |
| E4 | `e4_model_comparison` | CWTR-LightGBM against XGBoost, CatBoost, RF, LR, MLP, LSTM+attention, CNN-1D on the same 251-column input; ablation without the 115 CFIN features. | data |
| E5 | `e5_base_only` | The same models on the base features only, i.e. the gain of CWTR per model. E4 + E5 form the model × input table of the manuscript (Table 3). | data, E4 (summary) |
| E6 | `e6_ablation_windows` | Leave-one-group-out ablation of CWTR; control "base + weekly counts"; short observation windows (7–36 days); normalisation fitted on training rows only; classification metrics (accuracy, precision, recall, F1, confusion matrix). | data; E4, E5 (`metrics`) |
| E7 | `e7_leakage_lgbm` | Base-feature statistics computed from the training pool only (LightGBM, 3 cuts); bootstrap over **learners** on the official split. | data, E4–E6 |
| E8 | `e8_leakage_all_models` | The same training-pool-only audit for all six tabular models and three inputs. | data (E4, E5 for the summary) |
| E9 | `e9_representations` | CWTR against other temporal representations: weekly counts aligned to the course start vs to the learner's first active day, cumulative windows (in the spirit of Zhang & Duan; not a faithful re-implementation), combinations. | data, E6 (`params_*.json`) |

Where each result of the manuscript comes from:

| Manuscript | Experiment |
|---|---|
| Table 2 (split sizes) | printed by E4 / E5 (`cwtr.splits.make_split`) |
| Tables 3–4 (six models, Base vs. +CWTR+N); Sec. 4.6 sequence models | E4 + E5 |
| Sec. 4.1 (learner-level bootstrap, seed variation, macro-F1 threshold) | E7 (`stats`) |
| Table 5 (classification metrics) | E6 (`metrics`) |
| Tables 6–8, Fig. 5 (components, observation window) | E6 |
| Sec. 4.4, official split with training-only statistics | E6 (`inductive`) |
| Sec. 4.4, cluster labels removed | E7 (`inductive`) |
| Tables 9–10 (training-only statistics, all six models) | E8 |
| Table 11 (course- vs. learner-aligned, cumulative windows) | E9 |
| Sec. 4.6 (CFIN reproduction) | E1 |
| Table 12 (feature importance) | E2 (LightGBM `lgbm_temp`, official split) |
| Sec. 4.7 (recency groups) | E3 (LightGBM `lgbm_temp` / `lgbm_norm`, cut60) |

The CFIN-T models in E2 and E3 were exploratory and are not reported in the manuscript; only the LightGBM parts of E2 and E3 are used.

Typical commands (replace `$DATA`; GPU flags are optional):

```
python -m experiments.e1_cfin_replication cfin      --data $DATA --out runs_e1 --device cuda
python -m experiments.e1_cfin_replication baselines --data $DATA --out runs_e1
python -m experiments.e1_cfin_replication summarize --out runs_e1

python -m experiments.e2_temporal_cfint run       --data $DATA --out runs_e2 --device cuda
python -m experiments.e2_temporal_cfint summarize --out runs_e2

python -m experiments.e3_unseen_courses run       --data $DATA --out runs_e3 --device cuda
python -m experiments.e3_unseen_courses summarize --out runs_e3

python -m experiments.e4_model_comparison run       --data $DATA --out runs_e4 --device cuda --xgb_device cuda --cat_task GPU
python -m experiments.e4_model_comparison summarize --out runs_e4
python -m experiments.e5_base_only run       --data $DATA --out runs_e5 --device cuda --xgb_device cuda --cat_task GPU
python -m experiments.e5_base_only summarize --out runs_e5 --e4 runs_e4

python -m experiments.e6_ablation_windows run       --data $DATA --out runs_e6
python -m experiments.e6_ablation_windows summarize --out runs_e6
python -m experiments.e6_ablation_windows metrics   --e4 runs_e4 --e5 runs_e5 --out runs_e6

python -m experiments.e7_leakage_lgbm run       --data $DATA --out runs_e7 --e4 runs_e4 --e5 runs_e5 --e6 runs_e6
python -m experiments.e7_leakage_lgbm summarize --out runs_e7 --e6 runs_e6
python -m experiments.e8_leakage_all_models run       --data $DATA --out runs_e8 --device cuda --xgb_device cuda --cat_task GPU
python -m experiments.e8_leakage_all_models summarize --out runs_e8 --e4 runs_e4 --e5 runs_e5
python -m experiments.e9_representations run       --data $DATA --out runs_e9 --e6 runs_e6
python -m experiments.e9_representations summarize --out runs_e9
```

Each `summarize` writes `summary_eN.md` into its output directory (tables of AUC, differences with 95% bootstrap intervals, courses won).
Useful options: `--splits` (e.g. `0.6` for one cut), `--models`, `--n_seeds`, `--n_trials`, `--n_jobs`; E6 also has `--only` to run a few variants.
If CatBoost runs on the GPU it reserves most of the GPU memory: do not start other GPU jobs at the same time.
With 8 GB of GPU memory we ran XGBoost/CatBoost/neural jobs one at a time and the CPU-only models in parallel.

Rough wall-clock time on the machine above, with two to six processes running in parallel (indicative only): E1 about 20 min per CFIN run
(batch 32, 27 epochs), about 6 h for the full grid with six parallel processes; E2 about 1 h; E3 about 1.5 h; E4 a few hours; E5 about 1 h;
E6 about 3 h; E7 under 1 h; E8 about 2 h; E9 about 2 h; the `summarize` steps with bootstrap can take from a few minutes to about an hour
(E6, E8, E9). RAM: about 5–6 GB per process, up to ~12 GB for E3.

## Protocol shared by the experiments

* **Labels and splits.** `official` is the fixed train/test split of the dataset (random across courses). In `cut50/60/70` the courses are sorted by
  start date, the first 50/60/70% of enrollments form the training pool and the later courses are the test set, so test courses are never seen in
  training. The validation set is always held out from the training pool: 20% of enrollments (official) or randomly chosen courses (~15%, unseen-course splits).
* **Model selection.** GBDT models: 12 (E2/E3: 20) random hyper-parameter configurations scored by validation AUC with seed 0, best one retrained for
  5 seeds (10 for the neural models of E2/E3), early stopping on validation. RF: 3 seeds. LR: C ∈ {0.01, 0.1, 1, 10} chosen on validation. Neural models: early stopping on validation AUC.
  The test set is used once, for the final numbers. Differences are reported with 95% paired bootstrap intervals (E1, E2: over enrollments;
  unseen-course splits: over courses; E7: over learners) computed on the seed-averaged predictions.
* **No label information in the features.** CWTR features and the course-level normalisation are computed from activity inside the 36-day window only.
  Learner / course statistics of the base features follow the authors' code (computed on train + test, label-free); E6 (`inductive`), E7 and E8 repeat
  the evaluation with these statistics computed from the training pool only, and E6 does the same for the normalisation.
* **Base features.** `cwtr/data.py` re-implements the feature pipeline of the public CFIN code. `fix=0` keeps two bugs of that code (gender and
  course category are always 0); `fix=1` repairs them. All experiments after E1 use `fix=1`.

## Reproducibility

* LightGBM is deterministic given the seeds: the two sanity checks above match our original runs exactly. We did not re-check the other
  tabular models digit by digit (XGBoost and CatBoost on the GPU may also vary slightly).
* Neural models on the GPU are **not bit-reproducible** even with fixed seeds (non-deterministic CUDA kernels). Repeating the same configuration gave
  validation AUCs differing by about ±0.0005 (for example 0.8486 / 0.8488 / 0.8487 with the original code and 0.8490 / 0.8495 / 0.8484 with this
  cleaned code, same seed, 3 epochs), and test AUC of a single seed can move by up to ~0.001. Averages over seeds are stable. Expect the neural rows to
  agree with `reference_results/` within seed noise, not digit for digit.
* The cleaned code was checked against the code that produced the reported numbers: the base-feature tables (`cwtr.data.build`, all
  four combinations of `agg` ∈ {all, train} and `fix` ∈ {0, 1}) and the CWTR table (`temporal_table`) are identical value by value; the
  LightGBM runs in E4, E5 and E6 reproduce the original validation and test AUC to four decimals and the same number of boosting rounds, and the
  logistic regression of E8 (statistics from the training pool only) reproduces the original AUC to six decimals. Every script was also executed
  end to end on a reduced configuration (one cut, one seed) to check the `run` and `summarize` paths.
* `reference_results/` holds our per-seed results: `e1_*` … `e9_*` correspond to the scripts of the same number. `e6_params_<split>.json` are the
  LightGBM parameters selected once per split and reused by E6, E7 and E9. `e6_metrics.csv` has the classification metrics.
  The saved predictions (`preds/`) are large and not included; re-running a script regenerates them.

## Known limitations of the code

* The "cumulative windows" representation in E9 follows the idea of Zhang & Duan but is not a faithful re-implementation of their definitions.
* In the short-window part of E6 the base feature `session#count` is approximated by the number of events (the original code counts log rows that carry
  a session id); the correlation with the original column is ≥ 0.9998 for the 23 count features at the full window.
* The sequence baselines (LSTM+attention, CNN-1D) use a small fixed architecture and early stopping (max 30 epochs); they were not tuned further.
* E1 reproduces the authors' training recipe for CFIN (batch 32, lr 1e-4, L2 1e-5, 27 epochs) in PyTorch rather than the original TensorFlow code.

## License and citation

License: to be added. If you use this code, please cite the manuscript (full reference to be added after publication).
