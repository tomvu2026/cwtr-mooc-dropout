"""Command-line arguments shared by the experiment scripts."""
import torch


def add_data_args(ap):
    ap.add_argument("--data", required=True, help="XuetangX directory (base_counts.pkl, seq.npy, user_info.csv, course_info.csv)")
    ap.add_argument("--cluster", default="cfin_authors/cluster", help="directory with the authors' user-cluster files")
    ap.add_argument("--out", required=True, help="output directory (results CSV + preds/)")


def add_train_args(ap, n_seeds=5, n_seeds_rf=3, n_trials=12, max_epochs=30, patience=4, n_estimators=5000):
    ap.add_argument("--n_seeds", type=int, default=n_seeds, help="seeds for GBDT / MLP / neural models")
    ap.add_argument("--n_seeds_rf", type=int, default=n_seeds_rf)
    ap.add_argument("--n_trials", type=int, default=n_trials, help="random hyper-parameter configurations for GBDT")
    ap.add_argument("--n_estimators", type=int, default=n_estimators, help="maximum boosting rounds (early stopping on validation)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_epochs", type=int, default=max_epochs)
    ap.add_argument("--patience", type=int, default=patience)
    ap.add_argument("--n_jobs", type=int, default=16)
    ap.add_argument("--xgb_device", default="cpu", help="'cuda' to run XGBoost on the GPU")
    ap.add_argument("--cat_task", default="CPU", help="'GPU' to run CatBoost on the GPU")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device for the neural models")
