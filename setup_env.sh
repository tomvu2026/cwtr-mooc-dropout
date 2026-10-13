#!/usr/bin/env bash
# Checks the environment and fetches the user-cluster files of the CFIN authors (needed by the base features).
#   bash setup_env.sh
# Python packages: pip install -r requirements.txt   (use a virtual environment if your system Python is externally managed)
set -u
PY=${PYTHON:-python3}
echo "== Python: $($PY --version 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; else echo "[warning] no nvidia-smi: neural models and GPU boosting will run on CPU / are unavailable"; fi
$PY - <<'PYEOF'
import importlib.util, sys
mods = ("numpy", "pandas", "scipy", "sklearn", "torch", "lightgbm", "xgboost", "catboost", "tabulate")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print("missing packages:", missing or "none")
if importlib.util.find_spec("torch") is not None:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
sys.exit(1 if missing else 0)
PYEOF
echo "== User-cluster files of the CFIN authors (cfin_authors/cluster/)"
if [ ! -f cfin_authors/cluster/user_dict ]; then
  git clone --depth 1 https://github.com/wzfhaha/dropout_prediction cfin_authors || echo "[error] clone failed: download github.com/wzfhaha/dropout_prediction into cfin_authors/ by hand"
fi
ls -la cfin_authors/cluster 2>/dev/null
echo "== Done"
