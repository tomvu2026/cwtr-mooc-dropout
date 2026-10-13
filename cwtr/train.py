"""Training loops for the neural models (early stopping on validation AUC, best weights restored)."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .nets import CFINT, MLP, SeqNet

CFINT_KEYS = ("A", "UI", "UV", "CI", "CV", "S")
# CFIN-T variants: name -> (branch, day-level attention)
CFINT_VARIANTS = {"orig": ("orig", True), "temp": ("temp", True), "both": ("both", True), "both_mean": ("both", False)}


def _snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def predict_cfint(model, T, idx, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i:i + bs], device=T["y"].device)
        out.append(torch.sigmoid(model(*(T[k][j] for k in CFINT_KEYS))).float().cpu())
    return torch.cat(out).numpy()


def fit_cfint(D, T, variant, seed, a):
    """Train one CFIN-T variant. D: data dict with idx / S / A ...; T: the same arrays as tensors on a.device.
    Returns (validation predictions, test predictions, best epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    branch, day_attn = CFINT_VARIANTS[variant]
    idx = D["idx"]
    m = CFINT(D["A"].shape[1], D["u_dim"], D["c_dim"], D["UI"].shape[1], D["CI"].shape[1], seq_ch=D["S"].shape[2], days=D["S"].shape[1],
              branch=branch, day_attn=day_attn).to(a.device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
    tr = torch.as_tensor(idx["train"], device=a.device)
    best = (-1, None, 0)
    for ep in range(1, a.max_epochs + 1):
        m.train()
        perm = tr[torch.randperm(len(tr), device=a.device)]
        for i in range(len(perm) // a.batch):
            j = perm[i * a.batch:(i + 1) * a.batch]
            loss = F.binary_cross_entropy_with_logits(m(*(T[k][j] for k in CFINT_KEYS)), T["y"][j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = roc_auc_score(D["y"][idx["val"]], predict_cfint(m, T, idx["val"]))
        if v > best[0]:
            best = (v, _snapshot(m), ep)
        elif ep - best[2] >= a.patience:
            break
    m.load_state_dict(best[1])
    return predict_cfint(m, T, idx["val"]), predict_cfint(m, T, idx["test"]), best[2]


def train_nn(kind, arrays, y, idx, seed, a):
    """Train the 'mlp' (arrays: {'X'}) or a sequence baseline 'lstm_att' / 'cnn1d' (arrays: {'S', 'X'}).
    Returns (validation predictions, test predictions, best epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = a.device
    t = {k: torch.as_tensor(v, device=dev) for k, v in arrays.items()}
    yt = torch.as_tensor(y.astype(np.float32), device=dev)
    m = (MLP(arrays["X"].shape[1]) if kind == "mlp" else SeqNet(kind, arrays["S"].shape[2], arrays["X"].shape[1])).to(dev)
    call = (lambda j: m(t["X"][j])) if kind == "mlp" else (lambda j: m(t["S"][j], t["X"][j]))
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
    tr = torch.as_tensor(idx["train"], device=dev)

    @torch.no_grad()
    def predict(rows):
        m.eval()
        rows = torch.as_tensor(rows, device=dev)
        return torch.cat([torch.sigmoid(call(rows[i:i + 4096])).float().cpu() for i in range(0, len(rows), 4096)]).numpy()

    best = (-1, None, 0)
    for ep in range(1, a.max_epochs + 1):
        m.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(len(perm) // a.batch):
            j = perm[i * a.batch:(i + 1) * a.batch]
            loss = F.binary_cross_entropy_with_logits(call(j), yt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = roc_auc_score(y[idx["val"]], predict(idx["val"]))
        if v > best[0]:
            best = (v, _snapshot(m), ep)
        elif ep - best[2] >= a.patience:
            break
    m.load_state_dict(best[1])
    return predict(idx["val"]), predict(idx["test"]), best[2]
