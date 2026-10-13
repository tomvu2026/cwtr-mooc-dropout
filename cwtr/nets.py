"""Neural models: CFIN re-implemented in PyTorch (follows model.py of the authors' TensorFlow code), CFIN-T, and the MLP / sequence
baselines used in the model comparison."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CFIN(nn.Module):
    """Convolutional feature interaction network (Feng et al., AAAI 2019)."""

    def __init__(self, a_fields, u_dim, c_dim, u_fields, c_fields, emb=32, attn=16, conv=512, ctx=32, deep=256,
                 keep_deep=(0.9, 0.9), bn_decay=0.995, use_attn=True, group=5):
        super().__init__()
        self.use_attn, self.keep = use_attn, keep_deep
        self.a_emb = nn.Parameter(torch.randn(a_fields, emb) * 0.1)        # one embedding vector per activity feature
        self.u_emb = nn.Parameter(torch.randn(u_dim, emb) * 0.1)
        self.c_emb = nn.Parameter(torch.randn(c_dim, emb) * 0.1)
        self.bn_a = nn.BatchNorm1d(emb, eps=1e-3, momentum=1 - bn_decay)
        self.conv = nn.Conv1d(emb, conv, group, stride=group)               # merges (own, learner mean/max, course mean/max) into one vector
        self.ctx = nn.Linear((u_fields + c_fields) * emb, ctx)
        self.att1 = nn.Linear(conv + ctx, attn)
        self.att2 = nn.Linear(attn, 1, bias=False)
        self.fc = nn.Linear(conv + ctx, deep)
        self.bn = nn.BatchNorm1d(deep, eps=1e-3, momentum=1 - bn_decay)
        self.out = nn.Linear(deep, 1)
        for m, fan_in, fan_out in ((self.conv, group * emb, conv), (self.ctx, (u_fields + c_fields) * emb, ctx), (self.att1, conv + ctx, attn),
                                   (self.att2, attn, 1), (self.fc, conv + ctx, deep), (self.out, deep, 1)):
            s = float(np.sqrt(2.0 / (fan_in + fan_out)))
            nn.init.normal_(m.weight, 0, s)
            if m.bias is not None:
                nn.init.normal_(m.bias, 0, s)
        nn.init.constant_(self.out.bias, 0.01)

    def encode(self, av, ui, uv, ci, cv):
        """Returns (attention-pooled activity vector, context vector, attention weights over the 23 feature groups)."""
        a = self.a_emb.unsqueeze(0) * av.unsqueeze(2)                       # B, 115, emb
        a = F.relu(self.conv(self.bn_a(a.transpose(1, 2)))).transpose(1, 2)  # B, 23, conv
        uc = torch.cat([self.u_emb[ui] * uv.unsqueeze(2), self.c_emb[ci] * cv.unsqueeze(2)], 1).flatten(1)
        c = F.relu(self.ctx(uc))                                            # B, ctx
        h = torch.cat([c.unsqueeze(1).expand(-1, a.shape[1], -1), a], 2)
        w = torch.softmax(self.att2(F.relu(self.att1(h))).squeeze(2), 1)    # B, 23
        x = (w.unsqueeze(2) * a).sum(1) if self.use_attn else a.sum(1)
        return x, c, w

    def forward(self, av, ui, uv, ci, cv, return_attn=False):
        x, c, w = self.encode(av, ui, uv, ci, cv)
        x = F.dropout(torch.cat([x, c], 1), 1 - self.keep[0], self.training)
        x = F.dropout(F.relu(self.bn(self.fc(x))), 1 - self.keep[1], self.training)
        o = self.out(x).squeeze(1)
        return (o, w) if return_attn else o

    def l2(self):
        """tf.contrib.layers.l2_regularizer semantics: sum(w^2) / 2 over all non-batch-norm variables."""
        return sum((p ** 2).sum() for n, p in self.named_parameters() if not n.startswith("bn")) / 2

    def n_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("bn"))


class CFINT(nn.Module):
    """CFIN-T: keeps the CFIN statistics and context branches and adds a temporal branch.

    Temporal branch: 36-day x C-channel sequence -> 2-layer bidirectional GRU -> day-level attention conditioned on the context vector.
    branch: "orig" (CFIN only) | "temp" (temporal + context only) | "both".  day_attn=False averages the days instead of attending."""

    def __init__(self, a_fields, u_dim, c_dim, u_fields, c_fields, seq_ch=23, days=36, hid=64, deep=256, drop=0.2,
                 branch="both", day_attn=True, conv=512, ctx=32, attn=16):
        super().__init__()
        self.branch, self.day_attn, self.drop = branch, day_attn, drop
        self.base = CFIN(a_fields, u_dim, c_dim, u_fields, c_fields, conv=conv, ctx=ctx, attn=attn)
        d_in = ctx
        if branch in ("orig", "both"):
            d_in += conv
        if branch in ("temp", "both"):
            self.emb = nn.Linear(seq_ch, hid)
            self.pos = nn.Parameter(torch.zeros(days, hid))
            self.gru = nn.GRU(hid, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.t_att1 = nn.Linear(2 * hid + ctx, attn)
            self.t_att2 = nn.Linear(attn, 1, bias=False)
            d_in += 2 * hid
        self.fc = nn.Linear(d_in, deep)
        self.bn = nn.BatchNorm1d(deep)
        self.out = nn.Linear(deep, 1)

    def forward(self, av, ui, uv, ci, cv, seq, return_attn=False):
        x, c, _ = self.base.encode(av, ui, uv, ci, cv)
        parts, w = [], None
        if self.branch in ("orig", "both"):
            parts.append(x)
        if self.branch in ("temp", "both"):
            h, _ = self.gru(F.relu(self.emb(seq)) + self.pos)               # B, 36, 2*hid
            if self.day_attn:
                scores = self.t_att2(F.relu(self.t_att1(torch.cat([c.unsqueeze(1).expand(-1, h.shape[1], -1), h], 2)))).squeeze(2)
                w = torch.softmax(scores, 1)
                parts.append((w.unsqueeze(2) * h).sum(1))
            else:
                parts.append(h.mean(1))
        z = F.dropout(torch.cat(parts + [c], 1), self.drop, self.training)
        z = F.dropout(F.relu(self.bn(self.fc(z))), self.drop, self.training)
        o = self.out(z).squeeze(1)
        return (o, w) if return_attn else o


class MLP(nn.Module):
    """Two-hidden-layer MLP baseline on standardised tabular features."""

    def __init__(self, d, h=256, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))

    def forward(self, X):
        return self.net(X).squeeze(1)


class SeqNet(nn.Module):
    """Sequence baselines on the daily matrix, concatenated with a static context vector.
    "lstm_att": LSTM -> multi-head self-attention (residual + LayerNorm) -> attentive pooling (DP-SCL-style encoder without the contrastive loss).
    "cnn1d": multi-width 1-D CNN (kernels 3/5/7) with max + mean pooling."""

    def __init__(self, kind, ch, static_dim, hid=128, p=0.2):
        super().__init__()
        self.kind = kind
        if kind == "lstm_att":
            self.lstm = nn.LSTM(ch, hid, batch_first=True)
            self.mha = nn.MultiheadAttention(hid, 4, batch_first=True)
            self.ln = nn.LayerNorm(hid)
            self.q = nn.Parameter(torch.randn(hid) * 0.1)
            d = hid
        else:
            self.convs = nn.ModuleList([nn.Conv1d(ch, 64, k, padding=k // 2) for k in (3, 5, 7)])
            d = 64 * 3 * 2
        self.st = nn.Sequential(nn.Linear(static_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(d + 32, hid), nn.ReLU(), nn.Dropout(p), nn.Linear(hid, 1))

    def forward(self, S, X):
        if self.kind == "lstm_att":
            h, _ = self.lstm(S)
            z, _ = self.mha(h, h, h)
            h = self.ln(h + z)
            w = torch.softmax(h @ self.q / h.shape[2] ** 0.5, 1)
            v = (w.unsqueeze(2) * h).sum(1)
        else:
            c = [F.relu(cv(S.transpose(1, 2))) for cv in self.convs]
            v = torch.cat([x.amax(2) for x in c] + [x.mean(2) for x in c], 1)
        return self.head(torch.cat([v, self.st(X)], 1)).squeeze(1)
