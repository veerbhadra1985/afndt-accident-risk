"""AFNDT-v3.

Four changes over v2, each with a reason and each switchable for the ablation:
  1. relation-aware propagation: hour, month and year neighbours get separate learned weights
  2. auxiliary count regression: the model also predicts the (standardized log) accident count,
     which is a richer target than the binary label and regularizes the shared body
  3. mixed personalization: every state blends a personal head with the global head through a
     learned local gate, so data-rich states personalize and data-poor states fall back (Ditto-style)
  4. free depth (2/4/8 layers), the option v2 never had
"""
import torch, torch.nn as nn, torch.nn.functional as F
from .graph import N_EDGES
from .v3graph import REL

class Ops3:
    """Per-relation graph operators (+ the hypergraph operator) for one node subset."""
    def __init__(self, rel_props, hyper): self.rel, self.hyper = rel_props, hyper

class AFNDTv3(nn.Module):
    def __init__(self, d_in=6, h=64, layers=4, alpha=0.1, lam=0.5, use_rel=True, use_aux=True,
                 use_mix=True, use_hyper=False):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        self.ws = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.alpha = alpha
        self.betas = [float(torch.log(torch.tensor(lam / (l + 1) + 1))) for l in range(layers)]
        self.use_rel, self.use_aux, self.use_mix, self.use_hyper = use_rel, use_aux, use_mix, use_hyper
        self.rel_raw = nn.Parameter(torch.zeros(len(REL)))          # relation weights (softmax)
        if use_hyper:
            self.w_raw = nn.Parameter(torch.full((N_EDGES,), 0.5413)); self.g_raw = nn.Parameter(torch.tensor(-2.0))
        self.head = nn.Linear(h, 1)                                  # global head (shared)
        self.pers = nn.Linear(h, 1)                                  # personal head (kept local)
        self.mix = nn.Parameter(torch.tensor(0.0))                   # personal/global gate (kept local)
        self.aux = nn.Linear(h, 1)                                   # auxiliary count head (shared)

    PERSONAL = ("pers.", "mix")

    def propagate(self, H, ops):
        if self.use_rel:
            w = torch.softmax(self.rel_raw, 0)
            P = sum(w[i] * ops.rel[r](H) for i, r in enumerate(REL))
        else:
            P = sum(ops.rel[r](H) for r in REL) / len(REL)
        if self.use_hyper:
            g = torch.sigmoid(self.g_raw); P = (1 - g) * P + g * ops.hyper(H, F.softplus(self.w_raw))
        return P

    def forward(self, X, ops):
        H0 = F.relu(self.inp(X)); H = H0
        for W, b in zip(self.ws, self.betas):
            M = (1 - self.alpha) * self.propagate(H, ops) + self.alpha * H0
            H = F.dropout(F.relu((1 - b) * M + b * W(M)), 0.2, self.training)
        g_logit = self.head(H).squeeze(-1)
        if self.use_mix:
            m = torch.sigmoid(self.mix)
            logit = (1 - m) * g_logit + m * self.pers(H).squeeze(-1)
        else:
            logit = g_logit
        return dict(logit=logit, aux=self.aux(H).squeeze(-1), mix=torch.sigmoid(self.mix))
