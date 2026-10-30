"""AFNDT-v2 and depth-configurable GCNII. All neural models expose forward(X, ops) -> dict(logit=...)."""
import torch, torch.nn as nn, torch.nn.functional as F
from .graph import n_edges
from .models import SpikeFn

class GCNIIv(nn.Module):
    """GCNII (Chen et al., 2020) with configurable depth; final layer named `head`."""
    def __init__(self, d_in=6, h=64, layers=8, alpha=0.1, lam=0.5):
        super().__init__(); self.inp = nn.Linear(d_in, h); self.head = nn.Linear(h, 1)
        self.ws = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.alpha = alpha; self.betas = [float(torch.log(torch.tensor(lam / (l + 1) + 1))) for l in range(layers)]
    def forward(self, X, ops):
        H0 = F.relu(self.inp(X)); H = H0
        for W, b in zip(self.ws, self.betas):
            M = (1 - self.alpha) * ops.graph(H) + self.alpha * H0
            H = F.dropout(F.relu((1 - b) * M + b * W(M)), 0.2, self.training)
        return dict(logit=self.head(H).squeeze(-1))

class AFNDTv2(nn.Module):
    """Deep initial-residual propagation on the within-state graph, an optional gated hypergraph
    branch, an optional spiking (LIF) readout, and a `head` that can be kept personal per state."""
    def __init__(self, d_in=6, h=64, layers=8, alpha=0.1, lam=0.5, use_hyper=True, use_spike=False,
                 t_spike=8, beta=0.95, theta=0.5, k=10.0):
        super().__init__()
        self.inp = nn.Linear(d_in, h); self.ws = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.alpha = alpha; self.betas = [float(torch.log(torch.tensor(lam / (l + 1) + 1))) for l in range(layers)]
        self.use_hyper, self.use_spike = use_hyper, use_spike
        if use_hyper:
            self.w_raw = nn.Parameter(torch.full((n_edges(),), 0.5413))    # softplus -> 1
            self.g_raw = nn.Parameter(torch.tensor(-2.0))                 # gate starts at 0.12
        if use_spike:
            self.lif_in = nn.Linear(h, h); self.t_spike, self.beta, self.theta, self.k = t_spike, beta, theta, k
        self.head = nn.Linear(h, 1)

    def gate(self): return torch.sigmoid(self.g_raw) if self.use_hyper else torch.tensor(0.0)

    def forward(self, X, ops):
        H0 = F.relu(self.inp(X)); H = H0
        if self.use_hyper: g, w = torch.sigmoid(self.g_raw), F.softplus(self.w_raw)
        for W, b in zip(self.ws, self.betas):
            P = ops.graph(H)
            if self.use_hyper: P = (1 - g) * P + g * ops.hyper(H, w)
            M = (1 - self.alpha) * P + self.alpha * H0
            H = F.dropout(F.relu((1 - b) * M + b * W(M)), 0.2, self.training)
        if self.use_spike:
            I = self.lif_in(H); V = torch.zeros_like(I); rate = torch.zeros_like(I)
            for _ in range(self.t_spike):
                V = self.beta * V + I; s = SpikeFn.apply(V - self.theta, self.k); V = V - self.theta * s; rate = rate + s
            H = rate / self.t_spike
        return dict(logit=self.head(H).squeeze(-1))
