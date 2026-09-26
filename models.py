"""FedHSA and all neural baselines. Every model has forward(X, ops) -> dict(logit=..., ...)."""
import torch, torch.nn as nn, torch.nn.functional as F
from .graph import n_edges

class Ops:
    """Structure operators for one node subset (a client, or all nodes for centralized models)."""
    def __init__(self, hyper, graph, src, dst):
        self.hyper, self.graph = hyper, graph
        self.src, self.dst = torch.as_tensor(src), torch.as_tensor(dst)

class SpikeFn(torch.autograd.Function):
    """Heaviside forward; fast-sigmoid surrogate 1/(1+k|v|)^2 backward (Neftci et al., 2019)."""
    @staticmethod
    def forward(ctx, v, k):
        ctx.save_for_backward(v); ctx.k = k
        return (v >= 0).to(v.dtype)
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        return g / (1 + ctx.k * v.abs()) ** 2, None

class FedHSA(nn.Module):
    def __init__(self, cfg, d_in=6, use_hypergraph=True, use_residual=True, use_spiking=True, alpha=None, beta=None):
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.alpha if alpha is None else alpha
        self.beta = cfg.beta if beta is None else beta
        self.use_hypergraph, self.use_residual, self.use_spiking = use_hypergraph, use_residual, use_spiking
        self.w_raw = nn.Parameter(torch.full((n_edges(),), 0.5413))      # softplus(0.5413) = 1.0
        self.inp = nn.Linear(d_in, d_in)                                   # I_i = w_i^T x_hat
        self.emb = nn.Parameter(torch.randn(d_in, cfg.d_model) * 0.1)      # one token per feature channel
        self.att = nn.MultiheadAttention(cfg.d_model, cfg.heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(cfg.d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def propagate(self, X, ops):
        if self.use_hypergraph:
            PX = ops.hyper(X, F.softplus(self.w_raw))
        else:
            PX = ops.graph(X)
        return self.alpha * X + (1 - self.alpha) * PX if self.use_residual else PX

    def from_parts(self, Xhat):
        I = self.inp(Xhat)                                                 # (n,d)
        if self.use_spiking:
            V = torch.zeros_like(I); S = []
            for _ in range(self.cfg.t_spike):
                V = self.beta * V + I                                      # Eq. (5)
                s = SpikeFn.apply(V - self.cfg.theta_spike, self.cfg.surrogate_k)   # Eq. (6)
                V = V - self.cfg.theta_spike * s                           # soft reset
                S.append(s)
            S = torch.stack(S, 0)
            r = S.mean(0)                                                  # Eq. (7)
            spk = S.mean()
        else:
            r = torch.sigmoid(I); spk = torch.zeros((), dtype=I.dtype)
        tok = r.unsqueeze(-1) * self.emb.unsqueeze(0)                      # (n,d,dm)
        out, A = self.att(tok, tok, tok, need_weights=True, average_attn_weights=True)   # A: (n,d,d)
        a = A.mean(1)                                                      # attention received per channel
        q = r * a
        attr = q / q.sum(1, keepdim=True).clamp_min(1e-8)                  # Eq. (8)
        logit = self.head(out.mean(1)).squeeze(-1)
        return dict(logit=logit, rates=r, attr=attr, spk=spk)

    def forward(self, X, ops):
        return self.from_parts(self.propagate(X, ops))

class MLP(nn.Module):
    def __init__(self, d_in=6, h=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, X, ops=None): return dict(logit=self.net(X).squeeze(-1))

class HGNN(nn.Module):
    """Feng et al. (2019): two hypergraph convolutions with learnable hyperedge weights."""
    def __init__(self, d_in=6, h=64):
        super().__init__(); self.w_raw = nn.Parameter(torch.full((n_edges(),), 0.5413))
        self.l1, self.l2 = nn.Linear(d_in, h), nn.Linear(h, 1)
    def forward(self, X, ops):
        w = F.softplus(self.w_raw)
        H = F.dropout(F.relu(self.l1(ops.hyper(X, w))), 0.2, self.training)
        return dict(logit=self.l2(ops.hyper(H, w)).squeeze(-1))

class HGNNPlus(nn.Module):
    """HGNN+ (Gao et al., 2023) spatial convolution: vertex->hyperedge->vertex mean passing."""
    def __init__(self, d_in=6, h=64):
        super().__init__(); self.l1, self.l2 = nn.Linear(d_in, h), nn.Linear(h, 1)
    def forward(self, X, ops):
        H = F.dropout(F.relu(ops.hyper.v2e2v(self.l1(X))), 0.2, self.training)
        return dict(logit=ops.hyper.v2e2v(self.l2(H)).squeeze(-1))

class GCN(nn.Module):
    def __init__(self, d_in=6, h=64):
        super().__init__(); self.l1, self.l2 = nn.Linear(d_in, h), nn.Linear(h, 1)
    def forward(self, X, ops):
        H = F.dropout(F.relu(self.l1(ops.graph(X))), 0.2, self.training)
        return dict(logit=self.l2(ops.graph(H)).squeeze(-1))

class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, heads, concat=True):
        super().__init__(); self.h, self.d, self.concat = heads, d_out, concat
        self.W = nn.Linear(d_in, heads * d_out, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, d_out) * 0.1); self.a_dst = nn.Parameter(torch.randn(heads, d_out) * 0.1)
    def forward(self, X, src, dst):
        n = X.shape[0]; loops = torch.arange(n)
        src = torch.cat([src, loops]); dst = torch.cat([dst, loops])
        Z = self.W(X).view(n, self.h, self.d)
        e = F.leaky_relu((Z[src] * self.a_src).sum(-1) + (Z[dst] * self.a_dst).sum(-1), 0.2)   # (E,h)
        e = e - e.max()
        ex = e.exp()
        den = torch.zeros(n, self.h).index_add_(0, dst, ex)
        alpha = ex / den[dst].clamp_min(1e-12)
        out = torch.zeros(n, self.h, self.d).index_add_(0, dst, alpha.unsqueeze(-1) * Z[src])
        return out.reshape(n, -1) if self.concat else out.mean(1)

class GAT(nn.Module):
    """Veličković et al. (2018): two layers, four heads."""
    def __init__(self, d_in=6, h=16, heads=4):
        super().__init__(); self.g1 = GATLayer(d_in, h, heads); self.g2 = GATLayer(h * heads, 1, 1, concat=False)
    def forward(self, X, ops):
        H = F.dropout(F.elu(self.g1(X, ops.src, ops.dst)), 0.2, self.training)
        return dict(logit=self.g2(H, ops.src, ops.dst).squeeze(-1))

class GCNII(nn.Module):
    """Chen et al. (2020): initial residual + identity mapping, 8 layers."""
    def __init__(self, d_in=6, h=64, layers=8, alpha=0.1, lam=0.5):
        super().__init__(); self.inp, self.out = nn.Linear(d_in, h), nn.Linear(h, 1)
        self.ws = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(layers)])
        self.alpha, self.betas = alpha, [torch.log(torch.tensor(lam / (l + 1) + 1)).item() for l in range(layers)]
    def forward(self, X, ops):
        H0 = F.relu(self.inp(X)); H = H0
        for W, b in zip(self.ws, self.betas):
            M = (1 - self.alpha) * ops.graph(H) + self.alpha * H0
            H = F.dropout(F.relu((1 - b) * M + b * W(M)), 0.2, self.training)
        return dict(logit=self.out(H).squeeze(-1))
