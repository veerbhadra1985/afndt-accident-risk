"""Hypergraph and pairwise-graph operators restricted to an arbitrary node subset.

Every node belongs to exactly three hyperedges (its state, its month, its time-of-day bin),
so the incidence structure is stored as an (n, 3) array of global hyperedge ids.
Restricting to a subset of nodes (e.g. one client) automatically yields the induced
sub-hypergraph: hyperedge sums are taken only over the nodes that are present.
"""
import numpy as np, torch
from .config import STATES

N_EDGES = len(STATES) + 12 + 4          # 49 + 12 + 4 = 65

def edge_ids(ctx):
    s = ctx.state.map({s: i for i, s in enumerate(STATES)}).to_numpy()
    m = len(STATES) + (ctx.month.to_numpy() - 1)
    b = len(STATES) + 12 + ctx.hbin.to_numpy()
    return np.stack([s, m, b], 1).astype(np.int64)

class HyperProp:
    """Theta X = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X  (Eq. 1) on the induced sub-hypergraph."""
    def __init__(self, eids):
        self.eids = torch.as_tensor(eids)                     # (n,3)
        self.flat = self.eids.reshape(-1)
        deg = torch.zeros(N_EDGES).index_add_(0, self.flat, torch.ones(self.flat.numel()))
        self.de = deg.clamp_min(1.0)
        self.dv = 3.0                                          # every node has degree 3

    def __call__(self, X, w):                                  # w: (65,) positive weights
        Xs = X / self.dv ** 0.5
        E = torch.zeros(N_EDGES, X.shape[1], dtype=X.dtype).index_add_(0, self.flat, Xs.repeat_interleave(3, 0))
        E = E * (w / self.de).unsqueeze(1)                     # W De^-1 H^T
        return E[self.eids].sum(1) / self.dv ** 0.5            # Dv^-1/2 H (.)

    def v2e2v(self, X):
        """HGNN+ spatial two-stage mean message passing (vertex->hyperedge->vertex)."""
        E = torch.zeros(N_EDGES, X.shape[1], dtype=X.dtype).index_add_(0, self.flat, X.repeat_interleave(3, 0))
        E = E / self.de.unsqueeze(1)
        return E[self.eids].mean(1)

def lattice_edges(ctx):
    """Pairwise graph: contexts of the same state that are adjacent in hour (±1, cyclic)
    or month (±1, cyclic). Returns directed edge list (src, dst) without self loops."""
    key = {(s, m, h): i for i, (s, m, h) in enumerate(zip(ctx.state, ctx.month, ctx.hour))}
    src, dst = [], []
    for i, (s, m, h) in enumerate(zip(ctx.state, ctx.month, ctx.hour)):
        for nb in ((s, m, (h + 1) % 24), (s, m, (h - 1) % 24),
                   (s, (m % 12) + 1, h), (s, ((m - 2) % 12) + 1, h)):
            j = key.get(nb)
            if j is not None:
                src.append(j); dst.append(i)
    return np.array(src, np.int64), np.array(dst, np.int64)

class GraphProp:
    """Symmetric-normalized adjacency with self loops, D^-1/2 (A+I) D^-1/2, on a node subset."""
    def __init__(self, src, dst, n):
        loops = np.arange(n)
        self.src = torch.as_tensor(np.concatenate([src, loops])); self.dst = torch.as_tensor(np.concatenate([dst, loops]))
        deg = torch.zeros(n).index_add_(0, self.dst, torch.ones(self.dst.numel()))
        self.norm = (deg[self.src] * deg[self.dst]).rsqrt()
        self.n = n
    def __call__(self, X, w=None):
        out = torch.zeros_like(X)
        return out.index_add_(0, self.dst, X[self.src] * self.norm.unsqueeze(1))

def subset_edges(src, dst, nodes):
    """Re-index a global edge list to a node subset."""
    pos = -np.ones(max(src.max(), dst.max(), nodes.max()) + 1, np.int64); pos[nodes] = np.arange(len(nodes))
    keep = (pos[src] >= 0) & (pos[dst] >= 0)
    return pos[src[keep]], pos[dst[keep]]
