"""Attribution analysis: intrinsic spike-rate x attention, KernelSHAP, integrated gradients, ROAR."""
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import kendalltau
from .config import STATES, FEATURES

@torch.no_grad()
def neighbor_messages(model, D, S):
    """(Theta X)_j for every node, computed per state (local sub-hypergraph) as in federated inference."""
    X = D["X"]; M = np.zeros_like(X)
    for st in STATES:
        nodes = np.where(S.state == st)[0]
        if len(nodes) == 0: continue
        ops = S.ops(nodes, ("all", st)); Xt = torch.as_tensor(X[nodes])
        PX = ops.hyper(Xt, F.softplus(model.w_raw)) if model.use_hypergraph else ops.graph(Xt)
        M[nodes] = PX.numpy()
    return M

def _node_fn(model, m):
    a = model.alpha if model.use_residual else 0.0
    def f(Z):
        if not torch.is_tensor(Z): Z = torch.as_tensor(np.asarray(Z, np.float32))
        mm = torch.as_tensor(m).expand_as(Z)
        return torch.sigmoid(model.from_parts(a * Z + (1 - a) * mm)["logit"])
    return f

@torch.no_grad()
def intrinsic(model, D, S):
    X = D["X"]; A = np.zeros_like(X)
    for st in STATES:
        nodes = np.where(S.state == st)[0]
        if len(nodes): A[nodes] = model(torch.as_tensor(X[nodes]), S.ops(nodes, ("all", st)))["attr"].numpy()
    return A[D["test"]].mean(0)

def kernel_shap(model, D, S, n_nodes, nsamples, seed=0):
    import shap
    model.eval(); M = neighbor_messages(model, D, S); rng = np.random.default_rng(seed)
    nodes = rng.choice(D["test"], size=min(n_nodes, len(D["test"])), replace=False)
    bg = shap.kmeans(D["X"][D["train"]], 20); vals = []
    for j in nodes:
        f = _node_fn(model, M[j])
        ex = shap.KernelExplainer(lambda Z: f(Z).detach().numpy(), bg)
        vals.append(np.asarray(ex.shap_values(D["X"][j:j + 1], nsamples=nsamples, silent=True)).reshape(-1))
    return np.abs(np.array(vals)).mean(0)

def integrated_gradients(model, D, S, n_nodes, steps=50, seed=0):
    model.eval(); M = neighbor_messages(model, D, S); rng = np.random.default_rng(seed)
    nodes = rng.choice(D["test"], size=min(n_nodes, len(D["test"])), replace=False)
    base = torch.as_tensor(D["X"][D["train"]].mean(0)); out = []
    for j in nodes:
        x = torch.as_tensor(D["X"][j]); f = _node_fn(model, M[j])
        alphas = torch.linspace(0, 1, steps).unsqueeze(1)
        Z = (base + alphas * (x - base)).requires_grad_(True)
        f(Z).sum().backward()
        out.append(((x - base) * Z.grad.mean(0)).abs().numpy())
    return np.array(out).mean(0)

def rank_agreement(a, b):
    return float(kendalltau(a, b).statistic)

def top_k(imp, k=2):
    return [int(i) for i in np.argsort(-np.asarray(imp))[:k]]

def remove_features(D, idx):
    """ROAR (Hooker et al., 2019): replace features by their training mean, to be followed by retraining."""
    D2 = dict(D); X = D["X"].copy(); X[:, idx] = X[D["train"]][:, idx].mean(0); D2["X"] = X
    return D2

def feature_names(idx): return [FEATURES[i] for i in idx]
