"""Centralized and (hierarchical) federated training loops, prediction and model selection."""
import copy, time, numpy as np, torch, torch.nn.functional as F
from .graph import HyperProp, GraphProp, subset_edges, edge_ids, lattice_edges
from .models import Ops
from .ramal import objectives, DWA, perturb_temp
from .config import STATES, REGIONS
from .stats import auc

class Structure:
    """Caches operators for node subsets."""
    def __init__(self, ctx):
        self.eids = edge_ids(ctx); self.src, self.dst = lattice_edges(ctx); self.cache = {}
        self.state = ctx.state.to_numpy()
    def ops(self, nodes, key=None):
        if key is not None and key in self.cache: return self.cache[key]
        nodes = np.asarray(nodes)
        s, d = subset_edges(self.src, self.dst, nodes)
        o = Ops(HyperProp(self.eids[nodes]), GraphProp(s, d, len(nodes)), s, d)
        if key is not None: self.cache[key] = o
        return o

def _t(x, dt=torch.float32): return torch.as_tensor(x, dtype=dt)

# ------------------------------------------------------------------ prediction
@torch.no_grad()
def predict(model, D, S, X=None, federated=False):
    """Probabilities for all nodes. Federated models are evaluated per state on local sub-hypergraphs."""
    model.eval(); X = D["X"] if X is None else X
    p = np.zeros(len(X), np.float32)
    if federated:
        for st in STATES:
            nodes = np.where(S.state == st)[0]
            if len(nodes) == 0: continue
            p[nodes] = torch.sigmoid(model(_t(X[nodes]), S.ops(nodes, ("all", st)))["logit"]).numpy()
    else:
        allnodes = np.arange(len(X))
        p[:] = torch.sigmoid(model(_t(X), S.ops(allnodes, ("all", "central")))["logit"]).numpy()
    return p

# ------------------------------------------------------------------ centralized
def train_central(model, D, S, cfg, ramal=False, aug_sigma=0.0, seed=0):
    torch.manual_seed(seed)
    tr = D["train"]; ops = S.ops(tr)                     # not cached: training set differs by seed
    X = _t(D["X"][tr]); y = _t(D["y"][tr], torch.long); mask = torch.ones(len(tr), dtype=torch.bool)
    omega = float((y == 0).sum() / max((y == 1).sum(), 1))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad, hist = -1, None, 0, []
    dwa = DWA(cfg.t_w, adaptive=True); epoch_losses = []
    steps_per_round = cfg.local_steps
    for ep in range(cfg.central_epochs):
        model.train(); opt.zero_grad()
        Xin = perturb_temp(X, aug_sigma) if aug_sigma > 0 else X
        if ramal:
            Ls = objectives(model, Xin, ops, y, mask, omega, [p.detach().clone() for p in model.parameters()], cfg)
            lam, gam = dwa.weights(), dwa.scales()
            loss = sum(l * g * L for l, g, L in zip(lam, gam, Ls))
            epoch_losses.append([float(L) for L in Ls])
            if (ep + 1) % steps_per_round == 0:
                dwa.record(list(np.mean(epoch_losses, 0)), cfg.mu_prox); epoch_losses = []
        else:
            out = model(Xin, ops)
            loss = F.binary_cross_entropy_with_logits(out["logit"], y.float(), pos_weight=torch.tensor(omega))
        loss.backward(); opt.step()
        va = auc(D["y"][D["val"]], predict(model, D, S)[D["val"]])
        hist.append(dict(epoch=ep, loss=float(loss), val_auc=va))
        if va > best: best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience: break
    model.load_state_dict(best_state)
    return model, hist

def fit_sklearn(clf, D, aug_sigma=0.0, seed=0):
    X, y = D["X"][D["train"]], D["y"][D["train"]]
    if aug_sigma > 0:
        rng = np.random.default_rng(seed); Xs = [X]
        for _ in range(2):
            Xn = X.copy(); Xn[:, [4, 5]] += rng.normal(0, aug_sigma, (len(X), 2)); Xs.append(Xn)
        X, y = np.vstack(Xs), np.tile(y, 3)
    clf.fit(X, y); return clf

# ------------------------------------------------------------------ federated
def _clients(D, S):
    tr = set(D["train"].tolist()); out = {}
    for st in STATES:
        nodes = np.array([i for i in np.where(S.state == st)[0] if i in tr])
        if len(nodes): out[st] = nodes
    return out

def _flat(params): return torch.cat([p.detach().reshape(-1) for p in params])
def _assign(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(vec[i:i + n].view_as(p)); i += n

def train_federated(model, D, S, cfg, mode="hier_ramal", rounds=None, seed=0, aug_sigma=0.0):
    """mode: hier_ramal (FedHSA) | hier_fixed (w/o RAMAL adaptation) | fedavg | fedprox | scaffold."""
    torch.manual_seed(seed); rounds = rounds or cfg.rounds
    clients = _clients(D, S); ctx = D["ctx"]
    counts = {st: float(ctx["count"].to_numpy()[nodes].sum()) for st, nodes in clients.items()}
    dwa = {st: DWA(cfg.t_w, adaptive=(mode == "hier_ramal")) for st in clients}
    theta = _flat(model.parameters()); n_par = theta.numel()
    c_glob = torch.zeros(n_par); c_loc = {st: torch.zeros(n_par) for st in clients}
    hist, best, best_theta = [], -1, theta.clone()
    y_all = _t(D["y"], torch.long)
    client_ops = {st: S.ops(nodes) for st, nodes in clients.items()}   # built once per run
    for t in range(rounds):
        deltas, t0 = {}, time.time(); round_loss = []
        for st, nodes in clients.items():
            _assign(model, theta); model.train()
            gparams = [p.detach().clone() for p in model.parameters()]
            ops = client_ops[st]; X = _t(D["X"][nodes]); y = y_all[torch.as_tensor(nodes)]
            mask = torch.ones(len(nodes), dtype=torch.bool)
            omega = float((y == 0).sum() / max((y == 1).sum(), 1))
            if mode == "scaffold":
                opt = torch.optim.SGD(model.parameters(), lr=cfg.scaffold_lr)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            step_losses = []
            for _ in range(cfg.local_steps):
                opt.zero_grad()
                Xin = perturb_temp(X, aug_sigma) if aug_sigma > 0 else X
                if mode.startswith("hier"):
                    Ls = objectives(model, Xin, ops, y, mask, omega, gparams, cfg)
                    lam, gam = dwa[st].weights(), dwa[st].scales()
                    loss = sum(l * g * L for l, g, L in zip(lam, gam, Ls))
                    step_losses.append([float(L) for L in Ls])
                else:
                    out = model(Xin, ops)
                    loss = F.binary_cross_entropy_with_logits(out["logit"], y.float(), pos_weight=torch.tensor(omega))
                    if mode == "fedprox":
                        loss = loss + 0.5 * cfg.mu_prox * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), gparams))
                    step_losses.append([float(loss)])
                loss.backward()
                if mode == "scaffold":
                    corr = c_glob - c_loc[st]; i = 0
                    for p in model.parameters():
                        n = p.numel(); p.grad.add_(corr[i:i + n].view_as(p)); i += n
                opt.step()
            m = list(np.mean(step_losses, 0))
            if mode.startswith("hier"): dwa[st].record(m, cfg.mu_prox)
            round_loss.append(m[0] * len(nodes))
            new = _flat(model.parameters()); deltas[st] = new - theta
            if mode == "scaffold":
                c_new = c_loc[st] - c_glob + (theta - new) / (cfg.local_steps * cfg.scaffold_lr)
                deltas[st + "_c"] = c_new - c_loc[st]; c_loc[st] = c_new
        if mode.startswith("hier"):                     # Eqs. (3)-(4)
            reg_delta, reg_count = {}, {}
            for r, sts in REGIONS.items():
                present = [s for s in sts if s in clients]
                if not present: continue
                reg_delta[r] = torch.stack([deltas[s] for s in present]).mean(0)
                reg_count[r] = sum(counts[s] for s in present)
            tot = sum(reg_count.values())
            theta = theta + sum(reg_count[r] / tot * reg_delta[r] for r in reg_delta)
            msgs = dict(client_to_edge=len(clients), edge_to_cloud=len(reg_delta))
        else:                                            # flat FedAvg weighting by local data size
            tot = sum(len(v) for v in clients.values())
            theta = theta + sum(len(clients[s]) / tot * deltas[s] for s in clients)
            if mode == "scaffold":
                c_glob = c_glob + sum(deltas[s + "_c"] for s in clients) / len(clients)
            msgs = dict(client_to_server=len(clients))
        _assign(model, theta)
        va = auc(D["y"][D["val"]], predict(model, D, S, federated=True)[D["val"]])
        hist.append(dict(round=t + 1, train_loss=float(sum(round_loss) / sum(len(v) for v in clients.values())),
                         val_auc=va, seconds=time.time() - t0, **msgs,
                         bytes_per_update=int(n_par * 4)))
        if va > best: best, best_theta = va, theta.clone()
    _assign(model, best_theta)
    return model, hist
