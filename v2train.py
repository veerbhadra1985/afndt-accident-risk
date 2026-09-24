"""Training for the AFNDT-v2 protocol: centralized, flat federated (FedAvg/FedProx) and
personalized hierarchical federated training (shared body via 6 regions, personal heads per state)."""
import copy, numpy as np, torch, torch.nn.functional as F
from .graph import HyperProp, GraphProp, subset_edges, edge_ids
from .models import Ops
from .v2graph import year_lattice
from .config import STATES, REGIONS
from .stats import auc

class Structure2:
    def __init__(self, ctx):
        self.eids = edge_ids(ctx); self.src, self.dst = year_lattice(ctx)
        self.state = ctx.state.to_numpy(); self._cache = {}
    def ops(self, nodes, key=None):
        if key is not None and key in self._cache: return self._cache[key]
        nodes = np.asarray(nodes); s, d = subset_edges(self.src, self.dst, nodes)
        o = Ops(HyperProp(self.eids[nodes]), GraphProp(s, d, len(nodes)), s, d)
        if key is not None: self._cache[key] = o
        return o
    def state_nodes(self, scope, st): return scope[self.state[scope] == st]

def _t(x, dt=torch.float32): return torch.as_tensor(x, dtype=dt)
def _loss(logit, y):
    y = y.float(); pw = (y == 0).sum() / (y == 1).sum().clamp_min(1)
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)

@torch.no_grad()
def predict(model, D, S, scope_name, federated=False, heads=None):
    model.eval(); scope = D[scope_name]; p = np.full(len(D["X"]), np.nan, np.float32)
    if not federated:
        p[scope] = torch.sigmoid(model(_t(D["X"][scope]), S.ops(scope, (scope_name, "all")))["logit"]).numpy()
        return p
    base = copy.deepcopy(model.head.state_dict()) if heads else None
    for st in STATES:
        nodes = S.state_nodes(scope, st)
        if len(nodes) == 0: continue
        if heads: model.head.load_state_dict(heads[st])
        p[nodes] = torch.sigmoid(model(_t(D["X"][nodes]), S.ops(nodes, (scope_name, st)))["logit"]).numpy()
    if heads: model.head.load_state_dict(base)
    return p

def val_auc(model, D, S, **kw):
    p = predict(model, D, S, "scope_val", **kw); return auc(D["y"][D["val"]], p[D["val"]])

def train_central(model, D, S, lr, seed, epochs=300, eval_every=5, patience=6, wd=1e-4):
    torch.manual_seed(seed); tr = D["train"]; ops = S.ops(tr)
    X, y = _t(D["X"][tr]), _t(D["y"][tr], torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); best, state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss = _loss(model(X, ops)["logit"], y); loss.backward(); opt.step()
        if (ep + 1) % eval_every == 0:
            va = val_auc(model, D, S); hist.append(dict(epoch=ep + 1, loss=float(loss.detach()), val_auc=va))
            if va > best: best, state, bad = va, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience: break
    model.load_state_dict(state); return model, None, hist

def _shared_names(model, personal):
    return [n for n, _ in model.named_parameters() if not (personal and n.startswith("head."))]

def train_federated(model, D, S, lr, seed, mode="hier", personal=True, rounds=30, local_steps=10, mu=0.01, wd=1e-4, patience=None):
    """mode: 'hier' (6 regions, count-weighted) or 'flat' (FedAvg weighting by local size).
    personal=True keeps `head.*` parameters local to each state (FedPer-style personalization).
    prox=True adds a FedProx term on shared parameters (use mode='flat', personal=False, mu>0)."""
    torch.manual_seed(seed); ctx = D["ctx"]; trset = D["train"]
    clients = {st: S.state_nodes(trset, st) for st in STATES}; clients = {k: v for k, v in clients.items() if len(v)}
    counts = {st: float(ctx["count"].to_numpy()[v].sum()) for st, v in clients.items()}
    names = _shared_names(model, personal); params = dict(model.named_parameters())
    theta = {n: params[n].detach().clone() for n in names}
    heads = {st: copy.deepcopy(model.head.state_dict()) for st in clients} if personal else None
    ops = {st: S.ops(v) for st, v in clients.items()}
    best, best_snap, hist, bad = -1, None, [], 0
    for r in range(rounds):
        deltas, losses = {}, []
        for st, nodes in clients.items():
            with torch.no_grad():
                for n in names: params[n].copy_(theta[n])
            if personal: model.head.load_state_dict(heads[st])
            model.train(); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            X, y = _t(D["X"][nodes]), _t(D["y"][nodes], torch.long)
            for _ in range(local_steps):
                opt.zero_grad(); loss = _loss(model(X, ops[st])["logit"], y)
                if mu > 0 and not personal and mode == "flat":
                    loss = loss + 0.5 * mu * sum(((params[n] - theta[n]) ** 2).sum() for n in names)
                loss.backward(); opt.step()
            losses.append(float(loss.detach()) * len(nodes))
            deltas[st] = {n: params[n].detach() - theta[n] for n in names}
            if personal: heads[st] = copy.deepcopy(model.head.state_dict())
        if mode == "hier":
            reg, rc = {}, {}
            for rg, sts in REGIONS.items():
                pres = [s for s in sts if s in clients]
                if pres:
                    reg[rg] = {n: torch.stack([deltas[s][n] for s in pres]).mean(0) for n in names}; rc[rg] = sum(counts[s] for s in pres)
            tot = sum(rc.values())
            for n in names: theta[n] = theta[n] + sum(rc[g] / tot * reg[g][n] for g in reg)
        else:
            tot = sum(len(v) for v in clients.values())
            for n in names: theta[n] = theta[n] + sum(len(clients[s]) / tot * deltas[s][n] for s in clients)
        with torch.no_grad():
            for n in names: params[n].copy_(theta[n])
        va = val_auc(model, D, S, federated=True, heads=heads)
        hist.append(dict(round=r + 1, train_loss=sum(losses) / sum(len(v) for v in clients.values()), val_auc=va))
        if va > best:
            best, best_snap, bad = va, (copy.deepcopy(model.state_dict()), copy.deepcopy(heads)), 0
        else:
            bad += 1
            if patience is not None and bad >= patience: break      # same early-stopping rule as central models
    model.load_state_dict(best_snap[0]); return model, best_snap[1], hist
