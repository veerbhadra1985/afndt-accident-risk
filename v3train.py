"""Training for AFNDT-v3 and for the personalized federated baseline (FedAvg + local fine-tuning)."""
import copy, os, numpy as np, torch, torch.nn.functional as F
from .graph import HyperProp, GraphProp, subset_edges, edge_ids
from .v3graph import typed_lattice, REL
from .v3models import Ops3
from .config import STATES, REGIONS
from .stats import auc

SMOKE = os.environ.get("AFNDT_SMOKE") == "1"      # tiny settings to check the code runs

class Structure3:
    def __init__(self, ctx):
        self.eids = edge_ids(ctx); self.rel = typed_lattice(ctx)
        self.state = ctx.state.to_numpy(); self._cache = {}
    def ops(self, nodes, key=None):
        if key is not None and key in self._cache: return self._cache[key]
        nodes = np.asarray(nodes); n = len(nodes)
        props = {}
        for r in REL:
            s, d = subset_edges(self.rel[r][0], self.rel[r][1], nodes); props[r] = GraphProp(s, d, n)
        o = Ops3(props, HyperProp(self.eids[nodes]))
        if key is not None: self._cache[key] = o
        return o
    def state_nodes(self, scope, st): return scope[self.state[scope] == st]

def _t(x, dt=torch.float32): return torch.as_tensor(x, dtype=dt)

def _loss(out, y, z, aux_w):
    y = y.float(); pw = (y == 0).sum() / (y == 1).sum().clamp_min(1)
    L = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pw)
    if aux_w > 0 and z is not None: L = L + aux_w * F.mse_loss(out["aux"], z)
    return L

@torch.no_grad()
def predict(model, D, S, scope_name, personal=None):
    """personal: {state: state_dict of the personal parameters} or None for a single global model."""
    model.eval(); scope = D[scope_name]; p = np.full(len(D["X"]), np.nan, np.float32)
    if personal is None:
        p[scope] = torch.sigmoid(model(_t(D["X"][scope]), S.ops(scope, (scope_name, "all")))["logit"]).numpy()
        return p
    keep = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for st in STATES:
        nodes = S.state_nodes(scope, st)
        if len(nodes) == 0: continue
        if st in personal: model.load_state_dict({**model.state_dict(), **personal[st]})
        p[nodes] = torch.sigmoid(model(_t(D["X"][nodes]), S.ops(nodes, (scope_name, st)))["logit"]).numpy()
    model.load_state_dict(keep)
    return p

def _split_names(model, personal_prefixes):
    shared, pers = [], []
    for n, _ in model.named_parameters():
        (pers if any(n.startswith(p) for p in personal_prefixes) else shared).append(n)
    return shared, pers

def train_afndt_v3(model, D, S, lr, seed, rounds=60, local_steps=10, aux_w=0.3, mode="hier",
                   personal_prefixes=("pers.", "mix"), wd=1e-4, patience=15):
    torch.manual_seed(seed)
    if SMOKE: rounds, local_steps, patience = 3, 3, 99
    ctx = D["ctx"]; tr = D["train"]
    clients = {st: S.state_nodes(tr, st) for st in STATES}; clients = {k: v for k, v in clients.items() if len(v)}
    counts = {st: float(ctx["count"].to_numpy()[v].sum()) for st, v in clients.items()}
    ops = {st: S.ops(v) for st, v in clients.items()}
    params = dict(model.named_parameters())
    snames, pnames = _split_names(model, personal_prefixes)
    theta = {n: params[n].detach().clone() for n in snames}
    local = {st: {n: params[n].detach().clone() for n in pnames} for st in clients}
    z_all = _t(D["z"]) if "z" in D else None
    best, best_snap, bad, hist = -1, None, 0, []
    for r in range(rounds):
        deltas, losses = {}, []
        for st, nodes in clients.items():
            with torch.no_grad():
                for n in snames: params[n].copy_(theta[n])
                for n in pnames: params[n].copy_(local[st][n])
            model.train(); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            X, y = _t(D["X"][nodes]), _t(D["y"][nodes], torch.long)
            z = z_all[torch.as_tensor(nodes)] if z_all is not None else None
            for _ in range(local_steps):
                opt.zero_grad(); L = _loss(model(X, ops[st]), y, z, aux_w); L.backward(); opt.step()
            losses.append(float(L.detach()) * len(nodes))
            deltas[st] = {n: params[n].detach() - theta[n] for n in snames}
            local[st] = {n: params[n].detach().clone() for n in pnames}
        if mode == "hier":
            reg, rc = {}, {}
            for rg, sts in REGIONS.items():
                pres = [s for s in sts if s in clients]
                if pres:
                    reg[rg] = {n: torch.stack([deltas[s][n] for s in pres]).mean(0) for n in snames}
                    rc[rg] = sum(counts[s] for s in pres)
            tot = sum(rc.values())
            for n in snames: theta[n] = theta[n] + sum(rc[g] / tot * reg[g][n] for g in reg)
        else:
            tot = sum(len(v) for v in clients.values())
            for n in snames: theta[n] = theta[n] + sum(len(clients[s]) / tot * deltas[s][n] for s in clients)
        with torch.no_grad():
            for n in snames: params[n].copy_(theta[n])
        pers = {st: {n: v.clone() for n, v in local[st].items()} for st in local}
        va = auc(D["y"][D["val"]], predict(model, D, S, "scope_val", personal=pers)[D["val"]])
        hist.append(dict(round=r + 1, train_loss=sum(losses) / sum(len(v) for v in clients.values()), val_auc=va,
                         mix=float(torch.sigmoid(params["mix"].detach())) if "mix" in params else None))
        if va > best: best, best_snap, bad = va, (copy.deepcopy(model.state_dict()), pers), 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_snap[0]); return model, best_snap[1], hist

def finetune_heads(model, D, S, lr, steps=30, head_prefixes=("head.", "l2.", "out.", "g2.")):
    """Personalized federated baseline: after federated training each state fine-tunes its output
    layer locally (FedAvg+FT). Gives the baselines the same personalization advantage."""
    if SMOKE: steps = 3
    tr = D["train"]; pers = {}
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for st in STATES:
        nodes = S.state_nodes(tr, st)
        if len(nodes) == 0: continue
        model.load_state_dict(base)
        ps = [p for n, p in model.named_parameters() if any(n.startswith(h) for h in head_prefixes)]
        if not ps: return None
        opt = torch.optim.Adam(ps, lr=lr)
        X, y = _t(D["X"][nodes]), _t(D["y"][nodes], torch.long)
        ops = S.ops(nodes)
        model.train()
        for _ in range(steps):
            opt.zero_grad(); out = model(X, ops)
            out = out if isinstance(out, dict) else dict(logit=out)
            _loss(out, y, None, 0.0).backward(); opt.step()
        pers[st] = {n: p.detach().clone() for n, p in model.named_parameters() if any(n.startswith(h) for h in head_prefixes)}
    model.load_state_dict(base)
    return pers
