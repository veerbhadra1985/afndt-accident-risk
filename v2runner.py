"""Model families, their (equal-size) tuning grids, and a single fit/predict entry point."""
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from .models import MLP, GCN, GAT, HGNN
from .v2models import GCNIIv, AFNDTv2
from .v2train import train_central, train_federated, predict
from .v3models import AFNDTv3
from .v3train import Structure3, train_afndt_v3, predict as predict3, finetune_heads

LRS = (5e-3, 1e-2)          # phase B: 1e-3 was dominated for every family in phase A
DEPTHS = (2, 4, 8)          # equal-size grids: 6 configs for every neural family
GRID = {
    "State-only LR":       [dict()],
    "Logistic regression": [dict(C=c) for c in (0.1, 1.0, 10.0)],
    "XGBoost":             [dict(depth=d, lr=l) for d in (3, 4, 6) for l in (0.05, 0.1)],
    "MLP":                 [dict(h=h, lr=l) for h in (64, 128, 256) for l in LRS],
    "GCN":                 [dict(h=h, lr=l) for h in (64, 128, 256) for l in LRS],
    "GAT":                 [dict(h=h, lr=l) for h in (16, 32, 64) for l in LRS],
    "GCNII":               [dict(layers=L, lr=l) for L in DEPTHS for l in LRS],
    "HGNN":                [dict(h=h, lr=l) for h in (64, 128, 256) for l in LRS],
    "FedAvg-GCNII":        [dict(layers=L, lr=l) for L in DEPTHS for l in LRS],
    "FedProx-GCNII":       [dict(layers=L, lr=l) for L in DEPTHS for l in LRS],
    "FedAvg-GAT":          [dict(h=h, lr=l) for h in (16, 32, 64) for l in LRS],
    # the hypergraph gate was selected OFF in phase A (0.8506 vs 0.8479), so phase B tunes depth
    "AFNDT-v2":            [dict(layers=L, lr=l, hyper=False) for L in DEPTHS for l in LRS],
    "FedAvg-GCNII+FT":     [dict(layers=L, lr=l) for L in DEPTHS for l in LRS],   # personalized baseline
    "AFNDT-v3":            [dict(layers=L, lr=l) for L in DEPTHS for l in LRS],
}
# Phase C (budget fairness): every federated family peaked at the last of 30 rounds, while central
# models converged under early stopping. All federated families get up to 100 rounds with patience 15.
# lr 0.01 won for every federated family in phase B, so phase C uses it for all of them.
for _fam, _key, _vals in (("AFNDT-v2", "layers", DEPTHS), ("FedAvg-GCNII", "layers", DEPTHS),
                          ("FedProx-GCNII", "layers", DEPTHS), ("FedAvg-GCNII+FT", "layers", DEPTHS),
                          ("FedAvg-GAT", "h", (16, 32, 64))):
    for _v in _vals:
        _c = {_key: _v, "lr": 0.01, "rounds": 100}
        if _fam == "AFNDT-v2": _c["hyper"] = False
        GRID[_fam].append(_c)

class Structures:
    """Holds the operators both model generations need, built once per dataset."""
    def __init__(self, ctx):
        from .v2train import Structure2
        self.s2 = Structure2(ctx); self.s3 = Structure3(ctx)

def _s2(S): return S.s2 if hasattr(S, "s2") else S
def _s3(S): return S.s3 if hasattr(S, "s3") else S
PROPOSED = "AFNDT-v2"   # selected on validation: best federated model (0.8581)

class Fitted:
    def __init__(self, predict_fn, hist=None, extra=None): self.predict, self.hist, self.extra = predict_fn, hist, extra or {}

def _sk(clf, X_of, D):
    clf.fit(X_of(D["train"]), D["y"][D["train"]])
    def pred(scope_name):
        p = np.full(len(D["X"]), np.nan, np.float32); sc = D[scope_name]; p[sc] = clf.predict_proba(X_of(sc))[:, 1]; return p
    return Fitted(pred)

def fit(family, cfg, D, S, seed, variant=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if family == "State-only LR":
        enc = OneHotEncoder(handle_unknown="ignore").fit(D["ctx"][["state"]])
        return _sk(LogisticRegression(max_iter=2000, class_weight="balanced"), lambda idx: enc.transform(D["ctx"].iloc[idx][["state"]]), D)
    if family == "Logistic regression":
        return _sk(LogisticRegression(C=cfg["C"], max_iter=2000, class_weight="balanced"), lambda idx: D["X"][idx], D)
    if family == "XGBoost":
        ytr = D["y"][D["train"]]; spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        return _sk(XGBClassifier(n_estimators=300, max_depth=cfg["depth"], learning_rate=cfg["lr"], subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=spw, random_state=seed, n_jobs=-1, eval_metric="auc"), lambda idx: D["X"][idx], D)
    central = {"MLP": lambda: MLP(h=cfg["h"]), "GCN": lambda: GCN(h=cfg["h"]), "GAT": lambda: GAT(h=cfg["h"]),
               "GCNII": lambda: GCNIIv(layers=cfg["layers"]), "HGNN": lambda: HGNN(h=cfg["h"])}
    if family in central:
        m, _, h = train_central(central[family](), D, _s2(S), cfg["lr"], seed)
        return Fitted(lambda sc: predict(m, D, _s2(S), sc), h)
    if family in ("FedAvg-GCNII", "FedProx-GCNII", "FedAvg-GAT", "FedAvg-GCNII+FT"):
        model = GCNIIv(layers=cfg["layers"], h=cfg.get("h", 64)) if "GCNII" in family else GAT(h=cfg["h"])
        m, _, h = train_federated(model, D, _s2(S), cfg["lr"], seed, mode="flat", personal=False, mu=0.01 if "Prox" in family else 0.0,
                                  rounds=cfg.get("rounds", 30), patience=15 if "rounds" in cfg else None)
        if family == "FedAvg-GCNII+FT":
            pers = finetune_heads(m, D, _s2(S), cfg["lr"])
            return Fitted(lambda sc: predict3(m, D, _s2(S), sc, personal=pers), h)
        return Fitted(lambda sc: predict(m, D, _s2(S), sc, federated=True), h)
    if family == "AFNDT-v3":
        v = variant or "full"
        model = AFNDTv3(layers=cfg.get("layers", 4), use_rel=(v != "no relation weights"),
                        use_aux=(v != "no auxiliary target"), use_mix=(v != "no personalization"),
                        use_hyper=(v == "+ hypergraph branch"))
        mode = "flat" if v == "flat (no regions)" else "hier"
        aux_w = 0.0 if v == "no auxiliary target" else 0.3
        m, pers, h = train_afndt_v3(model, D, _s3(S), cfg["lr"], seed, aux_w=aux_w, mode=mode)
        if v == "no personalization": pers = None
        return Fitted(lambda sc: predict3(m, D, _s3(S), sc, personal=pers), h,
                      dict(mix=h[-1].get("mix"), rel=[float(x) for x in torch.softmax(m.rel_raw, 0)]))
    if family == "AFNDT-v2":
        v = variant or "full"
        hyper = cfg.get("hyper", False)
        model = AFNDTv2(layers=cfg.get("layers", 8), use_hyper=hyper if v != "toggle hypergraph" else not hyper,
                        use_spike=(v == "+ spiking readout"))
        if v == "centralized":
            m, _, h = train_central(model, D, _s2(S), cfg["lr"], seed)
            return Fitted(lambda sc: predict(m, D, _s2(S), sc), h, dict(gate=float(m.gate())))
        mode = "flat" if v == "flat (no regions)" else "hier"; personal = v != "no personalization"
        m, heads, h = train_federated(model, D, _s2(S), cfg["lr"], seed, mode=mode, personal=personal, mu=0.0,
                                      rounds=cfg.get("rounds", 30), patience=15 if "rounds" in cfg else None)
        return Fitted(lambda sc: predict(m, D, _s2(S), sc, federated=True, heads=heads), h, dict(gate=float(m.gate())))
    raise KeyError(family)

ABLATIONS = ["no personalization", "flat (no regions)", "toggle hypergraph", "+ spiking readout", "centralized"]
