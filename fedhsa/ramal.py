"""RAMAL objective, Eqs. (9)-(16)."""
import math, torch, torch.nn.functional as F
from .config import TEMP_COLS

NAMES = ["pred", "rob", "prox", "spk", "att"]

def perturb_temp(X, sigma, gen=None):
    Xn = X.clone()
    Xn[:, TEMP_COLS] += sigma * torch.randn(X.shape[0], len(TEMP_COLS), generator=gen, dtype=X.dtype)
    return Xn

def objectives(model, X, ops, y, mask, omega, global_params, cfg):
    out = model(X, ops)
    logit = out["logit"][mask]; yy = y[mask].float()
    pw = torch.tensor(omega, dtype=logit.dtype)
    L_pred = F.binary_cross_entropy_with_logits(logit, yy, pos_weight=pw)                    # Eq. (9)
    out_n = model(perturb_temp(X, cfg.sigma_r), ops)
    L_rob = ((torch.sigmoid(logit) - torch.sigmoid(out_n["logit"][mask])) ** 2).mean()       # Eq. (10)
    L_prox = 0.5 * sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))  # Eq. (11)
    L_spk = out["spk"]                                                                        # Eq. (12)
    a = out["attr"][mask].mean(0).clamp_min(1e-8)
    H = -(a * a.log()).sum() / math.log(a.numel())
    L_att = 1 - H                                                                             # Eq. (13)
    return [L_pred, L_rob, L_prox, L_spk, L_att]

class DWA:
    """Dynamic weight averaging (Liu et al., 2019) with first-round scale normalization, Eqs. (14)-(16)."""
    def __init__(self, t_w=2.0, adaptive=True):
        self.hist, self.t_w, self.adaptive, self.gamma = [], t_w, adaptive, None
    def weights(self):
        if not self.adaptive or len(self.hist) < 2:
            return [1.0] * 5
        rho = [self.hist[-1][m] / max(self.hist[-2][m], 1e-12) for m in range(5)]
        ex = [math.exp(r / self.t_w) for r in rho]
        return [5 * e / sum(ex) for e in ex]
    def scales(self):
        return [1.0] * 5 if self.gamma is None else self.gamma
    def record(self, round_means, mu_prox=0.01):
        if self.gamma is None:
            # scale normalization gamma_m = 1 / L_m(1); the proximal term keeps a fixed coefficient
            # mu, because normalizing a drift penalty by its own first-round value would freeze training
            self.gamma = [1.0 / max(v, 1e-8) for v in round_means]
            self.gamma[NAMES.index("prox")] = mu_prox
        self.hist.append(round_means)
