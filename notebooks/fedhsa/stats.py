"""Metrics and significance tests: DeLong (1988) for AUC, McNemar for paired accuracy."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

def best_threshold(y_val, p_val):
    grid = np.unique(np.quantile(p_val, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(y_val, p_val >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(f1s))])

def metrics(y, p, thr):
    yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, yh), prec=precision_score(y, yh, zero_division=0),
                rec=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                auc=auc(y, p), threshold=thr, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out

def delong(y, p1, p2):
    """Fast DeLong test (Sun & Xu, 2014 algorithm) for two correlated AUCs. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y); order = np.argsort(-y, kind="mergesort"); y = y[order]
    P = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]]); m = int(y.sum()); n = len(y) - m
    tx = np.array([_midrank(P[k, :m]) for k in range(2)]); ty = np.array([_midrank(P[k, m:]) for k in range(2)])
    tz = np.array([_midrank(P[k]) for k in range(2)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))

def mcnemar(y, yh1, yh2):
    """McNemar test; exact binomial if fewer than 25 discordant pairs, else chi-square with continuity correction."""
    c1 = yh1 == y; c2 = yh2 == y; b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    if b + c == 0: return dict(b=b, c=c, stat=0.0, p=1.0, method="none")
    if b + c < 25:
        return dict(b=b, c=c, stat=float(min(b, c)), p=float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))), method="exact")
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return dict(b=b, c=c, stat=float(chi), p=float(stats.chi2.sf(chi, 1)), method="chi2_cc")
