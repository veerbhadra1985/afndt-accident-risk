"""Recompute the DeLong tests in Table 6 from the released probability vectors.

The archive stores one predicted probability vector per test run, but not the test label
vector, because the labels are rebuilt deterministically from the public source data by
the released loader rather than redistributed.  That leaves one reported quantity - the
DeLong p-value - that a reader cannot recompute from the archive alone.  This script
closes that gap: regenerate the labels with the loader, point this script at them, and
every DeLong comparison in Table 6 is recomputed from the released vectors in one command.

The implementation is the fast algorithm of Sun and Xu for the DeLong covariance, which
agrees with the original formulation of DeLong, DeLong and Clarke-Pearson (1988) and is
what the study used.

Usage
-----
    python3 recompute_delong.py results/us/us_final_seeds0-4.json \
        results/us/test_probs us_test_labels.npy

`us_test_labels.npy` is a 1-D array of 0/1 labels in the same row order as the probability
vectors, as written by the loader.  The script checks the length against the stored
confusion matrices before doing anything else, so a misaligned label vector is caught
rather than silently producing wrong p-values.
"""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def midrank(x):
    """Midranks of x, as used by the DeLong covariance."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out


def delong_cov(preds, y):
    """AUCs and their DeLong covariance matrix for k predictors on shared labels."""
    y = np.asarray(y).astype(bool)
    pos = preds[:, y]
    neg = preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([midrank(pos[r]) for r in range(k)])
    ty = np.array([midrank(neg[r]) for r in range(k)])
    tz = np.array([midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_p(p1, p2, y):
    """Two-sided p-value for the difference between two correlated AUCs."""
    aucs, s = delong_cov(np.vstack([p1, p2]), y)
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return aucs, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(2 * stats.norm.sf(abs(z)))


def main(results_json, probs_dir, labels_npy):
    runs = json.load(open(results_json))["runs"]
    runs = list(runs.values()) if isinstance(runs, dict) else runs
    y = np.load(labels_npy).astype(int).ravel()
    probs_dir = pathlib.Path(probs_dir)

    n_expected = sorted({r["test"]["tp"] + r["test"]["fn"] + r["test"]["tn"] + r["test"]["fp"]
                         for r in runs if isinstance(r.get("test"), dict)})
    if len(n_expected) != 1 or len(y) != n_expected[0]:
        sys.exit(f"label vector has {len(y)} rows but the stored confusion matrices "
                 f"imply {n_expected}; check that the loader was run on the same period")
    pos_stored = {round((r["test"]["tp"] + r["test"]["fn"]) / len(y), 4) for r in runs}
    if abs(sorted(pos_stored)[0] - y.mean()) > 5e-4:
        sys.exit(f"label vector has positive proportion {y.mean():.4f} but the archive "
                 f"reports {sorted(pos_stored)}; the labels do not match these runs")
    print(f"{len(y):,} test rows, positive proportion {y.mean():.4f} - matches the archive")

    by = defaultdict(list)
    for r in runs:
        f = r.get("probs_file")
        if f and (probs_dir / f).exists():
            by[r.get("name") or r.get("family")].append(np.load(probs_dir / f).ravel())
    if not by:
        sys.exit(f"no probability vectors found under {probs_dir}")

    mean_auc = {}
    for name, vs in by.items():
        a, _ = delong_cov(np.vstack(vs), y)
        mean_auc[name] = float(np.mean(a))
    best = max(mean_auc, key=mean_auc.get)
    print(f"best model by mean test AUC: {best} ({mean_auc[best]:.4f})\n")
    print(f"{'model':34s} {'mean AUC':>9} {'DeLong p vs best':>18}")
    for name in sorted(mean_auc, key=mean_auc.get, reverse=True):
        if name == best:
            print(f"{name:34s} {mean_auc[name]:9.4f} {'-':>18}")
            continue
        ps = []
        for a, b in zip(by[best], by[name]):
            _, p = delong_p(a, b, y)
            ps.append(p)
        # the paper reports the least favourable p-value across seeds
        print(f"{name:34s} {mean_auc[name]:9.4f} {max(ps):18.3g}")
    print("\nReported in Table 6 as the least favourable value across seeds.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
