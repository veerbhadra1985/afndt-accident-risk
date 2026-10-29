"""Does exposure leakage survive at the sparsity of a city grid?

The measurements in the paper are made on jurisdiction-level contexts, which are
denser than the grid cells much of the audited literature uses. This simulation
asks whether the effect we report would be found at those sparsities, in a setting
where the answer is known by construction.

Design.  Forty-nine areas are given the real vehicle-miles-travelled of the
49 United States jurisdictions (FHWA table VM-2, a 92-fold spread).  Each area has
12 months x 24 hours x 7 years = 2,016 contexts.  Collision counts are drawn as

    c(a,t) ~ Poisson( kappa * E(a) * r(t) ),

where r(t) is a seasonal-plus-diurnal risk profile shared by every area and
perturbed by independent noise.  Risk therefore carries NO between-area signal:
by construction, every systematic difference in counts between areas is exposure.
Sweeping kappa moves the mean count per context from roughly 0.05, the sparsity of
an hourly city grid, to roughly 100, the density of our jurisdiction contexts.

At each density we fit the paper's area-only diagnostic - a classifier given the
area identifier and nothing else - under national and within-area labelling, and
report its pooled AUC and its macro per-area AUC.

Usage:  python3 density_simulation.py out.csv out.png
"""
import sys
import numpy as np
import csv

RNG = np.random.default_rng(20260927)
N_YEARS, N_MONTH, N_HOUR = 7, 12, 24
TOP = 0.40  # "high risk" = top 40%, as in the paper


def exposures(path="us_vmt_exposure.csv"):
    """Real VM-2 jurisdiction exposures, one value per area (2016 baseline)."""
    try:
        rows = list(csv.DictReader(open(path)))
        v = [float(r["TotalVMT_millions"]) for r in rows if r["Year"] == "2016"]
        if len(v) >= 40:
            return np.array(sorted(v), dtype=float)
    except Exception:
        pass
    # fallback: log-uniform spread matching the observed 92x range
    return np.exp(RNG.uniform(np.log(3421), np.log(315244), N_AREAS))


def auc(y, s):
    """Area under the ROC curve, with ties handled by midrank."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = np.asarray(s)[order]
    i = 0
    while i < len(s):                       # midrank for tied scores
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def label_top(values, frac=TOP):
    """Binary label marking the top `frac` of `values`, ties broken at random."""
    jitter = RNG.random(len(values)) * 1e-9
    thr = np.quantile(values + jitter, 1.0 - frac)
    return (values + jitter) > thr


def one_density(kappa, E, seed):
    N_AREAS = len(E)
    rng = np.random.default_rng(seed)
    n_ctx = N_YEARS * N_MONTH * N_HOUR
    month = np.repeat(np.tile(np.arange(N_MONTH), N_YEARS), N_HOUR)
    hour = np.tile(np.arange(N_HOUR), N_YEARS * N_MONTH)
    year = np.repeat(np.arange(N_YEARS), N_MONTH * N_HOUR)

    # risk profile shared across areas: no between-area risk signal by construction
    base = 1.0 + 0.30 * np.sin(2 * np.pi * month / 12) + 0.50 * np.sin(2 * np.pi * hour / 24)
    base = np.clip(base, 0.15, None)
    area_id = np.repeat(np.arange(N_AREAS), n_ctx)
    r = np.tile(base, N_AREAS) * np.exp(rng.normal(0, 0.35, N_AREAS * n_ctx))
    lam = kappa * np.repeat(E, n_ctx) * r
    c = rng.poisson(lam).astype(float)

    yr = np.tile(year, N_AREAS)
    y_nat = np.zeros(len(c), dtype=bool)
    y_win = np.zeros(len(c), dtype=bool)
    for y in range(N_YEARS):                      # label within each year
        m = yr == y
        y_nat[m] = label_top(c[m])
        for a in range(N_AREAS):                  # and, for within-area, each area
            ma = m & (area_id == a)
            y_win[ma] = label_top(c[ma])
    y_occ = c > 0          # binary occurrence: the label sparse-grid studies use

    out = {"kappa": kappa, "mean_count": c.mean(), "frac_zero": (c == 0).mean()}
    for name, y in (("national", y_nat), ("within", y_win), ("occurrence", y_occ)):
        p = np.array([y[area_id == a].mean() for a in range(N_AREAS)])
        score = p[area_id]                        # the area-only classifier
        out[f"{name}_pooled"] = auc(y, score)
        per = [auc(y[area_id == a], score[area_id == a]) for a in range(N_AREAS)]
        out[f"{name}_macro"] = float(np.nanmean(per))
    return out


def main(csv_out, png_out):
    E = exposures()
    global N_AREAS
    print(f"exposures: n={len(E)} spread={E.max()/E.min():.0f}x")
    rows = []
    for kappa in np.logspace(-7.5, -2.6, 14):
        res = [one_density(kappa, E, 1000 + s) for s in range(3)]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        rows.append(agg)
        print(f"  mean count {agg['mean_count']:8.3f}  zeros {agg['frac_zero']:5.1%}  "
              f"nat {agg['national_pooled']:.4f}  occ {agg['occurrence_pooled']:.4f}  "
              f"within-macro {agg['within_macro']:.4f}")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["mean_count"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=220)
    ax.semilogx(x, [r["national_pooled"] for r in rows], "o-", lw=3, ms=9,
                color="#55A868", label="national labels, area-only AUC")
    ax.semilogx(x, [r["occurrence_pooled"] for r in rows], "^-", lw=3, ms=9,
                color="#C44E52", label="binary-occurrence labels, area-only AUC")
    ax.semilogx(x, [r["within_macro"] for r in rows], "s-", lw=3, ms=9,
                color="#4C72B0", label="within-area labels, macro per-area AUC")
    ax.axhline(0.5, ls=":", c="0.4", lw=1.5)
    ax.axvspan(0.05, 0.3, color="#C44E52", alpha=0.10)
    ax.text(0.12, 0.455, "sparsity of an\nhourly city grid", ha="center", fontsize=13, color="#8d3a3d")
    ax.axvspan(9, 100, color="#4C72B0", alpha=0.08)
    ax.text(30, 0.455, "this paper's\ncontexts", ha="center", fontsize=13, color="#2f4a6d")
    ax.set_xlabel("mean collisions per spatiotemporal unit", fontsize=14)
    ax.set_ylabel("area-only classifier AUC", fontsize=14)
    ax.set_ylim(0.43, 1.02)
    ax.legend(fontsize=13, frameon=False, loc="center left")
    ax.grid(axis="y", color="0.9"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=13)
    fig.tight_layout(); fig.savefig(png_out, bbox_inches="tight", facecolor="white")
    print("wrote", csv_out, png_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
