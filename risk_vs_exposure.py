"""Could genuine between-area risk differences explain the area-only result?

Section 7.7 simulates a world in which risk carries no between-area signal, so that
any area-only AUC above chance is exposure leakage.  That design invites an obvious
objection: the conclusion is built into the construction.  Real jurisdictions do
differ in collisions per vehicle-mile, and a sceptic may argue that the 0.898 we
measure reflects those real differences rather than exposure.

This script answers that objection on its own terms.  Between-area risk is no longer
held flat.  Each area a is given a risk multiplier exp(u_a) with u_a ~ N(0, sigma^2),
so that areas genuinely differ in collisions per vehicle-mile, and sigma is swept from
0 upwards.  Counts are drawn as negative binomial rather than Poisson, so the result
does not depend on equidispersion either:

    c(a,t) ~ NegBin( mean = kappa * E(a) * exp(u_a) * r(t) ,  dispersion = phi )

Two worlds are simulated at each sigma, on the same draws of u:

    with exposure     E(a) = the real VM-2 vehicle-miles of area a (a 92-fold spread)
    without exposure  E(a) = constant, so counts differ between areas only through u

The second world contains genuine between-area risk and no exposure at all.  The
area-only AUC it produces is therefore the most that real risk variation could
contribute.  Comparing it with the observed 0.898 asks how large the real spread in
collisions per vehicle-mile would have to be for exposure to be unnecessary as an
explanation.

Usage:  python3 risk_vs_exposure.py out.csv out.png
"""
import sys
import csv
import numpy as np

RNG = np.random.default_rng(20261030)
N_YEARS, N_MONTH, N_HOUR = 7, 12, 24
TOP = 0.40            # "high risk" = top 40%, as in the paper
OBSERVED = 0.898      # the United States area-only AUC under national labels
PHI = 2.0             # negative-binomial dispersion; var = mu + mu^2/phi


def exposures(path="us_vmt_exposure.csv"):
    """Real VM-2 jurisdiction exposures for 2016, one value per area."""
    rows = list(csv.DictReader(open(path)))
    v = [float(r["TotalVMT_millions"]) for r in rows if r["Year"] == "2016"]
    return np.array(sorted(v), dtype=float)


def auc(y, s):
    """Area under the ROC curve, ties handled by midrank."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def label_top(values, rng, frac=TOP):
    jitter = rng.random(len(values)) * 1e-9
    thr = np.quantile(values + jitter, 1.0 - frac)
    return (values + jitter) > thr


def negbin(rng, mean, phi):
    """Negative-binomial counts with the given mean and dispersion phi."""
    mean = np.maximum(mean, 1e-12)
    gam = rng.gamma(shape=phi, scale=mean / phi)
    return rng.poisson(gam).astype(float)


def one_run(sigma, E, kappa, seed):
    """Area-only AUC with and without exposure, on the same risk draws."""
    n_areas = len(E)
    rng = np.random.default_rng(seed)
    n_ctx = N_YEARS * N_MONTH * N_HOUR
    month = np.repeat(np.tile(np.arange(N_MONTH), N_YEARS), N_HOUR)
    hour = np.tile(np.arange(N_HOUR), N_YEARS * N_MONTH)
    year = np.repeat(np.arange(N_YEARS), N_MONTH * N_HOUR)

    base = 1.0 + 0.30 * np.sin(2 * np.pi * month / 12) + 0.50 * np.sin(2 * np.pi * hour / 24)
    base = np.clip(base, 0.15, None)

    u = rng.normal(0.0, sigma, n_areas)          # genuine between-area risk
    area_id = np.repeat(np.arange(n_areas), n_ctx)
    yr = np.tile(year, n_areas)
    r = np.tile(base, n_areas) * np.exp(np.repeat(u, n_ctx))

    ratio = float(np.exp(np.quantile(u, 0.9)) / np.exp(np.quantile(u, 0.1)))
    out = {"sigma": sigma, "risk_spread": float(np.exp(u).max() / np.exp(u).min()),
           "p90_p10": ratio}
    for world, Evec in (("with_exposure", E), ("risk_only", np.full(n_areas, E.mean()))):
        lam = kappa * np.repeat(Evec, n_ctx) * r
        c = negbin(rng, lam, PHI)
        y_nat = np.zeros(len(c), dtype=bool)
        y_win = np.zeros(len(c), dtype=bool)
        for y in range(N_YEARS):
            m = yr == y
            y_nat[m] = label_top(c[m], rng)
            for a in range(n_areas):
                ma = m & (area_id == a)
                y_win[ma] = label_top(c[ma], rng)
        for name, lab in (("nat", y_nat), ("win", y_win)):
            p = np.array([lab[area_id == a].mean() for a in range(n_areas)])
            score = p[area_id]
            if name == "nat":
                out[f"{world}_national"] = auc(lab, score)
            else:
                per = [auc(lab[area_id == a], score[area_id == a]) for a in range(n_areas)]
                out[f"{world}_within_macro"] = float(np.nanmean(per))
        out[f"{world}_mean_count"] = float(c.mean())
    return out


def main(csv_out, png_out):
    E = exposures()
    kappa = 95.0 / E.mean()      # calibrated to ~95 collisions per context, as observed
    print(f"areas {len(E)}   exposure spread {E.max()/E.min():.0f}x   dispersion phi={PHI}")
    print(f"{'sigma':>6} {'p90/p10':>9} {'national AUC':>13} {'risk-only AUC':>14} "
          f"{'within macro':>13} {'risk-only within':>17}")
    rows = []
    for sigma in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00, 1.40]:
        res = [one_run(sigma, E, kappa, 4000 + s) for s in range(3)]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        rows.append(agg)
        print(f"{agg['sigma']:6.2f} {agg['p90_p10']:8.1f}x {agg['with_exposure_national']:13.4f} "
              f"{agg['risk_only_national']:14.4f} {agg['with_exposure_within_macro']:13.4f} "
              f"{agg['risk_only_within_macro']:17.4f}")

    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # how large would the real risk spread have to be to reach the observed value alone?
    xs = [r["risk_only_national"] for r in rows]
    sp = [r["p90_p10"] for r in rows]
    need = None
    for i in range(1, len(xs)):
        if xs[i - 1] < OBSERVED <= xs[i]:
            f = (OBSERVED - xs[i - 1]) / (xs[i] - xs[i - 1])
            need = sp[i - 1] + f * (sp[i] - sp[i - 1])
            break
    if need is None and xs[-1] < OBSERVED:
        need = float("inf")
    print(f"\nto reach {OBSERVED} from genuine risk alone, the ratio in collisions per "
          f"vehicle-mile between the 90th- and 10th-percentile jurisdiction would have to be "
          f"{'beyond the range simulated' if need == float('inf') else f'about {need:.0f}-fold'}")
    print(f"within-area macro AUC stays at {rows[-1]['with_exposure_within_macro']:.4f} "
          f"at the largest risk spread tested")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=220)
    ax.plot(sp, [r["with_exposure_national"] for r in rows], "o-", lw=3, ms=9,
            color="#55A868", label="national labels, real exposure present")
    ax.plot(sp, xs, "^-", lw=3, ms=9, color="#C44E52",
            label="national labels, genuine risk only (no exposure)")
    ax.plot(sp, [r["with_exposure_within_macro"] for r in rows], "s-", lw=3, ms=9,
            color="#4C72B0", label="within-area labels, macro per-area AUC")
    ax.axhline(OBSERVED, ls="--", c="#2f4a6d", lw=1.6)
    ax.text(sp[1], OBSERVED + 0.008, f"observed United States value, {OBSERVED}",
            fontsize=12, color="#2f4a6d")
    ax.axhline(0.5, ls=":", c="0.4", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("ratio in collisions per vehicle-mile, 90th to 10th percentile jurisdiction", fontsize=14)
    ax.set_ylabel("area-only classifier AUC", fontsize=14)
    ax.set_ylim(0.43, 1.02)
    ax.legend(fontsize=12.5, frameon=False, loc="center right")
    ax.grid(axis="y", color="0.9"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=12.5)
    fig.tight_layout(); fig.savefig(png_out, bbox_inches="tight", facecolor="white")
    print("wrote", csv_out, png_out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
