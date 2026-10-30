"""Pre-specified replication of the area-only diagnostic on sparse grid cells.

Section 7.7 characterises exposure leakage across unit densities by simulation.  A
simulation motivates a replication on real data; it does not substitute for one.  This
script is that replication, specified in advance and released so that it can be run by
anyone holding the source collision records, including on data this study never used.

What it does.  Raw collision records with coordinates and timestamps are aggregated to
square grid cells of a chosen size and to a chosen temporal resolution, giving units far
sparser than the jurisdiction-year-month-hour contexts of the main study.  The three
labellings of Section 4.1 are then applied - national top-q, within-area top-q, and the
binary-occurrence rule that sparse-grid studies actually use - and for each the area-only
classifier of Section 4.3 is fitted and scored.  The classifier sees the cell identifier
and nothing else.

What would falsify the paper's claim.  The paper predicts, for any such grid:

  1. within-cell labelling gives a macro per-cell AUC of 0.5000, to numerical precision,
     at every cell size and temporal resolution.  This is algebraic (Section 4.2); a
     value materially different from 0.5000 would refute it.
  2. binary-occurrence labelling gives a pooled area-only AUC well above 0.5, and does
     not fall away as cells become sparser (Section 7.7 predicts about 0.745 at hourly
     city-grid sparsity).
  3. national top-q labelling leaks at moderate density but degrades towards chance at
     extreme sparsity, where the top-q rule becomes ill-defined.

Reporting all three, at several cell sizes, is the intended output.  A negative result is
as publishable as a positive one, and the authors would like to hear of either.

Usage
-----
    python3 grid_replication.py records.csv out.csv \
        --lat LAT_COLUMN --lon LON_COLUMN --time TIME_COLUMN \
        --km 2 5 10 --freq H D --top 0.40

`records.csv` needs one row per collision, with latitude, longitude and a parseable
timestamp.  Nothing else is required: no exposure data, no features, no model fitting
beyond the area-only baseline.  The whole run is a few minutes on a laptop.
"""
import argparse
import sys

import numpy as np
import pandas as pd

EARTH_KM_PER_DEG_LAT = 110.574


def auc(y, s):
    """Area under the ROC curve, ties handled by midrank."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
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


def label_top(values, frac, rng):
    """Binary label marking the top `frac` of `values`, ties broken at random."""
    jitter = rng.random(len(values)) * 1e-9
    v = np.asarray(values, dtype=float) + jitter
    return v > np.quantile(v, 1.0 - frac)


def norm_freq(freq):
    """Accept either pandas alias spelling: 'H' and 'h', 'M' and 'ME', and so on."""
    try:
        pd.tseries.frequencies.to_offset(freq)
        return freq
    except ValueError:
        for cand in (freq.lower(), freq.upper(), freq + "E", freq.lower() + "e"):
            try:
                pd.tseries.frequencies.to_offset(cand)
                return cand
            except ValueError:
                continue
    raise SystemExit(f"unrecognised frequency {freq!r}; try 'h', 'D', 'W' or 'MS'")


def build_units(df, km, freq):
    """Aggregate collision records to (cell, time bucket) counts, zeros included."""
    freq = norm_freq(freq)
    dlat = km / EARTH_KM_PER_DEG_LAT
    mid = float(df["lat"].median())
    dlon = km / (EARTH_KM_PER_DEG_LAT * max(np.cos(np.radians(mid)), 1e-6))
    cell = (np.floor(df["lat"] / dlat).astype("int64").astype(str) + "_" +
            np.floor(df["lon"] / dlon).astype("int64").astype(str))
    bucket = df["ts"].dt.floor(freq)
    obs = (pd.DataFrame({"cell": cell, "bucket": bucket})
           .groupby(["cell", "bucket"]).size().rename("count").reset_index())
    # a prediction unit exists for every cell at every bucket in the period, not only
    # where a collision was recorded; the zeros are the point of a sparse grid
    cells = obs["cell"].unique()
    buckets = pd.date_range(obs["bucket"].min(), obs["bucket"].max(), freq=freq)
    full = pd.MultiIndex.from_product([cells, buckets], names=["cell", "bucket"]).to_frame(index=False)
    out = full.merge(obs, on=["cell", "bucket"], how="left")
    out["count"] = out["count"].fillna(0.0)
    out["year"] = out["bucket"].dt.year
    return out


def diagnose(units, top, rng):
    """Area-only AUC under the three labellings."""
    cell = units["cell"].to_numpy()
    cnt = units["count"].to_numpy(dtype=float)
    yr = units["year"].to_numpy()
    codes, cell_idx = np.unique(cell, return_inverse=True)

    y_nat = np.zeros(len(cnt), dtype=bool)
    y_win = np.zeros(len(cnt), dtype=bool)
    for y in np.unique(yr):                       # label within each year, as in the paper
        m = yr == y
        y_nat[m] = label_top(cnt[m], top, rng)
        for a in range(len(codes)):
            ma = m & (cell_idx == a)
            if ma.any():
                y_win[ma] = label_top(cnt[ma], top, rng)
    y_occ = cnt > 0

    res = {}
    for name, lab in (("national", y_nat), ("within_cell", y_win), ("occurrence", y_occ)):
        p = np.array([lab[cell_idx == a].mean() if (cell_idx == a).any() else 0.0
                      for a in range(len(codes))])
        score = p[cell_idx]                       # the area-only classifier
        res[f"{name}_pooled"] = auc(lab, score)
        per = [auc(lab[cell_idx == a], score[cell_idx == a]) for a in range(len(codes))]
        res[f"{name}_macro"] = float(np.nanmean(per))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records"); ap.add_argument("out_csv")
    ap.add_argument("--lat", default="Start_Lat"); ap.add_argument("--lon", default="Start_Lng")
    ap.add_argument("--time", default="Start_Time")
    ap.add_argument("--km", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    ap.add_argument("--freq", nargs="+", default=["H", "D"])
    ap.add_argument("--top", type=float, default=0.40)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(a.records, usecols=[a.lat, a.lon, a.time],
                     low_memory=False).rename(columns={a.lat: "lat", a.lon: "lon", a.time: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", format="mixed")
    df = df.dropna(subset=["lat", "lon", "ts"])
    print(f"{len(df):,} usable records, {df['ts'].min().date()} to {df['ts'].max().date()}")

    rows = []
    print(f"\n{'cell':>7} {'freq':>5} {'units':>12} {'mean count':>11} {'zeros':>7} "
          f"{'national':>9} {'occurrence':>11} {'within macro':>13}")
    for km in a.km:
        for freq in a.freq:
            u = build_units(df, km, freq)
            r = diagnose(u, a.top, rng)
            r.update(km=km, freq=freq, n_units=len(u),
                     mean_count=float(u["count"].mean()),
                     frac_zero=float((u["count"] == 0).mean()))
            rows.append(r)
            print(f"{km:6.1f}k {freq:>5} {r['n_units']:12,} {r['mean_count']:11.4f} "
                  f"{r['frac_zero']:6.1%} {r['national_pooled']:9.4f} "
                  f"{r['occurrence_pooled']:11.4f} {r['within_cell_macro']:13.4f}")

    pd.DataFrame(rows).to_csv(a.out_csv, index=False)
    worst = max(abs(r["within_cell_macro"] - 0.5) for r in rows)
    print(f"\nlargest departure of the within-cell macro AUC from 0.5000: {worst:.6f}")
    print("the paper predicts this to be zero to numerical precision at every cell size;"
          "\na materially larger value would refute the cancellation result of Section 4.2.")
    print("wrote", a.out_csv)


if __name__ == "__main__":
    sys.exit(main())
