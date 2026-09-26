"""UK STATS19 (Great Britain) loader, mirroring the US protocol exactly.

Clients are police forces (a real data-governance boundary, the UK analogue of US states).
Contexts are (force, year, month, hour). Labels: top 40% of collision count within each
force-year. Features: cyclic month, cyclic hour, and two environment shares computed from the
same records that define the labels, exactly as the US study used temperature from its records.

Data: https://www.data.gov.uk/dataset/.../road-accidents-safety-data
Yearly files: dft-road-casualty-statistics-collision-<year>.csv  (~19 MB each)
"""
import glob, os, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from .config import set_profile

UK_FEATURES = ["month_sin", "month_cos", "hour_sin", "hour_cos", "wet_z", "adverse_z"]
WET = {2, 3, 4, 5, 6, 7}        # road_surface_conditions: wet/damp, snow, frost/ice, flood, oil, mud
ADVERSE = {2, 3, 4, 5, 6, 7, 8} # weather_conditions: rain/snow/fog, with or without high winds

def _read_one(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"date", "time", "police_force"}
    missing = need - set(df.columns)
    assert not missing, f"{os.path.basename(path)} missing columns: {missing}"
    dt = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    hh = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
    out = pd.DataFrame({
        "force": df["police_force"].astype("Int64"),
        "year": dt.dt.year, "month": dt.dt.month, "hour": hh,
        "lat": pd.to_numeric(df.get("latitude"), errors="coerce"),
        "lon": pd.to_numeric(df.get("longitude"), errors="coerce"),
        "wet": pd.to_numeric(df.get("road_surface_conditions"), errors="coerce").isin(WET),
        "adverse": pd.to_numeric(df.get("weather_conditions"), errors="coerce").isin(ADVERSE)})
    return out.dropna(subset=["force", "year", "month", "hour"])

def load_uk(path_or_dir, min_year=2021, max_year=2025, min_context_rows=None):
    files = sorted(glob.glob(os.path.join(path_or_dir, "*collision*.csv"))) if os.path.isdir(path_or_dir) else [path_or_dir]
    files = [f for f in files if "vehicle" not in f and "casualty" not in os.path.basename(f).replace("casualty-statistics", "")]
    assert files, f"no collision CSV found in {path_or_dir}"
    df = pd.concat([_read_one(f) for f in files], ignore_index=True)
    df = df[(df.year >= min_year) & (df.year <= max_year)].copy()
    for c in ("year", "month", "hour", "force"):
        df[c] = df[c].astype(int)
    audit = dict(files=[os.path.basename(f) for f in files], rows_used=int(len(df)),
                 n_forces=int(df.force.nunique()),
                 years={int(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
                 forces={int(k): int(v) for k, v in df.force.value_counts().sort_index().items()})
    return df, audit

def uk_regions(df, n_regions=6, seed=0):
    """Group police forces into regions by the geographic centroid of their collisions,
    the UK analogue of the six US census-based regions. Data-driven, so no hand-coded map."""
    cen = df.dropna(subset=["lat", "lon"]).groupby("force")[["lat", "lon"]].mean()
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(cen[["lat", "lon"]])
    out = {}
    for force, lab in zip(cen.index, km.labels_):
        out.setdefault(f"Region {lab + 1}", []).append(f"F{int(force)}")
    for f in df.force.unique():                       # forces without coordinates join the largest region
        if f"F{int(f)}" not in {s for v in out.values() for s in v}:
            out[max(out, key=lambda k: len(out[k]))].append(f"F{int(f)}")
    return out

def uk_contexts(df):
    """(force, year, month, hour) contexts with the same column names the pipeline expects."""
    g = df.assign(state=["F%d" % f for f in df.force]).groupby(["state", "year", "month", "hour"])
    ctx = pd.concat([g.size().rename("count"), g.wet.mean().rename("wet"), g.adverse.mean().rename("adverse")], axis=1).reset_index()
    ctx["hbin"] = ctx.hour // 6
    ctx["month_sin"] = np.sin(2 * np.pi * ctx.month / 12); ctx["month_cos"] = np.cos(2 * np.pi * ctx.month / 12)
    ctx["hour_sin"] = np.sin(2 * np.pi * ctx.hour / 24);   ctx["hour_cos"] = np.cos(2 * np.pi * ctx.hour / 24)
    return ctx

def prepare_uk(path_or_dir, min_year=2021, max_year=2025, n_regions=6):
    """One call: load, define regions, switch the pipeline profile, build contexts."""
    from .config import set_split
    set_split(train_end=2023, val_year=2024, test_year=2025)   # DfT "last 5 years" file: 2021-2025
    df, audit = load_uk(path_or_dir, min_year, max_year)
    regions = uk_regions(df, n_regions)
    clients = sorted({s for v in regions.values() for s in v})
    n_cl, n_rg = set_profile(clients, regions, UK_FEATURES)
    ctx = uk_contexts(df)
    audit.update(n_clients=n_cl, n_regions=n_rg, n_contexts=int(len(ctx)))
    return ctx, audit
