"""Loading the full US Accidents CSV, auditing it, and building spatiotemporal contexts."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .config import STATES, STATE_REGION, TIME_BINS, FEATURES

EXCEL_ROW_LIMIT = 1_048_575

def load_records(csv, chunksize=1_000_000, min_year=2016, max_year=2022):
    cols = ["Start_Time", "State", "Temperature(F)"]
    parts, n_raw = [], 0
    for ch in pd.read_csv(csv, usecols=cols, chunksize=chunksize, low_memory=False):
        n_raw += len(ch)
        t = pd.to_datetime(ch["Start_Time"].astype(str).str.slice(0, 19), errors="coerce")
        df = pd.DataFrame({"state": ch["State"], "temp": pd.to_numeric(ch["Temperature(F)"], errors="coerce"),
                           "year": t.dt.year, "month": t.dt.month, "hour": t.dt.hour})
        parts.append(df.dropna(subset=["state", "year"]))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.state.isin(STATES) & (df.year >= min_year) & (df.year <= max_year)].copy()
    for c in ("year", "month", "hour"):
        df[c] = df[c].astype(int)
    audit = {
        "rows_in_csv": int(n_raw),
        "rows_used": int(len(df)),
        "rows_dropped_time_state_or_year": int(n_raw - len(df)),
        "excel_truncation_suspected": bool(n_raw == EXCEL_ROW_LIMIT),
        "temperature_missing_frac": float(df.temp.isna().mean()),
        "n_jurisdictions": int(df.state.nunique()),
        "years": {int(k): int(v) for k, v in df.year.value_counts().sort_index().items()},
        "states": {k: int(v) for k, v in df.state.value_counts().sort_index().items()},
    }
    if audit["excel_truncation_suspected"]:
        raise RuntimeError("CSV has exactly 1,048,575 rows: it was truncated by Excel. "
                           "Download the original file again and do not open it in Excel.")
    return df, audit

def hour_bin(h):
    for b, (lo, hi) in enumerate(TIME_BINS):
        if lo <= h < hi:
            return b
    raise ValueError(h)

def build_contexts(df, by_year=False):
    keys = ["state", "month", "hour"] + (["year"] if by_year else [])
    # fill missing temperatures with the state-month mean, then the global mean
    t = df.temp.fillna(df.groupby(["state", "month"]).temp.transform("mean")).fillna(df.temp.mean())
    g = df.assign(temp=t).groupby(keys).temp
    ctx = pd.concat([g.size().rename("count"), g.mean().rename("temp_mean"),
                     g.std(ddof=1).fillna(0.0).rename("temp_std")], axis=1).reset_index()
    ctx["region"] = ctx.state.map(STATE_REGION)
    ctx["hbin"] = ctx.hour.map(hour_bin)
    ctx["month_sin"] = np.sin(2 * np.pi * ctx.month / 12); ctx["month_cos"] = np.cos(2 * np.pi * ctx.month / 12)
    ctx["hour_sin"] = np.sin(2 * np.pi * ctx.hour / 24);   ctx["hour_cos"] = np.cos(2 * np.pi * ctx.hour / 24)
    return ctx

def _finalize(ctx, tr, te, seed, val_frac, label_top=0.40):
    ctx = ctx.copy()
    thr = np.quantile(ctx.loc[tr, "count"], 1 - label_top)
    ctx["y"] = (ctx["count"] >= thr).astype(int)
    strata = (ctx.loc[tr, "state"] + "_" + ctx.loc[tr, "y"].astype(str)).to_numpy()
    vc = pd.Series(strata).value_counts()
    labels = ctx.loc[tr, "y"].astype(str).to_numpy()
    for i in np.where(np.isin(strata, vc[vc < 2].index))[0]:   # singleton strata join the largest stratum with the same label
        same = vc[[k.endswith("_" + labels[i]) and vc[k] >= 2 for k in vc.index]]
        strata[i] = same.index[0] if len(same) else labels[i]
    tr_idx, va_idx = train_test_split(np.where(tr)[0], test_size=val_frac, random_state=seed, stratify=strata)
    fit = ctx.iloc[tr_idx]
    for raw, z in (("temp_mean", "temp_z"), ("temp_std", "temp_std_z")):
        mu, sd = fit[raw].mean(), fit[raw].std(ddof=0) + 1e-8
        ctx[z] = (ctx[raw] - mu) / sd
    te_idx = np.where(te)[0]
    info = {"label_threshold_count": float(thr),
            "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
            "pos_frac_train": float(ctx.y.iloc[tr_idx].mean()),
            "pos_frac_val": float(ctx.y.iloc[va_idx].mean()),
            "pos_frac_test": float(ctx.y.iloc[te_idx].mean())}
    return dict(ctx=ctx, X=ctx[FEATURES].to_numpy(np.float32), y=ctx.y.to_numpy(np.int64),
                train=tr_idx, val=va_idx, test=te_idx, info=info)

def seasonal_split(ctx, seed, val_frac=0.15, label_top=0.40):
    """Train on months 1-9, test on months 10-12; label threshold from training contexts only."""
    return _finalize(ctx, (ctx.month <= 9).to_numpy(), (ctx.month >= 10).to_numpy(), seed, val_frac, label_top)

def forward_split(ctx_y, seed, val_frac=0.15, label_top=0.40, test_year=2022):
    """Year-specific contexts: train (and validate) on years before test_year, test on test_year."""
    tr = (ctx_y.year < test_year).to_numpy(); te = (ctx_y.year == test_year).to_numpy()
    return _finalize(ctx_y, tr, te, seed, val_frac, label_top)
