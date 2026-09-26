"""AFNDT-v2 protocol: year-indexed contexts, two label definitions, temporal train/val/test.

Train: 2016-2021   Validation: 2022   Final test (lockbox): Jan-Mar 2023.
During development the 2023 rows are never loaded.
"""
import numpy as np
from .data import build_contexts
from .config import FEATURES, SPLIT

LABELS = ("within_state", "national", "rate_national", "rate_within_state")
# within_state   : top 40% of COUNT within each area-year   (primary; exposure cancels exactly)
# national       : top 40% of COUNT within each year        (leaks exposure - reported as a control)
# rate_national  : top 40% of COUNT/EXPOSURE within each year (exposure-adjusted national ranking)
# rate_within_state : top 40% of COUNT/EXPOSURE within each area-year (equals within_state by
#                  construction when exposure is constant inside an area-year; kept for completeness)

def make_labels(ctx, mode, top=0.40):
    if mode.startswith("rate"):
        assert "rate" in ctx.columns, "rate labels need exposure: call attach_exposure() first"
        col, keys = "rate", (["year"] if mode == "rate_national" else ["state", "year"])
        thr = ctx.groupby(keys)[col].transform(lambda c: c.quantile(1 - top))
        return (ctx[col] > thr).astype(np.int64).to_numpy()
    """High-risk = count strictly above the (1-top) quantile of its group.
    national:     group = year           (removes growth of data coverage over years)
    within_state: group = (state, year)  (also removes differences in state size/exposure)"""
    keys = ["year"] if mode == "national" else ["state", "year"]
    thr = ctx.groupby(keys)["count"].transform(lambda c: c.quantile(1 - top))
    return (ctx["count"] > thr).astype(np.int64).to_numpy()

def temporal_split(ctx, label_mode):
    ctx = ctx.copy()
    ctx["y"] = make_labels(ctx, label_mode)
    tr = np.where(ctx.year <= SPLIT["train_end"])[0]
    va = np.where(ctx.year == SPLIT["val_year"])[0]
    te = np.where(ctx.year == SPLIT["test_year"])[0]
    fit = ctx.iloc[tr]
    pairs = ((("temp_mean", "temp_z"), ("temp_std", "temp_std_z")) if "temp_z" in FEATURES
             else (("wet", "wet_z"), ("adverse", "adverse_z")))          # UK profile
    for raw, z in pairs:
        mu, sd = fit[raw].mean(), fit[raw].std(ddof=0) + 1e-8
        ctx[z] = (ctx[raw] - mu) / sd
    lg = np.log1p(ctx["count"].to_numpy(float))
    mu, sd = lg[tr].mean(), lg[tr].std() + 1e-8
    info = dict(label=label_mode, years=dict(SPLIT), n_train=len(tr), n_val=len(va), n_test=len(te),
                pos_train=float(ctx.y.iloc[tr].mean()), pos_val=float(ctx.y.iloc[va].mean()),
                pos_test=float(ctx.y.iloc[te].mean()) if len(te) else None)
    return dict(ctx=ctx, X=ctx[FEATURES].to_numpy(np.float32), y=ctx.y.to_numpy(np.int64), z=((lg - mu) / sd).astype(np.float32),
                train=tr, val=va, test=te, info=info,
                scope_val=np.where(ctx.year <= SPLIT["val_year"])[0],   # validation never sees test-year features
                scope_test=np.arange(len(ctx)))

def year_contexts(df):
    return build_contexts(df, by_year=True)
