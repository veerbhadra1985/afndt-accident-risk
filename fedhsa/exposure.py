"""Exposure normalization: turn accident COUNTS into accident RATES per unit of travel.

Why this matters. Labels defined on raw counts across areas partly measure how much traffic an
area carries, not how risky it is: a model that knows only the area can then score highly
(0.898 AUC in the US national-label setting). Two ways to remove that:
  * within-area labels (no external data needed) - used as the primary protocol; exposure is
    constant inside an area-year, so it cancels exactly;
  * rate labels (this module) - divide counts by vehicle-miles travelled, so areas become
    comparable and a national ranking becomes meaningful.

US   : FHWA Highway Statistics table VM-2, "Vehicle Miles of Travel by Functional System and
       State, 1980-2024", CSV from https://catalog.data.gov (search "VM-2").
UK   : DfT road traffic statistics by local authority (TRA89xx) aggregated to police force area
       with an ONS LAD -> PFA lookup, or any CSV with the columns below.

Any CSV works if it has an area column, a year column and an exposure column; `load_exposure`
detects them, and `attach_exposure` joins it to the contexts.
"""
import numpy as np, pandas as pd

US_ABBREV = {"alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO",
"connecticut":"CT","delaware":"DE","district of columbia":"DC","florida":"FL","georgia":"GA","hawaii":"HI",
"idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA","kansas":"KS","kentucky":"KY","louisiana":"LA",
"maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
"missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH","new jersey":"NJ",
"new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
"oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD",
"tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA",
"west virginia":"WV","wisconsin":"WI","wyoming":"WY"}

def _pick(cols, *words):
    for c in cols:
        lc = str(c).lower()
        if any(w in lc for w in words): return c
    return None

def load_exposure(path, area_col=None, year_col=None, value_col=None, us_state_names=True):
    """Return a tidy frame with columns area, year, exposure (one row per area-year)."""
    df = pd.read_csv(path, low_memory=False)
    area = area_col or _pick(df.columns, "state", "area", "force", "authority", "region")
    year = year_col or _pick(df.columns, "year")
    val = value_col or _pick(df.columns, "total", "vmt", "vehicle miles", "traffic", "exposure", "all motor")
    assert area and year and val, f"could not detect columns in {list(df.columns)[:12]}; pass them explicitly"
    out = pd.DataFrame({"area": df[area].astype(str).str.strip(),
                        "year": pd.to_numeric(df[year], errors="coerce"),
                        "exposure": pd.to_numeric(df[val].astype(str).str.replace(",", ""), errors="coerce")})
    if us_state_names:
        out["area"] = out.area.str.lower().map(US_ABBREV).fillna(out.area)
    out = out.dropna().groupby(["area", "year"], as_index=False).exposure.sum()
    out["year"] = out.year.astype(int)
    return out

def attach_exposure(ctx, exposure, strict=True):
    """Add ctx['exposure'] and ctx['rate'] = count / exposure. Areas are matched on ctx['state']."""
    e = exposure.rename(columns={"area": "state"})
    out = ctx.merge(e, on=["state", "year"], how="left")
    miss = out.exposure.isna()
    if miss.any():
        pairs = out.loc[miss, ["state", "year"]].drop_duplicates()
        msg = f"no exposure for {len(pairs)} area-year pairs, e.g. {pairs.head(3).to_dict('records')}"
        assert not strict, msg
        print("WARNING:", msg)
        out["exposure"] = out.groupby("state").exposure.transform(lambda s: s.fillna(s.mean()))
        out["exposure"] = out.exposure.fillna(out.exposure.mean())
    out["rate"] = out["count"] / out["exposure"]
    return out

def exposure_summary(ctx):
    """Spread of exposure across areas: the quantity that leaks into national count labels."""
    per = ctx.groupby("state").exposure.mean()
    return dict(n_areas=int(per.size), min=float(per.min()), median=float(per.median()),
                max=float(per.max()), ratio_max_min=float(per.max() / max(per.min(), 1e-9)))
