"""PHASE 1 - development. Uses 2016-2022 only (2023 is never loaded). Every model family gets
a grid of the same size; the configuration with the best mean validation AUC (2022) is selected.
Resumable: rerun the same command after a disconnect.

python dev_select.py --csv US_Accidents_March23.csv --out v2_results [--labels within_state,national] [--seeds 0,1,2]"""
import argparse, json, os, time, numpy as np, torch
from fedhsa.data import load_records
from fedhsa.uk_data import prepare_uk
from fedhsa.exposure import load_exposure, attach_exposure, exposure_summary
from fedhsa.v2data import year_contexts, temporal_split, LABELS
from fedhsa.v2runner import Structures
from fedhsa.v2runner import GRID, fit
from fedhsa.stats import auc

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True, help="US Accidents CSV, or the UK STATS19 file/directory when --uk is set")
ap.add_argument("--uk", action="store_true", help="use the UK STATS19 protocol (police forces as clients)")
ap.add_argument("--exposure", default="", help="CSV of area,year,exposure (e.g. FHWA VM-2) - enables rate_* labels"); ap.add_argument("--out", default="v2_results")
ap.add_argument("--labels", default=",".join(LABELS)); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--families", default="")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); path = os.path.join(a.out, "dev.json")
R = json.load(open(path)) if os.path.exists(path) else {"runs": {}, "info": {}}
# a run is identified by its CONTENT, so changing the grid never mislabels finished work
seen = {(r["label"], r["family"], json.dumps(r["config"], sort_keys=True), r["seed"]) for r in R["runs"].values()}
def save():
    json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
torch.set_num_threads(max(1, os.cpu_count() or 1))
if a.uk:                                    # UK: contexts are built directly by the loader
    from fedhsa.config import SPLIT
    ctx, audit = prepare_uk(a.csv)                       # loader also sets the UK split years
    ctx = ctx[ctx.year <= SPLIT["val_year"]].reset_index(drop=True)   # development never loads the test year
    assert int(ctx.year.max()) <= SPLIT["val_year"]
else:
    df, audit = load_records(a.csv, max_year=2022); assert df.year.max() <= 2022
    ctx = year_contexts(df)
R["info"]["audit"] = {k: v for k, v in audit.items() if k not in ("states", "forces")}
if a.exposure:
    _e = load_exposure(a.exposure, us_state_names=not a.uk)
    ctx = attach_exposure(ctx, _e, strict=False)
    print("exposure attached:", exposure_summary(ctx), flush=True)
S = Structures(ctx); R["info"]["n_contexts"] = int(len(ctx)); save()
fams = a.families.split(",") if a.families else list(GRID)
for label in a.labels.split(","):
    D = temporal_split(ctx, label); R["info"][label] = D["info"]; save()
    assert len(D["val"]) and len(np.unique(D["y"][D["val"]])) == 2, f"bad validation split: {D['info']}"
    assert not np.isnan(D["X"]).any(), "NaN in features"
    for fam in fams:
        for ci, cfg in enumerate(GRID[fam]):
            for seed in [int(s) for s in a.seeds.split(",")]:
                key = (label, fam, json.dumps(cfg, sort_keys=True), seed)
                tag = f"{label}|{fam}|{json.dumps(cfg, sort_keys=True)}|{seed}"
                if key in seen: continue
                t0 = time.time(); f = fit(fam, cfg, D, S, seed); p = f.predict("scope_val")
                R["runs"][tag] = dict(label=label, family=fam, config=cfg, seed=seed, val_auc=auc(D["y"][D["val"]], p[D["val"]]),
                                      seconds=time.time() - t0, hist=f.hist)
                seen.add(key); save(); print(f"{tag}  val AUC={R['runs'][tag]['val_auc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
# selection
sel = {}
for label in a.labels.split(","):
    sel[label] = {}
    for fam in GRID:
        scores = {}
        for r in R["runs"].values():
            if r["label"] == label and r["family"] == fam: scores.setdefault(json.dumps(r["config"], sort_keys=True), []).append(r["val_auc"])
        if scores:
            best = max(scores, key=lambda k: np.mean(scores[k]))
            sel[label][fam] = dict(config=json.loads(best), val_auc_mean=float(np.mean(scores[best])), n_seeds=len(scores[best]))
json.dump(sel, open(os.path.join(a.out, "selected.json"), "w"), indent=1)
print("DEVELOPMENT FINISHED - selected configurations written to selected.json")
