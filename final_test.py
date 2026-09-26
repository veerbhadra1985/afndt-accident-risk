"""PHASE 2 - one-time final test on January-March 2023 with the configurations frozen in selected.json.
Resumable. The first access to the test set is time-stamped in final.json.

python final_test.py --csv US_Accidents_March23.csv --out v2_results [--seeds 0,1,2,3,4]"""
import argparse, json, os, time, datetime, hashlib, numpy as np, torch
from fedhsa.data import load_records
from fedhsa.uk_data import prepare_uk
from fedhsa.exposure import load_exposure, attach_exposure, exposure_summary
from fedhsa.v2data import year_contexts, temporal_split
from fedhsa.v2runner import Structures
from fedhsa.v2runner import fit, PROPOSED, ABLATIONS
from fedhsa.stats import auc, best_threshold, metrics

ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True, help="US Accidents CSV, or the UK STATS19 file/directory when --uk is set")
ap.add_argument("--uk", action="store_true", help="use the UK STATS19 protocol (police forces as clients)")
ap.add_argument("--exposure", default="", help="CSV of area,year,exposure (e.g. FHWA VM-2) - enables rate_* labels"); ap.add_argument("--out", default="v2_results")
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--families", default="", help="comma list to run only some families (for splitting across sessions)")
ap.add_argument("--ablations", default="yes", choices=["yes", "no", "only"]); a = ap.parse_args()
selp = os.path.join(a.out, "selected.json"); sel = json.load(open(selp))
path = os.path.join(a.out, "final.json"); pdir = os.path.join(a.out, "test_probs"); os.makedirs(pdir, exist_ok=True)
R = json.load(open(path)) if os.path.exists(path) else {"runs": {}}
if "test_first_opened" not in R:
    R["test_first_opened"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    R["selected_sha256"] = hashlib.sha256(open(selp, "rb").read()).hexdigest()
def save():
    json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
assert R["selected_sha256"] == hashlib.sha256(open(selp, "rb").read()).hexdigest(), "selected.json changed after the test set was opened"
torch.set_num_threads(max(1, os.cpu_count() or 1))
if a.uk:
    from fedhsa.config import SPLIT
    ctx, audit = prepare_uk(a.csv)
    assert (ctx.year == SPLIT["test_year"]).any(), f"no {SPLIT['test_year']} UK records found"
else:
    df, audit = load_records(a.csv, max_year=2023); assert (df.year == 2023).any(), "no 2023 records found"
    ctx = year_contexts(df)
if a.exposure:
    _e = load_exposure(a.exposure, us_state_names=not a.uk)
    ctx = attach_exposure(ctx, _e, strict=False)
    print("exposure attached:", exposure_summary(ctx), flush=True)
S = Structures(ctx); R["audit"] = {k: v for k, v in audit.items() if k != "states"}; save()

def per_state_auc(D, p):
    st = D["ctx"].state.to_numpy()[D["test"]]; y = D["y"][D["test"]]; pt = p[D["test"]]; out = {}
    for s in np.unique(st):
        m = st == s
        if len(np.unique(y[m])) == 2: out[s] = auc(y[m], pt[m])
    return out

for label in sel:
    D = temporal_split(ctx, label); R.setdefault("info", {})[label] = D["info"]
    fams = [x for x in sel[label] if not a.families or x in a.families.split(",")]
    jobs = ([] if a.ablations == "only" else [(fam, None) for fam in fams]) + \
           ([] if a.ablations == "no" else [(PROPOSED, v) for v in ABLATIONS])
    for fam, variant in jobs:
        cfg = sel[label][fam]["config"]; name = fam if variant is None else f"{fam} [{variant}]"
        for seed in [int(s) for s in a.seeds.split(",")]:
            tag = f"{label}|{name}|{seed}"
            if tag in R["runs"]: continue
            t0 = time.time(); f = fit(fam, cfg, D, S, seed, variant=variant)
            pv, pt = f.predict("scope_val"), f.predict("scope_test")
            thr = best_threshold(D["y"][D["val"]], pv[D["val"]])
            m = metrics(D["y"][D["test"]], pt[D["test"]], thr); ps = per_state_auc(D, pt)
            np.save(os.path.join(pdir, hashlib.md5(tag.encode()).hexdigest() + ".npy"), pt[D["test"]])
            R["runs"][tag] = dict(label=label, name=name, seed=seed, config=cfg, test=m, val_auc=auc(D["y"][D["val"]], pv[D["val"]]),
                                  state_auc_macro=float(np.mean(list(ps.values()))), state_auc=ps, extra=f.extra,
                                  probs_file=hashlib.md5(tag.encode()).hexdigest() + ".npy", seconds=time.time() - t0)
            save(); print(f"{tag}  test AUC={m['auc']:.4f}  F1={m['f1']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(pdir, f"y_{label}.npy"), D["y"][D["test"]])
print("FINAL TEST FINISHED")
