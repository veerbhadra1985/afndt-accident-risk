"""Two supplementary analyses, run on VALIDATION DATA ONLY. The locked test sets are never loaded.

A. Protocol factorial: {random split, temporal split} x {national labels, within-area labels}.
   The random/national cell reproduces the convention used by much of the published literature;
   the temporal/within-area cell is the protocol of this paper. The difference between the cells
   is the amount of reported performance attributable to the conventions rather than to the model.

B. Exposure as a model feature: the same models with log vehicle-miles of the area-year added as a
   seventh input, to test whether giving models exposure explicitly changes the conclusions.

Usage:
  python protocol_study.py --csv US_Accidents_March23.csv --exposure vm2.csv --out protocol_results
"""
import argparse, json, os, time, torch
from fedhsa.data import load_records
from fedhsa.v2data import year_contexts, temporal_split, random_split
from fedhsa.exposure import load_exposure, attach_exposure
from fedhsa.config import FEATURES, STATES, REGIONS, set_profile
from fedhsa.v2runner import Structures, fit, GRID
from fedhsa.stats import auc, best_threshold, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--exposure", required=True)
ap.add_argument("--out", default="protocol_results"); ap.add_argument("--seeds", default="0,1,2,3")
ap.add_argument("--tops", default="0.40", help="comma list of label thresholds, e.g. 0.20,0.30,0.40")
ap.add_argument("--families", default="State-only LR,Logistic regression,XGBoost,MLP,GCNII")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
path = os.path.join(a.out, "protocol.json")
R = json.load(open(path)) if os.path.exists(path) else {"runs": {}}
def save(): json.dump(R, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
torch.set_num_threads(max(1, os.cpu_count() or 1))
seeds = [int(s) for s in a.seeds.split(",")]
tops = [float(x) for x in a.tops.split(",")]
fams = a.families.split(",")

df, audit = load_records(a.csv, max_year=2022)          # development years only
ctx = year_contexts(df)
ctx = attach_exposure(ctx, load_exposure(a.exposure), strict=False)
R["audit"] = {k: v for k, v in audit.items() if k not in ("states", "years")}
base_features = [f for f in FEATURES if f != "expo_z"]

for use_expo in (False, True):
    set_profile(STATES, REGIONS, base_features + (["expo_z"] if use_expo else []))
    S = Structures(ctx)                                  # graphs do not depend on the feature set
    for split in ("temporal", "random"):
        for label in ("within_state", "national"):
            if use_expo and (split == "random" or label == "national"):
                continue                                 # experiment B only under the paper's protocol
            for top in tops:
                if top != 0.40 and (use_expo or split == "random"):
                    continue                             # threshold sensitivity under the paper's protocol only
                for fam in fams:
                    if top != 0.40 and fam == "GCNII":
                        continue                         # thresholds are checked on the fast families only
                    for ci, cfg in enumerate(GRID[fam][:2]):
                        for seed in seeds:
                            tag = (f"{'expo' if use_expo else 'base'}|{split}|{label}|{fam}|{ci}|{seed}"
                                   + ("" if top == 0.40 else f"|top{int(top*100)}"))
                            if tag in R["runs"]: continue
                            D = (temporal_split(ctx, label, top=top) if split == "temporal"
                                 else random_split(ctx, label, seed=seed, top=top))
                            t0 = time.time(); f = fit(fam, cfg, D, S, seed)
                            p = f.predict("scope_val"); yv = D["y"][D["val"]]
                            thr = best_threshold(yv, p[D["val"]])
                            R["runs"][tag] = dict(features="exposure" if use_expo else "base", split=split, label=label,
                                                  family=fam, config=cfg, seed=seed, top=top,
                                                  val_auc=auc(yv, p[D["val"]]),
                                                  val_f1=metrics(yv, p[D["val"]], thr)["f1"],
                                                  n_val=int(len(yv)), pos_val=float(yv.mean()),
                                                  seconds=time.time() - t0)
                            save(); print(f"{tag}  val AUC={R['runs'][tag]['val_auc']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
set_profile(STATES, REGIONS, base_features)

# summary
import collections, statistics as st
def cells(feat, split, label, fam, top=0.40):
    v=[r["val_auc"] for r in R["runs"].values()
       if r["features"]==feat and r["split"]==split and r["label"]==label and r["family"]==fam and r.get("top",0.40)==top]
    return v
def bestmean(feat, split, label, fam, top=0.40):
    g=collections.defaultdict(list)
    for r in R["runs"].values():
        if (r["features"],r["split"],r["label"],r["family"],r.get("top",0.40))==(feat,split,label,fam,top):
            g[json.dumps(r["config"],sort_keys=True)].append(r["val_auc"])
    return max((st.mean(v), (max(v)-min(v)) if len(v)>1 else 0.0) for v in g.values()) if g else (float("nan"),0.0)
print("\n=== A. protocol factorial (validation AUC; seed spread in brackets) ===")
print(f"{'model':22s} {'random+national':>20s} {'random+within':>18s} {'temporal+national':>20s} {'temporal+within':>18s}")
for fam in fams:
    row=[bestmean('base',s_,l_,fam) for s_,l_ in (("random","national"),("random","within_state"),("temporal","national"),("temporal","within_state"))]
    print(f"{fam:22s} " + " ".join(f"{m:.4f} [{d:.4f}]".rjust(20 if i in (0,2) else 18) for i,(m,d) in enumerate(row)))
print("\n=== B. exposure as a model feature (temporal split, within-area labels) ===")
print(f"{'model':22s} {'base':>10s} {'+exposure':>11s} {'difference':>12s}")
for fam in fams:
    x=bestmean('base','temporal','within_state',fam)[0]; y=bestmean('exposure','temporal','within_state',fam)[0]
    print(f"{fam:22s} {x:10.4f} {y:11.4f} {y-x:+12.4f}")
print("\n=== C. label-threshold sensitivity (temporal split) ===")
print(f"{'model':22s} " + " ".join(f"{'top '+str(int(t*100))+'%':>22s}" for t in tops))
for fam in fams:
    out=[]
    for t in tops:
        w=bestmean('base','temporal','within_state',fam,t)[0]; n=bestmean('base','temporal','national',fam,t)[0]
        out.append(f"within {w:.4f} / nat {n:.4f}")
    print(f"{fam:22s} " + " ".join(o.rjust(22) for o in out))
print("\nDONE - download the output folder")
