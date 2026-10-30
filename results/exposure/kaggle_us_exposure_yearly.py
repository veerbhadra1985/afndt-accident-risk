# ============ US exposure analysis: how much of the national-label signal is traffic volume? ============
# Inputs needed: US Accidents dataset, the code zip (exposure-enabled), us_vmt_exposure.csv
# Runtime: ~15-20 minutes. Validation data only; the locked 2023 test set is NOT touched.
import os, glob, shutil, subprocess, sys, json, zipfile

py = glob.glob("/kaggle/input/**/dev_select.py", recursive=True)
if not py:
    z = max(glob.glob("/kaggle/input/**/*.zip", recursive=True), key=os.path.getsize)
    zipfile.ZipFile(z).extractall("/kaggle/working/unpacked")
    py = glob.glob("/kaggle/working/unpacked/**/dev_select.py", recursive=True)
shutil.copytree(os.path.dirname(py[0]), "/kaggle/working/code", dirs_exist_ok=True)
assert os.path.exists("/kaggle/working/code/fedhsa/exposure.py"), "need the exposure-enabled code zip"
print("code OK (exposure-enabled)")

csv = max([f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if os.path.getsize(f) > 5e8], key=os.path.getsize)
print("US Accidents:", round(os.path.getsize(csv)/1e9, 2), "GB")
vmt = [f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if "vmt" in f.lower() or "exposure" in f.lower()]
assert vmt, "attach us_vmt_exposure.csv as a Kaggle dataset"
vmt = sorted(vmt, key=lambda f: ("yearly" not in f.lower(), f))[0]   # prefer the year-specific file
print("exposure file:", os.path.basename(vmt))
import pandas as pd
_y = pd.read_csv(vmt); print("exposure years covered:", sorted(_y[[c for c in _y.columns if "year" in c.lower()][0]].unique()))

out = "/kaggle/working/exposure_results_yearly"; os.makedirs(out, exist_ok=True)
p = subprocess.run([sys.executable, "dev_select.py", "--csv", csv, "--exposure", vmt, "--out", out,
                    "--labels", "national,rate_national,within_state", "--seeds", "0,1",
                    "--families", "State-only LR,Logistic regression,XGBoost,MLP"], cwd="/kaggle/working/code")
print("exit code", p.returncode)

R = json.load(open(f"{out}/dev.json")); import collections, statistics as st
best = collections.defaultdict(dict)
for r in R["runs"].values():
    best[r["label"]].setdefault(r["family"], []).append(r["val_auc"])
print("\n=== validation AUC by labelling ===")
print(f"{'model':22s} {'within-area':>12s} {'national':>10s} {'rate (per VMT)':>15s}")
for fam in ["State-only LR", "Logistic regression", "XGBoost", "MLP"]:
    row = [max([st.mean(best[l][fam])]) if fam in best.get(l, {}) else float('nan')
           for l in ("within_state", "national", "rate_national")]
    print(f"{fam:22s} {row[0]:12.4f} {row[1]:10.4f} {row[2]:15.4f}")
print("\nDONE - download exposure_results_yearly from the Output tab")
