# ================= UK STATS19 study. Run as a Kaggle COMMIT. =================
# PHASE 1: development only (validation 2022). PHASE 2: locked test (Jan-Mar 2023), one seed per commit.
PHASE = 1
SEED  = "0"            # used only when PHASE = 2
# Kaggle stops a commit at 12 h, so development is split. Run these one per commit, in order:
#   A: FAMILIES = "AFNDT-v2"                          (~3 h)
#   B: FAMILIES = "FedAvg-GCNII+FT"                   (~4 h)
#   C: LABELS = "national", FAMILIES = "State-only LR,Logistic regression,XGBoost,MLP"  (~15 min)
#   D: FAMILIES = ""  -> rewrites selected.json only, nothing retrained (~2 min)
FAMILIES = "AFNDT-v2"
LABELS   = "within_state"

import os, glob, shutil, subprocess, sys, json, zipfile, urllib.request
# ---- code ----
py = glob.glob("/kaggle/input/**/dev_select.py", recursive=True)
if not py:
    z = max(glob.glob("/kaggle/input/**/*.zip", recursive=True), key=os.path.getsize)
    zipfile.ZipFile(z).extractall("/kaggle/working/unpacked")
    py = glob.glob("/kaggle/working/unpacked/**/dev_select.py", recursive=True)
shutil.copytree(os.path.dirname(py[0]), "/kaggle/working/code", dirs_exist_ok=True)
assert os.path.exists("/kaggle/working/code/fedhsa/uk_data.py"), "OLD CODE - upload the UK-enabled pipeline zip"
print("code OK (UK-enabled)")

# ---- data: use an attached dataset if present, otherwise download from DfT ----
uk_dir = "/kaggle/working/uk"; os.makedirs(uk_dir, exist_ok=True)
found = [f for f in glob.glob("/kaggle/input/**/*collision*.csv", recursive=True)]
if found:
    for f in found: shutil.copy(f, uk_dir)
    print("using attached UK files:", [os.path.basename(f) for f in found])
else:
    url = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv"
    dst = os.path.join(uk_dir, os.path.basename(url))
    try:
        urllib.request.urlretrieve(url, dst)
        print("downloaded", round(os.path.getsize(dst)/1e6, 1), "MB", flush=True)
    except Exception as e:
        print("download failed (Kaggle internet is probably OFF):", e)
        print("-> attach the CSV as a Kaggle dataset instead, then rerun")
files = sorted(glob.glob(uk_dir + "/*collision*.csv"))
assert files, "no UK collision CSVs - attach them as a Kaggle dataset instead"
print("UK files:", [os.path.basename(f) for f in files])
print("protocol: train 2021-2023, validate 2024, test 2025 (DfT last-5-years file)")

out = "/kaggle/working/uk_results"; os.makedirs(out, exist_ok=True)
dev = glob.glob("/kaggle/input/**/*dev*.json", recursive=True)
if dev:
    best = max(dev, key=lambda f: len(json.load(open(f))["runs"]))
    shutil.copy(best, f"{out}/dev.json"); print("carried over dev runs:", len(json.load(open(best))["runs"]))
for name in ("selected.json", "final.json"):
    c = [f for f in glob.glob(f"/kaggle/input/**/{name}", recursive=True)]
    if c: shutil.copy(max(c, key=os.path.getmtime), f"{out}/{name}"); print("carried over:", name)
p = [d for d in glob.glob("/kaggle/input/**/test_probs", recursive=True) if "uk" in d.lower()]
if p: shutil.copytree(max(p, key=os.path.getmtime), f"{out}/test_probs", dirs_exist_ok=True)

def run(*a):
    r = subprocess.run([sys.executable] + list(a), cwd="/kaggle/working/code"); print("exit code", r.returncode); return r.returncode

if PHASE == 1:
    args = ["dev_select.py", "--csv", uk_dir, "--uk", "--out", out, "--labels", LABELS, "--seeds", "0,1"]
    if FAMILIES: args += ["--families", FAMILIES]
    run(*args)
    allsel = json.load(open(f"{out}/selected.json"))
    for lab, sel in allsel.items():
        print(f"SELECTION (UK, {lab}):")
        for k, v in sorted(sel.items(), key=lambda kv: -kv[1]["val_auc_mean"]):
            print(f"  {k:20s} {v['val_auc_mean']:.4f}  {v['config']}")
else:
    run("final_test.py", "--csv", uk_dir, "--uk", "--out", out, "--seeds", SEED)
    if SEED == "4": run("report_v2.py", out)
print("DONE - download uk_results from the Output tab")
