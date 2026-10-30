# ============ FINAL TEST on Jan-Mar 2023. Run as a COMMIT, ONE SEED PER COMMIT. ============
# Commit 1: SEED = "0"   Commit 2: SEED = "1"   ... Commit 5: SEED = "4"
# From commit 2 onward, add the PREVIOUS commit's output (v2_results) as an input dataset.
SEED = "4"

import os, glob, shutil, subprocess, sys, json, zipfile
py = glob.glob("/kaggle/input/**/dev_select.py", recursive=True)
if not py:
    z = max(glob.glob("/kaggle/input/**/*.zip", recursive=True), key=os.path.getsize)
    zipfile.ZipFile(z).extractall("/kaggle/working/unpacked")
    py = glob.glob("/kaggle/working/unpacked/**/dev_select.py", recursive=True)
shutil.copytree(os.path.dirname(py[0]), "/kaggle/working/code", dirs_exist_ok=True)
assert '"rounds": 100' in open("/kaggle/working/code/fedhsa/v2runner.py").read(), "OLD CODE"
csv = max([f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if os.path.getsize(f) > 5e8], key=os.path.getsize)
out = "/kaggle/working/v2_results"; os.makedirs(out, exist_ok=True)

d = max(glob.glob("/kaggle/input/**/dev*.json", recursive=True), key=lambda f: len(json.load(open(f))["runs"]))
shutil.copy(d, f"{out}/dev.json"); n = len(json.load(open(f"{out}/dev.json"))["runs"])
print("dev runs:", n); assert n >= 322, "OLD dev.json - upload the 322-run file"
for name in ("selected.json", "final.json"):          # carry over from the previous commit
    c = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if c: shutil.copy(max(c, key=os.path.getmtime), f"{out}/{name}"); print("carried over:", name)
p = glob.glob("/kaggle/input/**/test_probs", recursive=True)
if p: shutil.copytree(max(p, key=os.path.getmtime), f"{out}/test_probs", dirs_exist_ok=True)

def run(*a):
    r = subprocess.run([sys.executable] + list(a), cwd="/kaggle/working/code"); print("exit code", r.returncode); return r.returncode

if not os.path.exists(f"{out}/selected.json"):        # freeze the selection (no new training)
    run("dev_select.py", "--csv", csv, "--out", out, "--labels", "within_state", "--seeds", "0,1")
sel = json.load(open(f"{out}/selected.json"))["within_state"]
print("FROZEN SELECTION:")
for k, v in sorted(sel.items(), key=lambda kv: -kv[1]["val_auc_mean"]): print(f"  {k:18s} {v['val_auc_mean']:.4f}  {v['config']}")

run("final_test.py", "--csv", csv, "--out", out, "--seeds", SEED)
if SEED == "4": run("report_v2.py", out)
print("DONE - download v2_results from the Output tab, then set SEED to the next number")
