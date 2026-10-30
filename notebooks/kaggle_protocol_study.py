# ===== Supplementary study, round 2: two more seeds + label-threshold sensitivity =====
# Inputs: US Accidents, afndt_pipeline_PROTOCOL2.zip, us_vmt_exposure_yearly.csv, protocol_90runs.json
# Validation data only; the locked test sets are never loaded. Resumable. Expect 3-5 hours.
import os, glob, shutil, subprocess, sys, json, zipfile
py = glob.glob("/kaggle/input/**/protocol_study.py", recursive=True)
if not py:
    zips = glob.glob("/kaggle/input/**/*.zip", recursive=True)
    assert zips, "upload afndt_pipeline_PROTOCOL2.zip"
    zipfile.ZipFile(max(zips, key=os.path.getsize)).extractall("/kaggle/working/unpacked")
    py = glob.glob("/kaggle/working/unpacked/**/protocol_study.py", recursive=True)
shutil.copytree(os.path.dirname(py[0]), "/kaggle/working/code", dirs_exist_ok=True)
assert "--tops" in open("/kaggle/working/code/protocol_study.py").read(), "OLD CODE - upload afndt_pipeline_PROTOCOL2.zip"
print("code OK (threshold study present)")

csv = max([f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if os.path.getsize(f) > 5e8], key=os.path.getsize)
vmt = [f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True) if "vmt" in f.lower() or "exposure" in f.lower()]
assert vmt, "attach us_vmt_exposure_yearly.csv"
vmt = sorted(vmt, key=lambda f: ("yearly" not in f.lower(), f))[0]
print("dataset:", round(os.path.getsize(csv)/1e9, 2), "GB | exposure:", os.path.basename(vmt))

out = "/kaggle/working/protocol_results"; os.makedirs(out, exist_ok=True)
prev = glob.glob("/kaggle/input/**/protocol*.json", recursive=True)
if prev:
    best = max(prev, key=lambda f: len(json.load(open(f))["runs"]))
    shutil.copy(best, os.path.join(out, "protocol.json"))
    print("carried over previous runs:", len(json.load(open(best))["runs"]))

p = subprocess.run([sys.executable, "protocol_study.py", "--csv", csv, "--exposure", vmt, "--out", out,
                    "--seeds", "0,1,2,3", "--tops", "0.20,0.30,0.40"], cwd="/kaggle/working/code")
print("exit code", p.returncode)
print("DONE - download protocol_results from the Output tab")
