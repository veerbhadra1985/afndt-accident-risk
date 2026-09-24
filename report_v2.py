"""Tables and figures for the AFNDT-v2 study. python report_v2.py v2_results"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fedhsa.stats import delong
out = sys.argv[1] if len(sys.argv) > 1 else "v2_results"; L = []
ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
dev = json.load(open(os.path.join(out, "dev.json"))); sel = json.load(open(os.path.join(out, "selected.json")))
L.append("# AFNDT-v2 study\n\n## Development (validation = 2022)\n")
for label in sel:
    L.append(f"\n### Label: {label}  —  {dev['info'].get(label)}\n\n| Family | Config | Val AUC (mean ± sd) | Selected |\n|---|---|---|---|")
    groups = {}
    for r in dev["runs"].values():
        if r["label"] == label: groups.setdefault((r["family"], json.dumps(r["config"], sort_keys=True)), []).append(r["val_auc"])
    for (fam, c), v in sorted(groups.items()):
        L.append(f"| {fam} | {c} | {ms(v)} | {'✔' if json.dumps(sel[label][fam]['config'], sort_keys=True) == c else ''} |")
fp = os.path.join(out, "final.json")
if os.path.exists(fp):
    F = json.load(open(fp)); L.append(f"\n## Final test (Jan–Mar 2023; opened {F['test_first_opened']})\n")
    for label in sel:
        y = np.load(os.path.join(out, "test_probs", f"y_{label}.npy")) if os.path.exists(os.path.join(out, "test_probs", f"y_{label}.npy")) else None
        runs = [r for r in F["runs"].values() if r["label"] == label]; names = sorted({r["name"] for r in runs})
        byname = {n: sorted([r for r in runs if r["name"] == n], key=lambda r: r["seed"]) for n in names}
        L.append(f"\n### Label: {label}  —  {F['info'].get(label)}\n\n| Model | AUC | F1 | Macro state AUC | DeLong p vs AFNDT-v2 (max over seeds) |\n|---|---|---|---|---|")
        prop = byname.get("AFNDT-v2", [])
        order = sorted(names, key=lambda n: -np.mean([r["test"]["auc"] for r in byname[n]]))
        for n in order:
            rs = byname[n]; p = "—"
            if n != "AFNDT-v2" and y is not None and prop:
                ps = []
                for r in rs:
                    q = [x for x in prop if x["seed"] == r["seed"]]
                    if q:
                        a_ = np.load(os.path.join(out, "test_probs", q[0]["probs_file"])); b_ = np.load(os.path.join(out, "test_probs", r["probs_file"]))
                        ps.append(delong(y, a_, b_)[3])
                p = f"{max(ps):.3g}" if ps else "—"
            L.append(f"| {n} | {ms([r['test']['auc'] for r in rs])} | {ms([r['test']['f1'] for r in rs])} | {ms([r['state_auc_macro'] for r in rs])} | {p} |")
        gates = [r["extra"].get("gate") for r in prop if r["extra"].get("gate") is not None]
        if gates: L.append(f"\nAFNDT-v2 hypergraph gate: {ms(gates)}")
        main = [n for n in order if "[" not in n]
        fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
        vals = [[r["test"]["auc"] for r in byname[n]] for n in main]
        ax.bar(range(len(main)), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], capsize=3,
               color=["#B94A48" if n == "AFNDT-v2" else "#2F5D8A" for n in main])
        ax.set_xticks(range(len(main))); ax.set_xticklabels(main, rotation=35, ha="right", fontsize=8); ax.set_ylabel("Test AUC (2023)")
        ax.set_ylim(max(0.4, min(np.mean(v) for v in vals) - 0.05), max(np.mean(v) for v in vals) + 0.03)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"fig_test_auc_{label}.png")); plt.close()
open(os.path.join(out, "tables_v2.md"), "w").write("\n".join(L)); print("wrote tables_v2.md")
