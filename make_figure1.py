"""Regenerate Figure 1 (labelling comparison) from the released result files.

Usage:
    python3 make_figure1.py results/exposure/exposure_yearly_dev_96runs.json \
                            results/uk/uk_dev_202runs.json  figs/fig2_exposure.png

Each bar is the mean validation AUC over the runs of that family under that
labelling, which is the aggregation reported in Table 2 of the paper.  Use the
year-specific exposure file (exposure_yearly_dev_96runs.json) for the per-vehicle-mile
column; the 2022-only file reproduces the same figure to within 0.001, which is the
insensitivity reported in Section 7.1.
"""
import json, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

US_JSON, GB_JSON, DST = sys.argv[1], sys.argv[2], sys.argv[3]
FAMS = ["State-only LR", "Logistic regression", "XGBoost", "MLP"]
SHORT = {"State-only LR": "Area-only", "Logistic regression": "LogReg",
         "XGBoost": "XGBoost", "MLP": "MLP"}
COL = {"within_state": "#4C72B0", "national": "#DD8452", "rate_national": "#55A868"}
LAB = {"within_state": "within-area counts", "national": "national counts",
       "rate_national": "national, per vehicle-mile"}


def means(path):
    runs = json.load(open(path))["runs"]
    agg = {}
    for r in (runs.values() if isinstance(runs, dict) else runs):
        name = r.get("name") or r.get("family")
        if name in FAMS:
            agg.setdefault((r["label"], name), []).append(r["val_auc"])
    return {k: st.mean(v) for k, v in agg.items()}


us, gb = means(US_JSON), means(GB_JSON)
panels = [("(a) United States", us, ["within_state", "national", "rate_national"]),
          ("(b) Great Britain", gb, ["within_state", "national"])]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=220, sharey=True)
for ax, (title, data, labels) in zip(axes, panels):
    x = np.arange(len(FAMS))
    w = 0.8 / len(labels)
    for i, lab in enumerate(labels):
        vals = [data.get((lab, f), np.nan) for f in FAMS]
        off = (i - (len(labels) - 1) / 2) * w
        ax.bar(x + off, vals, w, color=COL[lab], label=LAB[lab])
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=13)
    ax.axhline(0.5, ls=":", c="0.4", lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[f] for f in FAMS], fontsize=17)
    ax.set_ylim(0.45, 1.02); ax.set_title(title, fontsize=20)
    ax.grid(axis="y", color="0.92"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=16)
axes[0].set_ylabel("validation AUC", fontsize=18)
fig.suptitle("Performance of a classifier given only the area identifier", fontsize=22, y=0.99)
h, l = [], []
for ax in axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh); l.append(ll)
fig.legend(h, l, loc="lower center", ncol=3, fontsize=17, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=[0, 0.06, 1, 0.97])
fig.savefig(DST, bbox_inches="tight", facecolor="white")
print("wrote", DST)
for title, data, labels in panels:
    print(" ", title, {f: {l: round(data.get((l, f), float("nan")), 4) for l in labels} for f in FAMS})
