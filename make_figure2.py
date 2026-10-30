"""Regenerate Figure 2 (protocol factorial) directly from the protocol runs.

Usage:
    python3 make_figure2.py results/protocol/protocol_runs.json figs/fig3_protocol.png

Every value plotted is the mean validation AUC over four seeds of the
better of the two configurations tried for that family in that cell,
which is the aggregation reported in Table 2.
"""
import json, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
RUNS = json.load(open(SRC))["runs"]

FAMS = ["State-only LR", "Logistic regression", "XGBoost", "MLP", "GCNII"]
SHORT = {"State-only LR": "Area-only", "Logistic regression": "LogReg",
         "XGBoost": "XGBoost", "MLP": "MLP", "GCNII": "GCNII"}
CELLS = [("random", "national"), ("random", "within_state"),
         ("temporal", "national"), ("temporal", "within_state")]
CELL_LAB = ["random split\nnational labels", "random split\nwithin-area labels",
            "temporal split\nnational labels", "temporal split\nwithin-area labels"]
COL = ["#C44E52", "#DD8452", "#55A868", "#4C72B0"]


def cell(fam, split, label, feat="base", top=""):
    """Mean and seed spread of the better configuration, or None."""
    by_cfg = {}
    for key, r in RUNS.items():
        p = key.split("|")
        if (p[6] if len(p) > 6 else "") != top:
            continue
        if p[0] != feat or r["family"] != fam:
            continue
        if r["split"] != split or r["label"] != label:
            continue
        by_cfg.setdefault(p[4], []).append(r["val_auc"])
    if not by_cfg:
        return None
    best = max(by_cfg.values(), key=st.mean)
    return st.mean(best), (max(best) - min(best)) / 2.0


fig, ax = plt.subplots(1, 3, figsize=(20, 10), dpi=220,
                       gridspec_kw={"width_ratios": [2.05, 1.0, 1.0]})
plt.rcParams.update({"font.size": 20})

# ---- (a) reported performance by protocol -------------------------------
x = np.arange(len(FAMS)); w = 0.2
for i, (split, label) in enumerate(CELLS):
    vals = [cell(f, split, label) for f in FAMS]
    ax[0].bar(x + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals],
              color=COL[i], label=CELL_LAB[i], capsize=3,
              error_kw={"elinewidth": 1.4, "capthick": 1.4})
ax[0].axhline(0.5, ls=":", c="0.4", lw=1.5)
ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[f] for f in FAMS], fontsize=20)
ax[0].set_ylim(0.45, 1.0); ax[0].set_ylabel("validation AUC", fontsize=21)
ax[0].set_title("(a) Reported performance by protocol", fontsize=23)

# ---- (b) apparent advantage of the graph model --------------------------
gaps = [cell("GCNII", s, l)[0] - cell("XGBoost", s, l)[0] for s, l in CELLS]
ax[1].bar(range(4), gaps, 0.7, color=COL)
for i, g in enumerate(gaps):
    ax[1].text(i, g + 0.0035, f"+{g:.3f}", ha="center", fontsize=20)
ax[1].set_xticks(range(4))
ax[1].set_xticklabels(["rand.\nnat.", "rand.\nwithin", "temp.\nnat.", "temp.\nwithin"], fontsize=20)
ax[1].set_ylim(0, max(gaps) * 1.18)
ax[1].set_ylabel("GCNII − XGBoost (AUC)", fontsize=21)
ax[1].set_title("(b) Apparent advantage of\nthe graph model", fontsize=23)

# ---- (c) sensitivity to the label threshold -----------------------------
tops = [("top20", 20), ("top30", 30), ("", 40)]
nat = [cell("State-only LR", "temporal", "national", top=t)[0] for t, _ in tops]
wit = [cell("State-only LR", "temporal", "within_state", top=t)[0] for t, _ in tops]
xs = [v for _, v in tops]
ax[2].plot(xs, nat, "o-", color=COL[2], lw=3, ms=11, label="national labels")
ax[2].plot(xs, wit, "s-", color=COL[3], lw=3, ms=11, label="within-area labels")
ax[2].axhline(0.5, ls=":", c="0.4", lw=1.5)
ax[2].set_xticks(xs); ax[2].set_xticklabels([str(v) for v in xs], fontsize=20)
ax[2].set_ylim(0.45, 1.0)
ax[2].set_xlabel("label threshold (top %)", fontsize=21)
ax[2].set_ylabel("area-only AUC", fontsize=21)
ax[2].set_title("(c) Sensitivity to the\nlabel threshold", fontsize=23)
ax[2].legend(fontsize=19, frameon=False, loc="center right")

for a in ax:
    a.grid(axis="y", color="0.9", lw=1.0); a.set_axisbelow(True)
    a.tick_params(labelsize=20)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

h, lab = ax[0].get_legend_handles_labels()
ax[0].legend_ = None
fig.legend(h, lab, loc="lower center", ncol=4, fontsize=20, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(DST, bbox_inches="tight", facecolor="white")
print("wrote", DST)
print("panel (b) gaps:", [f"{g:+.4f}" for g in gaps])
