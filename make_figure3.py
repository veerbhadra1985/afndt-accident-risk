"""Regenerate Figure 3 (validation rank against one-time test rank).

Usage:
    python3 make_figure3.py results/us/us_selected_FROZEN.json results/us/us_final_seeds0-4.json \
                            results/uk/uk_selected_FROZEN.json results/uk/uk_final_seeds0-4.json \
                            figs/fig1_val_vs_test.png

Validation ranks come from the frozen selection written before the test set was
opened; test ranks are the mean over the five locked-test seeds.  Lines are drawn
in red where a model moves two or more places.
"""
import json, sys, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

US_SEL, US_FIN, GB_SEL, GB_FIN, DST = sys.argv[1:6]

# ablation variants are not separate models in the main comparison
DROP_PREFIX = "AFNDT-v2 ["
PRETTY = {"AFNDT-v2": "PF-GNN", "AFNDT-v3": "PF-GNN-ext", "State-only LR": "Area-only",
          "Logistic regression": "LogReg", "FedAvg-GCNII+FT": "FedAvg+FT",
          "FedAvg-GCNII": "FedAvg", "FedProx-GCNII": "FedProx"}


def ranks(sel_path, fin_path):
    sel = json.load(open(sel_path))
    sel = sel.get("within_state", sel)
    val = {k: v["val_auc_mean"] for k, v in sel.items() if not k.startswith(DROP_PREFIX)}
    runs = json.load(open(fin_path))["runs"]
    agg = {}
    for r in (runs.values() if isinstance(runs, dict) else runs):
        n = r.get("name") or r.get("family")
        if n.startswith(DROP_PREFIX):
            continue
        agg.setdefault(n, []).append(r["test"]["auc"])
    test = {k: st.mean(v) for k, v in agg.items() if k in val}
    val = {k: v for k, v in val.items() if k in test}
    vo = sorted(val, key=lambda k: -val[k])
    to = sorted(test, key=lambda k: -test[k])
    return {k: (vo.index(k) + 1, to.index(k) + 1) for k in val}


panels = [("(a) United States\nvalidation 2022 → test Jan–Mar 2023", ranks(US_SEL, US_FIN)),
          ("(b) Great Britain\nvalidation 2024 → test 2025", ranks(GB_SEL, GB_FIN))]

fig, axes = plt.subplots(1, 2, figsize=(17, 11), dpi=220)
for ax, (title, R) in zip(axes, panels):
    n = len(R)
    for name, (v, t) in R.items():
        moved = abs(v - t) >= 2
        ax.plot([0, 1], [v, t], "-", lw=3.2 if moved else 2.0,
                color="#C44E52" if moved else "#8d99ae", zorder=1)
        ax.scatter([0, 1], [v, t], s=90, color="#C44E52" if moved else "#8d99ae", zorder=2)
        label = PRETTY.get(name, name)
        ax.text(-0.04, v, label, ha="right", va="center", fontsize=16)
        ax.text(1.04, t, label, ha="left", va="center", fontsize=16)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(n + 0.6, 0.4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["validation", "test"], fontsize=18)
    ax.set_yticks(range(1, n + 1)); ax.tick_params(axis="y", labelsize=16)
    ax.set_title(title, fontsize=20)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("rank (1 = best)", fontsize=18)
fig.suptitle("Model rank before and after temporal separation", fontsize=23, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(DST, bbox_inches="tight", facecolor="white")
print("wrote", DST)
for title, R in panels:
    moved = {k: v for k, v in R.items() if abs(v[0] - v[1]) >= 2}
    print(f"  {title.splitlines()[0]}: {len(R)} models, {len(moved)} moved >=2 places")
