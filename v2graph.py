"""Year-aware within-state context graph. No edge ever connects two states, so the graph
splits exactly across state clients: federated training never shares graph structure."""
import numpy as np

def year_lattice(ctx):
    key = {(s, y, m, h): i for i, (s, y, m, h) in enumerate(zip(ctx.state, ctx.year, ctx.month, ctx.hour))}
    src, dst = [], []
    for i, (s, y, m, h) in enumerate(zip(ctx.state, ctx.year, ctx.month, ctx.hour)):
        nbs = [(s, y, m, (h + 1) % 24), (s, y, m, (h - 1) % 24), (s, y - 1, m, h), (s, y + 1, m, h)]
        if m < 12: nbs.append((s, y, m + 1, h))
        if m > 1: nbs.append((s, y, m - 1, h))
        for nb in nbs:
            j = key.get(nb)
            if j is not None: src.append(j); dst.append(i)
    return np.array(src, np.int64), np.array(dst, np.int64)
