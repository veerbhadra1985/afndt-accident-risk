"""Typed within-state graph: hour-adjacent, month-adjacent and year-adjacent edges are kept
apart so the model can weight each relation separately (relation-aware propagation)."""
import numpy as np
REL = ("hour", "month", "year")

def typed_lattice(ctx):
    key = {(s, y, m, h): i for i, (s, y, m, h) in enumerate(zip(ctx.state, ctx.year, ctx.month, ctx.hour))}
    out = {r: ([], []) for r in REL}
    for i, (s, y, m, h) in enumerate(zip(ctx.state, ctx.year, ctx.month, ctx.hour)):
        cand = [("hour", (s, y, m, (h + 1) % 24)), ("hour", (s, y, m, (h - 1) % 24)),
                ("year", (s, y - 1, m, h)), ("year", (s, y + 1, m, h))]
        if m < 12: cand.append(("month", (s, y, m + 1, h)))
        if m > 1:  cand.append(("month", (s, y, m - 1, h)))
        for r, nb in cand:
            j = key.get(nb)
            if j is not None: out[r][0].append(j); out[r][1].append(i)
    return {r: (np.array(a, np.int64), np.array(b, np.int64)) for r, (a, b) in out.items()}
