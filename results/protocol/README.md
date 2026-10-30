# Protocol study (validation data only; the locked test sets were never loaded)

`protocol_runs.json` holds the runs of three analyses reported in Section 7.2 of the paper:

1. **Protocol factorial** — {random split, temporal split} x {national labels, within-area labels}, five model
   families, two configurations, four seeds.
2. **Exposure as a model feature** — the same families with log area-year vehicle-miles as a seventh input.
3. **Label-threshold sensitivity** — the temporal-split cells at thresholds of 20%, 30% and 40%.

Reproduce with:

```bash
python protocol_study.py --csv US_Accidents_March23.csv --exposure ../exposure/us_vmt_exposure_yearly.csv \
    --out . --seeds 0,1,2,3 --tops 0.20,0.30,0.40
```

The file contains all 292 runs of the three analyses: 4 seeds, 3 label thresholds, 9 protocol cells.

The file contains all 292 runs: 36 per factorial cell (five families x two configurations x four seeds,
minus the area-only classifier which has a single configuration), 36 for the exposure-feature analysis and
28 per threshold cell (the graph model is omitted at the non-default thresholds).
