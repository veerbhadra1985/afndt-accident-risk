# US exposure analysis (validation 2022; the locked 2023 test set was not touched)

Exposure: FHWA Highway Statistics table VM-2, 'Functional system travel - annual vehicle-miles', 2022 values.
Spread across jurisdictions: 92x between the largest (California, 315,244 M VMT) and smallest (District of Columbia, 3,421 M).
Year-to-year stability of state shares (2016/2020/2022 tables): Spearman 0.996-0.998; one of 51 states moves more than three rank places.

| Model | within-area labels | national (counts) | rate labels (counts / VMT) |
|---|---|---|---|
| State-only LR | 0.5100 | 0.8979 | 0.7800 |
| Logistic regression | 0.7664 | 0.6873 | 0.6908 |
| XGBoost | 0.8525 | 0.7595 | 0.7681 |
| MLP | 0.8545 | 0.7640 | 0.7699 |

Positive-class fraction is ~0.39-0.40 under all three labellings, so the differences are not an imbalance artefact.

## Reading
1. Ranking accident counts ACROSS areas lets a model that knows only the area reach 0.898 AUC.
2. Dividing counts by vehicle-miles removes roughly a third of that signal (0.898 -> 0.780), but area identity still predicts the label strongly: US states differ materially in collisions per vehicle-mile.
3. Only within-area labelling removes the effect entirely (0.510 pooled; macro per-state AUC exactly 0.5000).
4. Both national labellings LOWER the apparent performance of real models (XGBoost 0.853 -> 0.760 / 0.768), because across-area ranking is dominated by area effects rather than by time and weather features.
5. Logistic regression moves the other way (0.766 -> 0.687), as it cannot represent area identity from cyclic time features - an internal check that the mechanism is as described.

## Robustness: single-year vs year-specific exposure

| Model | within-area | national (counts) | rate, 2022 VMT | rate, year-specific VMT | difference |
|---|---|---|---|---|---|
| State-only LR | 0.5100 | 0.8979 | 0.7800 | 0.7810 | +0.0010 |
| Logistic regression | 0.7664 | 0.6873 | 0.6908 | 0.6906 | -0.0002 |
| XGBoost | 0.8525 | 0.7595 | 0.7681 | 0.7679 | -0.0002 |
| MLP | 0.8545 | 0.7640 | 0.7699 | 0.7697 | -0.0002 |

Year-specific exposure uses FHWA VM-2 for 2016, 2020 and 2022 with intermediate years linearly interpolated.
The within-area and national columns are identical in both runs (they do not use exposure), which confirms that
nothing else differed between the two runs. Maximum change in the rate column: 0.0010 AUC.
State shares are highly stable across years (Spearman 0.996-0.998; one of 51 states moves more than three rank
places between 2016 and 2022), so either exposure series supports the same conclusion.

**Reported in the paper:** year-specific values, with the single-year run cited as a robustness check.