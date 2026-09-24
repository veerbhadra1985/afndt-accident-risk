# Accident risk prediction under temporal shift, exposure leakage, and federation

Code and results for a benchmark study of accident risk prediction on the US Accidents dataset
(7.73 M records, 49 jurisdictions, 2016–2023), with a pre-registered protocol and a locked test set.

## What this study reports

1. **Exposure leakage in labels.** A model that sees only the jurisdiction reaches 0.898 AUC when
   "high risk" is defined nationally, but 0.510 when it is defined within each state-year. Its macro
   per-state AUC is exactly 0.5000. National labels therefore largely measure exposure, not risk.
2. **Validation rankings do not survive a one-year gap.** GCNII ranks 1st on 2022 validation and 3rd
   on the 2023 test; a two-layer MLP rises from 4th to 1st and beats every other model (DeLong p ≤ 0.011).
3. **Federation costs 0.0127 AUC** under temporal shift (paired t-test over 5 seeds, p = 0.0012).
4. **Architectural complexity did not pay.** Hypergraph propagation, a spiking readout, relation-aware
   propagation, an auxiliary count target, personalized heads and hierarchical aggregation gave no
   significant gain over simpler baselines.

**Author:** Veer Bhadra Pratap Singh, Department of Cyber Security, School of Skill Sciences & Entrepreneurship, Galgotias University, Greater Noida, India (ORCID [0000-0002-3162-134X](https://orcid.org/0000-0002-3162-134X)).

Full tables, p-values and figures: [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

## Protocol (fixed before the test set was opened)

| Item | Setting |
|---|---|
| Contexts | (state, year, month, hour); 79,198 non-empty contexts |
| Features | cyclic month, cyclic hour, mean and s.d. of temperature (standardized on training contexts) |
| Labels | top 40% of accident count **within each state-year** (`within_state`); the national variant is reported only to demonstrate exposure leakage |
| Split | train 2016–2021, validation 2022, **test Jan–Mar 2023** |
| Graph | same-state edges only: adjacent hour, adjacent month, same slot in the adjacent year. No edge crosses a state, so the graph partitions exactly across federated clients |
| Federation | 49 state clients; hierarchical variants aggregate through 6 regional servers |
| Tuning | 6 configurations per neural family, selected on validation AUC only; federated models get up to 100 rounds with the same early stopping as centralized models |
| Metrics | AUC, F1 at the validation-optimal threshold, macro per-state AUC; DeLong for AUC, paired t-tests over seeds |
| Test discipline | `final_test.py` records a SHA-256 of the frozen selection and the timestamp of first access; it refuses to run if the selection changes afterwards |

Test first opened: 2026-09-23T02:27:07Z. Selection SHA-256: `c88d51bae0cd9cc3…`

## Reproducing

```bash
pip install -r requirements.txt
# 1. development (validation only; 2023 is never loaded)
python dev_select.py  --csv US_Accidents_March23.csv --out results_new --labels within_state --seeds 0,1
# 2. locked test, one seed at a time
python final_test.py  --csv US_Accidents_March23.csv --out results_new --seeds 0
# 3. tables and figures
python report_v2.py results_new
```

The dataset is available at https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents.
The loader refuses any CSV with exactly 1,048,575 rows, the Excel row limit, which silently truncates
this file to about 14% of its records.

`notebooks/kaggle_final_test.py` is the single-cell runner used on Kaggle (one seed per commit).
`notebooks/run_all.py` reproduces the earlier study of the original architecture.

## Hardware

All experiments ran on CPU only: Intel Xeon @ 2.20 GHz, 4 logical cores, 33.7 GB RAM,
Linux 6.12, Python 3.12.13, PyTorch 2.10.0+cpu, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1,
XGBoost 3.2.0. Total recorded compute: ≈ 65 h (322 development runs + 95 test runs).

## Contents

```
fedhsa/        library: data, graphs, models, federated trainers, statistics, explanations
dev_select.py  development phase (validation only), resumable
final_test.py  locked test phase, resumable, with the selection fingerprint check
report_v2.py   tables and figures
results/       322 development runs, frozen selection, 95 test runs, per-run predicted
               probabilities (test_probs/), final tables and three figures
notebooks/     Kaggle runner and the scripts for the earlier study
```

## License

MIT (see LICENSE).

## Citation

See `CITATION.cff`.
