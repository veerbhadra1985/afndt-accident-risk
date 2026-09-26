# Do accident risk models learn risk or exposure?

Code and complete results for a two-country study of how labelling and evaluation choices shape reported
performance in machine-learning accident risk prediction. United States (US Accidents, 7.48 M records) and
Great Britain (DfT STATS19, 2021–2025), with pre-registered protocols and locked test sets.

**Author:** Veer Bhadra Pratap Singh, Department of Cyber Security, School of Skill Sciences & Entrepreneurship,
Galgotias University, Greater Noida, India · [ORCID 0000-0002-3162-134X](https://orcid.org/0000-0002-3162-134X)
**Archive:** [10.5281/zenodo.22932569](https://doi.org/10.5281/zenodo.22932569)

## Findings

**1. Count-based labels leak exposure, in both countries.** A model given only the area identifier — no time, no
weather — reaches **0.898** AUC (US) and **0.778** (GB) when "high risk" is defined by ranking counts nationally.
Under within-area labels the same model scores **0.510** and **0.530**, with macro per-area AUC of exactly
**0.5000** in both. Dividing counts by vehicle-miles removes only part of the effect (US: 0.898 → 0.781).

**2. Whether conclusions survive temporal shift is dataset-specific.** The US ranking reordered sharply across a
one-year gap (GCNII 1st → 3rd, MLP 4th → 1st, DeLong p ≤ 0.011); the GB ranking barely moved.

**3. So is the cost of federated training.** US: −0.0127 AUC (paired t, p = 0.0012). GB: **+0.0039** (p = 0.004).

**4. Architectural complexity did not pay.** XGBoost wins in GB, an MLP in the US, ahead of every graph,
hypergraph, spiking and federated model. Of the proposed model's components only personalization helps
(GB +0.0041, p = 0.0007); the learned hypergraph gate converged to exactly 0.0 in every seed of both studies.

**Practical recommendation:** report an area-only baseline as a routine diagnostic. If it scores well above
chance, the labels are leaking exposure.

## Protocols

| | United States | Great Britain |
|---|---|---|
| Source | US Accidents (Moosavi et al.), March 2023 release | DfT STATS19, "last 5 years" collision file |
| Records | 7,481,761 (2016–2022) + Jan–Mar 2023 for the test | 2021–2025 |
| Clients | 49 jurisdictions (48 states + DC) | 43 police forces |
| Contexts | (area, year, month, hour); 79,198 | (area, year, month, hour); 43,492 |
| Features | cyclic month, cyclic hour, temperature mean and s.d. | cyclic month, cyclic hour, wet-road and adverse-weather shares |
| Split | train 2016–2021, validate 2022, **test Jan–Mar 2023** (2,287 contexts) | train 2021–2023, validate 2024, **test 2025** (11,173 contexts) |
| Test opened | 2026-09-23T02:27:07Z | 2026-09-25T12:05:17Z |

Labels: top 40% of the accident count **within each area-year**. The national variant is reported only to
demonstrate exposure leakage. Graphs connect contexts of the same area only (adjacent hour, adjacent month, same
slot in the adjacent year), so they partition exactly across federated clients and no graph structure is shared.

Every family receives the same tuning budget, selected on validation data alone; federated models get up to 100
rounds with the same early stopping the centralized models use. `final_test.py` records a SHA-256 of the frozen
selection and the timestamp of first access, and refuses to run if the selection changes afterwards.

## Reproducing

```bash
pip install -r requirements.txt

# United States
python dev_select.py --csv US_Accidents_March23.csv --out results_us --labels within_state --seeds 0,1
python final_test.py --csv US_Accidents_March23.csv --out results_us --seeds 0

# Great Britain
python dev_select.py --csv dft-road-casualty-statistics-collision-last-5-years.csv --uk --out results_uk --labels within_state --seeds 0,1
python final_test.py --csv dft-road-casualty-statistics-collision-last-5-years.csv --uk --out results_uk --seeds 0

# Exposure analysis
python dev_select.py --csv US_Accidents_March23.csv --exposure results/exposure/us_vmt_exposure_yearly.csv \
    --out results_exposure --labels within_state,national,rate_national --seeds 0,1

python report_v2.py results_us
```

Data: US Accidents at https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents;
STATS19 at https://www.gov.uk/government/statistical-data-sets/road-safety-open-data;
exposure from FHWA Highway Statistics table VM-2 (transcriptions in `results/exposure/`).
**Never open the US CSV in Excel**: it truncates at 1,048,575 rows, about 14% of the data, and the loader refuses
such files.

## Contents

```
fedhsa/            library: data loaders (US + UK), graphs, models, federated trainers, statistics, exposure
dev_select.py      development phase (validation only), resumable
final_test.py      locked test phase, with the frozen-selection fingerprint check
report_v2.py       tables and figures
results/us/        322 development runs, 95 test runs (19 models x 5 seeds), 96 prediction files, figures
results/uk/        202 development runs, 90 test runs (18 models x 5 seeds), 91 prediction files
results/exposure/  96 runs x 2 exposure series, FHWA VM-2 transcriptions (2016/2020/2022), both CSVs
notebooks/         the Kaggle cells used for each study
```

`results/us/US_RESULTS.md`, `results/uk/UK_RESULTS.md` and `results/exposure/EXPOSURE_RESULTS.md` contain the
full tables with DeLong and paired t-tests.

## Hardware

All runs on Kaggle, CPU only: Intel Xeon @ 2.20 GHz, 4 logical cores, 33.7 GB RAM, Python 3.12.13,
PyTorch 2.10.0+cpu, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.2.0. Total ≈ 97 h.

## Licence

MIT (see LICENSE). Citation details in `CITATION.cff`.
