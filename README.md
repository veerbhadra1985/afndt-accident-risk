# Do machine-learning benchmarks for accident-risk prediction compare models or protocols?

Code and complete results for a two-country study of how labelling and evaluation choices shape reported
performance in machine-learning accident risk prediction. United States (US Accidents, 7.73 M records,
2016–2023) and Great Britain (DfT STATS19, 0.51 M collisions, 2021–2025), under pre-registered protocols
with test sets that were opened once.

**Author:** Veer Bhadra Pratap Singh, Department of Cyber Security, School of Skill Sciences & Entrepreneurship,
Galgotias University, Greater Noida, India · [ORCID 0000-0002-3162-134X](https://orcid.org/0000-0002-3162-134X)
**Archive:** [10.5281/zenodo.22932569](https://doi.org/10.5281/zenodo.22932569)

## Findings

**1. Count-based labels encode exposure, in both countries.** A classifier given only the area identifier — no
time, no weather — reaches **0.898** AUC (US) and **0.778** (GB) when "high risk" is defined by ranking counts
nationally, against **0.510** and **0.530** under within-area labels, where macro per-area AUC is exactly
**0.500**. The effect is larger at tighter thresholds (0.916 at a 20% threshold). Normalising by vehicle-miles
removes only part of it (0.898 → 0.781), and supplying exposure as a model feature changes AUC by at most 0.002.

**2. The conventions decide which model appears best.** Under a random split with national labels, a graph
network appears to beat gradient-boosted trees by **0.129 AUC**; under temporal separation with within-area
labels the same comparison is **0.007**. The convention rewards models able to infer area identity (the
area-only classifier gains 0.357, GCNII 0.060) and penalises those that cannot (−0.057 to −0.061).

**3. Temporal-shift effects are setting-specific.** The US ranking reordered across a one-year gap (a two-layer
perceptron rose from 4th to 1st, DeLong p ≤ 0.011); the British ranking barely moved.

**4. So is the cost of federated training.** US: −0.0127 AUC (paired t, p = 0.001). GB: **+0.0039** (p = 0.004).

**5. Architectural complexity did not pay.** Boosted trees win in GB, a perceptron in the US, ahead of every
graph, hypergraph, spiking and federated model. Of the proposed model's components only personalisation helps;
the learned hypergraph gate converged to exactly 0.0 in every seed of both studies.

**6. The mechanism holds across five orders of magnitude of unit density.** A simulation in which
risk carries no between-area signal by construction reproduces the observed 0.898 at the observed
density, keeps the within-area macro per-area AUC at exactly 0.500 throughout, and shows that at the
sparsity of an hourly city grid the binary-occurrence labelling such studies use still leaks at
**0.745**. Run `python3 density_simulation.py out.csv out.png` (about twelve seconds).

## Reproducing the figures

Every figure in the paper is regenerated from the files in `results/` by a script in this repository:

| Figure | Script | Inputs |
|---|---|---|
| 1 | `make_figure1.py` | `results/exposure/exposure_yearly_dev_96runs.json`, `results/uk/uk_dev_202runs.json` |
| 2 | `make_figure2.py` | `results/protocol/protocol_runs.json` |
| 3 | `make_figure3.py` | `results/us/us_selected_FROZEN.json`, `results/us/us_final_seeds0-4.json`, and the UK equivalents |
| 4 | `density_simulation.py` | `us_vmt_exposure.csv` only |

Each prints the values it plots, so the numbers in the paper can be checked against the
figures without opening them.

**Recommendation:** report an area-only baseline as a routine diagnostic. If it scores well above chance, the
labels are encoding exposure.

## Protocols

| | United States | Great Britain |
|---|---|---|
| Source | US Accidents (Moosavi et al.), March 2023 release | DfT STATS19, "last 5 years" collision file |
| Clients | 49 jurisdictions (48 states + DC) | 44 police forces |
| Contexts | (area, year, month, hour); 79,198 | (area, year, month, hour); 55,965 |
| Features | cyclic month, cyclic hour, temperature mean and s.d. | cyclic month, cyclic hour, wet-road and adverse-weather shares |
| Split | train 2016–2021, validate 2022, **test Jan–Mar 2023** (2,287 contexts) | train 2021–2023, validate 2024, **test 2025** (11,173 contexts) |
| Test opened | 2026-09-23T02:27:07Z | 2026-09-25T12:05:17Z |

Labels: top 40% of the collision count **within each area-year**. Graphs connect contexts of the same area only,
so they partition exactly across federated clients and no graph structure is shared. Every family receives the
same tuning budget, selected on validation data alone. `final_test.py` records a SHA-256 of the frozen selection
and the timestamp of first access, and refuses to run if the selection changes afterwards.

## Reproducing

```bash
pip install -r requirements.txt

# United States
python dev_select.py --csv US_Accidents_March23.csv --out results_us --labels within_state --seeds 0,1
python final_test.py --csv US_Accidents_March23.csv --out results_us --seeds 0

# Great Britain
python dev_select.py --csv dft-road-casualty-statistics-collision-last-5-years.csv --uk --out results_uk --labels within_state --seeds 0,1
python final_test.py --csv dft-road-casualty-statistics-collision-last-5-years.csv --uk --out results_uk --seeds 0

# Exposure analysis (validation data only)
python dev_select.py --csv US_Accidents_March23.csv --exposure results/exposure/us_vmt_exposure_yearly.csv \
    --out results_exposure --labels within_state,national,rate_national --seeds 0,1

# Protocol factorial, exposure feature and label-threshold sensitivity (validation data only)
python protocol_study.py --csv US_Accidents_March23.csv --exposure results/exposure/us_vmt_exposure_yearly.csv \
    --out results_protocol --seeds 0,1,2,3 --tops 0.20,0.30,0.40

python report_v2.py results_us
```

Data: US Accidents at https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents; STATS19 at
https://www.gov.uk/government/statistical-data-sets/road-safety-open-data; exposure from FHWA Highway
Statistics table VM-2 (transcriptions in `results/exposure/`).
**Never open the US CSV in Excel**: it truncates at 1,048,575 rows, about 86% of the records, and the loader
refuses such files.

## Contents

```
fedhsa/              library: data loaders (US + UK), graphs, models, federated trainers, statistics, exposure
dev_select.py        development phase (validation only), resumable
final_test.py        locked test phase, with the frozen-selection fingerprint check
protocol_study.py    protocol factorial, exposure feature, label-threshold sensitivity (validation only)
report_v2.py         tables and figures
results/us/          322 development runs, 95 test runs (19 models x 5 seeds), 96 prediction files, figures
results/uk/          202 development runs, 90 test runs (18 models x 5 seeds), 91 prediction files
results/exposure/    exposure analysis, FHWA VM-2 transcriptions (2016/2020/2022), both exposure series
results/protocol/    protocol factorial, exposure-feature and threshold runs
notebooks/           the Kaggle cells used for each study
```

Each run in the `*_final_seeds0-4.json` files carries a `probs_file` field holding the predicted
probability vector for that run's test set. The name of that file is a hash of the run configuration
(model, labelling, seed), **not** of its contents, so the same name occurs under both
`results/us/test_probs/` and `results/uk/test_probs/` while referring to different vectors: 91 of the
96 United States names also occur among the 91 Great Britain names. Resolve `probs_file` against the
`test_probs` directory of the country whose result file names it, and preserve the directory layout
when copying the archive; collapsing it into a single folder silently overwrites 91 of the 187 vectors.

`results/us/US_RESULTS.md`, `results/uk/UK_RESULTS.md` and `results/exposure/EXPOSURE_RESULTS.md` contain the
full tables with DeLong and paired t-tests.

## Hardware

All runs on Kaggle, CPU only: Intel Xeon @ 2.20 GHz, 4 logical cores, 33.7 GB RAM, Python 3.12.13,
PyTorch 2.10.0+cpu, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.2.0. Total ≈ 92 h across 1,193 model runs.

## Licence

MIT (see LICENSE). Citation details in `CITATION.cff`.
