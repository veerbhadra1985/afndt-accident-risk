# FINAL RESULTS (5 seeds, locked test Jan–Mar 2023)

Test contexts: 2287, positives 0.396. Test opened 2026-09-23T02:27:07.662039+00:00, selection sha256 c88d51bae0cd9cc3…

| Model | Test AUC | F1 | Macro state AUC | Val AUC (2022) | DeLong p vs MLP (worst seed) |
|---|---|---|---|---|---|
| MLP | 0.8135 ± 0.0019 | 0.6932 ± 0.0035 | 0.8358 ± 0.0016 | 0.8549 | — |
| XGBoost | 0.8034 ± 0.0005 | 0.6804 ± 0.0013 | 0.8276 ± 0.0007 | 0.8540 | 5.4e-05 |
| GCNII | 0.8002 ± 0.0016 | 0.6924 ± 0.0020 | 0.8264 ± 0.0027 | 0.8614 | 0.00024 |
| FedAvg-GAT | 0.7966 ± 0.0031 | 0.6869 ± 0.0023 | 0.8195 ± 0.0023 | 0.8534 | 0.011 |
| GAT | 0.7934 ± 0.0039 | 0.6794 ± 0.0030 | 0.8128 ± 0.0037 | 0.8513 | 0.00081 |
| AFNDT-v2 | 0.7874 ± 0.0048 | 0.6736 ± 0.0063 | 0.8239 ± 0.0035 | 0.8594 | 9.4e-06 |
| GCN | 0.7859 ± 0.0008 | 0.6778 ± 0.0027 | 0.8136 ± 0.0010 | 0.8437 | 1e-10 |
| FedProx-GCNII | 0.7829 ± 0.0020 | 0.6791 ± 0.0026 | 0.8150 ± 0.0009 | 0.8531 | 1e-13 |
| FedAvg-GCNII+FT | 0.7774 ± 0.0024 | 0.6735 ± 0.0052 | 0.8215 ± 0.0028 | 0.8576 | 1.1e-11 |
| FedAvg-GCNII | 0.7718 ± 0.0038 | 0.6644 ± 0.0036 | 0.8147 ± 0.0029 | 0.8530 | 2.7e-17 |
| AFNDT-v3 | 0.7614 ± 0.0022 | 0.6508 ± 0.0063 | 0.8044 ± 0.0021 | 0.8513 | 9e-16 |
| Logistic regression | 0.7426 ± 0.0000 | 0.6648 ± 0.0000 | 0.7547 ± 0.0000 | 0.7664 | 2.2e-14 |
| HGNN | 0.7397 ± 0.0631 | 0.6081 ± 0.0361 | 0.7629 ± 0.0610 | 0.7725 | 0.011 |
| State-only LR | 0.5040 ± 0.0000 | 0.5670 ± 0.0000 | 0.5000 ± 0.0000 | 0.5100 | 5.4e-88 |

## AFNDT-v2: ablations and comparisons (paired t-test over 5 seeds)

| Comparison | ΔAUC | paired-t p | verdict |
|---|---|---|---|
| AFNDT-v2 vs AFNDT-v2 [centralized] | -0.0127 | 0.0012 | cost of federation: significant |
| AFNDT-v2 vs AFNDT-v2 [no personalization] | +0.0033 | 0.211 | personalization: not significant |
| AFNDT-v2 vs AFNDT-v2 [flat (no regions)] | +0.0009 | 0.484 | hierarchical aggregation: not significant |
| AFNDT-v2 vs AFNDT-v2 [toggle hypergraph] | -0.0013 | 0.144 | hypergraph branch: not significant |
| AFNDT-v2 vs AFNDT-v2 [+ spiking readout] | -0.0008 | 0.786 | spiking readout: not significant |
| AFNDT-v2 vs FedAvg-GAT | -0.0092 | 0.0239 | vs best federated baseline: significant |
| AFNDT-v2 vs FedAvg-GCNII+FT | +0.0100 | 0.029 | vs personalized baseline: significant |
| AFNDT-v2 vs FedAvg-GCNII | +0.0156 | 0.0114 | vs FedAvg: significant |
| AFNDT-v2 vs GCNII | -0.0128 | 0.00175 | vs best central graph model: significant |

## Headline findings
1. Exposure leakage: state-only AUC 0.898 with national labels vs 0.510 with within-state labels; macro per-state AUC exactly 0.5000.
2. Temporal shift reorders models: GCNII 1st on validation -> 3rd on test; MLP 4th -> 1st (p<=0.011 vs all).
3. Federation costs 0.0127 AUC (p=0.0012) under a one-year shift.
4. AFNDT-v2 beats FedAvg-GCNII (+0.0156, p=0.011) and FedAvg-GCNII+FT (+0.0100, p=0.029) but loses to FedAvg-GAT (-0.0092, p=0.024).
5. Personalization (+0.0033, p=0.21), hierarchy (+0.0009, p=0.48), hypergraph and spiking readout: no significant effect.
6. Graph, hypergraph and spiking machinery does not beat a two-layer MLP on this task.