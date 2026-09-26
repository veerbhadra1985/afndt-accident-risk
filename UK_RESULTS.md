# UK STATS19 study - FINAL RESULTS (5 seeds)

Data: DfT STATS19 'last 5 years' collision file. Clients: 43 police forces. Contexts (force, year, month, hour).
Protocol: train 2021-2023, validate 2024, test 2025 (11173 contexts, 0.354 positive).
Test opened 2026-09-25T12:05:17.592177+00:00; frozen-selection SHA-256 ddfb807d41acaf22...
Compute: development 16.7 h + test 11.9 h.

## Final test (mean ± sd over 5 seeds)

| Model | Test AUC | F1 | Macro force AUC | Val AUC (2024) | DeLong p vs XGBoost (worst seed) |
|---|---|---|---|---|---|
| XGBoost | 0.9171 ± 0.0002 | 0.7732 ± 0.0024 | 0.9110 ± 0.0002 | 0.9277 | — |
| MLP | 0.9100 ± 0.0010 | 0.7657 ± 0.0017 | 0.9028 ± 0.0009 | 0.9183 | 8.6e-09 |
| AFNDT-v2 [flat (no regions)] | 0.9051 ± 0.0004 | 0.7619 ± 0.0015 | 0.8963 ± 0.0005 | — | 1.2e-14 |
| FedAvg-GCNII+FT | 0.9049 ± 0.0003 | 0.7607 ± 0.0016 | 0.8981 ± 0.0002 | 0.9142 | 6.8e-16 |
| AFNDT-v2 | 0.9046 ± 0.0007 | 0.7612 ± 0.0022 | 0.8952 ± 0.0008 | 0.9125 | 2e-15 |
| AFNDT-v2 [toggle hypergraph] | 0.9045 ± 0.0005 | 0.7616 ± 0.0009 | 0.8952 ± 0.0010 | — | 4.5e-16 |
| GCNII | 0.9039 ± 0.0008 | 0.7552 ± 0.0018 | 0.8951 ± 0.0008 | 0.9119 | 5.2e-20 |
| AFNDT-v2 [+ spiking readout] | 0.9027 ± 0.0004 | 0.7607 ± 0.0018 | 0.8967 ± 0.0002 | — | 2.1e-19 |
| FedAvg-GCNII | 0.9025 ± 0.0011 | 0.7545 ± 0.0040 | 0.8957 ± 0.0009 | 0.9096 | 2.4e-25 |
| FedProx-GCNII | 0.9010 ± 0.0004 | 0.7528 ± 0.0013 | 0.8934 ± 0.0002 | 0.9082 | 3.8e-34 |
| AFNDT-v2 [centralized] | 0.9007 ± 0.0011 | 0.7503 ± 0.0010 | 0.8914 ± 0.0012 | — | 2.2e-28 |
| AFNDT-v2 [no personalization] | 0.9005 ± 0.0004 | 0.7552 ± 0.0010 | 0.8939 ± 0.0003 | — | 5.5e-32 |
| GAT | 0.8850 ± 0.0015 | 0.7312 ± 0.0028 | 0.8746 ± 0.0013 | 0.8946 | 6.9e-73 |
| FedAvg-GAT | 0.8850 ± 0.0010 | 0.7344 ± 0.0016 | 0.8752 ± 0.0007 | 0.8923 | 2e-69 |
| GCN | 0.8843 ± 0.0007 | 0.7374 ± 0.0012 | 0.8771 ± 0.0005 | 0.8890 | 1.4e-69 |
| Logistic regression | 0.8426 ± 0.0000 | 0.6997 ± 0.0000 | 0.8400 ± 0.0000 | 0.8495 | 8e-155 |
| HGNN | 0.8027 ± 0.0010 | 0.6460 ± 0.0007 | 0.8043 ± 0.0002 | 0.8075 | 1e-226 |
| State-only LR | 0.5270 ± 0.0000 | 0.5205 ± 0.0000 | 0.5000 ± 0.0000 | 0.5300 | 0 |

## AFNDT-v2 comparisons (paired t-test over 5 seeds; DeLong range)

| Comparison | ΔAUC | paired-t p | DeLong p range | verdict |
|---|---|---|---|---|
| federation vs centralized | +0.0039 | 0.00407 | 2.8e-08 – 0.021 | significant |
| personalization | +0.0041 | 0.000694 | 6.2e-08 – 0.0044 | significant |
| hierarchical aggregation | -0.0006 | 0.0305 | 0.00024 – 0.76 | significant |
| hypergraph branch | +0.0001 | 0.651 | 0.16 – 0.35 | not significant |
| spiking readout | +0.0019 | 0.0114 | 0.00068 – 0.19 | significant |
| vs personalized baseline | -0.0004 | 0.346 | 0.05 – 0.72 | not significant |
| vs FedAvg | +0.0021 | 0.0278 | 0.00038 – 0.91 | significant |
| vs centralized GCNII | +0.0006 | 0.251 | 0.11 – 0.59 | not significant |
| vs XGBoost | -0.0125 | 4.27e-07 | 1.5e-17 – 2e-15 | significant |

## Exposure leakage (validation 2024)

| Model | within-force labels | national labels | inflation |
|---|---|---|---|
| State-only LR | 0.5300 | 0.7776 | +0.2476 |
| Logistic regression | 0.8495 | 0.7690 | -0.0805 |
| XGBoost | 0.9262 | 0.9639 | +0.0377 |
| MLP | 0.9169 | 0.9292 | +0.0123 |

## Findings (UK, 5 seeds)
1. Exposure leakage replicates: the force-only model rises from 0.530 (within-force labels) to 0.778 (national labels). Macro per-force AUC under within-force labels is exactly 0.5000.
2. The ranking is STABLE under a one-year shift: every model loses 0.007-0.010 AUC and the order is nearly unchanged (contrast: the US ranking reordered sharply).
3. Federated training BEATS centralized training (+0.0039, p=0.004) - opposite in sign to the US result (-0.0127, p=0.001).
4. Of AFNDT-v2's components: personalization helps (+0.0041, p=0.0007); hierarchical aggregation slightly hurts (-0.0006, p=0.031); the spiking readout hurts (+0.0019 when removed, p=0.011); the hypergraph branch does nothing (+0.0001, p=0.65) and its gate converged to exactly 0.0 in every seed.
5. AFNDT-v2 is statistically tied with FedAvg-GCNII+FT (p=0.35) and with centralized GCNII (p=0.25).
6. XGBoost beats every neural, graph and federated model (+0.0125 over AFNDT-v2, p=4e-07; DeLong p<1e-14 in every seed).