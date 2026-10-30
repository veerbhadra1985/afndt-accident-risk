# Adding exposure data (closes the "thin features / raw counts" objection)

## Why
Labels built on raw counts across areas partly measure traffic volume, not risk. Our primary
protocol already removes this by ranking **within** each area-year (exposure is constant there, so it
cancels exactly). Exposure data lets us go further and show what happens to a **national** ranking
when it is divided by vehicle-miles travelled.

## US data (one CSV, free)
1. Go to https://catalog.data.gov and search: **"Vehicle Miles of Travel by Functional System and State VM-2"**
   (publisher: Federal Highway Administration; covers 1980–2024).
2. Download the **CSV** resource. Alternatively use the Excel table VM-2 from
   https://www.fhwa.dot.gov/policyinformation/statistics/2023/vm2.cfm and save it as CSV.
3. The loader detects the area, year and total-VMT columns automatically; state names are mapped to
   two-letter codes.

## UK data (two files)
1. DfT road traffic statistics by local authority ("TRA89" tables), from
   https://www.gov.uk/government/collections/road-traffic-statistics — gives vehicle-miles per local
   authority per year.
2. ONS Open Geography lookup "Local Authority District to Community Safety Partnership to Police
   Force Area" — maps local authorities to the 43 police forces.
3. Aggregate traffic by police force area and save a CSV with columns: area,year,exposure
   (area = F1, F3, … matching the client names used by the pipeline).

## Running with exposure
```bash
python dev_select.py --csv US_Accidents_March23.csv --exposure vm2.csv \
    --labels within_state,national,rate_national --seeds 0,1 --out results_exposure
```
`rate_national` ranks contexts by count / vehicle-miles within each year.
`rate_within_state` is provided for completeness and is identical to `within_state` by construction.

## What to report in the paper
| Labelling | What the area-only baseline scores | Interpretation |
|---|---|---|
| national (counts) | 0.898 (US) | the ranking is largely exposure |
| rate_national (counts / VMT) | to be measured | how much of it survives exposure adjustment |
| within_state (counts) | 0.510 (US), macro 0.5000 | exposure removed by construction |
