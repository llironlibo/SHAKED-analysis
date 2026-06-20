# SHAKED - Analysis Code

Statistical analysis code for the SHAKED pilot study: a DECIDE-AI Stage 1
evaluation of a multi-model large language model clinical decision support
system, deployed in the emergency department of Rambam Health Care Campus.
Patients were allocated to an intervention wing (with SHAKED access) or a
control wing by strict alternating A/B assignment at triage by the charge nurse
(not bed-availability/occupancy-based), with baseline balance verified after the
fact. The pre-specified focus is implementation and
adoption; the consultation cycle time (time from a specialist-consultation order
to the recorded recommendation) is reported as an **exploratory** efficacy
signal. Borderline efficacy results are treated as hypothesis-generating, consistent with a Stage 1 pilot.

Reference: [Nature Medicine citation - to be added on publication].

This repository contains the **statistical analysis code only**. The SHAKED
system / deployment code is proprietary and is not part of this release.

## Data availability

The patient-level data are **not included** in this repository (protected health
information). The de-identified analytic dataset is available under **controlled
access** - see the Data Availability statement in the published article and
[`data/README.md`](data/README.md) for the request and approval procedure.

To reproduce the analyses, obtain the dataset through that procedure and place it
at:

```
data/cleaned/unified_pilot_cohort.csv
```

Everything under `data/` (except the README) is git-ignored, so data files
cannot be committed accidentally.

## Requirements and how to run

- Python 3.10+ with the packages in `requirements.txt`
  (pandas, numpy, scipy, statsmodels, scikit-learn, linearmodels, lifelines,
  matplotlib; plus `openpyxl` only if re-running data preparation from raw files).
- R 4.5+ with ggplot2 4.0+ (and patchwork, scales, ggtext, ragg, dplyr, readr,
  pROC) for the R figures. The R figures are optional; the analysis stages and
  the Python figures run without R.

```bash
pip install -r requirements.txt
# place the de-identified cohort at data/cleaned/unified_pilot_cohort.csv
python run_all.py
```

`run_all.py` runs the numbered stages `01`..`16` in order, then the figures
(Python first, then the R figures if `Rscript` is available). Every stage writes
its tables to `results/` and figures to `results/figures/`. Each stage is also
runnable on its own, e.g. `python 03_sex_subgroup_sager.py`.

Stage `01` rebuilds the cohort from the raw hospital exports; those raw files are
controlled-access and are not shipped, so stage `01` simply reports that and
exits when they are absent. Stages `02`-`16` load the cohort CSV directly.

## Stage-by-stage map

| File | Computes | Reproduces |
|--|--|--|
| `01_data_preparation.py` | Builds the analytic cohort from the raw exports (filters, SHAKED usage flags, outcome derivation, consultation merge) | The analytic dataset underlying Table 1 and all downstream analyses |
| `02_baseline_and_itt.py` | Baseline characteristics by wing (incl. P1-P5 triage) and primary ITT subgroup Hodges-Lehmann effects | Table 1; primary ITT subgroup estimates |
| `03_sex_subgroup_sager.py` | Sex-disaggregated primary ITT + sex-by-allocation interaction (SAGER) | Sex-disaggregated ITT (Round-2 addition) |
| `04_adjusted_primary.py` | Multivariable-adjusted log-linear, quantile, and winsorized models | eTable 1 (adjusted robustness rows) |
| `05_iptw_aipw.py` | Inverse-probability weighting + AIPW; propensity balance | eTable 1 (IPTW / AIPW); eFigure 1 love-plot data |
| `06_iv_cace.py` | Instrumental-variable / CACE (Wald, 2SLS, cluster-bootstrap) | Table 2C; eTable 1 (causal rows) |
| `07_cace_anderson_rubin.py` | Anderson-Rubin confidence set for the CACE | CACE Anderson-Rubin CI (Round-2 addition) |
| `08_negative_controls.py` | Negative-control outcomes (IV falsification) | Table 3 / eTable 6 (falsification) |
| `09_survival_rmst.py` | RMST, AFT, and Cox models | Figure 2A RMST (-11.8 min); eTable 1 (AFT/Cox) |
| `10_clustering_robustness.py` | Mixed-effects, cluster-robust SE, fixed effects, jackknife | eTable 1 (clustering / jackknife) |
| `11_dose_response.py` | Engagement exposure-level contrasts and within-wing trend | Table 2 D/E; Supp Fig 2A exposure-level |
| `12_adoption_dynamics.py` | Adoption decay, workload, and radiology models | eTable 3; Figure 3 data |
| `13_evalues.py` | E-values for unmeasured confounding | eTable 11 (Round-2 addition) |
| `14_rct_sizing.py` | Sample-size estimation for a definitive trial | eTable 5 |
| `15_multiplicity.py` | Holm-Bonferroni multiplicity correction | eTable 13 |
| `16_weekly_los_contamination.py` | Weekly effects, length of stay, contamination bounding, design checks | eTable 2 (weekly); eTable 4 (LOS); design validation |
| `figures/figure1_consort.py` | CONSORT participant-flow diagram | Figure 1 |
| `figures/figure2_forest_dose_response.py` (.R) | Forest plot of effect estimates + engagement exposure-level | Supp Fig 2A |
| `figures/figure3_adoption_dynamics.R` | Adoption dynamics + weekly ITT effects | Figure 3 |
| `figures/efigure1_propensity_balance.R` | Propensity-score distribution + SMD love plot | eFigure 1 |

`figures/_common.R` and `_shared.py` are shared infrastructure (theme/save
helpers and data loading); they are imported by the scripts above and produce no
output of their own.

## License

MIT - see [LICENSE](LICENSE).
