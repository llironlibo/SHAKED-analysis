"""
Stage 09 -- Survival / accelerated-failure-time / RMST analysis.

Treats consultation cycle time as time-to-event data: restricted mean
survival time (primary tau at the 90th percentile, plus sensitivities), a
log-normal AFT time ratio, and a Cox model (the latter feeds the E-values).

Reproduces: the RMST difference shown in Figure 2A (-11.8 min) and the
AFT / Cox rows of eTable 1.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable8_Survival_AFT_RMST.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 4: SURVIVAL / AFT / RMST ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

# PP subset: Users vs Wing B controls
sub = df[(df['is_user'] == 1) | (df['wing'] == 'B')].dropna(
    subset=['consultation_cycle_min']
).copy()
sub['event'] = 1  # All consultations complete (no censoring)
print(f"PP sample: {len(sub)} patients (Users: {sub['is_user'].sum()}, Controls: {(sub['wing']=='B').sum()})")

covars = ['triage', 'diagnosis_count', 'has_radiology', 'is_admitted',
          'n_consultations', 'is_ambulance', 'admission_hour']  # NOTE: admission_hour = ED arrival hour, not hospital admission
sub_cov = sub.dropna(subset=covars).copy()

# ============================================================================
# 1. RMST DIFFERENCE
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: RESTRICTED MEAN SURVIVAL TIME (RMST)")
print("=" * 80)

# Pre-specified primary tau: 90th percentile of observed consultation times
tau_primary = np.percentile(sub['consultation_cycle_min'], 90)
print(f"Primary tau: {tau_primary:.0f} min (90th percentile of observed consultation times)")
print("Clinical rationale: captures the vast majority of consultation completions")

taus = [tau_primary, 360, 480, 600]

for tau in taus:
    is_primary = tau == tau_primary
    label = f"tau={tau:.0f} min {'(PRIMARY - 90th pctl)' if is_primary else '(sensitivity)'}"
    print(f"\n  --- RMST at {label} ---")

    # Compute RMST for each group via Kaplan-Meier integration
    for group_name, g_val in [('Users', 1), ('Controls', 0)]:
        g = sub[sub['is_user'] == g_val]
        kmf = KaplanMeierFitter()
        # Cap times at tau for RMST
        t_capped = np.minimum(g['consultation_cycle_min'].values, tau)
        e_capped = np.where(g['consultation_cycle_min'].values <= tau, 1, 1)
        kmf.fit(t_capped, e_capped)

    # Bootstrap RMST difference
    n_boot = 2000
    rmst_diffs = np.zeros(n_boot)
    users_data = sub[sub['is_user'] == 1]['consultation_cycle_min'].values
    ctrl_data = sub[sub['is_user'] == 0]['consultation_cycle_min'].values

    for b in range(n_boot):
        bu = np.random.choice(users_data, size=len(users_data), replace=True)
        bc = np.random.choice(ctrl_data, size=len(ctrl_data), replace=True)
        # RMST = mean of min(T, tau)
        rmst_u = np.mean(np.minimum(bu, tau))
        rmst_c = np.mean(np.minimum(bc, tau))
        rmst_diffs[b] = rmst_u - rmst_c

    rmst_diff = np.mean(np.minimum(users_data, tau)) - np.mean(np.minimum(ctrl_data, tau))
    ci = np.percentile(rmst_diffs, [2.5, 97.5])
    boot_p = 2 * min(np.mean(rmst_diffs >= 0), np.mean(rmst_diffs <= 0))

    print(f"  RMST difference: {rmst_diff:.1f} min [{ci[0]:.1f}, {ci[1]:.1f}], p={boot_p:.4f}")
    print(f"  Interpretation: Users complete consultations {abs(rmst_diff):.1f} min {'faster' if rmst_diff < 0 else 'slower'} (within {tau:.0f} min window)")

    results_rows.append({
        'Model': f'RMST (tau={tau:.0f})', 'Analysis': 'PP (Users vs Wing B)',
        'N': len(sub), 'Estimate': f"{rmst_diff:.1f} min",
        'CI_Lower': f"{ci[0]:.1f}", 'CI_Upper': f"{ci[1]:.1f}",
        'p_value': f"{boot_p:.4f}",
        'Note': f"{'PRIMARY' if is_primary else 'Sensitivity'}; 2000 bootstrap replicates"
    })

# ============================================================================
# 2. AFT MODEL (Log-Normal)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: ACCELERATED FAILURE TIME (Log-Normal)")
print("=" * 80)

aft_cols = ['consultation_cycle_min', 'event', 'is_user'] + covars
aft_data = sub_cov[aft_cols].copy()

# Unadjusted
aft_unadj = LogNormalAFTFitter()
aft_unadj.fit(aft_data[['consultation_cycle_min', 'event', 'is_user']],
               duration_col='consultation_cycle_min', event_col='event')
print("\n  Unadjusted AFT:")
tr_unadj = np.exp(aft_unadj.params_.loc[('mu_', 'is_user')])
ci_unadj = aft_unadj.confidence_intervals_.loc[('mu_', 'is_user')]
p_unadj = aft_unadj.summary.loc[('mu_', 'is_user'), 'p']
print(f"  Time Ratio: {tr_unadj:.3f} [{np.exp(ci_unadj.iloc[0]):.3f}, {np.exp(ci_unadj.iloc[1]):.3f}], p={p_unadj:.4f}")

results_rows.append({
    'Model': 'AFT Log-Normal (unadjusted)', 'Analysis': 'PP',
    'N': len(aft_data), 'Estimate': f"TR={tr_unadj:.3f}",
    'CI_Lower': f"{np.exp(ci_unadj.iloc[0]):.3f}", 'CI_Upper': f"{np.exp(ci_unadj.iloc[1]):.3f}",
    'p_value': f"{p_unadj:.4f}",
    'Note': 'TR<1 = faster completion'
})

# Adjusted
aft_adj = LogNormalAFTFitter()
aft_adj.fit(aft_data, duration_col='consultation_cycle_min', event_col='event')
tr_adj = np.exp(aft_adj.params_.loc[('mu_', 'is_user')])
ci_adj = aft_adj.confidence_intervals_.loc[('mu_', 'is_user')]
p_adj = aft_adj.summary.loc[('mu_', 'is_user'), 'p']
print(f"\n  Adjusted AFT:")
print(f"  Time Ratio: {tr_adj:.3f} [{np.exp(ci_adj.iloc[0]):.3f}, {np.exp(ci_adj.iloc[1]):.3f}], p={p_adj:.4f}")

results_rows.append({
    'Model': 'AFT Log-Normal (adjusted)', 'Analysis': 'PP',
    'N': len(aft_data), 'Estimate': f"TR={tr_adj:.3f}",
    'CI_Lower': f"{np.exp(ci_adj.iloc[0]):.3f}", 'CI_Upper': f"{np.exp(ci_adj.iloc[1]):.3f}",
    'p_value': f"{p_adj:.4f}",
    'Note': f'Adjusted for {", ".join(covars)}; TR<1 = faster'
})

# ============================================================================
# 3. COX PH (for E-value input)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: COX PROPORTIONAL HAZARDS")
print("=" * 80)

cox = CoxPHFitter()
cox_data = sub_cov[['consultation_cycle_min', 'event', 'is_user'] + covars].copy()
cox.fit(cox_data, duration_col='consultation_cycle_min', event_col='event')

hr = np.exp(cox.params_['is_user'])
cox_ci = cox.confidence_intervals_.loc['is_user']
cox_p = cox.summary.loc['is_user', 'p']

print(f"  HR: {hr:.3f} [{np.exp(cox_ci.iloc[0]):.3f}, {np.exp(cox_ci.iloc[1]):.3f}], p={cox_p:.4f}")
print(f"  HR > 1 means faster consultation completion (beneficial)")

# PH assumption test
ph_test = cox.check_assumptions(cox_data, p_value_threshold=0.05, show_plots=False)

results_rows.append({
    'Model': 'Cox PH (adjusted)', 'Analysis': 'PP',
    'N': len(cox_data), 'Estimate': f"HR={hr:.3f}",
    'CI_Lower': f"{np.exp(cox_ci.iloc[0]):.3f}", 'CI_Upper': f"{np.exp(cox_ci.iloc[1]):.3f}",
    'p_value': f"{cox_p:.4f}",
    'Note': f'HR>1 = faster completion; for E-value input in Phase 8'
})

# Log-rank test (unadjusted)
lr = logrank_test(
    sub[sub['is_user']==1]['consultation_cycle_min'],
    sub[sub['is_user']==0]['consultation_cycle_min'],
    event_observed_A=np.ones(sub['is_user'].sum()),
    event_observed_B=np.ones((sub['is_user']==0).sum())
)
print(f"\n  Log-rank test: chi2={lr.test_statistic:.2f}, p={lr.p_value:.4f}")

results_rows.append({
    'Model': 'Log-rank test', 'Analysis': 'PP',
    'N': len(sub), 'Estimate': f"chi2={lr.test_statistic:.2f}",
    'CI_Lower': '', 'CI_Upper': '',
    'p_value': f"{lr.p_value:.4f}",
    'Note': 'Unadjusted comparison'
})

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable8")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable8_Survival_AFT_RMST.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print(f"""
PHASE 4 COMPLETE

Key findings:
  RMST (primary, tau={tau_primary:.0f} min): {results_rows[0]['Estimate']} [{results_rows[0]['CI_Lower']}, {results_rows[0]['CI_Upper']}]
  AFT adjusted: TR={tr_adj:.3f} (p={p_adj:.4f})
  Cox adjusted: HR={hr:.3f} (p={cox_p:.4f})

  Cox HR is saved for E-value computation in Phase 8.
""")
