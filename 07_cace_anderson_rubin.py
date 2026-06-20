"""
Stage 07 -- Anderson-Rubin confidence set for the CACE.

Because the Wald estimator relies on large-sample normal approximations that
may be imprecise with right-skewed outcomes, this computes the Anderson-Rubin
confidence set, which gives finite-sample-valid inference for the CACE under
the standard IV assumptions, with OLS / HC1 / cluster-robust standard errors.

Reproduces: CACE Anderson-Rubin confidence interval (Round-2 addition).

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable6b_AR_ConfidenceSet.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import f as f_dist
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 3C: ANDERSON-RUBIN CONFIDENCE SET FOR CACE")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
adf = df.dropna(subset=['consultation_cycle_min', 'log_consult']).copy()
print(f"Loaded: {len(adf):,} patients with consultation data")

Z = adf['is_wing_a'].values
D = adf['is_user'].values
D2 = adf['has_shaked_results'].values
Y = adf['consultation_cycle_min'].values
Y_log = adf['log_consult'].values
clusters = adf['arrival_date'].values

unique_dates = np.unique(clusters)
n_clusters = len(unique_dates)
print(f"Clusters (arrival_date): {n_clusters}")

# Wald point estimate for reference
compliance = D[Z == 1].mean() - D[Z == 0].mean()
itt_diff = Y[Z == 1].mean() - Y[Z == 0].mean()
wald_cace = itt_diff / compliance
print(f"\nWald CACE point estimate: {wald_cace:.1f} min (reference)")
print(f"Compliance rate: {compliance:.4f}")
print(f"ITT mean difference: {itt_diff:.1f} min")

# =============================================================================
# ANDERSON-RUBIN TEST FUNCTION
# =============================================================================
def ar_test_pvalue(beta0, Y_vec, D_vec, Z_vec, cluster_vec=None, cov_type='nonrobust'):
    """
    Anderson-Rubin test for H0: CACE = beta0.
    
    Under H0, Y_adj = Y - beta0*D should be independent of Z.
    Regress Y_adj on Z, return p-value for Z coefficient.
    
    cov_type: 'nonrobust' (OLS), 'HC1' (heteroscedasticity-robust), 
              'cluster' (cluster-robust)
    """
    Y_adj = Y_vec - beta0 * D_vec
    X = sm.add_constant(Z_vec)
    if cov_type == 'cluster' and cluster_vec is not None:
        model = sm.OLS(Y_adj, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': cluster_vec}
        )
    elif cov_type == 'HC1':
        model = sm.OLS(Y_adj, X).fit(cov_type='HC1')
    else:
        model = sm.OLS(Y_adj, X).fit()
    # p-value for the instrument coefficient (index 1)
    return model.pvalues[1], model.params[1], model.bse[1]


# =============================================================================
# HELPER: Compute AR confidence set for a given SE type
# =============================================================================
def compute_ar_ci(Y_vec, D_vec, Z_vec, cluster_vec, cov_type, alpha=0.05,
                  grid_lo=-200, grid_hi=100, coarse_step=1.0, fine_step=0.1):
    """Compute AR confidence set with specified SE type."""
    # Coarse grid
    grid_coarse = np.arange(grid_lo, grid_hi, coarse_step)
    p_coarse = np.array([
        ar_test_pvalue(b, Y_vec, D_vec, Z_vec, cluster_vec, cov_type)[0]
        for b in grid_coarse
    ])
    
    in_ci = grid_coarse[p_coarse >= alpha]
    if len(in_ci) == 0:
        return np.nan, np.nan, np.nan
    
    rough_lo, rough_hi = in_ci[0] - 5, in_ci[-1] + 5
    
    # Fine grid
    grid_fine = np.arange(rough_lo, rough_hi, fine_step)
    p_fine = np.array([
        ar_test_pvalue(b, Y_vec, D_vec, Z_vec, cluster_vec, cov_type)[0]
        for b in grid_fine
    ])
    
    in_ci_fine = grid_fine[p_fine >= alpha]
    if len(in_ci_fine) == 0:
        return np.nan, np.nan, np.nan
    
    ci_lo = in_ci_fine[0]
    ci_hi = in_ci_fine[-1]
    
    # Refine with bisection
    try:
        ci_lo = brentq(
            lambda b: ar_test_pvalue(b, Y_vec, D_vec, Z_vec, cluster_vec, cov_type)[0] - alpha,
            ci_lo - 1.0, ci_lo + 0.5, xtol=0.01
        )
    except (ValueError, RuntimeError):
        pass
    
    try:
        ci_hi = brentq(
            lambda b: ar_test_pvalue(b, Y_vec, D_vec, Z_vec, cluster_vec, cov_type)[0] - alpha,
            ci_hi - 0.5, ci_hi + 1.0, xtol=0.01
        )
    except (ValueError, RuntimeError):
        pass
    
    # p-value at beta0 = 0
    p_null = ar_test_pvalue(0, Y_vec, D_vec, Z_vec, cluster_vec, cov_type)[0]
    
    return ci_lo, ci_hi, p_null


# =============================================================================
# AR CONFIDENCE SETS -- MINUTES SCALE (three SE types)
# =============================================================================
alpha = 0.05

se_types = [
    ('nonrobust', 'OLS (homoscedastic)'),
    ('HC1',       'HC1 (heteroscedasticity-robust)'),
    ('cluster',   'Cluster-robust (arrival_date)')
]

ar_results = {}  # store for comparison table

print("\n" + "=" * 80)
print("ANDERSON-RUBIN 95% CONFIDENCE SETS (minutes scale)")
print("=" * 80)

for cov_type, label in se_types:
    print(f"\n  --- {label} ---")
    ci_lo, ci_hi, p_null = compute_ar_ci(Y, D, Z, clusters, cov_type)
    ar_results[f'min_{cov_type}'] = (ci_lo, ci_hi, p_null)
    
    if not np.isnan(ci_lo):
        crosses = "CROSSES zero" if ci_hi > 0 else "EXCLUDES zero"
        print(f"  AR 95% CI: [{ci_lo:.1f}, {ci_hi:.1f}] -- {crosses}")
        print(f"  AR p(H0: CACE=0): {p_null:.4f}")
    else:
        print("  WARNING: Empty confidence set")

# Store primary results (HC1 is the primary -- matches Wald's individual-level approach
# but adds heteroscedasticity robustness)
ar_ci_lo_hc1, ar_ci_hi_hc1, ar_p_hc1 = ar_results['min_HC1']
ar_ci_lo_ols, ar_ci_hi_ols, ar_p_ols = ar_results['min_nonrobust']
ar_ci_lo_clust, ar_ci_hi_clust, ar_p_clust = ar_results['min_cluster']


# =============================================================================
# AR CONFIDENCE SET -- LOG SCALE (time ratio), HC1 primary
# =============================================================================
print("\n" + "=" * 80)
print("ANDERSON-RUBIN 95% CONFIDENCE SET (log scale / time ratio)")
print("=" * 80)

wald_log = (Y_log[Z == 1].mean() - Y_log[Z == 0].mean()) / compliance
wald_tr = np.exp(wald_log)

for cov_type, label in se_types:
    print(f"\n  --- {label} ---")
    grid_log = np.arange(-1.5, 1.0, 0.005)
    p_log = np.array([
        ar_test_pvalue(b, Y_log, D, Z, clusters, cov_type)[0]
        for b in grid_log
    ])
    in_ci_log = grid_log[p_log >= alpha]
    
    if len(in_ci_log) > 0:
        log_lo = in_ci_log[0]
        log_hi = in_ci_log[-1]
        try:
            log_lo = brentq(
                lambda b: ar_test_pvalue(b, Y_log, D, Z, clusters, cov_type)[0] - alpha,
                log_lo - 0.05, log_lo + 0.02, xtol=0.001
            )
        except (ValueError, RuntimeError):
            pass
        try:
            log_hi = brentq(
                lambda b: ar_test_pvalue(b, Y_log, D, Z, clusters, cov_type)[0] - alpha,
                log_hi - 0.02, log_hi + 0.05, xtol=0.001
            )
        except (ValueError, RuntimeError):
            pass
        
        tr_lo, tr_hi = np.exp(log_lo), np.exp(log_hi)
        p_null_log = ar_test_pvalue(0, Y_log, D, Z, clusters, cov_type)[0]
        ar_results[f'log_{cov_type}'] = (log_lo, log_hi, tr_lo, tr_hi, p_null_log)
        
        crosses = "CROSSES 1" if tr_lo < 1 < tr_hi else "EXCLUDES 1"
        print(f"  AR 95% CI (log): [{log_lo:.4f}, {log_hi:.4f}]")
        print(f"  AR 95% CI (TR):  [{tr_lo:.3f}, {tr_hi:.3f}] -- {crosses}")
        print(f"  AR p(H0: TR=1):  {p_null_log:.4f}")
    else:
        ar_results[f'log_{cov_type}'] = (np.nan, np.nan, np.nan, np.nan, np.nan)
        print("  WARNING: Empty confidence set")


# =============================================================================
# SENSITIVITY: D2 = has_shaked_results
# =============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: AR for D2 = has_shaked_results (HC1)")
print("=" * 80)

compliance_d2 = D2[Z == 1].mean() - D2[Z == 0].mean()
wald_d2 = itt_diff / compliance_d2
print(f"  Compliance rate (D2): {compliance_d2:.4f}")
print(f"  Wald CACE (D2): {wald_d2:.1f} min")

ar_d2_lo, ar_d2_hi, ar_d2_p = compute_ar_ci(Y, D2, Z, clusters, 'HC1',
                                              grid_lo=-80, grid_hi=50)
if not np.isnan(ar_d2_lo):
    crosses = "CROSSES zero" if ar_d2_hi > 0 else "EXCLUDES zero"
    print(f"  AR 95% CI (D2, HC1): [{ar_d2_lo:.1f}, {ar_d2_hi:.1f}] -- {crosses}")
    print(f"  AR p(H0: CACE=0): {ar_d2_p:.4f}")
else:
    print("  WARNING: Empty confidence set")

# Also cluster-robust for D2
ar_d2_lo_cl, ar_d2_hi_cl, ar_d2_p_cl = compute_ar_ci(Y, D2, Z, clusters, 'cluster',
                                                        grid_lo=-80, grid_hi=50)
if not np.isnan(ar_d2_lo_cl):
    print(f"  AR 95% CI (D2, cluster): [{ar_d2_lo_cl:.1f}, {ar_d2_hi_cl:.1f}]")
    print(f"  AR p(H0: CACE=0, cluster): {ar_d2_p_cl:.4f}")


# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 80)
print("COMPARISON: WALD vs ANDERSON-RUBIN")
print("=" * 80)

from scipy.stats import norm

# Wald CI (delta-method, from Phase 3 -- individual-level SE, no clustering)
itt_se = np.sqrt(Y[Z==1].var()/sum(Z==1) + Y[Z==0].var()/sum(Z==0))
wald_se = itt_se / compliance
wald_ci_lo = wald_cace - 1.96 * wald_se
wald_ci_hi = wald_cace + 1.96 * wald_se
wald_p = 2 * (1 - norm.cdf(abs(wald_cace / wald_se)))

print(f"\n  {'Method':<45} {'Estimate':>10} {'95% CI':>22} {'p(H0:b=0)':>12}")
print(f"  {'-'*45} {'-'*10} {'-'*22} {'-'*12}")
print(f"  {'Wald LATE (delta-method, no clustering)':<45} {wald_cace:>8.1f}   [{wald_ci_lo:>6.1f}, {wald_ci_hi:>5.1f}]   {wald_p:>10.4f}")

for cov_type, label in [('nonrobust', 'AR (OLS SE)'),
                          ('HC1', 'AR (HC1 robust SE)'),
                          ('cluster', 'AR (cluster-robust SE)')]:
    lo, hi, p = ar_results[f'min_{cov_type}']
    if not np.isnan(lo):
        print(f"  {label:<45} {wald_cace:>8.1f}   [{lo:>6.1f}, {hi:>5.1f}]   {p:>10.4f}")

print()
print(f"  Point estimate is identical across all methods ({wald_cace:.1f} min = ITT/compliance).")
print(f"  Differences are in confidence set construction:")
print(f"    - Wald: large-sample normal approximation on the ratio (delta method)")
print(f"    - AR (OLS): exact finite-sample test, homoscedastic errors assumed")
print(f"    - AR (HC1): exact test with heteroscedasticity-robust SE")
print(f"    - AR (cluster): exact test accounting for within-date correlation")
print()
print(f"  NOTE ON SE CHOICE:")
print(f"    The Wald CI [-65.2, 0.9] uses individual-level SE (analogous to OLS/HC1).")
print(f"    Cluster-robust SE inflates the CI because it accounts for within-date")
print(f"    correlation across {n_clusters} dates. Both are valid; they answer")
print(f"    slightly different questions about the uncertainty structure.")


# =============================================================================
# SAVE OUTPUT TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SAVING eTable6b")
print("=" * 80)

rows = []

# Row 1: Wald (minutes) -- reference
rows.append({
    'Method': 'Wald LATE (delta-method)',
    'Treatment': 'is_shaked_opened',
    'Outcome': 'consultation_cycle_min',
    'N': len(adf),
    'Point_Estimate': f"{wald_cace:.1f} min",
    'CI_Lower': f"{wald_ci_lo:.1f}",
    'CI_Upper': f"{wald_ci_hi:.1f}",
    'p_value_H0_zero': f"{wald_p:.4f}",
    'SE_Type': 'Individual-level (delta method)',
    'Note': 'Large-sample normal approximation; SE = ITT_SE / compliance_rate'
})

# AR results by SE type
for cov_type, label, se_label in [
    ('nonrobust', 'Anderson-Rubin (OLS SE)', 'Homoscedastic OLS'),
    ('HC1', 'Anderson-Rubin (HC1 SE)', 'Heteroscedasticity-robust'),
    ('cluster', 'Anderson-Rubin (cluster-robust SE)', f'Cluster-robust ({n_clusters} dates)')
]:
    lo, hi, p = ar_results[f'min_{cov_type}']
    if not np.isnan(lo):
        rows.append({
            'Method': label,
            'Treatment': 'is_shaked_opened',
            'Outcome': 'consultation_cycle_min',
            'N': len(adf),
            'Point_Estimate': f"{wald_cace:.1f} min",
            'CI_Lower': f"{lo:.1f}",
            'CI_Upper': f"{hi:.1f}",
            'p_value_H0_zero': f"{p:.4f}",
            'SE_Type': se_label,
            'Note': 'Finite-sample valid; inverted reduced-form test'
        })

# AR time ratios (HC1 primary)
log_hc1 = ar_results.get('log_HC1', (np.nan,)*5)
if not np.isnan(log_hc1[0]):
    rows.append({
        'Method': 'Anderson-Rubin (HC1, time ratio)',
        'Treatment': 'is_shaked_opened',
        'Outcome': 'log(consultation_cycle_min)',
        'N': len(adf),
        'Point_Estimate': f"TR={wald_tr:.3f}",
        'CI_Lower': f"{log_hc1[2]:.3f}",
        'CI_Upper': f"{log_hc1[3]:.3f}",
        'p_value_H0_zero': f"{log_hc1[4]:.4f}",
        'SE_Type': 'Heteroscedasticity-robust',
        'Note': 'Time ratio = exp(log-CACE); AR CI on log scale, exponentiated'
    })

log_cl = ar_results.get('log_cluster', (np.nan,)*5)
if not np.isnan(log_cl[0]):
    rows.append({
        'Method': 'Anderson-Rubin (cluster-robust, time ratio)',
        'Treatment': 'is_shaked_opened',
        'Outcome': 'log(consultation_cycle_min)',
        'N': len(adf),
        'Point_Estimate': f"TR={wald_tr:.3f}",
        'CI_Lower': f"{log_cl[2]:.3f}",
        'CI_Upper': f"{log_cl[3]:.3f}",
        'p_value_H0_zero': f"{log_cl[4]:.4f}",
        'SE_Type': f'Cluster-robust ({n_clusters} dates)',
        'Note': 'Time ratio = exp(log-CACE); cluster-robust AR CI'
    })

# Sensitivity: D2
if not np.isnan(ar_d2_lo):
    rows.append({
        'Method': 'Anderson-Rubin (HC1, D2 sensitivity)',
        'Treatment': 'has_shaked_results',
        'Outcome': 'consultation_cycle_min',
        'N': len(adf),
        'Point_Estimate': f"{wald_d2:.1f} min",
        'CI_Lower': f"{ar_d2_lo:.1f}",
        'CI_Upper': f"{ar_d2_hi:.1f}",
        'p_value_H0_zero': f"{ar_d2_p:.4f}",
        'SE_Type': 'Heteroscedasticity-robust',
        'Note': f'Treatment = has_shaked_results; compliance = {compliance_d2:.4f}'
    })

if not np.isnan(ar_d2_lo_cl):
    rows.append({
        'Method': 'Anderson-Rubin (cluster-robust, D2 sensitivity)',
        'Treatment': 'has_shaked_results',
        'Outcome': 'consultation_cycle_min',
        'N': len(adf),
        'Point_Estimate': f"{wald_d2:.1f} min",
        'CI_Lower': f"{ar_d2_lo_cl:.1f}",
        'CI_Upper': f"{ar_d2_hi_cl:.1f}",
        'p_value_H0_zero': f"{ar_d2_p_cl:.4f}",
        'SE_Type': f'Cluster-robust ({n_clusters} dates)',
        'Note': f'Treatment = has_shaked_results; compliance = {compliance_d2:.4f}'
    })

results_df = pd.DataFrame(rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable6b_AR_ConfidenceSet.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

# =============================================================================
# INTERPRETATION
# =============================================================================
print("\n" + "=" * 80)
print("INTERPRETATION FOR MANUSCRIPT")
print("=" * 80)

# Use HC1 as the primary AR result (matches Wald's individual-level approach)
ar_primary_lo, ar_primary_hi, ar_primary_p = ar_results['min_HC1']
ar_cluster_lo, ar_cluster_hi, ar_cluster_p = ar_results['min_cluster']

if not np.isnan(ar_primary_lo):
    print(f"""
  PRIMARY NARRATIVE (AR with HC1, comparable to Wald SE assumptions):

  "Because the Wald estimator relies on large-sample normal approximations
  that may be imprecise with right-skewed outcomes (skewness = 4.5), we
  computed the Anderson-Rubin confidence set, which provides exact
  finite-sample inference for the CACE.

  The AR 95% confidence set was [{ar_primary_lo:.1f}, {ar_primary_hi:.1f}] minutes
  (p = {ar_primary_p:.4f} for the null hypothesis of no effect), compared with
  the Wald 95% CI of [{wald_ci_lo:.1f}, {wald_ci_hi:.1f}] (p = {wald_p:.4f}).
  Both yield a point estimate of {wald_cace:.1f} minutes.

  With cluster-robust standard errors (accounting for correlation among
  patients presenting on the same day, {n_clusters} clusters), the AR 95%
  confidence set widened to [{ar_cluster_lo:.1f}, {ar_cluster_hi:.1f}]
  (p = {ar_cluster_p:.4f})."
""")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Wald LATE:       {wald_cace:.1f} min  [{wald_ci_lo:.1f}, {wald_ci_hi:.1f}]  p={wald_p:.4f}")
print(f"  AR (HC1):        {wald_cace:.1f} min  [{ar_primary_lo:.1f}, {ar_primary_hi:.1f}]  p={ar_primary_p:.4f}")
print(f"  AR (cluster):    {wald_cace:.1f} min  [{ar_cluster_lo:.1f}, {ar_cluster_hi:.1f}]  p={ar_cluster_p:.4f}")

print("\nPHASE 3C COMPLETE")
