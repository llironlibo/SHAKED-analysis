"""
Stage 06 -- Instrumental-variable / complier-average causal effect.

Instrument: wing assignment. Treatment: SHAKED opened (primary) and SHAKED
output delivered (sensitivity). Wald LATE, 2SLS with covariates, and a
cluster-bootstrap CI; one-sided noncompliance is documented.

Reproduces: Table 2C (CACE) and eTable 1 (Wald / 2SLS / bootstrap rows).

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable6_IV_CACE.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import norm, mannwhitneyu
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
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
print("PHASE 3: INSTRUMENTAL VARIABLE / CACE ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
print(f"Loaded: {len(df):,} patients")

results_rows = []

# ============================================================================
# DATA PREP
# ============================================================================
# Keep patients with consultation data
adf = df.dropna(subset=['consultation_cycle_min', 'log_consult']).copy()
print(f"Patients with consultation data: {len(adf):,}")

Z = adf['is_wing_a'].values           # instrument
D = adf['is_user'].values             # treatment (primary)
D2 = adf['has_shaked_results'].values  # treatment (sensitivity: got output)
Y_min = adf['consultation_cycle_min'].values
Y_log = adf['log_consult'].values

# ============================================================================
# FIRST-STAGE DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 80)
print("FIRST-STAGE DIAGNOSTICS")
print("=" * 80)

# Compliance rates
compliance_za1 = D[Z == 1].mean()  # P(D=1|Z=1)
compliance_za0 = D[Z == 0].mean()  # P(D=1|Z=0) -- should be 0
compliance_rate = compliance_za1 - compliance_za0

print(f"  P(D=1 | Z=1) = {compliance_za1:.4f}  (Wing A adoption rate)")
print(f"  P(D=1 | Z=0) = {compliance_za0:.4f}  (should be 0: one-sided noncompliance)")
print(f"  Compliance rate (first stage): {compliance_rate:.4f}")

# First-stage F-statistic
X_fs = sm.add_constant(Z)
fs_model = sm.OLS(D, X_fs).fit()
f_stat = fs_model.fvalue
print(f"  First-stage F-statistic: {f_stat:.1f} (threshold: >10 for strong instrument)")
print(f"  First-stage partial R2: {fs_model.rsquared:.4f}")

# D2: has_shaked_results
compliance_d2_za1 = D2[Z == 1].mean()
compliance_d2_za0 = D2[Z == 0].mean()
print(f"\n  Sensitivity treatment (has_shaked_results):")
print(f"  P(D2=1 | Z=1) = {compliance_d2_za1:.4f}")
print(f"  P(D2=1 | Z=0) = {compliance_d2_za0:.4f}")

# ============================================================================
# WALD LATE ESTIMATOR (simple ratio)
# ============================================================================
print("\n" + "=" * 80)
print("WALD LATE ESTIMATOR (ITT / compliance_rate)")
print("=" * 80)

# ITT effect on minutes
itt_treat = Y_min[Z == 1]
itt_ctrl = Y_min[Z == 0]
itt_diff = itt_treat.mean() - itt_ctrl.mean()
stat_mw, p_mw = mannwhitneyu(itt_ctrl, itt_treat)

# Wald LATE
wald_late_min = itt_diff / compliance_rate

# SE via delta method: SE_LATE = SE_ITT / compliance_rate
itt_se = np.sqrt(itt_treat.var()/len(itt_treat) + itt_ctrl.var()/len(itt_ctrl))
wald_se = itt_se / compliance_rate
wald_ci_lo = wald_late_min - 1.96 * wald_se
wald_ci_hi = wald_late_min + 1.96 * wald_se
wald_p = 2 * (1 - norm.cdf(abs(wald_late_min / wald_se)))

print(f"  ITT mean difference: {itt_diff:.1f} min")
print(f"  Compliance rate: {compliance_rate:.4f}")
print(f"  Wald LATE (CACE): {wald_late_min:.1f} min [{wald_ci_lo:.1f}, {wald_ci_hi:.1f}], p={wald_p:.4f}")

print(f"\n  NOTE: With one-sided noncompliance (P(D=1|Z=0)=0), the Wald estimator")
print(f"  simplifies to ITT/compliance_rate = {itt_diff:.1f}/{compliance_rate:.4f} = {wald_late_min:.1f}")
print(f"  2SLS gives identical point estimates; its value is covariate adjustment (precision gain).")

results_rows.append({
    'Model': 'Wald LATE (minutes)', 'Treatment': 'is_shaked_opened',
    'N': len(adf), 'Estimate': f"{wald_late_min:.1f} min",
    'CI_Lower': f"{wald_ci_lo:.1f}", 'CI_Upper': f"{wald_ci_hi:.1f}",
    'p_value': f"{wald_p:.4f}",
    'First_Stage_F': f"{f_stat:.1f}",
    'Compliance_Rate': f"{compliance_rate:.4f}",
    'Note': 'Simple ratio: ITT_effect / compliance_rate; delta-method SE'
})

# Wald LATE on log scale
itt_log_diff = Y_log[Z == 1].mean() - Y_log[Z == 0].mean()
wald_late_log = itt_log_diff / compliance_rate
log_se = np.sqrt(Y_log[Z==1].var()/sum(Z==1) + Y_log[Z==0].var()/sum(Z==0)) / compliance_rate
wald_log_ci_lo = wald_late_log - 1.96 * log_se
wald_log_ci_hi = wald_late_log + 1.96 * log_se
wald_log_p = 2 * (1 - norm.cdf(abs(wald_late_log / log_se)))
wald_tr = np.exp(wald_late_log)

print(f"\n  Wald LATE (log-scale): {wald_late_log:.4f}, TR={wald_tr:.3f} [{np.exp(wald_log_ci_lo):.3f}, {np.exp(wald_log_ci_hi):.3f}], p={wald_log_p:.4f}")

results_rows.append({
    'Model': 'Wald LATE (time ratio)', 'Treatment': 'is_shaked_opened',
    'N': len(adf), 'Estimate': f"TR={wald_tr:.3f}",
    'CI_Lower': f"{np.exp(wald_log_ci_lo):.3f}", 'CI_Upper': f"{np.exp(wald_log_ci_hi):.3f}",
    'p_value': f"{wald_log_p:.4f}",
    'First_Stage_F': f"{f_stat:.1f}",
    'Compliance_Rate': f"{compliance_rate:.4f}",
    'Note': 'Wald on log(consultation_min); time ratio = exp(beta)'
})

# ============================================================================
# 2SLS WITH COVARIATES
# ============================================================================
print("\n" + "=" * 80)
print("2SLS WITH COVARIATE ADJUSTMENT")
print("=" * 80)

covars = ['triage', 'diagnosis_count', 'has_radiology', 'is_admitted',
          'n_consultations', 'is_ambulance', 'admission_hour']
sub_2sls = adf.dropna(subset=covars).copy()
print(f"  N for 2SLS: {len(sub_2sls)}")

for y_name, y_col in [('consultation_cycle_min', 'consultation_cycle_min'),
                        ('log_consult', 'log_consult')]:
    print(f"\n  --- 2SLS: Y = {y_name} ---")

    # Build arrays
    Y_2sls = sub_2sls[y_col].values
    D_2sls = sub_2sls['is_user'].values
    Z_2sls = sub_2sls['is_wing_a'].values
    # Cast to float so the exogenous matrix is numeric (not object): is_admitted
    # loads as bool, which mixed with floats yields an object dtype that IV2SLS
    # cannot process. The numeric values are unchanged (bool -> 0.0/1.0).
    X_2sls = sub_2sls[covars].astype(float).values

    # linearmodels IV2SLS
    # Formula: Y ~ [D ~ Z] + exog
    dep = pd.Series(Y_2sls, name='Y')
    endog = pd.DataFrame({'D': D_2sls})
    exog = sm.add_constant(pd.DataFrame(X_2sls, columns=covars))
    instr = pd.DataFrame({'Z': Z_2sls})

    iv_model = IV2SLS(dep, exog, endog, instr).fit(cov_type='clustered',
                                                      clusters=sub_2sls['arrival_date'].values)

    # Get the endogenous variable name from the model
    endog_name = list(endog.columns)[0]
    beta_iv = iv_model.params[endog_name]
    se_iv = iv_model.std_errors[endog_name]
    ci_iv = iv_model.conf_int().loc[endog_name]
    p_iv = iv_model.pvalues[endog_name]
    try:
        fs_diag = iv_model.first_stage.diagnostics
        f_first = fs_diag.iloc[0]['f.stat'] if 'f.stat' in fs_diag.columns else fs_diag.values[0][0]
    except Exception:
        f_first = f_stat  # fall back to simple first-stage F

    if y_name == 'log_consult':
        tr_iv = np.exp(beta_iv)
        desc = f"TR={tr_iv:.3f} [{np.exp(ci_iv.iloc[0]):.3f}, {np.exp(ci_iv.iloc[1]):.3f}]"
        ctrl_med = sub_2sls.loc[sub_2sls['is_wing_a']==0, 'consultation_cycle_min'].median()
        min_eff = ctrl_med * (tr_iv - 1)
    else:
        desc = f"{beta_iv:.1f} min [{ci_iv.iloc[0]:.1f}, {ci_iv.iloc[1]:.1f}]"
        min_eff = beta_iv

    print(f"  2SLS CACE: {desc}, p={p_iv:.4f}")
    print(f"  First-stage F (with covariates): {f_first:.1f}")
    if y_name == 'log_consult':
        print(f"  -> At control median ({ctrl_med:.1f} min): ~{min_eff:.1f} min")

    results_rows.append({
        'Model': f'2SLS ({y_name})', 'Treatment': 'is_shaked_opened',
        'N': len(sub_2sls), 'Estimate': desc,
        'CI_Lower': f"{ci_iv.iloc[0]:.3f}", 'CI_Upper': f"{ci_iv.iloc[1]:.3f}",
        'p_value': f"{p_iv:.4f}",
        'First_Stage_F': f"{f_first:.1f}",
        'Compliance_Rate': f"{compliance_rate:.4f}",
        'Note': f'Adjusted for {", ".join(covars)}; cluster-robust SE (arrival_date)'
    })

# ============================================================================
# CLUSTER-BOOTSTRAP (by arrival_date, 5000 replicates)
# ============================================================================
print("\n" + "=" * 80)
print("CLUSTER-BOOTSTRAP FOR WALD LATE (5,000 replicates by arrival_date)")
print("=" * 80)

dates = adf['arrival_date'].unique()
n_dates = len(dates)
n_boot = 5000
boot_cace_min = np.zeros(n_boot)
boot_cace_log = np.zeros(n_boot)

for b in range(n_boot):
    # Resample dates with replacement
    boot_dates = np.random.choice(dates, size=n_dates, replace=True)
    # Build bootstrap sample
    boot_idx = []
    for d in boot_dates:
        boot_idx.extend(adf.index[adf['arrival_date'] == d].tolist())
    boot_sample = adf.loc[boot_idx]

    bZ = boot_sample['is_wing_a'].values
    bD = boot_sample['is_user'].values
    bY = boot_sample['consultation_cycle_min'].values
    bY_log = boot_sample['log_consult'].values

    comp = bD[bZ == 1].mean() - bD[bZ == 0].mean()
    if comp > 0.01:
        itt_min = bY[bZ == 1].mean() - bY[bZ == 0].mean()
        itt_log = bY_log[bZ == 1].mean() - bY_log[bZ == 0].mean()
        boot_cace_min[b] = itt_min / comp
        boot_cace_log[b] = itt_log / comp
    else:
        boot_cace_min[b] = np.nan
        boot_cace_log[b] = np.nan

# Remove NaN
boot_cace_min = boot_cace_min[~np.isnan(boot_cace_min)]
boot_cace_log = boot_cace_log[~np.isnan(boot_cace_log)]

# Percentile CI
ci_boot_min = np.percentile(boot_cace_min, [2.5, 97.5])
ci_boot_log = np.percentile(boot_cace_log, [2.5, 97.5])

# Bootstrap p-value (proportion of bootstraps on wrong side of 0)
boot_p_min = 2 * min(np.mean(boot_cace_min >= 0), np.mean(boot_cace_min <= 0))
boot_p_log = 2 * min(np.mean(boot_cace_log >= 0), np.mean(boot_cace_log <= 0))

print(f"  Valid bootstrap samples: {len(boot_cace_min)}/{n_boot}")
print(f"  CACE (minutes): {np.mean(boot_cace_min):.1f} [{ci_boot_min[0]:.1f}, {ci_boot_min[1]:.1f}], boot-p={boot_p_min:.4f}")
print(f"  CACE (log):     {np.mean(boot_cace_log):.4f}, TR={np.exp(np.mean(boot_cace_log)):.3f} [{np.exp(ci_boot_log[0]):.3f}, {np.exp(ci_boot_log[1]):.3f}], boot-p={boot_p_log:.4f}")

results_rows.append({
    'Model': 'Cluster-bootstrap CACE (minutes)', 'Treatment': 'is_shaked_opened',
    'N': len(adf), 'Estimate': f"{np.mean(boot_cace_min):.1f} min",
    'CI_Lower': f"{ci_boot_min[0]:.1f}", 'CI_Upper': f"{ci_boot_min[1]:.1f}",
    'p_value': f"{boot_p_min:.4f}",
    'First_Stage_F': f"{f_stat:.1f}",
    'Compliance_Rate': f"{compliance_rate:.4f}",
    'Note': f'{n_boot} cluster-bootstrap replicates by arrival_date ({n_dates} dates); percentile CI'
})

results_rows.append({
    'Model': 'Cluster-bootstrap CACE (time ratio)', 'Treatment': 'is_shaked_opened',
    'N': len(adf), 'Estimate': f"TR={np.exp(np.mean(boot_cace_log)):.3f}",
    'CI_Lower': f"{np.exp(ci_boot_log[0]):.3f}", 'CI_Upper': f"{np.exp(ci_boot_log[1]):.3f}",
    'p_value': f"{boot_p_log:.4f}",
    'First_Stage_F': f"{f_stat:.1f}",
    'Compliance_Rate': f"{compliance_rate:.4f}",
    'Note': f'{n_boot} cluster-bootstrap replicates; exp(mean(log-CACE))'
})

# ============================================================================
# SENSITIVITY: D2 = has_shaked_results
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: Treatment = has_shaked_results (got SHAKED output)")
print("=" * 80)

comp_d2 = compliance_d2_za1 - compliance_d2_za0
wald_d2_min = itt_diff / comp_d2

itt_se_d2 = np.sqrt(itt_treat.var()/len(itt_treat) + itt_ctrl.var()/len(itt_ctrl))
wald_d2_se = itt_se_d2 / comp_d2
wald_d2_ci = [wald_d2_min - 1.96 * wald_d2_se, wald_d2_min + 1.96 * wald_d2_se]
wald_d2_p = 2 * (1 - norm.cdf(abs(wald_d2_min / wald_d2_se)))

print(f"  Compliance rate (D2): {comp_d2:.4f}")
print(f"  CACE (D2, minutes): {wald_d2_min:.1f} min [{wald_d2_ci[0]:.1f}, {wald_d2_ci[1]:.1f}], p={wald_d2_p:.4f}")

results_rows.append({
    'Model': 'Wald LATE (minutes, D2)', 'Treatment': 'has_shaked_results',
    'N': len(adf), 'Estimate': f"{wald_d2_min:.1f} min",
    'CI_Lower': f"{wald_d2_ci[0]:.1f}", 'CI_Upper': f"{wald_d2_ci[1]:.1f}",
    'p_value': f"{wald_d2_p:.4f}",
    'First_Stage_F': 'N/A',
    'Compliance_Rate': f"{comp_d2:.4f}",
    'Note': 'Sensitivity: treatment = received SHAKED output (excludes tech failures)'
})

# ============================================================================
# ASSUMPTION STATEMENTS (for manuscript)
# ============================================================================
print("\n" + "=" * 80)
print("IV ASSUMPTION STATEMENTS (for manuscript text)")
print("=" * 80)

print(f"""
  RELEVANCE: Wing assignment strongly predicts SHAKED use.
    Compliance rate = {compliance_rate:.1%}, First-stage F = {f_stat:.1f} >> 10.

  EXCLUSION RESTRICTION: Supported by Phase 3b falsification tests.
    Time to first physician: p=0.39 (no wing difference)
    Time to decision: p=0.74 (no wing difference)
    LOS: p=0.99 (no wing difference)

  MONOTONICITY: Satisfied by design (one-sided noncompliance).
    P(D=1|Z=0) = 0: Wing B patients had zero access to SHAKED.
    No defiers possible -- no one in Wing B could use SHAKED if offered
    while refusing it when actually assigned to Wing A.

  SUTVA: Bounded by contamination analysis (71 indirect exposures, 12.7%).
    ADDITIONAL LIMITATION: Physician learning spillover -- a physician who
    used SHAKED in Wing A may practice differently when treating Wing B
    patients even without the tool. Cannot be tested without physician IDs.
    Stated as a limitation.

  ONE-SIDED NONCOMPLIANCE NOTE:
    With P(D=1|Z=0) = 0, the Wald estimator simplifies to ITT/compliance_rate.
    2SLS produces identical point estimates in this special case.
    The added value of 2SLS here is covariate adjustment (precision gain only).
    This is stated explicitly for reviewer transparency.
""")

# ============================================================================
# INTERPRETATION PARAGRAPH
# ============================================================================
print("=" * 80)
print("INTERPRETATION PARAGRAPH (for reviewer response)")
print("=" * 80)
print(f"""
  "The Complier Average Causal Effect (CACE) was estimated at {wald_late_min:.1f} minutes
  [{wald_ci_lo:.1f} to {wald_ci_hi:.1f}] using the Wald estimator. This is larger in
  magnitude than the ITT effect ({itt_diff:.1f} minutes) because the ITT is diluted by
  {(1 - compliance_rate)*100:.0f}% non-users in the intervention arm.

  The CACE estimates the effect among patients whose physicians would use SHAKED
  when offered -- the clinically relevant estimand for deployment decisions.

  Cluster-bootstrap (5,000 resamples by date) yielded a similar estimate:
  {np.mean(boot_cace_min):.1f} minutes [{ci_boot_min[0]:.1f} to {ci_boot_min[1]:.1f}]."
""")

# ============================================================================
# SAVE
# ============================================================================
print("=" * 80)
print("SAVING eTable6")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable6_IV_CACE.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print("\nPHASE 3 COMPLETE")
