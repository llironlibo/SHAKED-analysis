"""
Stage 02 -- Baseline characteristics and primary intention-to-treat analysis.

Computes the published Table 1 (baseline characteristics by wing, including the
P1-P5 triage rows) and the primary ITT subgroup Hodges-Lehmann estimates.

Reproduces:
  - Table 1 (Baseline Characteristics, ITT panel; P5 triage cells folded in)
  - Primary ITT subgroup HL effects (radiology / non-radiology / admitted /
    discharged), reported in the appendix subgroup table.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/Table1_Baseline_Characteristics_ITT.csv
         results/eTable_ITT_Subgroups_HL.csv

The leading cohort columns carry the original Hebrew electronic-health-record
field names; a few are addressed here (by position, plus the triage field by
name) to derive English analysis variables.
"""

import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, norm
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================
# Repo root resolved from this file's location so the script runs unchanged on
# any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("SHAKED STAGE 02: BASELINE + PRIMARY ITT")
print("=" * 80)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded unified cohort: {len(df):,} patients")

# age_years comes from the leading Hebrew age column (addressed by position);
# is_female and the numeric triage level (including P5 -> 5) are precomputed
# English columns produced by stage 01.
cols = df.columns.tolist()
age_col = cols[2]

df['age_years'] = pd.to_numeric(df[age_col], errors='coerce')
df['is_female'] = df['is_female'].astype(bool)
df['triage_level'] = df['triage_acuity_numeric']

df['is_shaked_opened'] = df['is_shaked_opened'].astype(bool)
df['has_radiology_consult'] = df['has_radiology_consult'].fillna(False).astype(bool)
df['is_admitted'] = df['is_admitted'].fillna(False).astype(bool)
df['is_ambulance_arrival'] = df['is_ambulance_arrival'].fillna(False).astype(bool)

# Apply the consultation exclusion (same set as every other analysis).
df = df[df['n_consultations'] > 0].copy()
print(f"After consultation exclusions: {len(df)} patients")

df_wing_a = df[df['wing'] == 'A'].copy()
df_wing_b = df[df['wing'] == 'B'].copy()

print(f"Wing A (ITT): n = {len(df_wing_a)}")
print(f"Wing B (ITT): n = {len(df_wing_b)}")
print(f"SHAKED users (within Wing A): n = {df_wing_a['is_shaked_opened'].sum()}")

assert len(df_wing_a) == 584, f"Expected 584 Wing A patients, got {len(df_wing_a)}"
assert len(df_wing_b) == 554, f"Expected 554 Wing B patients, got {len(df_wing_b)}"
assert df_wing_a['is_shaked_opened'].sum() == 259, f"Expected 259 users, got {df_wing_a['is_shaked_opened'].sum()}"

print("\n[VALIDATION PASSED] Sample sizes correct")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_smd(group1, group2):
    """Standardized mean difference (Cohen's d), absolute value."""
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std == 0:
        return 0
    return abs((mean1 - mean2) / pooled_std)

def hodges_lehmann_ci_vectorized(x, y, alpha=0.05):
    """Hodges-Lehmann estimator and CI, consistent with the Mann-Whitney U test."""
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)

    diffs = (x[:, None] - y).flatten()
    diffs.sort()

    est = np.median(diffs)

    m_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_crit = norm.ppf(1 - alpha/2)
    k = int(z_crit * sigma_u)

    M = n1 * n2
    lower_rank = int(M/2 - k)
    upper_rank = int(M/2 + k) + 1

    lower_rank = max(0, lower_rank)
    upper_rank = min(M, upper_rank)

    ci_low = diffs[lower_rank]
    ci_high = diffs[upper_rank] if upper_rank < M else diffs[-1]

    return est, ci_low, ci_high

def format_mean_sd(series):
    """Format as mean +/- SD."""
    return f"{series.mean():.1f} +/- {series.std():.1f}"

def format_n_percent(series, total):
    """Format as n (%)."""
    if isinstance(series, pd.Series):
        count = series.sum()
    else:
        count = series
    return f"{count} ({count/total*100:.1f}%)"

def triage_cell(df_group, triage_val):
    """Count of patients at one numeric triage level (P1-P5) in one wing."""
    return int((df_group['triage_level'] == triage_val).sum())

# =============================================================================
# TABLE 1: ITT BASELINE CHARACTERISTICS (Wing A vs Wing B)
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 1: ITT BASELINE CHARACTERISTICS (Wing A vs Wing B)")
print("=" * 80)

table1_itt = []

# Age.
wing_a_age = df_wing_a['age_years'].dropna()
wing_b_age = df_wing_b['age_years'].dropna()
smd_age = calculate_smd(wing_a_age, wing_b_age)
_, p_age = mannwhitneyu(wing_a_age, wing_b_age, alternative='two-sided')

# Female sex.
wing_a_female = df_wing_a['is_female'].fillna(False)
wing_b_female = df_wing_b['is_female'].fillna(False)
pct_a_female = wing_a_female.sum() / len(wing_a_female)
pct_b_female = wing_b_female.sum() / len(wing_b_female)
smd_female = abs(pct_a_female - pct_b_female) / np.sqrt((pct_a_female*(1-pct_a_female) + pct_b_female*(1-pct_b_female))/2)
ct_female = pd.crosstab(
    df[df['wing'].isin(['A', 'B'])]['wing'],
    df[df['wing'].isin(['A', 'B'])]['is_female']
)
_, p_female, _, _ = chi2_contingency(ct_female)

# Triage acuity P1-P5, all from the numeric level (Fisher's exact test is used
# for the small P5 cell counts).
triage_levels = {'P1': 1.0, 'P2': 2.0, 'P3': 3.0, 'P4': 4.0, 'P5': 5.0}
triage_stats = {}
for triage_name, triage_val in triage_levels.items():
    a_count = triage_cell(df_wing_a, triage_val)
    b_count = triage_cell(df_wing_b, triage_val)
    p_a = a_count / len(df_wing_a)
    p_b = b_count / len(df_wing_b)
    pooled = np.sqrt((p_a*(1-p_a) + p_b*(1-p_b)) / 2)
    smd_triage = abs(p_a - p_b) / pooled if pooled > 0 else 0.0
    table = [[a_count, len(df_wing_a) - a_count], [b_count, len(df_wing_b) - b_count]]
    if a_count < 5 or b_count < 5:
        _, p_triage = fisher_exact(table)
    else:
        _, p_triage, _, _ = chi2_contingency(table, correction=False)
    triage_stats[triage_name] = {'wing_a_count': a_count, 'wing_b_count': b_count,
                                 'smd': smd_triage, 'p': p_triage}

# Radiology consultation.
wing_a_rad = df_wing_a['has_radiology_consult'].fillna(False).astype(int)
wing_b_rad = df_wing_b['has_radiology_consult'].fillna(False).astype(int)
smd_rad = calculate_smd(wing_a_rad.astype(float), wing_b_rad.astype(float))
_, p_rad, _, _ = chi2_contingency([[wing_a_rad.sum(), len(wing_a_rad)-wing_a_rad.sum()],
                                   [wing_b_rad.sum(), len(wing_b_rad)-wing_b_rad.sum()]])

# Admission.
wing_a_admit = df_wing_a['is_admitted'].fillna(False)
wing_b_admit = df_wing_b['is_admitted'].fillna(False)
smd_admit = calculate_smd(wing_a_admit.astype(float), wing_b_admit.astype(float))
_, p_admit, _, _ = chi2_contingency([[wing_a_admit.sum(), len(wing_a_admit)-wing_a_admit.sum()],
                                     [wing_b_admit.sum(), len(wing_b_admit)-wing_b_admit.sum()]])

# Diagnosis count.
wing_a_dx = df_wing_a['diagnosis_count'].dropna()
wing_b_dx = df_wing_b['diagnosis_count'].dropna()
smd_dx = calculate_smd(wing_a_dx, wing_b_dx)
_, p_dx = mannwhitneyu(wing_a_dx, wing_b_dx, alternative='two-sided')

# Build table rows.
table1_itt.append({'Variable': 'Age (years), mean +/- SD',
                   'Wing A (n=584)': format_mean_sd(wing_a_age),
                   'Wing B (n=554)': format_mean_sd(wing_b_age),
                   'SMD': f"{smd_age:.3f}", 'p-value': f"{p_age:.3f}"})

table1_itt.append({'Variable': 'Female sex, n (%)',
                   'Wing A (n=584)': format_n_percent(wing_a_female, len(df_wing_a)),
                   'Wing B (n=554)': format_n_percent(wing_b_female, len(df_wing_b)),
                   'SMD': f"{smd_female:.3f}", 'p-value': f"{p_female:.3f}"})

table1_itt.append({'Variable': 'Triage acuity, n (%)', 'Wing A (n=584)': '',
                   'Wing B (n=554)': '', 'SMD': '', 'p-value': ''})

label_map = {'P1': '  P1 (Immediate)', 'P2': '  P2 (Emergent)', 'P3': '  P3 (Urgent)',
             'P4': '  P4 (Less Urgent)', 'P5': '  P5 (Non-urgent)'}
for triage_name in ['P1', 'P2', 'P3', 'P4', 'P5']:
    stats = triage_stats[triage_name]
    table1_itt.append({
        'Variable': label_map[triage_name],
        'Wing A (n=584)': f"{stats['wing_a_count']} ({stats['wing_a_count']/len(df_wing_a)*100:.1f}%)",
        'Wing B (n=554)': f"{stats['wing_b_count']} ({stats['wing_b_count']/len(df_wing_b)*100:.1f}%)",
        'SMD': f"{stats['smd']:.3f}", 'p-value': f"{stats['p']:.3f}"})

table1_itt.append({'Variable': 'Any radiology consultation, n (%)',
                   'Wing A (n=584)': format_n_percent(wing_a_rad, len(df_wing_a)),
                   'Wing B (n=554)': format_n_percent(wing_b_rad, len(df_wing_b)),
                   'SMD': f"{smd_rad:.3f}", 'p-value': f"{p_rad:.3f}"})

table1_itt.append({'Variable': 'Admitted to hospital, n (%)',
                   'Wing A (n=584)': format_n_percent(wing_a_admit, len(df_wing_a)),
                   'Wing B (n=554)': format_n_percent(wing_b_admit, len(df_wing_b)),
                   'SMD': f"{smd_admit:.3f}", 'p-value': f"{p_admit:.3f}"})

table1_itt.append({'Variable': 'Diagnosis count, mean +/- SD',
                   'Wing A (n=584)': format_mean_sd(wing_a_dx),
                   'Wing B (n=554)': format_mean_sd(wing_b_dx),
                   'SMD': f"{smd_dx:.3f}", 'p-value': f"{p_dx:.3f}"})

df_table1_itt = pd.DataFrame(table1_itt)
df_table1_itt.to_csv(os.path.join(RESULTS_DIR, 'Table1_Baseline_Characteristics_ITT.csv'),
                     index=False, encoding='utf-8-sig')

print("\n--- ITT Baseline Characteristics ---")
print(df_table1_itt.to_string(index=False))

# Validation: Wing A radiology rate should be 72.1% (421/584 after exclusions).
wing_a_rad_rate = wing_a_rad.sum() / len(wing_a_rad) * 100
print(f"\n[VALIDATION] Wing A radiology rate: {wing_a_rad_rate:.1f}% ({wing_a_rad.sum()}/{len(wing_a_rad)})")
if abs(wing_a_rad_rate - 72.1) > 0.1:
    print(f"  WARNING: Expected 72.1%, got {wing_a_rad_rate:.1f}%")
else:
    print("  [OK] Matches canonical 72.1%")

# Report the P5 cells explicitly (small-count Fisher cells in the published table).
p5 = triage_stats['P5']
print(f"[INFO] P5 (Non-urgent): Wing A {p5['wing_a_count']} ({p5['wing_a_count']/len(df_wing_a)*100:.1f}%) "
      f"vs Wing B {p5['wing_b_count']} ({p5['wing_b_count']/len(df_wing_b)*100:.1f}%), "
      f"SMD={p5['smd']:.2f}, p={p5['p']:.3f}")

# =============================================================================
# PRIMARY ITT SUBGROUP HODGES-LEHMANN ESTIMATES
# =============================================================================
print("\n" + "=" * 80)
print("ITT SUBGROUP HODGES-LEHMANN ESTIMATES")
print("=" * 80)

subgroup_results = []

subgroups = [
    ('Radiology consultations', df[df['has_radiology_consult'] == True]),
    ('Non-radiology consultations', df[df['has_radiology_consult'] == False]),
    ('Admitted patients', df[df['is_admitted'] == True]),
    ('Discharged patients', df[df['is_admitted'] == False])
]

for subgroup_name, subgroup_df in subgroups:
    print(f"\n--- {subgroup_name} ---")

    wing_a_sub = subgroup_df[subgroup_df['wing'] == 'A']['consultation_cycle_min'].dropna()
    wing_b_sub = subgroup_df[subgroup_df['wing'] == 'B']['consultation_cycle_min'].dropna()

    n_a = len(wing_a_sub)
    n_b = len(wing_b_sub)

    if n_a < 5 or n_b < 5:
        print(f"  SKIPPED: insufficient sample (Wing A: {n_a}, Wing B: {n_b})")
        continue

    hl_est, ci_low, ci_high = hodges_lehmann_ci_vectorized(wing_a_sub, wing_b_sub)
    _, p_value = mannwhitneyu(wing_a_sub, wing_b_sub, alternative='two-sided')

    median_a = wing_a_sub.median()
    q1_a, q3_a = wing_a_sub.quantile(0.25), wing_a_sub.quantile(0.75)
    median_b = wing_b_sub.median()
    q1_b, q3_b = wing_b_sub.quantile(0.25), wing_b_sub.quantile(0.75)

    print(f"  Wing A: n={n_a}, Median={median_a:.1f} [IQR: {q1_a:.1f}-{q3_a:.1f}]")
    print(f"  Wing B: n={n_b}, Median={median_b:.1f} [IQR: {q1_b:.1f}-{q3_b:.1f}]")
    print(f"  HL estimate: {hl_est:.1f} min [95% CI: {ci_low:.1f}, {ci_high:.1f}]")
    print(f"  p-value: {p_value:.3f}")

    assert ci_low <= hl_est <= ci_high, f"CI does not contain estimate for {subgroup_name}"

    subgroup_results.append({
        'Subgroup': subgroup_name,
        'N (Wing A / Wing B)': f"{n_a} / {n_b}",
        'HL Estimate (min)': f"{hl_est:.1f}",
        '95% CI': f"[{ci_low:.1f}, {ci_high:.1f}]",
        'p-value': f"{p_value:.3f}",
        'Wing A Median [IQR]': f"{median_a:.1f} [{q1_a:.1f}-{q3_a:.1f}]",
        'Wing B Median [IQR]': f"{median_b:.1f} [{q1_b:.1f}-{q3_b:.1f}]"
    })

df_subgroups = pd.DataFrame(subgroup_results)
df_subgroups.to_csv(os.path.join(RESULTS_DIR, 'eTable_ITT_Subgroups_HL.csv'),
                    index=False, encoding='utf-8-sig')

print("\n--- ITT Subgroup Summary Table ---")
print(df_subgroups.to_string(index=False))

print("\n[NOTE] ITT subgroup effects are smaller than per-protocol estimates due to")
print("       compliance dilution (44.3% adoption in Wing A).")

print("\n" + "=" * 80)
print("STAGE 02 COMPLETE")
print("=" * 80)
