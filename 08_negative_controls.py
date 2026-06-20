"""
Stage 08 -- Negative-control outcomes (IV falsification).

Tests the exclusion restriction: wing assignment should not affect outcomes
SHAKED cannot plausibly influence (time to first physician, time to decision,
length of stay), plus selection and technical-failure checks.

Reproduces: Table 3 / eTable 6 falsification rows.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable7_Negative_Controls.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 3b: NEGATIVE CONTROL OUTCOMES (IV FALSIFICATION)")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

def hl_estimate(x, y):
    """Hodges-Lehmann shift estimate."""
    diffs = np.subtract.outer(y, x).ravel()
    return np.median(diffs)

def analyze_outcome(data, outcome_col, label, group_col, group_treat, group_ctrl):
    """Mann-Whitney + Hodges-Lehmann for a negative control outcome."""
    treat = data.loc[data[group_col] == group_treat, outcome_col].dropna()
    ctrl  = data.loc[data[group_col] == group_ctrl, outcome_col].dropna()

    stat, p = mannwhitneyu(ctrl, treat)
    hl = hl_estimate(ctrl.values, treat.values)

    print(f"\n  {label}")
    print(f"    {group_treat}: n={len(treat)}, median={treat.median():.1f}")
    print(f"    {group_ctrl}: n={len(ctrl)}, median={ctrl.median():.1f}")
    print(f"    HL shift: {hl:.1f} min, MWU p={p:.4f}")

    return {
        'Outcome': label,
        'Comparison': f"{group_treat} vs {group_ctrl}",
        'N_treat': len(treat), 'N_ctrl': len(ctrl),
        'Median_treat': f"{treat.median():.1f}",
        'Median_ctrl': f"{ctrl.median():.1f}",
        'HL_shift': f"{hl:.1f}",
        'p_value': f"{p:.4f}",
        'Interpretation': 'Supports exclusion restriction' if p > 0.10 else 'POTENTIAL CONCERN'
    }

# ============================================================================
# 1. ITT-LEVEL NEGATIVE CONTROLS (Wing A vs Wing B)
# ============================================================================
print("\n--- ITT-Level: Wing A vs Wing B (instrument-level) ---")

for outcome, label in [
    ('tfp_min', 'Time to First Physician (TFP)'),
    ('time_to_decision_min', 'Time to Physician Decision'),
    ('los_hours', 'Length of Stay (hours)'),
]:
    row = analyze_outcome(df, outcome, label, 'wing', 'A', 'B')
    row['Level'] = 'ITT (Wing A vs B)'
    results_rows.append(row)

# ============================================================================
# 2. SELECTION BIAS CHECK: Non-Users vs Wing B
# ============================================================================
print("\n--- Selection Bias Check: Wing A Non-Users vs Wing B ---")

non_users = df[(df['wing'] == 'A') & (df['is_user'] == 0)].copy()
non_users['group'] = 'NonUser_A'
wing_b = df[df['wing'] == 'B'].copy()
wing_b['group'] = 'WingB'
combined = pd.concat([non_users, wing_b])

for outcome, label in [
    ('tfp_min', 'TFP: Non-Users(A) vs Wing B'),
    ('time_to_decision_min', 'Decision: Non-Users(A) vs Wing B'),
    ('consultation_cycle_min', 'Consult Cycle: Non-Users(A) vs Wing B'),
]:
    row = analyze_outcome(combined, outcome, label, 'group', 'NonUser_A', 'WingB')
    row['Level'] = 'Selection bias check'
    results_rows.append(row)

# ============================================================================
# 3. TECH-FAILURE NEGATIVE CONTROL
# ============================================================================
print("\n--- Tech-Failure Negative Control: TechFailure vs Wing B ---")

tech_fail = df[df['engagement_level'] == '3_TechFailure'].copy()
tech_fail['group'] = 'TechFail'
wing_b2 = df[df['wing'] == 'B'].copy()
wing_b2['group'] = 'WingB'
combined_tf = pd.concat([tech_fail, wing_b2])

for outcome, label in [
    ('consultation_cycle_min', 'Consult Cycle: TechFailure vs Wing B'),
    ('tfp_min', 'TFP: TechFailure vs Wing B'),
]:
    row = analyze_outcome(combined_tf, outcome, label, 'group', 'TechFail', 'WingB')
    row['Level'] = 'Tech-failure negative control'
    results_rows.append(row)

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable7")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable7_Negative_Controls.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

# Summary
print("\n" + "=" * 80)
print("PHASE 3b SUMMARY")
print("=" * 80)
for _, row in results_df.iterrows():
    print(f"  {row['Outcome']:<45} HL={row['HL_shift']:>6}, p={row['p_value']:<8} {row['Interpretation']}")

print(f"""
Manuscript text:
  "Wing assignment was not associated with time to first physician
  (HL shift: {results_df.iloc[0]['HL_shift']} min, p={results_df.iloc[0]['p_value']}),
  time to physician decision (p={results_df.iloc[1]['p_value']}), or length of stay
  (p={results_df.iloc[2]['p_value']}), supporting the exclusion restriction for
  instrumental variable analysis."
""")
