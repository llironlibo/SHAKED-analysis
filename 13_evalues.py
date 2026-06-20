"""
Stage 13 -- E-values for unmeasured confounding.

Quantifies the minimum strength of an unmeasured confounder needed to explain
away the observed associations: primary from the responder risk ratio, with
Cox-HR and adjusted-time-ratio sensitivities.

Reproduces: eTable 11 (E-values; Round-2 addition).

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable11_EValues.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 8: E-VALUES FOR UNMEASURED CONFOUNDING")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

def compute_evalue(rr):
    """Compute E-value from a risk ratio (RR >= 1)."""
    if rr < 1:
        rr = 1 / rr  # flip so RR > 1
    return rr + np.sqrt(rr * (rr - 1))

def compute_evalue_ci(rr_ci_bound):
    """E-value for the CI bound closest to null."""
    if rr_ci_bound < 1:
        rr_ci_bound = 1 / rr_ci_bound
    if rr_ci_bound <= 1:
        return 1.0  # CI includes null
    return rr_ci_bound + np.sqrt(rr_ci_bound * (rr_ci_bound - 1))

# ============================================================================
# PP subset
# ============================================================================
sub = df[(df['is_user'] == 1) | (df['wing'] == 'B')].dropna(subset=['consultation_cycle_min']).copy()

# ============================================================================
# 1. PRIMARY: E-value from responder binary RR
# ============================================================================
print("\n" + "=" * 80)
print("PRIMARY: E-VALUES FROM RESPONDER BINARY RR")
print("=" * 80)
print("(Clean and assumption-free -- consultation completion is not rare)")

for threshold, label in [(120, '<120 min'), (180, '<180 min')]:
    col = f'consult_le_{threshold}'
    users = sub[sub['is_user'] == 1][col]
    ctrls = sub[sub['is_user'] == 0][col]

    rr = users.mean() / ctrls.mean() if ctrls.mean() > 0 else np.nan

    # Bootstrap CI for RR
    n_boot = 5000
    boot_rr = np.zeros(n_boot)
    for b in range(n_boot):
        bu = np.random.choice(users.values, size=len(users), replace=True)
        bc = np.random.choice(ctrls.values, size=len(ctrls), replace=True)
        if bc.mean() > 0:
            boot_rr[b] = bu.mean() / bc.mean()
        else:
            boot_rr[b] = np.nan
    boot_rr = boot_rr[~np.isnan(boot_rr)]
    ci = np.percentile(boot_rr, [2.5, 97.5])

    # E-value
    ev_point = compute_evalue(rr)
    # CI bound closest to null
    ci_near_null = min(ci, key=lambda x: abs(x - 1))
    ev_ci = compute_evalue_ci(ci_near_null)

    print(f"\n  Responder {label}:")
    print(f"    Users: {users.mean():.1%}, Controls: {ctrls.mean():.1%}")
    print(f"    RR = {rr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"    E-value (point): {ev_point:.2f}")
    print(f"    E-value (CI bound): {ev_ci:.2f}")

    results_rows.append({
        'Source': f'Responder RR ({label})', 'Priority': 'PRIMARY',
        'Estimate': f"RR={rr:.3f}", 'CI': f"[{ci[0]:.3f}, {ci[1]:.3f}]",
        'E_value_point': f"{ev_point:.2f}",
        'E_value_CI': f"{ev_ci:.2f}",
        'Note': 'Binary outcome; no rare-disease approximation needed'
    })

# ============================================================================
# 2. SENSITIVITY: E-value from Cox HR (demoted)
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: E-VALUES FROM COX HR (demoted)")
print("=" * 80)
print("Note: Consultation completion is universal (not rare), so HR-to-RR")
print("approximation is poor. Included for completeness only.")

# Use Cox HR from Phase 4: HR=1.123 [0.963, 1.309]
# (hardcoded from Phase 4 output; in production, read from eTable8)
cox_hr = 1.123
cox_ci_lo = 0.963
cox_ci_hi = 1.309

ev_cox = compute_evalue(cox_hr)
ev_cox_ci = compute_evalue_ci(cox_ci_lo)  # lower bound closest to null

print(f"  Cox HR: {cox_hr:.3f} [{cox_ci_lo:.3f}, {cox_ci_hi:.3f}]")
print(f"  E-value (point): {ev_cox:.2f}")
print(f"  E-value (CI bound): {ev_cox_ci:.2f}")

results_rows.append({
    'Source': 'Cox PH HR', 'Priority': 'SENSITIVITY',
    'Estimate': f"HR={cox_hr:.3f}", 'CI': f"[{cox_ci_lo:.3f}, {cox_ci_hi:.3f}]",
    'E_value_point': f"{ev_cox:.2f}",
    'E_value_CI': f"{ev_cox_ci:.2f}",
    'Note': 'Demoted: HR-to-RR approximation unreliable (non-rare outcome)'
})

# ============================================================================
# 3. SENSITIVITY: E-value from adjusted time ratio (Phase 1)
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY: E-VALUES FROM ADJUSTED TIME RATIO")
print("=" * 80)

# PP adjusted TR from Phase 1: TR=0.984 [0.848, 1.142]
adj_tr = 0.984
adj_ci_lo = 0.848
adj_ci_hi = 1.142

# Convert TR to approximate RR: if TR<1, completion is faster
# Approximate: RR ~ 1/TR (completion rate ratio)
approx_rr = 1 / adj_tr
ev_adj = compute_evalue(approx_rr)
ev_adj_ci = compute_evalue_ci(1 / adj_ci_hi)  # CI bound closest to null when TR CI crosses 1

print(f"  Adjusted TR: {adj_tr:.3f} [{adj_ci_lo:.3f}, {adj_ci_hi:.3f}]")
print(f"  Approximate RR: {approx_rr:.3f}")
print(f"  E-value (point): {ev_adj:.2f}")
print(f"  E-value (CI bound): {ev_adj_ci:.2f}")

results_rows.append({
    'Source': 'Adjusted time ratio', 'Priority': 'SENSITIVITY',
    'Estimate': f"TR={adj_tr:.3f} (RR~{approx_rr:.3f})", 'CI': f"[{adj_ci_lo:.3f}, {adj_ci_hi:.3f}]",
    'E_value_point': f"{ev_adj:.2f}",
    'E_value_CI': f"{ev_adj_ci:.2f}",
    'Note': 'Approximate RR from time ratio inversion'
})

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable11")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable11_EValues.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print(f"""
PHASE 8 COMPLETE

Interpretation:
  "An unmeasured confounder would need to be associated with both SHAKED use
  and consultation cycle time by a risk ratio of at least [E-value] to explain
  away the observed association, above and beyond the measured covariates."
""")
