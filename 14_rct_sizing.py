"""
Stage 14 -- Sample-size estimation for a definitive trial.

Power calculations from the observed effect sizes and the date-level ICC for
three designs: individual-level RCT, cluster-RCT (by physician), and a
stepped-wedge design.

Reproduces: eTable 5 (future-RCT sample size).

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable12_FutureRCT_SampleSize.csv
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
print("PHASE 9: FUTURE RCT SAMPLE SIZE ESTIMATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

# ============================================================================
# EFFECT SIZE INPUTS
# ============================================================================
sub = df[(df['is_user'] == 1) | (df['wing'] == 'B')].dropna(subset=['consultation_cycle_min'])
users = sub[sub['is_user'] == 1]['consultation_cycle_min']
ctrls = sub[sub['is_user'] == 0]['consultation_cycle_min']

# Observed effects
mean_diff = users.mean() - ctrls.mean()
pooled_sd = np.sqrt((users.var() + ctrls.var()) / 2)
cohen_d = abs(mean_diff) / pooled_sd

# CACE from Phase 3: -32.2 min (Wald)
cace_min = 32.2
cace_d = cace_min / pooled_sd

# Log-scale effect
log_effect = np.log(users).mean() - np.log(ctrls).mean()
log_sd = np.sqrt((np.log(users).var() + np.log(ctrls).var()) / 2)
log_d = abs(log_effect) / log_sd

# ICC from Phase 5
icc_date = 0.0127  # date-level (lower bound on physician-level)
icc_literature_lo = 0.05
icc_literature_hi = 0.10

print(f"Observed PP effect size inputs:")
print(f"  Mean difference: {mean_diff:.1f} min")
print(f"  Pooled SD: {pooled_sd:.1f} min")
print(f"  Cohen's d (PP): {cohen_d:.3f}")
print(f"  CACE effect: {cace_min:.1f} min -> d = {cace_d:.3f}")
print(f"  Date-level ICC: {icc_date:.4f}")
print(f"  Literature physician-level ICC range: {icc_literature_lo}-{icc_literature_hi}")
print(f"\n  NOTE: Date-level ICC of {icc_date:.4f} provides a LOWER BOUND on")
print(f"  physician-level ICC, since dates pool multiple physicians.")
print(f"  Physician-level ICC is expected to be higher, yielding a more")
print(f"  conservative (larger) sample size estimate.")

# ============================================================================
# INDIVIDUAL-LEVEL RCT
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 1: INDIVIDUAL-LEVEL RCT")
print("=" * 80)

def sample_size_2group(effect, sd, alpha=0.05, power_vals=[0.80, 0.90]):
    """Two-group sample size (per arm) for difference in means."""
    results = []
    for power in power_vals:
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        n = ((z_alpha + z_beta)**2 * 2 * sd**2) / effect**2
        results.append((power, int(np.ceil(n))))
    return results

# Using CACE as effect size (primary for deployment-relevant design)
print(f"\n  Using CACE effect ({cace_min:.1f} min, SD={pooled_sd:.1f}):")
for power, n in sample_size_2group(cace_min, pooled_sd):
    total = 2 * n
    print(f"    Power={power:.0%}: n={n}/arm, total={total}")
    results_rows.append({
        'Design': 'Individual RCT', 'Effect_Size': f"CACE={cace_min:.1f} min",
        'Power': f"{power:.0%}", 'N_per_arm': n, 'N_total': total,
        'Clusters': '', 'Cluster_Size': '', 'ICC': '',
        'Note': f'Based on observed CACE; pooled SD={pooled_sd:.1f}'
    })

# Using adjusted PP effect (conservative)
adj_effect = abs(mean_diff)  # ~15.5 min
print(f"\n  Using PP mean difference ({adj_effect:.1f} min, SD={pooled_sd:.1f}):")
for power, n in sample_size_2group(adj_effect, pooled_sd):
    total = 2 * n
    print(f"    Power={power:.0%}: n={n}/arm, total={total}")
    results_rows.append({
        'Design': 'Individual RCT', 'Effect_Size': f"PP mean diff={adj_effect:.1f} min",
        'Power': f"{power:.0%}", 'N_per_arm': n, 'N_total': total,
        'Clusters': '', 'Cluster_Size': '', 'ICC': '',
        'Note': f'Conservative; uses unadjusted PP mean difference'
    })

# ============================================================================
# CLUSTER-RCT (by physician)
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 2: CLUSTER-RCT (randomize by physician)")
print("=" * 80)

cluster_sizes = [20, 30, 50]  # patients per physician

for icc_val, icc_label in [(icc_date, f"date-level ICC={icc_date}"),
                             (icc_literature_lo, f"literature ICC={icc_literature_lo}"),
                             (icc_literature_hi, f"literature ICC={icc_literature_hi}")]:
    print(f"\n  --- {icc_label} ---")

    for m in cluster_sizes:
        de = 1 + (m - 1) * icc_val  # design effect
        # Individual n at 80% power using CACE
        z_alpha = norm.ppf(1 - 0.025)
        z_beta = norm.ppf(0.80)
        n_indiv = ((z_alpha + z_beta)**2 * 2 * pooled_sd**2) / cace_min**2
        n_cluster = n_indiv * de
        k = int(np.ceil(n_cluster / m))  # clusters per arm
        total_patients = 2 * k * m

        print(f"    m={m} pts/physician: DE={de:.2f}, k={k} physicians/arm, total={total_patients} pts")

        results_rows.append({
            'Design': 'Cluster-RCT', 'Effect_Size': f"CACE={cace_min:.1f} min",
            'Power': '80%', 'N_per_arm': int(np.ceil(n_cluster)),
            'N_total': total_patients,
            'Clusters': f"{k}/arm", 'Cluster_Size': m, 'ICC': f"{icc_val:.4f}",
            'Note': f'{icc_label}; DE={de:.2f}'
        })

# ============================================================================
# STEPPED-WEDGE DESIGN
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 3: STEPPED-WEDGE DESIGN")
print("=" * 80)

# Simplified: stepped wedge gains ~1/3 efficiency over parallel cluster
for icc_val, icc_label in [(icc_literature_lo, f"ICC={icc_literature_lo}"),
                             (icc_literature_hi, f"ICC={icc_literature_hi}")]:
    m = 30  # patients per cluster-period
    de = 1 + (m - 1) * icc_val
    n_indiv = ((z_alpha + z_beta)**2 * 2 * pooled_sd**2) / cace_min**2
    # SW efficiency gain ~33%
    n_sw = n_indiv * de * 0.67
    k_sw = int(np.ceil(n_sw / m))
    steps = max(3, k_sw // 2)

    print(f"  {icc_label}, m={m}: ~{k_sw} clusters, {steps} steps, ~{k_sw * m * 2} total pts")

    results_rows.append({
        'Design': 'Stepped-Wedge', 'Effect_Size': f"CACE={cace_min:.1f} min",
        'Power': '80%', 'N_per_arm': '',
        'N_total': f"~{k_sw * m * 2}",
        'Clusters': f"~{k_sw} total", 'Cluster_Size': m,
        'ICC': f"{icc_val:.4f}",
        'Note': f'Approximate; ~33% efficiency gain over parallel cluster; {steps} steps'
    })

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable12")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable12_FutureRCT_SampleSize.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print(f"""
PHASE 9 COMPLETE

DECIDE-AI statement:
  "Based on the observed CACE of {cace_min:.1f} minutes (SD={pooled_sd:.1f}) and date-level
  ICC of {icc_date:.4f} (lower bound on physician-level ICC), a definitive
  parallel-group RCT would require [N] patients per arm for 80% power.
  A cluster-RCT randomizing by physician (ICC={icc_literature_lo}-{icc_literature_hi})
  would require [K] clusters of [M] patients each."
""")
