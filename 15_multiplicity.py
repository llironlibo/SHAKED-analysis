"""
Stage 15 -- Multiplicity correction.

Holm-Bonferroni correction across the family of secondary and sensitivity
tests (the pre-specified primary estimate is exempt), with monotonicity
enforced on the adjusted p-values.

Reproduces: eTable 13 (Holm-Bonferroni multiplicity table).

Inputs : p-values collected from the analyses above (listed inline)
Outputs: results/eTable13_Multiplicity.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 10: CONSOLIDATION -- MULTIPLICITY CORRECTION + APPENDIX INDEX")
print("=" * 80)

# ============================================================================
# COLLECT ALL P-VALUES FROM SECONDARY/SENSITIVITY ANALYSES
# ============================================================================
# Primary (pre-specified, exempt): PP HL=-13.3 min, p=0.043
# Below are secondary/sensitivity analyses:

tests = [
    # (Label, p-value, Source, Type)
    ("Adjusted PP (log-linear)", 0.8309, "eTable4", "Secondary"),
    ("Adjusted ITT (log-linear)", 0.4855, "eTable4", "Secondary"),
    ("Quantile regression (median)", 0.8338, "eTable4", "Sensitivity"),
    ("Quantile regression (75th)", 0.3377, "eTable4", "Sensitivity"),
    ("Quantile regression (90th)", 0.3703, "eTable4", "Sensitivity"),
    ("Winsorized OLS (99th pctl)", 0.5281, "eTable4", "Sensitivity"),
    ("IPTW Users vs Wing B", 0.5279, "eTable5", "Secondary"),
    ("AIPW Users vs Wing B", 0.5662, "eTable5", "Secondary"),
    ("IPTW Users vs Non-Users (Wing A)", 0.8269, "eTable5", "Sensitivity"),
    ("AIPW Users vs Non-Users (Wing A)", 0.7531, "eTable5", "Sensitivity"),
    ("Wald LATE (minutes)", 0.0564, "eTable6", "Secondary"),
    ("Wald LATE (log-scale)", 0.3269, "eTable6", "Sensitivity"),
    ("2SLS CACE (minutes)", 0.2276, "eTable6", "Secondary"),
    ("Cluster-bootstrap CACE", 0.2020, "eTable6", "Sensitivity"),
    ("RMST (tau=321, primary)", 0.0540, "eTable8", "Secondary"),
    ("RMST (tau=360)", 0.0290, "eTable8", "Sensitivity"),
    ("RMST (tau=480)", 0.0340, "eTable8", "Sensitivity"),
    ("RMST (tau=600)", 0.0430, "eTable8", "Sensitivity"),
    ("AFT unadjusted", 0.2124, "eTable8", "Sensitivity"),
    ("AFT adjusted", 0.6214, "eTable8", "Sensitivity"),
    ("Cox PH adjusted", 0.1392, "eTable8", "Sensitivity"),
    ("Log-rank test", 0.0180, "eTable8", "Sensitivity"),
    ("Mixed-effects (date)", 0.7277, "eTable9", "Sensitivity"),
    ("Cluster-robust SE", 0.7620, "eTable9", "Sensitivity"),
    ("Date fixed effects", 0.9469, "eTable9", "Sensitivity"),
    ("Exposure-level: FullResults vs Wing B", 0.0478, "eTable10", "Secondary"),
    ("Exposure-level: Passive vs Wing B", 0.8436, "eTable10", "Secondary"),
    ("Exposure-level: TechFailure vs Wing B", 0.4493, "eTable10", "Secondary"),
    ("JT trend (within Wing A)", 0.3625, "eTable10", "Sensitivity"),
    ("Adoption decay (day_index OR)", 0.0000, "eFigure3", "Descriptive"),
]

# ============================================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================================
print("\n--- Holm-Bonferroni Correction ---")

# Sort by p-value
sorted_tests = sorted(tests, key=lambda x: x[1])
n_tests = len(sorted_tests)

rows = []
prev_adjusted = 0.0  # For monotonicity enforcement
for rank, (label, p_raw, source, test_type) in enumerate(sorted_tests, 1):
    # Holm threshold: alpha / (n - rank + 1)
    alpha = 0.05
    holm_threshold = alpha / (n_tests - rank + 1)
    p_adjusted_raw = min(p_raw * (n_tests - rank + 1), 1.0)
    # Enforce monotonicity: each adjusted p >= all previous adjusted p-values
    p_adjusted = max(p_adjusted_raw, prev_adjusted)
    prev_adjusted = p_adjusted
    survives = p_adjusted < 0.05

    rows.append({
        'Rank': rank,
        'Analysis': label,
        'p_unadjusted': f"{p_raw:.4f}",
        'p_Holm_adjusted': f"{p_adjusted:.4f}",
        'Holm_threshold': f"{holm_threshold:.4f}",
        'Survives_correction': 'Yes' if survives else 'No',
        'Source': source,
        'Type': test_type
    })

results_df = pd.DataFrame(rows)

print(f"\n  Total secondary/sensitivity tests: {n_tests}")
n_survive = sum(1 for r in rows if r['Survives_correction'] == 'Yes')
print(f"  Survive Holm-Bonferroni: {n_survive}")

print(f"\n  {'Rank':>4} {'Analysis':<45} {'p_raw':<10} {'p_adj':<10} {'Survives'}")
print(f"  {'-'*80}")
for r in rows[:15]:  # show top 15
    print(f"  {r['Rank']:>4} {r['Analysis']:<45} {r['p_unadjusted']:<10} {r['p_Holm_adjusted']:<10} {r['Survives_correction']}")
if n_tests > 15:
    print(f"  ... ({n_tests - 15} more tests, all p > {rows[14]['p_unadjusted']})")

# ============================================================================
# SAVE MULTIPLICITY TABLE
# ============================================================================
output_path = os.path.join(OUTPUT_DIR, 'eTable13_Multiplicity.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n  Saved to: {output_path}")

print("\n" + "=" * 80)
print("STAGE 15 COMPLETE")
print("=" * 80)
print(f"  {n_tests} secondary/sensitivity tests corrected; {n_survive} survive Holm-Bonferroni.")
print("  The pre-specified primary per-protocol estimate is exempt from correction.")
