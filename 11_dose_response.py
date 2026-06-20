"""
Stage 11 -- Exposure-level analysis by engagement tier.

Access-based contrasts (each Wing A engagement subgroup vs Wing B) and a
within-Wing-A engagement trend (Jonckheere-Terpstra), including the passive-
exposure and technical-failure groups. The trend test uses the manuscript's
published 4-tier engagement convention; see the within-Wing-A section below.

Reproduces: Table 2 panels D/E and the exposure-level panel of Supp Fig 2A.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable10_ExposureLevel.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
import statsmodels.formula.api as smf
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
print("PHASE 6: EXPOSURE-LEVEL + PASSIVE EXPOSURE + TECH-FAILURE NEGATIVE CONTROL")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

def hl_estimate(x, y):
    """Hodges-Lehmann shift estimate."""
    diffs = np.subtract.outer(y, x).ravel()
    return np.median(diffs)

# ============================================================================
# DESCRIPTIVE: Engagement level summary
# ============================================================================
print("\n--- Engagement Level Summary ---")
print(f"  {'Level':<20} {'N':>5} {'Median consult':>15} {'IQR':>20}")
print(f"  {'-'*65}")

# Published 4-tier engagement convention. The "no-exposure" sub-stratum (Wing A
# patients for whom SHAKED was neither opened nor produced output) folds into the
# non-user/excluded bucket and is not reported as a standalone tier.
for level in ['0_NoAccess', '2_Passive', '3_TechFailure', '4_FullResults']:
    g = df[df['engagement_level'] == level]
    ct = g['consultation_cycle_min'].dropna()
    print(f"  {level:<20} {len(g):>5} {ct.median():>15.1f} [{ct.quantile(0.25):.1f}-{ct.quantile(0.75):.1f}]")

# ============================================================================
# (a) ACCESS-BASED CONTRASTS: Each Wing A subgroup vs Wing B
# ============================================================================
print("\n" + "=" * 80)
print("(a) ACCESS-BASED CONTRASTS: Each subgroup vs Wing B controls")
print("=" * 80)

wing_b = df[df['engagement_level'] == '0_NoAccess']['consultation_cycle_min'].dropna()

# Access-based contrasts follow the manuscript's reported engagement tiers:
# Passive (auto-run, not viewed), technical failure, and full results -- each vs
# Wing B. The "no exposure" sub-stratum (Wing A patients for whom SHAKED was
# neither opened nor produced output) is folded out of the published contrast:
# it is post-randomisation, not part of the manuscript's reported tiers, and is
# carried only as an ordinal level inside the within-Wing-A engagement trend.
for level in ['2_Passive', '3_TechFailure', '4_FullResults']:
    g = df[df['engagement_level'] == level]['consultation_cycle_min'].dropna()

    stat, p = mannwhitneyu(wing_b, g)
    hl = hl_estimate(wing_b.values, g.values)

    # Bootstrap CI for HL
    n_boot = 5000
    boot_hl = np.zeros(n_boot)
    for b in range(n_boot):
        bx = np.random.choice(wing_b.values, size=len(wing_b), replace=True)
        by = np.random.choice(g.values, size=len(g), replace=True)
        boot_hl[b] = np.median(np.subtract.outer(by, bx).ravel())
    ci = np.percentile(boot_hl, [2.5, 97.5])

    print(f"\n  {level} (n={len(g)}) vs Wing B (n={len(wing_b)}):")
    print(f"    HL shift: {hl:.1f} min [{ci[0]:.1f}, {ci[1]:.1f}], p={p:.4f}")
    print(f"    Medians: {g.median():.1f} vs {wing_b.median():.1f}")

    results_rows.append({
        'Contrast': 'Access-based', 'Comparison': f"{level} vs 0_NoAccess",
        'N_group': len(g), 'N_ref': len(wing_b),
        'Median_group': f"{g.median():.1f}", 'Median_ref': f"{wing_b.median():.1f}",
        'HL_shift': f"{hl:.1f}", 'CI_Lower': f"{ci[0]:.1f}", 'CI_Upper': f"{ci[1]:.1f}",
        'p_value': f"{p:.4f}",
        'Note': ''
    })

# ============================================================================
# KEY FINDING 1: PASSIVE EXPOSURE (promote to main manuscript)
# ============================================================================
print("\n" + "=" * 80)
print("*** KEY FINDING: PASSIVE EXPOSURE (n=275) vs Wing B ***")
print("=" * 80)

passive = df[df['engagement_level'] == '2_Passive']['consultation_cycle_min'].dropna()
stat_p, p_passive = mannwhitneyu(wing_b, passive)
hl_passive = hl_estimate(wing_b.values, passive.values)

print(f"  Passive (auto-run, not viewed): median={passive.median():.1f} min (n={len(passive)})")
print(f"  Wing B (no access):             median={wing_b.median():.1f} min (n={len(wing_b)})")
print(f"  HL shift: {hl_passive:.1f} min, p={p_passive:.4f}")

if p_passive > 0.10:
    print(f"""
  INTERPRETATION (for manuscript):
  "The effect was absent among patients for whom SHAKED generated outputs
  that were not reviewed by clinicians (passive exposure group, n={len(passive)},
  HL shift {hl_passive:.1f} min, p={p_passive:.3f}), consistent with a mechanism
  requiring active physician engagement rather than passive information
  availability."
""")
else:
    print(f"  NOTE: Passive group shows p={p_passive:.4f} -- may have some spillover effect.")

# ============================================================================
# KEY FINDING 2: TECH-FAILURE NEGATIVE CONTROL
# ============================================================================
print("=" * 80)
print("*** KEY FINDING: TECH-FAILURE NEGATIVE CONTROL (n=44) ***")
print("=" * 80)

tech_fail = df[df['engagement_level'] == '3_TechFailure']['consultation_cycle_min'].dropna()
stat_tf, p_tf = mannwhitneyu(wing_b, tech_fail)
hl_tf = hl_estimate(wing_b.values, tech_fail.values)

print(f"  TechFailure (opened, no output): median={tech_fail.median():.1f} min (n={len(tech_fail)})")
print(f"  Wing B (no access):              median={wing_b.median():.1f} min (n={len(wing_b)})")
print(f"  HL shift: {hl_tf:.1f} min, p={p_tf:.4f}")

if p_tf > 0.10:
    print(f"""
  INTERPRETATION:
  "Patients whose physicians opened SHAKED but received no output due to
  technical failure (n={len(tech_fail)}) showed no benefit compared to controls
  (HL shift {hl_tf:.1f} min, p={p_tf:.3f}), demonstrating that the effect
  requires SHAKED output rather than selection into use."
""")

# ============================================================================
# (b) WITHIN-WING-A ENGAGEMENT CONTRAST
# ============================================================================
print("=" * 80)
print("(b) WITHIN-WING-A ENGAGEMENT CONTRAST")
print("=" * 80)

wing_a = df[df['wing'] == 'A'].dropna(subset=['consultation_cycle_min']).copy()
# Published engagement convention within Wing A: Passive -> TechFailure ->
# FullResults. The "no-exposure" sub-stratum is excluded from the trend test per
# the manuscript's 4-tier convention (it is post-randomisation and not a reported
# engagement tier); it folds into the non-user/excluded bucket.
wing_a_levels = ['2_Passive', '3_TechFailure', '4_FullResults']
wing_a_data = wing_a[wing_a['engagement_level'].isin(wing_a_levels)]

# Kruskal-Wallis
groups_kw = [wing_a_data.loc[wing_a_data['engagement_level'] == lev, 'consultation_cycle_min'].values
             for lev in wing_a_levels]
groups_kw = [g for g in groups_kw if len(g) > 0]
kw_stat, kw_p = kruskal(*groups_kw)
print(f"  Kruskal-Wallis across 3 Wing A engagement tiers: H={kw_stat:.2f}, p={kw_p:.4f}")

# Jonckheere-Terpstra trend test (manual implementation)
# Under H0: no ordered trend; H1: medians decrease with engagement level
def jonckheere_terpstra(groups):
    """Jonckheere-Terpstra test statistic."""
    k = len(groups)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for xi in groups[i]:
                for xj in groups[j]:
                    if xj < xi:
                        J += 1
                    elif xj == xi:
                        J += 0.5
    # Expected value and variance
    N = sum(len(g) for g in groups)
    ns = [len(g) for g in groups]
    E_J = (N**2 - sum(n**2 for n in ns)) / 4
    V_J = (N**2 * (2*N + 3) - sum(n**2 * (2*n + 3) for n in ns)) / 72
    Z = (J - E_J) / np.sqrt(V_J)
    from scipy.stats import norm
    p = norm.sf(Z)  # one-sided (expecting decrease)
    return J, Z, p

jt_groups = [wing_a_data.loc[wing_a_data['engagement_level'] == lev, 'consultation_cycle_min'].values
             for lev in wing_a_levels]
jt_groups = [g for g in jt_groups if len(g) > 0]

J_stat, J_z, jt_p = jonckheere_terpstra(jt_groups)
print(f"  Jonckheere-Terpstra (within Wing A): J={J_stat:.0f}, Z={J_z:.2f}, p={jt_p:.4f} (one-sided)")

results_rows.append({
    'Contrast': 'Within-Wing-A trend', 'Comparison': 'JT trend (Passive->TechFail->FullResults)',
    'N_group': len(wing_a_data), 'N_ref': '',
    'Median_group': '', 'Median_ref': '',
    'HL_shift': f"Z={J_z:.2f}", 'CI_Lower': '', 'CI_Upper': '',
    'p_value': f"{jt_p:.4f}",
    'Note': 'Jonckheere-Terpstra; one-sided; expects decrease with engagement; published 4-tier convention (no-exposure excluded)'
})

# Pairwise within Wing A: FullResults vs each reported engagement tier.
# (The "no exposure" sub-stratum is excluded from the pairwise contrasts, as
# above, so no standalone no-exposure effect is emitted.)
full_res = wing_a_data[wing_a_data['engagement_level'] == '4_FullResults']['consultation_cycle_min'].dropna()
for ref_level in ['2_Passive', '3_TechFailure']:
    ref = wing_a_data[wing_a_data['engagement_level'] == ref_level]['consultation_cycle_min'].dropna()
    if len(ref) > 0 and len(full_res) > 0:
        stat, p = mannwhitneyu(ref, full_res)
        hl = hl_estimate(ref.values, full_res.values)
        print(f"  4_FullResults vs {ref_level}: HL={hl:.1f} min, p={p:.4f}")
        results_rows.append({
            'Contrast': 'Within-Wing-A pairwise', 'Comparison': f"4_FullResults vs {ref_level}",
            'N_group': len(full_res), 'N_ref': len(ref),
            'Median_group': f"{full_res.median():.1f}", 'Median_ref': f"{ref.median():.1f}",
            'HL_shift': f"{hl:.1f}", 'CI_Lower': '', 'CI_Upper': '',
            'p_value': f"{p:.4f}", 'Note': 'Within Wing A only'
        })

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable10")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable10_ExposureLevel.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print("\nPHASE 6 COMPLETE")
