"""
Stage 10 -- Date-level clustering and robustness.

Mixed-effects model (random intercept by date), cluster-robust SE, within-date
fixed effects, leave-one-date-out jackknife, and a consulting-specialist
random effect.

Reproduces: eTable 1 clustering / jackknife rows.

Inputs : data/cleaned/unified_pilot_cohort.csv, data/cleaned/consultation_data.csv
Outputs: results/eTable9_Clustering.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
CONSULT_PATH = os.path.join(BASE, 'data', 'cleaned', 'consultation_data.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 5: DATE-LEVEL CLUSTERING + JACKKNIFE + FIXED EFFECTS")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
results_rows = []

# PP subset
sub = df[(df['is_user'] == 1) | (df['wing'] == 'B')].dropna(
    subset=['consultation_cycle_min', 'log_consult', 'triage', 'diagnosis_count']
).copy()
print(f"PP sample: {len(sub)} (21 unique dates)")

covars_str = 'triage + diagnosis_count + has_radiology + is_admitted + n_consultations + is_ambulance + admission_hour'  # NOTE: admission_hour = ED arrival hour, not hospital admission

# ============================================================================
# 1. MIXED-EFFECTS MODEL (random intercept by arrival_date)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: MIXED-EFFECTS (Random Intercept by Date)")
print("=" * 80)

formula_me = f'log_consult ~ is_user + {covars_str}'
try:
    me_model = smf.mixedlm(formula_me, data=sub, groups=sub['arrival_date']).fit(reml=True)
    beta_me = me_model.fe_params['is_user']
    se_me = me_model.bse_fe['is_user']
    ci_me = [beta_me - 1.96*se_me, beta_me + 1.96*se_me]
    p_me = me_model.pvalues['is_user']
    tr_me = np.exp(beta_me)

    # ICC
    var_date = float(me_model.cov_re.iloc[0, 0])
    var_resid = me_model.scale
    icc = var_date / (var_date + var_resid)

    print(f"  Treatment effect: beta={beta_me:.4f}, TR={tr_me:.3f}, p={p_me:.4f}")
    print(f"  Random intercept variance (date): {var_date:.4f}")
    print(f"  Residual variance: {var_resid:.4f}")
    print(f"  ICC (date-level): {icc:.4f}")
    print(f"  Note: Date-level ICC provides a LOWER BOUND on physician-level ICC")
    print(f"        (dates pool multiple physicians).")

    results_rows.append({
        'Model': 'Mixed-effects (1|date)', 'Estimate': f"TR={tr_me:.3f}",
        'CI_Lower': f"{np.exp(ci_me[0]):.3f}", 'CI_Upper': f"{np.exp(ci_me[1]):.3f}",
        'p_value': f"{p_me:.4f}", 'ICC': f"{icc:.4f}",
        'Note': 'Random intercept by arrival_date; REML'
    })
except Exception as e:
    print(f"  Mixed-effects model failed: {e}")
    icc = None

# ============================================================================
# 2. CLUSTER-ROBUST SE (OLS with HC1 by arrival_date)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: CLUSTER-ROBUST SE")
print("=" * 80)

formula_ols = f'log_consult ~ is_user + {covars_str}'
ols_clust = smf.ols(formula_ols, data=sub).fit(
    cov_type='cluster', cov_kwds={'groups': sub['arrival_date']}
)
beta_cr = ols_clust.params['is_user']
ci_cr = ols_clust.conf_int().loc['is_user']
p_cr = ols_clust.pvalues['is_user']
tr_cr = np.exp(beta_cr)

print(f"  TR={tr_cr:.3f} [{np.exp(ci_cr[0]):.3f}, {np.exp(ci_cr[1]):.3f}], p={p_cr:.4f}")

results_rows.append({
    'Model': 'OLS + cluster-robust SE', 'Estimate': f"TR={tr_cr:.3f}",
    'CI_Lower': f"{np.exp(ci_cr[0]):.3f}", 'CI_Upper': f"{np.exp(ci_cr[1]):.3f}",
    'p_value': f"{p_cr:.4f}", 'ICC': '',
    'Note': 'HC1 cluster-robust SE by arrival_date (21 clusters)'
})

# ============================================================================
# 3. WITHIN-DATE FIXED EFFECTS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: WITHIN-DATE FIXED EFFECTS")
print("=" * 80)

formula_fe = f'log_consult ~ is_user + {covars_str} + C(arrival_date)'
fe_model = smf.ols(formula_fe, data=sub).fit(cov_type='HC1')
beta_fe = fe_model.params['is_user']
ci_fe = fe_model.conf_int().loc['is_user']
p_fe = fe_model.pvalues['is_user']
tr_fe = np.exp(beta_fe)

print(f"  TR={tr_fe:.3f} [{np.exp(ci_fe[0]):.3f}, {np.exp(ci_fe[1]):.3f}], p={p_fe:.4f}")
print(f"  This implicitly controls for all day-level confounders (staffing, volume, etc.)")

results_rows.append({
    'Model': 'Date fixed effects', 'Estimate': f"TR={tr_fe:.3f}",
    'CI_Lower': f"{np.exp(ci_fe[0]):.3f}", 'CI_Upper': f"{np.exp(ci_fe[1]):.3f}",
    'p_value': f"{p_fe:.4f}", 'ICC': '',
    'Note': 'Date dummies absorb all day-level confounders; HC1 robust SE'
})

# ============================================================================
# 4. LEAVE-ONE-DATE-OUT JACKKNIFE
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 4: LEAVE-ONE-DATE-OUT JACKKNIFE (21 dates)")
print("=" * 80)

dates = sub['arrival_date'].unique()
jack_effects = []

for d in dates:
    sub_jack = sub[sub['arrival_date'] != d].copy()
    try:
        m = smf.ols(f'log_consult ~ is_user + {covars_str}', data=sub_jack).fit()
        jack_effects.append({
            'date_dropped': d,
            'beta': m.params['is_user'],
            'tr': np.exp(m.params['is_user']),
            'p': m.pvalues['is_user'],
            'n': len(sub_jack)
        })
    except Exception:
        pass

jack_df = pd.DataFrame(jack_effects)
tr_full = np.exp(smf.ols(f'log_consult ~ is_user + {covars_str}', data=sub).fit().params['is_user'])

print(f"  Full sample TR: {tr_full:.3f}")
print(f"  Jackknife TR range: [{jack_df['tr'].min():.3f}, {jack_df['tr'].max():.3f}]")
print(f"  Max deviation from full: {(jack_df['tr'] - tr_full).abs().max():.3f}")
print(f"  Dates where dropping changes significance:")
for _, row in jack_df.iterrows():
    if (row['p'] < 0.05) != (p_cr < 0.05):
        print(f"    Dropping {row['date_dropped']}: TR={row['tr']:.3f}, p={row['p']:.4f}")
if all((jack_df['p'] < 0.05) == (p_cr < 0.05)):
    print(f"    None -- significance is robust to dropping any single date")

results_rows.append({
    'Model': 'Jackknife (leave-one-date-out)', 'Estimate': f"TR range: [{jack_df['tr'].min():.3f}, {jack_df['tr'].max():.3f}]",
    'CI_Lower': '', 'CI_Upper': '',
    'p_value': f"range: [{jack_df['p'].min():.4f}, {jack_df['p'].max():.4f}]",
    'ICC': '',
    'Note': f'21 iterations; max deviation from full sample = {(jack_df["tr"] - tr_full).abs().max():.3f}'
})

# ============================================================================
# 5. CONSULTING-SPECIALIST CLUSTERING
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 5: CONSULTING-SPECIALIST RANDOM EFFECT")
print("=" * 80)
print("NOTE: This captures consultant-side variation (how fast the specialist responds),")
print("NOT ED-side variation (which ED physician ordered the consult).")

try:
    consult_df = pd.read_csv(CONSULT_PATH, encoding='utf-8')
    # Consultant-name column. Prefer an English-named column if present;
    # otherwise fall back to its fixed position (column 4) in the export.
    consult_col = None
    for c in consult_df.columns:
        if 'consultant' in c.lower():
            consult_col = c
            break
    if consult_col is None:
        consult_col = consult_df.columns[4]  # consultant-name column

    case_col = 'case_num_str'

    # Take first consultant per patient
    first_consult = consult_df.groupby(case_col).first().reset_index()
    first_consult = first_consult[[case_col, consult_col]].rename(
        columns={case_col: 'case_id', consult_col: 'consultant'}
    )

    # Merge
    sub_cons = sub.merge(first_consult, on='case_id', how='inner')
    n_consultants = sub_cons['consultant'].nunique()
    print(f"  Merged: {len(sub_cons)} patients, {n_consultants} unique consultants")

    if n_consultants > 2:
        me_cons = smf.mixedlm(formula_me, data=sub_cons, groups=sub_cons['consultant']).fit(reml=True)
        beta_cons = me_cons.fe_params['is_user']
        se_cons = me_cons.bse_fe['is_user']
        p_cons = me_cons.pvalues['is_user']
        tr_cons = np.exp(beta_cons)

        var_cons = float(me_cons.cov_re.iloc[0, 0])
        var_res_cons = me_cons.scale
        icc_cons = var_cons / (var_cons + var_res_cons)

        print(f"  TR={tr_cons:.3f}, p={p_cons:.4f}")
        print(f"  ICC (consultant): {icc_cons:.4f}")

        results_rows.append({
            'Model': 'Mixed-effects (1|consultant)', 'Estimate': f"TR={tr_cons:.3f}",
            'CI_Lower': f"{np.exp(beta_cons - 1.96*se_cons):.3f}",
            'CI_Upper': f"{np.exp(beta_cons + 1.96*se_cons):.3f}",
            'p_value': f"{p_cons:.4f}", 'ICC': f"{icc_cons:.4f}",
            'Note': 'Consultant-side variation only (not ED physician); REML'
        })
except Exception as e:
    print(f"  Consultant clustering failed: {e}")

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING eTable9")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable9_Clustering.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  Saved to: {output_path}")

print(f"""
PHASE 5 COMPLETE

Limitation statement:
  "ED treating physician identifiers were not available in the administrative
  dataset. Date-level clustering (21 unique dates, ICC={f'{icc:.4f}' if icc else 'N/A'}) was used as
  a proxy for physician-and-shift-level correlation. Date-level ICC provides
  a lower bound on physician-level ICC since dates pool multiple physicians."
""")
