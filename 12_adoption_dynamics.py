"""
Stage 12 -- Adoption dynamics over time.

Date-level adoption rates with binomial CIs, a patient-level logistic decay
model (time, workload, arrival hour), a piecewise break check, and Spearman
correlations.

Reproduces: eTable 3 (adoption dynamics) and the data underlying Figure 3.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable_Adoption_Decay_Model.csv,
         results/eFigure3_Adoption_TimeSeries_data.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIG_DIR    = os.path.join(BASE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 7: ADOPTION DECAY MODELING")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"

# Wing A only
wing_a = df[df['wing'] == 'A'].copy()
print(f"Wing A patients: {len(wing_a)}")

results_rows = []

# ============================================================================
# 1. DATE-LEVEL ADOPTION RATES
# ============================================================================
print("\n--- Date-Level Adoption Rates ---")

daily = wing_a.groupby('arrival_date').agg(
    n_patients=('case_id', 'count'),
    n_users=('is_user', 'sum'),
    daily_volume=('daily_volume', 'first'),
    day_index=('day_index', 'first'),
    mean_hour=('admission_hour', 'mean'),
).reset_index()
daily['adoption_rate'] = daily['n_users'] / daily['n_patients']

# Exact binomial CI
from scipy.stats import binom
daily['ci_lo'] = daily.apply(lambda r: binom.ppf(0.025, int(r['n_patients']), r['adoption_rate']) / r['n_patients']
                              if r['n_patients'] > 0 else 0, axis=1)
daily['ci_hi'] = daily.apply(lambda r: binom.ppf(0.975, int(r['n_patients']), r['adoption_rate']) / r['n_patients']
                              if r['n_patients'] > 0 else 0, axis=1)

print(f"\n  {'Date':<12} {'N':>4} {'Users':>6} {'Rate':>7} {'95% CI':>20}")
print(f"  {'-'*55}")
for _, row in daily.sort_values('day_index').iterrows():
    print(f"  {str(row['arrival_date']):<12} {int(row['n_patients']):>4} {int(row['n_users']):>6} {row['adoption_rate']:>7.1%} [{row['ci_lo']:.1%}-{row['ci_hi']:.1%}]")

# ============================================================================
# 2. PATIENT-LEVEL LOGISTIC MODEL
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: PATIENT-LEVEL LOGISTIC (adoption ~ time + volume)")
print("=" * 80)

# Linear time trend
formula = 'is_user ~ day_index + daily_volume + admission_hour'  # NOTE: admission_hour = ED arrival hour, not hospital admission
logit = smf.logit(formula, data=wing_a).fit(disp=0)
print(logit.summary2().tables[1].to_string())

results_rows.append({
    'Model': 'Logistic (linear time)',
    'Variable': 'day_index',
    'Coefficient': f"{logit.params['day_index']:.4f}",
    'OR': f"{np.exp(logit.params['day_index']):.3f}",
    'CI_Lower': f"{np.exp(logit.conf_int().loc['day_index'][0]):.3f}",
    'CI_Upper': f"{np.exp(logit.conf_int().loc['day_index'][1]):.3f}",
    'p_value': f"{logit.pvalues['day_index']:.4f}",
    'Note': 'OR per additional day; <1 = declining adoption'
})

for var in ['daily_volume', 'admission_hour']:
    results_rows.append({
        'Model': 'Logistic (linear time)',
        'Variable': var,
        'Coefficient': f"{logit.params[var]:.4f}",
        'OR': f"{np.exp(logit.params[var]):.3f}",
        'CI_Lower': f"{np.exp(logit.conf_int().loc[var][0]):.3f}",
        'CI_Upper': f"{np.exp(logit.conf_int().loc[var][1]):.3f}",
        'p_value': f"{logit.pvalues[var]:.4f}",
        'Note': ''
    })

# Piecewise break at day 7
wing_a['post_week1'] = (wing_a['day_index'] > 7).astype(int)
wing_a['day_post_week1'] = np.maximum(wing_a['day_index'] - 7, 0)

formula_pw = 'is_user ~ day_index + post_week1 + day_post_week1 + daily_volume + admission_hour'
logit_pw = smf.logit(formula_pw, data=wing_a).fit(disp=0)
print(f"\n  Piecewise model (break at day 7):")
print(f"    AIC linear: {logit.aic:.1f}")
print(f"    AIC piecewise: {logit_pw.aic:.1f}")
print(f"    Better model: {'Piecewise' if logit_pw.aic < logit.aic else 'Linear'}")

# ============================================================================
# 3. SPEARMAN CORRELATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

rho_time, p_time = spearmanr(daily['day_index'], daily['adoption_rate'])
rho_vol, p_vol = spearmanr(daily['daily_volume'], daily['adoption_rate'])

print(f"  Adoption vs Time: rho={rho_time:.3f}, p={p_time:.4f}")
print(f"  Adoption vs Volume: rho={rho_vol:.3f}, p={p_vol:.4f}")

results_rows.append({
    'Model': 'Spearman correlation', 'Variable': 'day_index vs adoption_rate',
    'Coefficient': f"rho={rho_time:.3f}", 'OR': '', 'CI_Lower': '', 'CI_Upper': '',
    'p_value': f"{p_time:.4f}", 'Note': f'Date-level (n={len(daily)} dates)'
})
results_rows.append({
    'Model': 'Spearman correlation', 'Variable': 'daily_volume vs adoption_rate',
    'Coefficient': f"rho={rho_vol:.3f}", 'OR': '', 'CI_Lower': '', 'CI_Upper': '',
    'p_value': f"{p_vol:.4f}", 'Note': f'Date-level (n={len(daily)} dates)'
})

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
model_path = os.path.join(OUTPUT_DIR, 'eTable_Adoption_Decay_Model.csv')
results_df.to_csv(model_path, index=False, encoding='utf-8-sig')
print(f"  Model table: {model_path}")

# Figure data
fig_path = os.path.join(FIG_DIR, 'eFigure3_Adoption_TimeSeries_data.csv')
daily.to_csv(fig_path, index=False, encoding='utf-8-sig')
print(f"  Figure data: {fig_path}")

print(f"""
PHASE 7 COMPLETE

Key findings:
  Time trend: OR={np.exp(logit.params['day_index']):.3f} per day (p={logit.pvalues['day_index']:.4f})
  Volume association: rho={rho_vol:.3f} (p={p_vol:.4f})
  Adoption range: {daily['adoption_rate'].min():.1%} to {daily['adoption_rate'].max():.1%}
""")
