"""
Stage 05 -- Inverse-probability weighting and doubly-robust estimation.

Two propensity models: users vs Wing B (headline PP) and users vs non-users
within Wing A. Each runs a positivity check, a lasso-logistic PS model,
stabilized ATE weights, a weighted outcome model, and an AIPW estimate.

Reproduces: eTable 1 (IPTW / AIPW rows) and the propensity-balance data
underlying eFigure 1 (love plot).

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable5_IPTW.csv, results/eFigure4_IPTW_Balance_data.csv
"""

import sys, os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
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
FIG_DIR    = os.path.join(BASE, 'results')
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 2: DUAL IPTW + DOUBLY ROBUST ESTIMATION")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding='utf-8')
df = df[df['n_consultations'] > 0]  # Exclude 8 patients with 0 consultations
assert len(df) == 1138, f"Expected 1138 rows after exclusion, got {len(df)}"
print(f"Loaded: {len(df):,} patients")

COVARIATES = ['age', 'is_female', 'triage', 'diagnosis_count', 'has_radiology',
              'is_admitted', 'n_consultations', 'is_ambulance', 'admission_hour']

results_rows = []
love_plot_rows = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def compute_smd(treat, ctrl, binary=False):
    """Standardized mean difference."""
    if binary:
        p1, p2 = treat.mean(), ctrl.mean()
        pooled = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
        return (p1 - p2) / pooled if pooled > 0 else 0
    else:
        pooled_sd = np.sqrt((treat.var() + ctrl.var()) / 2)
        return (treat.mean() - ctrl.mean()) / pooled_sd if pooled_sd > 0 else 0

def run_iptw_analysis(data, treat_col, label, covariates):
    """Full IPTW pipeline: PS model -> weighting -> outcome estimation."""
    print(f"\n{'='*70}")
    print(f"IPTW MODEL: {label}")
    print(f"{'='*70}")

    sub = data.dropna(subset=['consultation_cycle_min', 'log_consult'] + covariates).copy()
    n_treat = sub[treat_col].sum()
    n_ctrl = (1 - sub[treat_col]).sum()
    print(f"  N = {len(sub)} (treatment: {n_treat}, control: {n_ctrl})")

    X = sub[covariates].copy()
    y = sub[treat_col].values

    # ----------------------------------------------------------
    # STEP 1: POSITIVITY CHECK
    # ----------------------------------------------------------
    print(f"\n  --- Step 1: Positivity Check ---")
    binary_vars = [c for c in covariates if sub[c].nunique() <= 2]
    cont_vars = [c for c in covariates if sub[c].nunique() > 2]

    smd_pre = {}
    for var in covariates:
        is_bin = var in binary_vars
        t_vals = sub.loc[sub[treat_col] == 1, var]
        c_vals = sub.loc[sub[treat_col] == 0, var]
        smd_pre[var] = compute_smd(t_vals, c_vals, binary=is_bin)

    max_smd = max(abs(v) for v in smd_pre.values())
    print(f"  Max pre-weighting |SMD|: {max_smd:.3f}")
    if max_smd > 0.8:
        print("  *** WARNING: SMD > 0.8 suggests poor overlap. IPTW may not be credible.")
        print("  *** Consider relying on IV/CACE as primary causal estimator.")

    # Check for structural separation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Quick logistic to check PS range
    lr_check = LogisticRegressionCV(penalty='l1', solver='saga', max_iter=5000, cv=5,
                                     random_state=42, Cs=20)
    lr_check.fit(X_scaled, y)
    ps_check = lr_check.predict_proba(X_scaled)[:, 1]
    print(f"  PS range: [{ps_check.min():.4f}, {ps_check.max():.4f}]")
    print(f"  PS < 0.01: {(ps_check < 0.01).sum()}, PS > 0.99: {(ps_check > 0.99).sum()}")

    positivity_ok = ps_check.min() > 0.001 and ps_check.max() < 0.999
    if not positivity_ok:
        print("  *** POSITIVITY CONCERN: PS near boundaries. Results should be interpreted cautiously.")

    # ----------------------------------------------------------
    # STEP 2: PROPENSITY SCORE MODEL (Lasso-logistic)
    # ----------------------------------------------------------
    print(f"\n  --- Step 2: Lasso-Logistic PS Model ---")
    ps = ps_check  # Already fitted
    sub = sub.copy()
    sub['ps'] = ps

    # AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, ps)
    print(f"  AUC: {auc:.3f}")

    # ----------------------------------------------------------
    # STEP 3: STABILIZED ATE WEIGHTS (truncated)
    # ----------------------------------------------------------
    print(f"\n  --- Step 3: Stabilized ATE Weights ---")
    prev = y.mean()  # P(D=1)

    # Stabilized weights: w_i = D_i * prev/ps + (1-D_i) * (1-prev)/(1-ps)
    w = np.where(y == 1, prev / ps, (1 - prev) / (1 - ps))

    # Truncate at 1st/99th percentile
    lo, hi = np.percentile(w, [1, 99])
    w_trunc = np.clip(w, lo, hi)
    sub['iptw'] = w_trunc

    ess_treat = (w_trunc[y == 1].sum())**2 / (w_trunc[y == 1]**2).sum()
    ess_ctrl  = (w_trunc[y == 0].sum())**2 / (w_trunc[y == 0]**2).sum()

    print(f"  Weight range: [{w_trunc.min():.3f}, {w_trunc.max():.3f}]")
    print(f"  Mean weight: {w_trunc.mean():.3f}")
    print(f"  ESS treatment: {ess_treat:.0f} / {n_treat}")
    print(f"  ESS control:   {ess_ctrl:.0f} / {n_ctrl}")

    # ----------------------------------------------------------
    # STEP 4: POST-WEIGHTING BALANCE (Love plot data)
    # ----------------------------------------------------------
    print(f"\n  --- Step 4: Post-Weighting Balance ---")
    smd_post = {}
    for var in covariates:
        is_bin = var in binary_vars
        t_idx = sub[treat_col] == 1
        c_idx = sub[treat_col] == 0
        # Weighted means
        t_wt_mean = np.average(sub.loc[t_idx, var], weights=sub.loc[t_idx, 'iptw'])
        c_wt_mean = np.average(sub.loc[c_idx, var], weights=sub.loc[c_idx, 'iptw'])
        # Weighted variance (approximate)
        t_wt_var = np.average((sub.loc[t_idx, var] - t_wt_mean)**2, weights=sub.loc[t_idx, 'iptw'])
        c_wt_var = np.average((sub.loc[c_idx, var] - c_wt_mean)**2, weights=sub.loc[c_idx, 'iptw'])
        pooled_sd = np.sqrt((t_wt_var + c_wt_var) / 2)
        smd_post[var] = (t_wt_mean - c_wt_mean) / pooled_sd if pooled_sd > 0 else 0

    print(f"  {'Covariate':<20} {'SMD pre':<12} {'SMD post':<12} {'Balanced?'}")
    print(f"  {'-'*56}")
    for var in covariates:
        pre = smd_pre[var]
        post = smd_post[var]
        balanced = abs(post) < 0.1
        print(f"  {var:<20} {pre:>8.3f}     {post:>8.3f}     {'Yes' if balanced else 'NO'}")
        love_plot_rows.append({
            'Model': label, 'Covariate': var,
            'SMD_Pre': round(pre, 4), 'SMD_Post': round(post, 4),
            'Balanced': balanced
        })

    max_smd_post = max(abs(v) for v in smd_post.values())
    print(f"  Max post-weighting |SMD|: {max_smd_post:.3f}")

    # ----------------------------------------------------------
    # STEP 5: WEIGHTED OUTCOME (log-linear)
    # ----------------------------------------------------------
    print(f"\n  --- Step 5: Weighted Outcome Model ---")

    # WLS (weighted least squares)
    cov_str_simple = ' + '.join([c for c in covariates if c != 'admission_hour'])
    formula = f'log_consult ~ {treat_col} + {cov_str_simple}'

    wls = smf.wls(formula, data=sub, weights=sub['iptw']).fit(
        cov_type='HC1'  # robust SE
    )

    beta = wls.params[treat_col]
    ci = wls.conf_int().loc[treat_col]
    p_val = wls.pvalues[treat_col]
    tr = np.exp(beta)
    tr_lo = np.exp(ci[0])
    tr_hi = np.exp(ci[1])

    ctrl_median = sub.loc[sub[treat_col] == 0, 'consultation_cycle_min'].median()
    min_effect = ctrl_median * (tr - 1)

    print(f"  IPTW-weighted: TR={tr:.3f} [{tr_lo:.3f}, {tr_hi:.3f}], p={p_val:.4f}")
    print(f"  -> At control median ({ctrl_median:.1f} min): ~{min_effect:.1f} min")

    results_rows.append({
        'Analysis': label, 'Model': 'IPTW (weighted log-linear)',
        'N': len(sub), 'Estimate': f"TR={tr:.3f}",
        'CI_Lower': f"{tr_lo:.3f}", 'CI_Upper': f"{tr_hi:.3f}",
        'p_value': f"{p_val:.4f}",
        'Minutes_at_median': f"{min_effect:.1f}",
        'PS_AUC': f"{auc:.3f}",
        'ESS_treat': f"{ess_treat:.0f}", 'ESS_ctrl': f"{ess_ctrl:.0f}",
        'Max_SMD_post': f"{max_smd_post:.3f}",
        'Note': 'Stabilized ATE weights, truncated 1st/99th, HC1 robust SE'
    })

    # ----------------------------------------------------------
    # STEP 6: AIPW / DOUBLY ROBUST
    # ----------------------------------------------------------
    print(f"\n  --- Step 6: Augmented IPW (Doubly Robust) ---")

    # Outcome model: fit separate regressions for treated and control
    cov_cols = [c for c in covariates if c != 'admission_hour']
    # Cast to float so the design matrix is a numeric (not object) array: some
    # covariates load as bool, and mixing them with floats yields an object
    # dtype that the WLS fit below cannot process. The numeric values are
    # unchanged (bool -> 0.0/1.0, exactly as the regression would use them).
    X_out = sub[cov_cols].astype(float).values
    Y = sub['log_consult'].values
    D = sub[treat_col].values
    W = sub['iptw'].values

    # Fit outcome models
    X_out_c = sm.add_constant(X_out)

    # Treated outcome model
    t_mask = D == 1
    m1 = sm.WLS(Y[t_mask], X_out_c[t_mask], weights=W[t_mask]).fit()
    mu1_hat = m1.predict(X_out_c)  # predicted Y(1) for all

    # Control outcome model
    c_mask = D == 0
    m0 = sm.WLS(Y[c_mask], X_out_c[c_mask], weights=W[c_mask]).fit()
    mu0_hat = m0.predict(X_out_c)  # predicted Y(0) for all

    # AIPW estimator
    aipw_1 = D * W * (Y - mu1_hat) / (D * W).mean() + mu1_hat.mean()
    aipw_0 = (1 - D) * W * (Y - mu0_hat) / ((1 - D) * W).mean() + mu0_hat.mean()

    # Simplified: compute individual-level AIPW scores
    tau_i = (D * (Y - mu1_hat) / ps + mu1_hat) - ((1 - D) * (Y - mu0_hat) / (1 - ps) + mu0_hat)
    aipw_ate = tau_i.mean()
    aipw_se = tau_i.std() / np.sqrt(len(tau_i))
    aipw_ci_lo = aipw_ate - 1.96 * aipw_se
    aipw_ci_hi = aipw_ate + 1.96 * aipw_se
    aipw_p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(aipw_ate / aipw_se)))

    aipw_tr = np.exp(aipw_ate)
    aipw_tr_lo = np.exp(aipw_ci_lo)
    aipw_tr_hi = np.exp(aipw_ci_hi)
    aipw_min = ctrl_median * (aipw_tr - 1)

    print(f"  AIPW: TR={aipw_tr:.3f} [{aipw_tr_lo:.3f}, {aipw_tr_hi:.3f}], p={aipw_p:.4f}")
    print(f"  -> At control median: ~{aipw_min:.1f} min")

    results_rows.append({
        'Analysis': label, 'Model': 'AIPW (doubly robust)',
        'N': len(sub), 'Estimate': f"TR={aipw_tr:.3f}",
        'CI_Lower': f"{aipw_tr_lo:.3f}", 'CI_Upper': f"{aipw_tr_hi:.3f}",
        'p_value': f"{aipw_p:.4f}",
        'Minutes_at_median': f"{aipw_min:.1f}",
        'PS_AUC': f"{auc:.3f}",
        'ESS_treat': f"{ess_treat:.0f}", 'ESS_ctrl': f"{ess_ctrl:.0f}",
        'Max_SMD_post': f"{max_smd_post:.3f}",
        'Note': 'Augmented IPW; protects against misspecification of either PS or outcome model'
    })

    return positivity_ok

# ============================================================================
# RUN BOTH MODELS
# ============================================================================

# Model A: Users vs Wing B
sub_a = df[(df['is_user'] == 1) | (df['wing'] == 'B')].copy()
sub_a['treatment'] = sub_a['is_user']
pos_a = run_iptw_analysis(sub_a, 'treatment', 'Model A: Users vs Wing B (headline PP)', COVARIATES)

# Model B: Users vs Non-Users within Wing A
sub_b = df[df['wing'] == 'A'].copy()
sub_b['treatment'] = sub_b['is_user']
pos_b = run_iptw_analysis(sub_b, 'treatment', 'Model B: Users vs Non-Users (Wing A only)', COVARIATES)

# ============================================================================
# SAVE
# ============================================================================
print("\n" + "=" * 80)
print("SAVING OUTPUTS")
print("=" * 80)

results_df = pd.DataFrame(results_rows)
output_path = os.path.join(OUTPUT_DIR, 'eTable5_IPTW.csv')
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  eTable5: {output_path}")

love_df = pd.DataFrame(love_plot_rows)
love_path = os.path.join(FIG_DIR, 'eFigure4_IPTW_Balance_data.csv')
love_df.to_csv(love_path, index=False, encoding='utf-8-sig')
print(f"  eFigure4 data: {love_path}")

print("\n" + "=" * 80)
print("PHASE 2 COMPLETE")
print("=" * 80)
print(f"""
Summary:
  Model A positivity: {'OK' if pos_a else 'CONCERN -- consider IV/CACE as primary'}
  Model B positivity: {'OK' if pos_b else 'CONCERN'}

Reviewer narrative:
  "After inverse probability weighting to balance measured confounders between
   SHAKED users and controls, the treatment effect was [X]. The doubly robust
   (AIPW) estimate, which protects against misspecification of either the
   propensity score or outcome model, yielded [X]."
""")
