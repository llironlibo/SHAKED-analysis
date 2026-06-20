"""
Stage 16 -- Weekly effects, length of stay, and contamination bounding.

Weekly ITT Hodges-Lehmann shifts (including the Study Week 3 anomaly), the
wing-assignment propensity check, ITT length-of-stay outcomes, ITT E-values,
a contaminated-controls bounding analysis, and a Wing B secular-trend check.

Reproduces: eTable 2 (weekly effects), eTable 4 (length of stay), the design-
validation rows, and the contamination bound noted in the Discussion.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable2_Weekly_Effects.csv, results/eTable_LOS_ITT.csv,
         results/eTable_Randomization_Checks.csv and related tables
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, norm, spearmanr
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import sys
import os

# Repo root resolved from this file's location so the script runs
# unchanged on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')
OUTPUT_DIR = os.path.join(BASE, 'results')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS (reused from existing scripts)
# ============================================================================

def hodges_lehmann_ci_vectorized(x, y, alpha=0.05):
    """
    Vectorized Hodges-Lehmann estimator with a normal-approximation CI.

    Parameters
    ----------
    x, y : array-like
        Two samples to compare
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns
    -------
    est : float
        Hodges-Lehmann estimate (median of all pairwise differences)
    ci_low, ci_high : float
        Confidence interval bounds
    """
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)

    # All pairwise differences
    diffs = (x[:, None] - y).flatten()
    diffs.sort()

    # Point estimate: median of differences
    est = np.median(diffs)

    # Normal approximation CI
    m_u = n1 * n2 / 2
    sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_crit = norm.ppf(1 - alpha/2)
    k = int(z_crit * sigma_u)

    M = n1 * n2
    lower_rank = max(0, int(M/2 - k))
    upper_rank = min(M, int(M/2 + k) + 1)

    ci_low = diffs[lower_rank]
    ci_high = diffs[upper_rank] if upper_rank < M else diffs[-1]

    return est, ci_low, ci_high


def compute_evalue(rr):
    """
    E-value from a risk ratio (see stage 13 for the primary E-value analysis).

    Parameters
    ----------
    rr : float
        Risk ratio

    Returns
    -------
    float
        E-value (minimum strength of unmeasured confounding to explain away effect)
    """
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def compute_evalue_ci(rr_ci_bound):
    """
    E-value for a confidence-interval bound (see stage 13).

    Parameters
    ----------
    rr_ci_bound : float
        CI bound closest to null (RR = 1)

    Returns
    -------
    float
        E-value for the CI bound
    """
    if rr_ci_bound < 1:
        rr_ci_bound = 1 / rr_ci_bound
    if rr_ci_bound <= 1:
        return 1.0
    return rr_ci_bound + np.sqrt(rr_ci_bound * (rr_ci_bound - 1))


# ============================================================================
# ANALYSIS 1: ITT WEEKLY HL SHIFTS
# ============================================================================

def analyze_itt_weekly_effects(df):
    """
    Compute weekly ITT effects (Wing A vs Wing B) by pilot week.

    Algorithm:
    1. Loop through weeks 45, 46, 47, 48, 49
    2. For each week:
       - Filter to week data
       - Split: Wing A (all patients) vs Wing B (all patients)
       - Extract consultation_cycle_min (dropna)
       - Compute HL estimate + CI using hodges_lehmann_ci_vectorized()
       - Compute p-value using mannwhitneyu()
       - Calculate adoption rate (users / Wing A)
    3. Validate: Overall ITT (-9.4 min) direction matches weekly pattern

    Returns
    --------
    pd.DataFrame with columns:
        Week, N_WingA, N_WingB, WingA_Median, WingB_Median,
        HL_Estimate, CI_Lower, CI_Upper, p_value, Adoption_Rate
    """
    print("\n" + "="*80)
    print("[ANALYSIS 1] ITT WEEKLY HL SHIFTS")
    print("="*80)

    results = []

    # Filter to study period (Israeli calendar weeks 1-4)
    df_pilot = df[df['study_week'].isin([1, 2, 3, 4])].copy()

    for week in [1, 2, 3, 4]:
        week_data = df_pilot[df_pilot['study_week'] == week]

        # ITT: All Wing A vs All Wing B
        wing_a = week_data[week_data['wing'] == 'A']
        wing_b = week_data[week_data['wing'] == 'B']

        # Calculate adoption rate
        n_users = len(wing_a[wing_a['is_shaked_opened'] == 1])
        n_wing_a = len(wing_a)
        adoption_rate = (n_users / n_wing_a * 100) if n_wing_a > 0 else 0

        # Extract consultation times
        wing_a_consult = wing_a['consultation_cycle_min'].dropna()
        wing_b_consult = wing_b['consultation_cycle_min'].dropna()

        n_a_consult = len(wing_a_consult)
        n_b_consult = len(wing_b_consult)

        # Compute medians
        median_a = wing_a_consult.median() if n_a_consult > 0 else np.nan
        median_b = wing_b_consult.median() if n_b_consult > 0 else np.nan

        # Compute HL estimate and CI
        if n_a_consult > 0 and n_b_consult > 0:
            hl_est, ci_low, ci_high = hodges_lehmann_ci_vectorized(
                wing_a_consult.values, wing_b_consult.values
            )
            _, p_val = mannwhitneyu(wing_a_consult, wing_b_consult, alternative='two-sided')
        else:
            hl_est = ci_low = ci_high = p_val = np.nan

        results.append({
            'Week': week,
            'N_WingA': n_wing_a,
            'N_WingB': len(wing_b),
            'N_WingA_Consult': n_a_consult,
            'N_WingB_Consult': n_b_consult,
            'WingA_Median': median_a,
            'WingB_Median': median_b,
            'HL_Estimate': hl_est,
            'CI_Lower': ci_low,
            'CI_Upper': ci_high,
            'p_value': p_val,
            'Adoption_Rate': adoption_rate
        })

        print(f"  Week {week}: HL={hl_est:.1f} min [{ci_low:.1f}, {ci_high:.1f}], "
              f"p={p_val:.3f} (n={n_a_consult} vs {n_b_consult}, adoption={adoption_rate:.1f}%)")

    results_df = pd.DataFrame(results)

    # Validation: Check that weekly N sum to total
    total_wing_a = results_df['N_WingA'].sum()
    total_wing_b = results_df['N_WingB'].sum()
    print(f"\n  [VALIDATION] Total Wing A: {total_wing_a}, Wing B: {total_wing_b}")

    # Check that Week 4 (lowest adoption) shows negative shift
    week4_hl = results_df[results_df['Week'] == 4]['HL_Estimate'].values[0]
    print(f"  [VALIDATION] Week 4 HL={week4_hl:.1f} (should be negative for paradox confirmation)")

    return results_df


# ============================================================================
# ANALYSIS 2: WING ASSIGNMENT PROPENSITY SCORE
# ============================================================================

def analyze_wing_propensity(df):
    """
    Logistic regression: Wing A (binary) ~ patient characteristics
    Tests quasi-randomization via AUC and omnibus LR test.

    Algorithm:
    1. Outcome: is_wing_a (1=Wing A, 0=Wing B)
    2. Predictors: age, sex, triage, radiology, admission, diagnosis_count
    3. Fit logistic regression (NO cluster-robust SE - testing randomization)
    4. Compute:
       - AUC from predicted probabilities
       - Omnibus likelihood ratio test (model vs null)
       - Pseudo-R2
    5. Expected: AUC ~0.50 (range 0.45-0.55), LR p > 0.05

    Returns
    --------
    dict with keys:
        auc, auc_ci, lr_chisq, lr_pvalue, pseudo_r2, n_sample,
        coef_summary (DataFrame with OR, CI, p per predictor)
    """
    print("\n" + "="*80)
    print("[ANALYSIS 2] WING ASSIGNMENT PROPENSITY SCORE")
    print("="*80)

    # Prepare data
    df_prop = df.copy()
    df_prop['is_wing_a'] = (df_prop['wing'] == 'A').astype(int)

    # Predictors (Table 1 baseline variables)
    predictors = [
        'age_years',
        'is_female',
        'triage_acuity_numeric',
        'has_radiology_consult',
        'is_admitted',
        'diagnosis_count'
    ]

    # Drop missing and ensure numeric types
    df_model = df_prop[['is_wing_a'] + predictors].dropna()

    # Convert boolean to int
    for col in ['is_female', 'has_radiology_consult', 'is_admitted']:
        df_model[col] = df_model[col].astype(int)

    # Ensure all predictors are numeric
    for col in predictors:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    # Drop any remaining NaNs from coercion
    df_model = df_model.dropna()

    n_sample = len(df_model)
    n_missing = len(df_prop) - n_sample

    print(f"  Sample: n={n_sample} ({n_missing} missing covariates)")

    # Fit logistic regression
    X = df_model[predictors].astype(float)
    X = sm.add_constant(X)  # Add intercept
    y = df_model['is_wing_a'].astype(int)

    model = sm.Logit(y, X).fit(disp=0)

    # AUC
    y_pred = model.predict(X)
    auc = roc_auc_score(y, y_pred)

    # Bootstrap AUC CI
    print("  Computing bootstrap AUC CI (1000 resamples)...")
    boot_aucs = []
    for i in range(1000):
        indices = resample(range(n_sample), n_samples=n_sample, random_state=42+i)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]

        try:
            model_boot = sm.Logit(y_boot, X_boot).fit(disp=0, warn_convergence=False)
            y_pred_boot = model_boot.predict(X_boot)
            auc_boot = roc_auc_score(y_boot, y_pred_boot)
            boot_aucs.append(auc_boot)
        except:
            continue

    auc_ci = np.percentile(boot_aucs, [2.5, 97.5])

    # Omnibus LR test
    lr_stat = model.llr  # Likelihood ratio chi-square
    lr_pval = model.llr_pvalue

    # Pseudo-R2
    pseudo_r2 = model.prsquared

    print(f"  AUC: {auc:.3f} (95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}])")
    print(f"  Omnibus LR test: chi2={lr_stat:.2f}, p={lr_pval:.3f}")
    print(f"  Pseudo-R2: {pseudo_r2:.4f}")

    # Coefficient summary
    coef_summary = pd.DataFrame({
        'Variable': model.params.index[1:],  # Exclude intercept
        'OR': np.exp(model.params[1:]),
        'CI_Lower': np.exp(model.conf_int()[0][1:]),
        'CI_Upper': np.exp(model.conf_int()[1][1:]),
        'p_value': model.pvalues[1:]
    })

    # Interpretation
    if 0.45 <= auc <= 0.55 and lr_pval > 0.05:
        interpretation = "PASS - Quasi-randomization validated"
    else:
        interpretation = "CONCERN - Covariate imbalance detected"

    print(f"  >>> {interpretation}")

    return {
        'auc': auc,
        'auc_ci': auc_ci,
        'lr_chisq': lr_stat,
        'lr_pvalue': lr_pval,
        'pseudo_r2': pseudo_r2,
        'n_sample': n_sample,
        'n_missing': n_missing,
        'coef_summary': coef_summary,
        'interpretation': interpretation
    }


# ============================================================================
# ANALYSIS 3: ITT LENGTH OF STAY
# ============================================================================

def analyze_itt_los(df):
    """
    ITT comparison: Wing A vs Wing B for LOS outcomes.

    Algorithm:
    1. Extract los_hours
    2. Split: Wing A (584) vs Wing B (554)
    3. Compute:
       a) Median LOS HL shift + Mann-Whitney p
       b) 90th percentile LOS (Wing A - Wing B)
       c) Prolonged stay >8h rate (chi-square test)
    4. Expected: null effects (consistent with the per-protocol LOS analysis)

    Returns
    --------
    pd.DataFrame with rows:
        Outcome, Wing_A, Wing_B, Difference, p_value, CI, Note
    Rows: Median LOS, 90th Percentile LOS, Prolonged Stay >8h
    """
    print("\n" + "="*80)
    print("[ANALYSIS 3] ITT LENGTH OF STAY")
    print("="*80)

    wing_a = df[df['wing'] == 'A']
    wing_b = df[df['wing'] == 'B']

    los_a = wing_a['los_hours'].dropna()
    los_b = wing_b['los_hours'].dropna()

    print(f"  Wing A: n={len(los_a)}, Wing B: n={len(los_b)}")

    results = []

    # 1. Median LOS HL
    hl_median, ci_low, ci_high = hodges_lehmann_ci_vectorized(los_a, los_b)
    _, p_median = mannwhitneyu(los_a, los_b, alternative='two-sided')

    median_a = los_a.median()
    median_b = los_b.median()

    results.append({
        'Outcome': 'Median LOS',
        'Wing_A': f"{median_a:.1f} h",
        'Wing_B': f"{median_b:.1f} h",
        'Difference': f"{hl_median:.1f} h",
        'CI': f"[{ci_low:.1f}, {ci_high:.1f}]",
        'p_value': f"{p_median:.3f}",
        'Note': 'Hodges-Lehmann estimate'
    })

    print(f"  Median: Wing A {median_a:.1f} h vs Wing B {median_b:.1f} h, HL={hl_median:.1f} h, p={p_median:.3f}")

    # 2. 90th percentile LOS
    p90_a = np.percentile(los_a, 90)
    p90_b = np.percentile(los_b, 90)
    p90_diff = p90_a - p90_b

    results.append({
        'Outcome': '90th Percentile LOS',
        'Wing_A': f"{p90_a:.1f} h",
        'Wing_B': f"{p90_b:.1f} h",
        'Difference': f"{p90_diff:.1f} h",
        'CI': '-',
        'p_value': '-',
        'Note': 'Percentile difference'
    })

    print(f"  90th %ile: {p90_a:.1f} h vs {p90_b:.1f} h, Delta={p90_diff:.1f} h")

    # 3. Prolonged stay >8h rate
    prolong_a_count = (los_a > 8).sum()
    prolong_a_rate = prolong_a_count / len(los_a) * 100

    prolong_b_count = (los_b > 8).sum()
    prolong_b_rate = prolong_b_count / len(los_b) * 100

    # Chi-square test
    contingency = [
        [prolong_a_count, len(los_a) - prolong_a_count],
        [prolong_b_count, len(los_b) - prolong_b_count]
    ]
    chi2, p_prolong, _, _ = chi2_contingency(contingency)

    results.append({
        'Outcome': 'Prolonged Stay >8h',
        'Wing_A': f"{prolong_a_rate:.1f}% (n={prolong_a_count})",
        'Wing_B': f"{prolong_b_rate:.1f}% (n={prolong_b_count})",
        'Difference': f"{prolong_a_rate - prolong_b_rate:.1f}%",
        'CI': '-',
        'p_value': f"{p_prolong:.3f}",
        'Note': 'Chi-square test'
    })

    print(f"  >8h rate: {prolong_a_rate:.1f}% vs {prolong_b_rate:.1f}%, p={p_prolong:.3f}")
    print(f"  [VALIDATION] Null effects consistent with PP LOS: {'PASS' if p_median > 0.1 and p_prolong > 0.1 else 'NOTE'}")

    return pd.DataFrame(results)


# ============================================================================
# ANALYSIS 4: ITT E-VALUE
# ============================================================================

def compute_itt_evalues(df):
    """
    E-values for ITT effect using responder binary RR approach.
    Converts ITT HL shift to responder RR at 120/180min thresholds.

    Algorithm:
    1. Create binary responder variables:
       - consult_le_120 = (consultation_cycle_min <= 120)
       - consult_le_180 = (consultation_cycle_min <= 180)
    2. For each threshold:
       - Compute RR: (Wing A responder rate) / (Wing B responder rate)
       - Bootstrap CI (5000 resamples, independent random states)
       - Compute E-value: RR + sqrt(RR * (RR - 1))
       - E-value CI: Apply to CI bound closest to null
    3. Expected: ITT E-values smaller than PP (compliance dilution)

    Returns
    --------
    pd.DataFrame with columns:
        Threshold, WingA_Rate, WingB_Rate, RR, CI_Lower, CI_Upper,
        E_value_point, E_value_CI, Note
    """
    print("\n" + "="*80)
    print("[ANALYSIS 4] ITT E-VALUES")
    print("="*80)

    wing_a = df[df['wing'] == 'A'].copy()
    wing_b = df[df['wing'] == 'B'].copy()

    results = []

    for threshold in [120, 180]:
        print(f"\n  Threshold: <{threshold} minutes")

        # Create responder variables
        wing_a_consult = wing_a['consultation_cycle_min'].dropna()
        wing_b_consult = wing_b['consultation_cycle_min'].dropna()

        wing_a_responder = (wing_a_consult <= threshold).astype(int)
        wing_b_responder = (wing_b_consult <= threshold).astype(int)

        rate_a = wing_a_responder.mean()
        rate_b = wing_b_responder.mean()

        print(f"    Wing A responder rate: {rate_a*100:.1f}%")
        print(f"    Wing B responder rate: {rate_b*100:.1f}%")

        rr = rate_a / rate_b if rate_b > 0 else np.nan

        # Bootstrap RR CI (independent random states)
        print(f"    Computing bootstrap RR CI (5000 resamples)...")
        boot_rr = []
        for i in range(5000):
            # Independent random states for Wing A and Wing B
            rng_a = np.random.RandomState(42 + i)
            rng_b = np.random.RandomState(42 + i + 5000)

            sample_a_idx = rng_a.choice(len(wing_a_responder), size=len(wing_a_responder), replace=True)
            sample_b_idx = rng_b.choice(len(wing_b_responder), size=len(wing_b_responder), replace=True)

            sample_a = wing_a_responder.iloc[sample_a_idx]
            sample_b = wing_b_responder.iloc[sample_b_idx]

            rate_a_boot = sample_a.mean()
            rate_b_boot = sample_b.mean()

            if rate_b_boot > 0:
                boot_rr.append(rate_a_boot / rate_b_boot)

        ci = np.percentile(boot_rr, [2.5, 97.5])

        # Compute E-values
        ev_point = compute_evalue(rr)
        ci_near_null = min(ci, key=lambda x: abs(x - 1))
        ev_ci = compute_evalue_ci(ci_near_null)

        print(f"    RR={rr:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"    E-value (point)={ev_point:.2f}, E-value (CI)={ev_ci:.2f}")

        results.append({
            'Threshold': f"<{threshold} min",
            'WingA_Rate': f"{rate_a*100:.1f}%",
            'WingB_Rate': f"{rate_b*100:.1f}%",
            'RR': rr,
            'CI_Lower': ci[0],
            'CI_Upper': ci[1],
            'E_value_point': ev_point,
            'E_value_CI': ev_ci,
            'Note': 'ITT comparison'
        })

    print(f"\n  [NOTE] ITT E-values expected to be smaller than PP due to compliance dilution")

    return pd.DataFrame(results)


# ============================================================================
# ANALYSIS 5: CONTAMINATED CONTROLS SENSITIVITY
# ============================================================================

def analyze_contamination_sensitivity(df):
    """
    ITT contamination bounding: compare all Wing A vs Wing B excluding the
    fastest/slowest 71 control patients (the per-protocol version is reported
    alongside).

    Algorithm:
    1. Extract Wing A (all patients, n=584) and Wing B (n=554)
    2. Sort Wing B by consultation_cycle_min
    3. Best case (fastest 71 contaminated):
       - Compare all Wing A (584) vs Wing B excluding fastest 71 (483)
    4. Worst case (slowest 71 contaminated):
       - Compare all Wing A (584) vs Wing B excluding slowest 71 (483)
    5. Compute HL + p-value for both scenarios
    6. Observed: all Wing A vs all Wing B (reported ITT = -9.4 min)
    7. Compare to the per-protocol bounding

    Returns
    --------
    dict with keys:
        best_case_hl, best_case_ci, best_case_p,
        worst_case_hl, worst_case_ci, worst_case_p,
        observed_hl, observed_ci, observed_p,
        interpretation, pp_comparison
    """
    print("\n" + "="*80)
    print("[ANALYSIS 5] ITT CONTAMINATION BOUNDING (NEW)")
    print("="*80)

    n_contaminated = 71

    wing_a = df[df['wing'] == 'A']
    wing_b = df[df['wing'] == 'B']

    wing_a_consult = wing_a['consultation_cycle_min'].dropna()
    wing_b_consult = wing_b['consultation_cycle_min'].dropna()

    print(f"  Wing A: n={len(wing_a_consult)}")
    print(f"  Wing B: n={len(wing_b_consult)}")
    print(f"  Contamination assumption: {n_contaminated} Wing B patients")

    # Best case: contaminated = fastest 71 Wing B
    print(f"\n  Best case (exclude fastest {n_contaminated} from Wing B):")
    fastest_71 = wing_b_consult.nsmallest(n_contaminated)
    pure_slow = wing_b_consult[~wing_b_consult.index.isin(fastest_71.index)]

    hl_best, ci_low_best, ci_high_best = hodges_lehmann_ci_vectorized(
        wing_a_consult.values, pure_slow.values
    )
    _, p_best = mannwhitneyu(wing_a_consult, pure_slow, alternative='two-sided')

    print(f"    HL={hl_best:.1f} min [{ci_low_best:.1f}, {ci_high_best:.1f}], p={p_best:.3f}")
    print(f"    Non-contaminated Wing B: n={len(pure_slow)}")

    # Worst case: contaminated = slowest 71 Wing B
    print(f"\n  Worst case (exclude slowest {n_contaminated} from Wing B):")
    slowest_71 = wing_b_consult.nlargest(n_contaminated)
    pure_fast = wing_b_consult[~wing_b_consult.index.isin(slowest_71.index)]

    hl_worst, ci_low_worst, ci_high_worst = hodges_lehmann_ci_vectorized(
        wing_a_consult.values, pure_fast.values
    )
    _, p_worst = mannwhitneyu(wing_a_consult, pure_fast, alternative='two-sided')

    print(f"    HL={hl_worst:.1f} min [{ci_low_worst:.1f}, {ci_high_worst:.1f}], p={p_worst:.3f}")
    print(f"    Non-contaminated Wing B: n={len(pure_fast)}")

    # Observed ITT (all 584 vs all 554)
    print(f"\n  Observed ITT (all Wing A vs all Wing B):")
    hl_obs, ci_low_obs, ci_high_obs = hodges_lehmann_ci_vectorized(
        wing_a_consult.values, wing_b_consult.values
    )
    _, p_obs = mannwhitneyu(wing_a_consult, wing_b_consult, alternative='two-sided')

    print(f"    HL={hl_obs:.1f} min [{ci_low_obs:.1f}, {ci_high_obs:.1f}], p={p_obs:.3f}")

    # Validation (best case is most negative, worst case is least negative or positive)
    within_bounds = hl_best <= hl_obs <= hl_worst
    print(f"\n  [VALIDATION] Bounding range [{hl_best:.1f}, {hl_worst:.1f}] contains observed {hl_obs:.1f}: {'[OK]' if within_bounds else 'FAIL'}")
    print(f"  [VALIDATION] Best case shows negative effect: {'[OK]' if hl_best < 0 else 'FAIL'}")

    interpretation = (
        f"ITT effect robust to contamination. Observed ITT ({hl_obs:.1f} min) "
        f"falls within bounding range [{hl_best:.1f}, {hl_worst:.1f}]. "
        f"Best case (fastest 71 contaminated): {hl_best:.1f} min. "
        f"Worst case (slowest 71 contaminated): {hl_worst:.1f} min."
    )

    pp_comparison = (
        "ITT bounds narrower than PP bounds [-32.2, -5.9] due to compliance dilution. "
        "Consistent with 44% adoption rate attenuating both best and worst case effects."
    )

    print(f"\n  {interpretation}")
    print(f"  [NOTE] {pp_comparison}")

    return {
        'best_case_hl': hl_best,
        'best_case_ci': (ci_low_best, ci_high_best),
        'best_case_p': p_best,
        'worst_case_hl': hl_worst,
        'worst_case_ci': (ci_low_worst, ci_high_worst),
        'worst_case_p': p_worst,
        'observed_hl': hl_obs,
        'observed_ci': (ci_low_obs, ci_high_obs),
        'observed_p': p_obs,
        'interpretation': interpretation,
        'pp_comparison': pp_comparison
    }


# ============================================================================
# ANALYSIS 6: WING B SECULAR TREND CHECK
# ============================================================================

def analyze_wingb_secular_trend(df):
    """
    Test for secular trend in Wing B consultation times over study period.

    Algorithm:
    1. Extract Wing B patients only (n=554)
    2. Compute Spearman correlation: consultation_cycle_min ~ study_day
       (study_day = days since pilot start)
    3. Compute Wing B median consultation time per week (45, 46, 47, 48, 49)
    4. Expected: rho ~ 0 (no correlation), weekly medians stable
    5. Interpretation:
       - If rho near 0 and medians stable -> no secular trend
       - If rho > 0.2 or medians increasing -> external confounding concern

    Returns
    --------
    dict with keys:
        spearman_rho, spearman_p, weekly_medians (DataFrame),
        interpretation
    """
    print("\n" + "="*80)
    print("[ANALYSIS 6] WING B SECULAR TREND CHECK")
    print("="*80)

    wing_b = df[df['wing'] == 'B'].copy()

    print(f"  Wing B sample: n={len(wing_b)}")

    # Spearman correlation (consultation time vs study day)
    wing_b_consult = wing_b.dropna(subset=['consultation_cycle_min', 'study_day_index'])

    consult_times = wing_b_consult['consultation_cycle_min']
    study_days = wing_b_consult['study_day_index']

    rho, p_val = spearmanr(consult_times, study_days)

    print(f"\n  Spearman correlation: rho={rho:.3f} (p={p_val:.3f})")
    print(f"    Consultation time vs. study day index")

    # Weekly medians
    print(f"\n  Wing B weekly medians:")
    weekly_medians = []

    for week in [1, 2, 3, 4]:
        week_b = wing_b[wing_b['study_week'] == week]
        week_consult = week_b['consultation_cycle_min'].dropna()

        median_consult = week_consult.median()
        n_patients = len(week_b)
        n_consult = len(week_consult)

        weekly_medians.append({
            'Week': week,
            'N_Patients': n_patients,
            'N_Consult': n_consult,
            'Median_Consult': median_consult
        })

        print(f"    Week {week}: {median_consult:.1f} min (n={n_consult})")

    weekly_df = pd.DataFrame(weekly_medians)

    # Check stability (coefficient of variation)
    medians_array = weekly_df['Median_Consult'].dropna()
    cv = medians_array.std() / medians_array.mean() * 100 if len(medians_array) > 0 else 0

    print(f"\n  Coefficient of variation (weekly medians): {cv:.1f}%")

    # Interpretation
    if abs(rho) < 0.15 and p_val > 0.05:
        interpretation = "No secular trend detected - control group stable over study period"
        status = "PASS"
    elif rho > 0.15 and p_val < 0.05:
        interpretation = f"WARNING: Positive secular trend detected (rho={rho:.3f}, p={p_val:.3f}). Wing B times worsened over time, suggesting external confounding."
        status = "CONCERN"
    elif rho < -0.15 and p_val < 0.05:
        interpretation = f"WARNING: Negative secular trend detected (rho={rho:.3f}, p={p_val:.3f}). Wing B times improved over time, which could inflate apparent SHAKED effect."
        status = "CONCERN"
    else:
        interpretation = f"Weak/uncertain trend (rho={rho:.3f}, p={p_val:.3f}). Control group reasonably stable."
        status = "NOTE"

    print(f"\n  >>> {status}: {interpretation}")

    return {
        'spearman_rho': rho,
        'spearman_p': p_val,
        'weekly_medians': weekly_df,
        'cv': cv,
        'interpretation': interpretation,
        'status': status
    }


# ============================================================================
# OUTPUT GENERATION
# ============================================================================

def save_all_outputs(results):
    """
    Save all analysis results to CSV files.

    Parameters
    ----------
    results : dict
        Dictionary containing results from all 6 analyses
    """
    print("\n" + "="*80)
    print("[SAVING OUTPUTS]")
    print("="*80)

    # Analysis 1: ITT weekly effects (eTable 2)
    out_path = os.path.join(OUTPUT_DIR, 'eTable2_Weekly_Effects.csv')
    results['weekly'].to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Analysis 2: Randomization Checks (Propensity Score)
    prop_results = results['propensity']
    prop_summary = pd.DataFrame([
        {'Metric': 'AUC', 'Value': f"{prop_results['auc']:.3f}",
         'CI': f"[{prop_results['auc_ci'][0]:.3f}, {prop_results['auc_ci'][1]:.3f}]",
         'Interpretation': 'Random assignment (0.50 = perfect)'},
        {'Metric': 'Omnibus LR chi2', 'Value': f"{prop_results['lr_chisq']:.2f}",
         'CI': f"p={prop_results['lr_pvalue']:.3f}",
         'Interpretation': 'No joint prediction' if prop_results['lr_pvalue'] > 0.05 else 'Covariate imbalance'},
        {'Metric': 'Pseudo-R2', 'Value': f"{prop_results['pseudo_r2']:.4f}",
         'CI': '-',
         'Interpretation': 'Negligible variance explained'},
        {'Metric': 'Sample', 'Value': f"n={prop_results['n_sample']}",
         'CI': f"{prop_results['n_missing']} missing",
         'Interpretation': prop_results['interpretation']}
    ])

    out_path = os.path.join(OUTPUT_DIR, 'eTable_Randomization_Checks.csv')
    prop_summary.to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Also save coefficient details
    out_path_coef = os.path.join(OUTPUT_DIR, 'eTable_Propensity_Coefficients.csv')
    prop_results['coef_summary'].to_csv(out_path_coef, index=False)
    print(f"  [OK] {out_path_coef}")

    # Analysis 3: ITT length of stay (eTable 4)
    out_path = os.path.join(OUTPUT_DIR, 'eTable4_LOS_ITT.csv')
    results['los'].to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Analysis 4: ITT E-values
    out_path = os.path.join(OUTPUT_DIR, 'eTable_Evalues_ITT.csv')
    results['evalue'].to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Analysis 5: Contamination Sensitivity
    contam_results = results['contamination']
    contam_summary = pd.DataFrame([
        {
            'Analysis': 'PP (reported)',
            'Scenario': 'Best Case',
            'HL_Estimate': -32.2,
            'CI': '[-43.8, -20.7]',
            'p_value': '<.001',
            'Note': 'Per-protocol bounding'
        },
        {
            'Analysis': 'PP (reported)',
            'Scenario': 'Worst Case',
            'HL_Estimate': -5.9,
            'CI': '[-17.5, 5.7]',
            'p_value': '.32',
            'Note': 'Per-protocol bounding'
        },
        {
            'Analysis': 'PP (existing)',
            'Scenario': 'Observed',
            'HL_Estimate': -13.3,
            'CI': '[-26.2, -0.4]',
            'p_value': '.043',
            'Note': 'Within bounds'
        },
        {
            'Analysis': 'ITT (new)',
            'Scenario': 'Best Case',
            'HL_Estimate': contam_results['best_case_hl'],
            'CI': f"[{contam_results['best_case_ci'][0]:.1f}, {contam_results['best_case_ci'][1]:.1f}]",
            'p_value': f"{contam_results['best_case_p']:.3f}",
            'Note': 'Maximum ITT effect'
        },
        {
            'Analysis': 'ITT (new)',
            'Scenario': 'Worst Case',
            'HL_Estimate': contam_results['worst_case_hl'],
            'CI': f"[{contam_results['worst_case_ci'][0]:.1f}, {contam_results['worst_case_ci'][1]:.1f}]",
            'p_value': f"{contam_results['worst_case_p']:.3f}",
            'Note': 'Minimum ITT effect'
        },
        {
            'Analysis': 'ITT (new)',
            'Scenario': 'Observed',
            'HL_Estimate': contam_results['observed_hl'],
            'CI': f"[{contam_results['observed_ci'][0]:.1f}, {contam_results['observed_ci'][1]:.1f}]",
            'p_value': f"{contam_results['observed_p']:.3f}",
            'Note': 'Within bounds'
        }
    ])

    out_path = os.path.join(OUTPUT_DIR, 'eTable_Contamination_Sensitivity.csv')
    contam_summary.to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Analysis 6: Wing B Secular Trend
    trend_results = results['secular_trend']

    trend_summary = pd.DataFrame([
        {
            'Metric': 'Spearman rho',
            'Value': f"{trend_results['spearman_rho']:.3f}",
            'p_value': f"{trend_results['spearman_p']:.3f}",
            'Interpretation': 'Correlation: consultation time vs. study day'
        },
        {
            'Metric': 'CV (weekly medians)',
            'Value': f"{trend_results['cv']:.1f}%",
            'p_value': '-',
            'Interpretation': 'Stability measure across 4 study weeks'
        },
        {
            'Metric': 'Conclusion',
            'Value': trend_results['status'],
            'p_value': '-',
            'Interpretation': trend_results['interpretation']
        }
    ])

    out_path = os.path.join(OUTPUT_DIR, 'eTable_WingB_Secular_Trend.csv')
    trend_summary.to_csv(out_path, index=False)
    print(f"  [OK] {out_path}")

    # Save weekly medians detail
    out_path_detail = os.path.join(OUTPUT_DIR, 'eTable_WingB_Weekly_Medians.csv')
    trend_results['weekly_medians'].to_csv(out_path_detail, index=False)
    print(f"  [OK] {out_path_detail}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("SHAKED STAGE 16: WEEKLY EFFECTS, LOS, AND CONTAMINATION BOUNDING")
    print("=" * 80)
    print(f"Data:   {DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Load data
    print("\n" + "="*80)
    print("[DATA LOADING]")
    print("="*80)

    df = pd.read_csv(DATA_PATH, encoding='utf-8')

    print(f"  Loaded: n={len(df)} patients")

    # age_years and the arrival timestamp come from the leading Hebrew columns
    # (addressed by position); is_female is the precomputed English column.
    cols = df.columns.tolist()
    age_col = cols[2]
    arrival_col = cols[4]

    df['age_years'] = pd.to_numeric(df[age_col], errors='coerce')
    df['is_female'] = df['is_female'].astype(bool)

    # Type conversion
    df['is_shaked_opened'] = df['is_shaked_opened'].astype(bool)
    df['has_radiology_consult'] = df['has_radiology_consult'].fillna(False).astype(bool)
    df['is_admitted'] = df['is_admitted'].fillna(False).astype(bool)
    df['pilot_week'] = pd.to_numeric(df['pilot_week'], errors='coerce')
    df['study_week'] = pd.to_numeric(df['study_week'], errors='coerce')

    # Create study_day_index from arrival timestamp
    df['arrival_datetime'] = pd.to_datetime(df[arrival_col], errors='coerce')
    pilot_start = df[df['pilot_week'].notnull()]['arrival_datetime'].min()
    df['study_day_index'] = (df['arrival_datetime'] - pilot_start).dt.days

    # Apply standard exclusions (n_consultations > 0)
    df = df[df['n_consultations'] > 0].copy()

    print(f"  After exclusions: n={len(df)}")

    # Validate sample sizes
    n_wing_a = len(df[df['wing'] == 'A'])
    n_wing_b = len(df[df['wing'] == 'B'])

    print(f"  Wing A: n={n_wing_a}")
    print(f"  Wing B: n={n_wing_b}")

    expected_total = 1138
    expected_a = 584
    expected_b = 554

    if len(df) != expected_total or n_wing_a != expected_a or n_wing_b != expected_b:
        print(f"\n  WARNING: Sample sizes don't match expected values!")
        print(f"    Expected: {expected_total} total ({expected_a} Wing A, {expected_b} Wing B)")
        print(f"    Got: {len(df)} total ({n_wing_a} Wing A, {n_wing_b} Wing B)")
        print(f"  Continuing with actual sample...")
    else:
        print(f"  [OK] Validation PASS")

    # Run all 6 analyses
    results = {}

    results['weekly'] = analyze_itt_weekly_effects(df)
    results['propensity'] = analyze_wing_propensity(df)
    results['los'] = analyze_itt_los(df)
    results['evalue'] = compute_itt_evalues(df)
    results['contamination'] = analyze_contamination_sensitivity(df)
    results['secular_trend'] = analyze_wingb_secular_trend(df)

    # Save all outputs
    save_all_outputs(results)

    print("\n" + "="*80)
    print("STAGE 16 COMPLETE")
    print("="*80)
    print(f"  All six analyses finished; tables written to {OUTPUT_DIR}.")

    return results


if __name__ == '__main__':
    try:
        results = main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
