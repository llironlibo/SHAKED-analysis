"""
Stage 03 -- SAGER sex-disaggregated intention-to-treat analysis.

Reports the consultation-cycle-time effect (Wing A vs Wing B) separately for
female and male patients, plus a formal sex-by-allocation interaction test, in
line with SAGER (Sex And Gender Equity in Research) reporting guidance.

Reproduces: sex-disaggregated primary ITT (Round-2 addition).
  Female ~ -11.5 min (p~0.14); Male ~ -8.3 min (p~0.25);
  sex-by-allocation interaction p ~ 0.88.

Inputs : data/cleaned/unified_pilot_cohort.csv
Outputs: results/eTable_SAGER_Sex_Subgroup_ITT.csv

Conventions match the other stages:
  - Cohort loaded via _shared.load_analysis_data (n_consultations>0 exclusion,
    584/554 wing split asserted).
  - Subgroup effect = Hodges-Lehmann estimator with rank-based 95% CI and a
    Mann-Whitney two-sided p-value (_shared.hodges_lehmann_ci).
  - HL sign convention: median(Wing A - Wing B); negative => intervention faster.
  - Interaction tested with OLS on consultation_cycle_min and cluster-robust
    standard errors clustered on arrival_date.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from _shared import load_analysis_data, hodges_lehmann_ci

OUTCOME = 'consultation_cycle_min'

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')


def subgroup_itt(df, label):
    """Wing A vs Wing B Hodges-Lehmann ITT effect within one subgroup."""
    a = df[df['wing'] == 'A'][OUTCOME].dropna().values
    b = df[df['wing'] == 'B'][OUTCOME].dropna().values
    res = hodges_lehmann_ci(a, b)
    print(f"  {label}: n(A)={len(a)}, n(B)={len(b)}, "
          f"HL={res['hl']:.1f} min [{res['ci_lower']:.1f}, {res['ci_upper']:.1f}], "
          f"p={res['p_value']:.3f}")
    return {
        'Subgroup': label,
        'N_WingA': len(a),
        'N_WingB': len(b),
        'HL_Estimate_min': round(res['hl'], 1),
        'CI_Lower': round(res['ci_lower'], 1),
        'CI_Upper': round(res['ci_upper'], 1),
        'p_value': round(res['p_value'], 3),
    }


def main():
    print("=" * 80)
    print("SHAKED STAGE 03: SAGER SEX-DISAGGREGATED ITT")
    print("=" * 80)

    df = load_analysis_data()
    df = df[df[OUTCOME].notna()].copy()
    df['is_wing_a'] = (df['wing'] == 'A').astype(int)
    df['is_female'] = df['is_female'].astype(int)

    female = df[df['is_female'] == 1]
    male = df[df['is_female'] == 0]

    print("\n[Sex-disaggregated ITT effects]")
    rows = [
        subgroup_itt(female, 'Female'),
        subgroup_itt(male, 'Male'),
    ]

    # Sex-by-allocation interaction: does the Wing A vs Wing B effect on
    # consultation_cycle_min differ by sex? Cluster-robust SE on arrival_date.
    print("\n[Sex x allocation interaction test]")
    model = smf.ols(f'{OUTCOME} ~ is_wing_a * is_female', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['arrival_date']}
    )
    term = 'is_wing_a:is_female'
    beta = model.params[term]
    ci = model.conf_int().loc[term]
    p_int = model.pvalues[term]
    print(f"  Interaction (is_wing_a x is_female): beta={beta:.2f} min "
          f"[{ci[0]:.2f}, {ci[1]:.2f}], p={p_int:.3f}")
    print(f"  Interpretation: {'No significant heterogeneity by sex' if p_int > 0.05 else 'Significant sex heterogeneity'}")

    summary = pd.DataFrame(rows)
    print("\n[Summary]")
    print(summary.to_string(index=False))
    print(f"\n  Sex-by-allocation interaction p = {p_int:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'eTable_SAGER_Sex_Subgroup_ITT.csv')
    summary.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n[Saved] {out_path}")


if __name__ == '__main__':
    main()
