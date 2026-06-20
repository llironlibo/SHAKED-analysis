"""
Shared utilities for the SHAKED analysis scripts.

Provides the canonical data loader, the Hodges-Lehmann estimator with a
rank-based confidence interval, and standardized-mean-difference helpers.
Every analysis stage imports from this module (`from _shared import ...`).

Inputs : data/cleaned/unified_pilot_cohort.csv (controlled access; not shipped)
Outputs: none (library module)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# Repo root is the directory that holds this module; the de-identified analytic
# cohort lives under data/cleaned/. Resolving from the module's own location
# keeps the path portable on any machine after a fresh clone.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'cleaned', 'unified_pilot_cohort.csv')


def load_analysis_data(apply_exclusion=True):
    """Load the unified cohort, optionally apply the consultation exclusion,
    and assert the published wing counts (584 / 554)."""
    df = pd.read_csv(DATA_PATH)
    if apply_exclusion:
        df = df[df['n_consultations'] > 0]
        assert len(df) == 1138, f"Expected 1138 rows, got {len(df)}"
        assert (df['wing'] == 'A').sum() == 584, "Wing A count mismatch"
        assert (df['wing'] == 'B').sum() == 554, "Wing B count mismatch"
    return df


def hodges_lehmann_ci(x, y, alpha=0.05):
    """Hodges-Lehmann estimator with a rank-based CI and a Mann-Whitney p-value.

    Returns a dict with keys: hl, ci_lower, ci_upper, p_value.
    Sign convention: median(x - y); negative means group x is faster.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    pairwise = np.subtract.outer(x, y).ravel()
    hl = float(np.median(pairwise))

    n1, n2 = len(x), len(y)
    N = n1 * n2
    z = stats.norm.ppf(1 - alpha / 2)
    combined = np.concatenate([x, y])
    ranks = stats.rankdata(combined)
    rank_x = ranks[:n1]
    var_U = (n1 * n2 / 12.0) * ((n1 + n2 + 1)
             - np.sum((np.unique(combined, return_counts=True)[1] ** 3
                       - np.unique(combined, return_counts=True)[1]))
             / ((n1 + n2) * (n1 + n2 - 1)))
    se = np.sqrt(var_U) / np.sqrt(N) * np.sqrt(N)
    margin = z * se / np.sqrt(N) * 2

    sorted_pw = np.sort(pairwise)
    K = int(np.round(N / 2 - z * np.sqrt(var_U)))
    K = max(0, min(K, N - 1))
    ci_lower = float(sorted_pw[K])
    ci_upper = float(sorted_pw[N - 1 - K])

    stat_u, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
    return {'hl': hl, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'p_value': float(p_value)}


def smd_continuous(x1, x2):
    """Standardized mean difference for continuous variables."""
    m1, m2 = np.nanmean(x1), np.nanmean(x2)
    s1, s2 = np.nanstd(x1, ddof=1), np.nanstd(x2, ddof=1)
    pooled = np.sqrt((s1**2 + s2**2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0


def smd_binary(p1, p2):
    """Standardized mean difference for binary variables."""
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else 0.0
