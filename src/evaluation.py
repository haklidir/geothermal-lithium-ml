"""
evaluation.py — Metrics and uncertainty quantification utilities.

Implements:
- Point metrics (RMSE, MAE, R², MAPE).
- Empirical prediction intervals sized from cross-validation residuals
  (Section 3.5). Inspired by split-conformal prediction but does not
  satisfy the exchangeability condition — see manuscript discussion.
- Paired bootstrap test for model comparison (Section 4.2).
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def scores(y_true, y_hat):
    """Return (RMSE, MAE, R²)."""
    return (
        np.sqrt(mean_squared_error(y_true, y_hat)),
        mean_absolute_error(y_true, y_hat),
        r2_score(y_true, y_hat),
    )


def full_scores(y_true, y_hat):
    """Return (RMSE, MAE, R², MAPE)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mae = mean_absolute_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)
    mape = np.mean(np.abs((y_true - y_hat) / np.maximum(y_true, 1e-6))) * 100
    return rmse, mae, r2, mape


def empirical_pi_stats(y_true, y_hat, q):
    """Compute empirical prediction interval statistics for half-width q.

    Parameters
    ----------
    y_true, y_hat : array-like
        Observed and predicted values.
    q : float
        Half-width of the symmetric interval.

    Returns
    -------
    (lo, hi, coverage, mean_width)
    """
    lo, hi = y_hat - q, y_hat + q
    coverage = np.mean((y_true >= lo) & (y_true <= hi))
    mean_width = np.mean(hi - lo)
    return lo, hi, coverage, mean_width


# Backward-compatible alias used in notebook
cp_stats = empirical_pi_stats


def paired_bootstrap(abs_err_a, abs_err_b, n_iter=10000, seed=42):
    """Paired bootstrap test for MAE difference: model B minus model A.

    A positive difference indicates A has lower MAE (i.e. A is better).

    Returns
    -------
    dict with keys: observed_diff, ci_lo, ci_hi, p_value
    """
    rng = np.random.RandomState(seed)
    n = len(abs_err_a)
    obs = abs_err_b.mean() - abs_err_a.mean()
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx_a = rng.choice(n, n, replace=True)
        idx_b = rng.choice(n, n, replace=True)
        diffs[i] = abs_err_b[idx_b].mean() - abs_err_a[idx_a].mean()
    p_val = np.mean(diffs <= 0)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return dict(observed_diff=obs, ci_lo=ci_lo, ci_hi=ci_hi, p_value=p_val)


def clopper_pearson_ci(k, n, alpha=0.05):
    """Exact binomial confidence interval for empirical coverage rate.

    Used in Section 4.2 to qualify the 91.7-100% empirical coverage on
    the small test set (n=12).
    """
    from scipy import stats
    if k == 0:
        ci_lo = 0.0
    else:
        ci_lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        ci_hi = 1.0
    else:
        ci_hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return ci_lo, ci_hi
