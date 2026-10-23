"""Classical inference for the mean of d. Owner: Ma Tianjiao (B1).

Independence-based baseline for the mean daily excess return d, plus helpers for
checking how autocorrelation changes Var(d_bar).

Used by:
    notebooks/02_classical_baseline.ipynb   (Ma)
    notebooks/03_block_bootstrap.ipynb      (Deng, side-by-side table)
    notebooks/04_monte_carlo_coverage.ipynb (Ma)
    notebooks/05_robustness.ipynb           (Ma)
"""
import numpy as np
import pandas as pd
from scipy import stats

DAYS_PER_YEAR = 365   # BTC trades every calendar day


def t_test_mean(d, level: float = 0.95) -> dict:
    """One-sample t-test of H0: mu_d = 0 against H1: mu_d != 0.

    Assumes d_t are independent, identically distributed, with finite variance.

    Returns a dict with:
        n, mean, sd (ddof=1), se = sd / sqrt(n), t, df, p (two-sided),
        t_lo, t_hi     : CI using the t distribution with n - 1 df
        clt_lo, clt_hi : CI using the normal critical value (1.96 at 95%)
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = d.size
    if n < 2:
        raise ValueError("need at least two observations")

    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    t_stat = mean / se
    p = 2 * stats.t.sf(abs(t_stat), df)

    alpha = 1 - level
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return {
        "n": n, "mean": mean, "sd": sd, "se": se,
        "t": t_stat, "df": df, "p": p, "level": level,
        "t_lo": mean - t_crit * se, "t_hi": mean + t_crit * se,
        "clt_lo": mean - z_crit * se, "clt_hi": mean + z_crit * se,
    }


def annualize(res: dict, days: int = DAYS_PER_YEAR) -> dict:
    """Scale mean, SE and interval endpoints by `days` (log returns add up).

    t and p do not change, so they are copied as is.
    """
    out = dict(res)
    for key in ("mean", "se", "t_lo", "t_hi", "clt_lo", "clt_hi"):
        out[key] = res[key] * days
    out["sd"] = res["sd"] * np.sqrt(days)
    return out


def sample_acf(x, max_lag: int) -> np.ndarray:
    """Sample autocorrelations rho_1 ... rho_max_lag (same estimator as statsmodels.acf)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:-k]) / denom for k in range(1, max_lag + 1)])


def variance_factor(d, max_lag: int, bartlett: bool = True) -> float:
    """Estimate the factor in Var(d_bar) ~= (sigma^2 / n) * (1 + 2 * sum_k w_k * rho_k).

    The t-test assumes this factor equals 1. A factor below 1 means the t-test SE
    is too large (conservative); above 1 means it is too small (over-confident).
    Bartlett weights w_k = 1 - k / (max_lag + 1) keep the estimate positive.
    """
    rho = sample_acf(d, max_lag)
    k = np.arange(1, max_lag + 1)
    w = 1 - k / (max_lag + 1) if bartlett else np.ones(max_lag)
    return 1 + 2 * np.sum(w * rho)


def ar1_t_coverage(phi: float, level: float = 0.95) -> float:
    """Large-sample coverage of the nominal t interval when d follows AR(1) with coefficient phi.

    For AR(1), 1 + 2 * sum rho_k = (1 + phi) / (1 - phi), so the true SE is
    sqrt((1 + phi) / (1 - phi)) times the i.i.d. SE.
    """
    z = stats.norm.ppf(1 - (1 - level) / 2)
    ratio = np.sqrt((1 + phi) / (1 - phi))
    return 2 * stats.norm.cdf(z / ratio) - 1


def result_row(label: str, res: dict, keys=("n", "mean", "se", "t", "p", "t_lo", "t_hi", "clt_lo", "clt_hi")) -> pd.Series:
    """Turn a t_test_mean dict into a labelled row for a summary table."""
    return pd.Series({k: res[k] for k in keys}, name=label)
