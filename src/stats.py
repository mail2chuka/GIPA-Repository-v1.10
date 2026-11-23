"""
stats.py
========
Seed-level statistics used throughout the paper: percentile-bootstrap
confidence intervals, paired differences, two one-sided tests (TOST) for
equivalence and one-sided Clopper-Pearson upper bounds for miss rates.

STATUS: rebuilt for the first revision of manuscript ITS-2026-08-0579. The
original statistics script behind ``results/res_main_statistics.json`` was
not in the repository; ``experiments/stats_main.py`` uses these functions to
regenerate every value of that file (byte-identical under the pinned NumPy
2.4.4 / SciPy 1.17.1), which pins down every convention below.

Conventions (as in the archived analysis)
-----------------------------------------
* **Bootstrap.** Percentile bootstrap over seeds: 10,000 resamples, 95%
  interval from ``np.percentile`` at 2.5 and 97.5 (linear interpolation).
  Resample indices are drawn as ``rng.integers(0, n, (reps, n))`` from ONE
  generator (``np.random.default_rng(20260804)``) that is shared, in a fixed
  order, by every interval of an analysis. Values that are NaN (e.g. Jain
  fairness when no RSU received a task) are dropped first; n is the number
  of remaining values. If none remain, the interval is (None, None). If all
  remaining values are equal, the interval is [mean, mean] (the floating
  mean, as archived) and no random numbers are drawn.
* **Paired comparisons.** The per-seed difference a - b (same environment
  and action seeds), NaN pairs dropped, then the bootstrap above.
* **TOST.** Paired t-based two one-sided tests with a stated margin:
  90% CI mean +/- t_{0.95, n-1} * sd / sqrt(n) (sd with ddof = 1);
  p_lower = P(T > (mean + margin) / se), p_upper = P(T < (mean - margin) / se);
  equivalence if both p < 0.05.
* **Zero-miss bounds.** One-sided 95% Clopper-Pearson upper bound on a
  binomial proportion: Beta(0.95; x + 1, n - x), and 1 when x = n.
"""

import numpy as np
from scipy import stats as sps

BOOT_SEED = 20260804
BOOT_REPS = 10000


class Bootstrap:
    """Percentile bootstrap that shares one generator across calls (see module docstring)."""

    def __init__(self, seed=BOOT_SEED, reps=BOOT_REPS, level=0.95):
        self.rng = np.random.default_rng(seed)
        self.reps = reps
        self.level = level

    def ci(self, values):
        """(mean, [lo, hi], n) for the mean of ``values`` (NaN dropped)."""
        v = np.asarray(values, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n == 0:
            return None, [None, None], 0
        mean = float(np.mean(v))
        if np.all(v == v[0]):
            return mean, [mean, mean], n
        idx = self.rng.integers(0, n, (self.reps, n))
        a = (1 - self.level) / 2 * 100
        lo, hi = np.percentile(v[idx].mean(1), [a, 100 - a])
        return mean, [float(lo), float(hi)], n

    def summary(self, values):
        """Dict with mean, ci95 and n."""
        m, ci, n = self.ci(values)
        return dict(mean=m, ci95=ci, n=n)

    def paired(self, a, b):
        """Dict with mean_difference, ci95 and n for a - b (NaN pairs dropped)."""
        m, ci, n = self.ci(np.asarray(a, float) - np.asarray(b, float))
        return dict(mean_difference=m, ci95=ci, n=n)


def tost_paired(a, b, margin, alpha=0.05):
    """Paired two one-sided tests for |mean(a - b)| < margin."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    n = len(d)
    mean = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    tq = sps.t.ppf(1 - alpha, n - 1)
    p_lower = float(sps.t.sf((mean + margin) / se, n - 1))
    p_upper = float(sps.t.cdf((mean - margin) / se, n - 1))
    return dict(n=n, margin=float(margin), mean_difference=mean, sd_difference=sd,
                ci90=[float(mean - tq * se), float(mean + tq * se)],
                p_lower=p_lower, p_upper=p_upper, equivalent=bool(max(p_lower, p_upper) < alpha))


def cp_upper(x, n, level=0.95):
    """One-sided Clopper-Pearson upper bound for x events in n trials."""
    if x >= n:
        return 1.0
    return float(sps.beta.ppf(level, x + 1, n - x))
