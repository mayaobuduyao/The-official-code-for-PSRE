"""Statistics utilities: bootstrap CIs, two-proportion z-test (paper Sec.
Metrics)."""
from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple


def bootstrap_ci(samples: List[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) of a (1-alpha) bootstrap percentile CI."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = arr.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def two_proportion_ztest(s1: int, n1: int, s2: int, n2: int) -> Tuple[float, float]:
    """Two-proportion z-test. Returns (z, two-sided p)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided p via normal CDF
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_val


def paired_t_log_latency(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test on log-transformed latencies (paper Sec. Metrics)."""
    la = np.log(np.maximum(np.asarray(a, float), 1e-9))
    lb = np.log(np.maximum(np.asarray(b, float), 1e-9))
    d = la - lb
    n = d.size
    if n < 2:
        return 0.0, 1.0
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0, 1.0
    t = mean / (sd / math.sqrt(n))
    # approximate two-sided p with normal for large n
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(t), float(p)