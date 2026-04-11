from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def mann_whitney_u_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(u_stat), float(p_value)


def pvalue_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"
