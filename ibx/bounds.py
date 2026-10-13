"""Identified sets for the probability of necessity (PN) of a binary cause.

Tian and Pearl (2000): with observational and experimental data from ONE population,
    max{0, [P(y) - P(y_x')] / P(x, y)}  <=  PN  <=  min{1, [P(y'_x') - P(x', y')] / P(x, y)}.
Without experimental data, P(y_x') is only known to lie in [P(x', y), P(x', y) + P(x)].
"""
import numpy as np


def tian_pearl(p_y, p_xy, p_xpyp, p_yxp_lo, p_yxp_hi):
    """PN bounds, unioned over an interval for P(y_x')."""
    if p_xy <= 0:
        return 0.0, 1.0
    lo = max(0.0, (p_y - p_yxp_hi) / p_xy)
    hi = min(1.0, ((1 - p_yxp_lo) - p_xpyp) / p_xy)
    lo = min(lo, 1.0)
    return lo, max(lo, hi)


def pn_bounds(obs, exp=None, ignorable=False):
    """Plug-in identified set for PN of D on Y in the population that `obs` samples.

    obs, exp: DataFrames with binary columns D, Y (and B in `exp` and `obs` when exp is given).
    exp:        randomized rows; P(y_x') is the D=0 response reweighted to obs's mix of the logged
                actuator B, so that both data sources describe one population.
    ignorable:  treat obs as unconfounded (used within proxy strata).
    """
    y, d = obs.Y.to_numpy(), obs.D.to_numpy()
    p_y, p_xy = y.mean(), np.mean((d == 1) & (y == 1))
    p_xpyp, p_x = np.mean((d == 0) & (y == 0)), d.mean()
    if ignorable:
        if not (d == 0).any():
            return 0.0, 1.0
        v = y[d == 0].mean()
        return tian_pearl(p_y, p_xy, p_xpyp, v, v)
    if exp is not None:
        w = obs.B.value_counts(normalize=True)
        v = sum(w[k] * exp.Y[(exp.D == 0) & (exp.B == k)].mean() for k in w.index)
        return tian_pearl(p_y, p_xy, p_xpyp, v, v)
    p_xpy = np.mean((d == 0) & (y == 1))
    return tian_pearl(p_y, p_xy, p_xpyp, p_xpy, p_xpy + p_x)


def _pn_bounds_arrays(d, y, b=None, ed=None, ey=None, eb=None, ignorable=False):
    """Array version of pn_bounds (same formulas), used by the bootstrap."""
    p_y, p_xy = y.mean(), np.mean((d == 1) & (y == 1))
    p_xpyp, p_x = np.mean((d == 0) & (y == 0)), d.mean()
    if ignorable:
        if not (d == 0).any():
            return 0.0, 1.0
        v = y[d == 0].mean()
        return tian_pearl(p_y, p_xy, p_xpyp, v, v)
    if ed is not None:
        v = 0.0
        for k in np.unique(b):
            m = (ed == 0) & (eb == k)
            v += np.mean(b == k) * (ey[m].mean() if m.any() else 0.5)
        return tian_pearl(p_y, p_xy, p_xpyp, v, v)
    p_xpy = np.mean((d == 0) & (y == 1))
    return tian_pearl(p_y, p_xy, p_xpyp, p_xpy, p_xpy + p_x)


def pn_bounds_boot(obs, exp=None, ignorable=False, reps=200, alpha=0.05, seed=0):
    """Outer (1 - alpha) interval: alpha/2 quantile of the lower bound, 1 - alpha/2 of the upper.

    Plug-in bounds under-cover: sampling error in the experimental term is divided by P(x, y).
    """
    r = np.random.default_rng(seed)
    d, y = obs.D.to_numpy(), obs.Y.to_numpy()
    b = obs.B.to_numpy() if exp is not None else None
    if exp is not None:
        ed, ey, eb = exp.D.to_numpy(), exp.Y.to_numpy(), exp.B.to_numpy()
    los, his = [], []
    for _ in range(reps):
        i = r.integers(0, len(d), len(d))
        if exp is not None:
            j = r.integers(0, len(ed), len(ed))
            lo, hi = _pn_bounds_arrays(d[i], y[i], b[i], ed[j], ey[j], eb[j])
        else:
            lo, hi = _pn_bounds_arrays(d[i], y[i], ignorable=ignorable)
        los.append(lo), his.append(hi)
    lo = min(1.0, float(np.quantile(los, alpha / 2)))
    return lo, max(lo, min(1.0, float(np.quantile(his, 1 - alpha / 2))))


def claim_side(lo, hi, threshold=0.5):
    """1: claim PN >= threshold holds on the whole set; 0: fails on the whole set; -1: ambiguous."""
    if lo >= threshold:
        return 1
    if hi < threshold:
        return 0
    return -1
