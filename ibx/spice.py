"""Second system: natural confounding with a noisy proxy (Causal Chambers lt_spice_v1).

Light Tunnel Mk2, configuration linked_leds (config doc v1.0). The chamber sets
    led_2_uv <- 0.0624 * ir_1,
so the UV LED above the second sensor is driven by the first sensor's reading. The variables:
    U = green        brightness of the main light source (randomized, hidden from explainers)
    W = current_ls   current drawn by the light source: a proxy of U with ADC noise whose level is
                     set by sps_current_ls (4 in experiment I, 7 in experiment II)
    X = 1[led_2_uv above its median]          (the cause under study)
    Y = 1[ir_2 above a fixed quantile]        (the outcome: a bright reading at sensor 2)
Documented edges: U -> ir_1 -> led_2_uv -> ir_2 and U -> ir_2, U -> W. So U confounds X and Y.
Nothing is constructed: the confounding is physical.

Experiments III-1 and III-2 set green = 0 and record W alone, which gives the noise level of the
proxy (the "error mechanism" of Vollmer, Pfister and Weichwald, 2026). The noise is homoscedastic:
its standard deviation within each green level matches experiment III to the third digit.

Population. Rows whose green level has both values of X with probability at least 5%
(green 52 to 71 in experiment I), so that P(Y_{x'} | U) is defined.

Truth. ir_2 = a(green) + b * led_2_uv + r, fitted on all rows with green observed (R^2 0.968,
b stable at 2.32 and 2.36 across the two experiments; the residual r is uncorrelated with ir_1's
residual, i.e. no X-Y confounding beyond green). The intervention do(X = 0) draws led_2_uv from
its distribution given X = 0 and the unit's own green level; individual PN keeps r.

Identified sets for PN within proxy deciles:
    C0  observational, Manski range for P(y_x')
    C1  proxy treated as a deconfounder (ignorability given the W decile)
    C2  SPICE: proxy with its known error mechanism; P(Y_x' | W decile) from a latent-variable
        model W = L + E, E ~ N(0, sigma^2) with sigma from experiment III, and W independent of
        (X, Y) given L, fitted by EM on a grid (identified because Gaussian error is complete)
    C3  oracle: U observed, P(Y_x' | W decile) = sum_g P(g | decile) P(Y = 1 | X = 0, g)
"""
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import bounds as bd

FILES = {"I": ("exp_1_low_noise", "exp_3_spice_net_noise_1"),
         "II": ("exp_2_noisy", "exp_3_spice_net_noise_2")}
COLS = ["green", "current_ls", "ir_1", "led_2_uv", "ir_2"]
BLOCK = 5500          # 20 repetitions of 5,500 measurements (5,000 train + 500 test in the paper)
POSITIVITY = 0.05


def load(zip_path, experiment):
    main, noise = FILES[experiment]
    with zipfile.ZipFile(zip_path) as z:
        d = pd.read_csv(io.BytesIO(z.read(f"lt_spice_v1/{main}.csv")), usecols=COLS)
        e = pd.read_csv(io.BytesIO(z.read(f"lt_spice_v1/{noise}.csv")), usecols=["current_ls"])
    d["block"] = np.arange(len(d)) // BLOCK
    return d, float(e.current_ls.std())


def prepare(d, y_quantile=0.6):
    """Binary X and Y, the fitted ir_2 mechanism, and the positivity-restricted population."""
    d = d.copy()
    d["X"] = (d.led_2_uv > d.led_2_uv.median()).astype(int)
    thr = float(d.ir_2.quantile(y_quantile))
    d["Y"] = (d.ir_2 > thr).astype(int)
    G = pd.get_dummies(d.green).to_numpy(float)
    coef, *_ = np.linalg.lstsq(np.c_[G, d.led_2_uv.to_numpy()], d.ir_2.to_numpy(), rcond=None)
    levels = np.sort(d.green.unique())
    a = dict(zip(levels, coef[:-1]))
    slope = float(coef[-1])
    fitted = d.green.map(a).to_numpy() + slope * d.led_2_uv.to_numpy()
    d["r"] = d.ir_2.to_numpy() - fitted
    r2 = 1 - d.r.var() / d.ir_2.var()
    px = d.groupby("green").X.mean()
    keep = px[(px > POSITIVITY) & (px < 1 - POSITIVITY)].index
    pop = d[d.green.isin(keep)].reset_index(drop=True)
    p_y_x0_g = pop[pop.X == 0].groupby("green").Y.mean()          # for the oracle condition
    mech = dict(a=a, slope=slope, thr=thr, r2=float(r2),
                low_led={g: grp.led_2_uv.to_numpy() for g, grp in pop[pop.X == 0].groupby("green")},
                p_y_x0_g=p_y_x0_g)
    return pop, mech


def individual_pn(units, mech, rng, k=200):
    """Share of draws of led_2_uv from P(led | X = 0, own green) under which the unit is not bright."""
    out = np.empty(len(units))
    for i, (g, r) in enumerate(zip(units.green.to_numpy(), units.r.to_numpy())):
        led = rng.choice(mech["low_led"][g], k)
        out[i] = np.mean(mech["a"][g] + mech["slope"] * led + r <= mech["thr"])
    return out


# ---------------------------------------------------------------- SPICE latent model
def spice_em(w, cell, sigma, K=80, iters=300, smooth=1.5, init=None):
    """EM for W = L + E with known Gaussian E and a discrete grid for L.

    w: proxy values; cell: 2*X + Y in {0,1,2,3}. Returns grid, pi (P(L)), q (P(cell | L)) and the
    responsibilities P(L | w_i, cell_i).
    """
    grid = np.linspace(w.min() - 2 * sigma, w.max() + 2 * sigma, K)
    lik_w = norm.pdf((w[:, None] - grid[None, :]) / sigma)                    # n x K
    onehot = np.eye(4)[cell]                                                  # n x 4
    if init is None:
        pi = np.full(K, 1.0 / K)
        q = np.tile(onehot.mean(0), (K, 1))
    else:
        pi, q = init
    kern = norm.pdf(np.arange(-4, 5) / smooth)
    kern /= kern.sum()
    for _ in range(iters):
        R = lik_w * pi[None, :] * (onehot @ q.T)                              # n x K
        R /= R.sum(1, keepdims=True) + 1e-300
        pi = R.mean(0)
        num = R.T @ onehot + 1e-3                                             # K x 4
        num = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, num)
        q = num / num.sum(1, keepdims=True)
    return grid, pi, q, R


def p_y_x0_given_w(w, cell, sigma, bins, n_bins, init=None):
    """P(Y_{x'} = 1 | W decile) = sum_L P(L | decile) P(Y = 1 | X = 0, L), from the SPICE model."""
    grid, pi, q, _ = spice_em(w, cell, sigma, init=init)
    p_y1_x0 = q[:, 1] / (q[:, 0] + q[:, 1])                                   # cells 0: x0y0, 1: x0y1
    lik_w = norm.pdf((w[:, None] - grid[None, :]) / sigma) * pi[None, :]
    post_w = lik_w / lik_w.sum(1, keepdims=True)                              # P(L | w), not | cell
    out = np.array([post_w[bins == b].mean(0) @ p_y1_x0 if (bins == b).any() else np.nan
                    for b in range(n_bins)])
    return out, (pi, q)


# ---------------------------------------------------------------- identified sets
def _tp(x, y, v_lo, v_hi):
    p_y, p_xy = y.mean(), np.mean((x == 1) & (y == 1))
    return bd.tian_pearl(p_y, p_xy, np.mean((x == 0) & (y == 0)), v_lo, v_hi)


def theta_by_bin(obs, sigma, n_bins=10, reps=200, seed=0, alpha=0.05):
    """Bootstrap outer intervals of PN within each proxy decile, for C0 to C3."""
    rng = np.random.default_rng(seed)
    x, y, w, g = obs.X.to_numpy(), obs.Y.to_numpy(), obs.current_ls.to_numpy(), obs.green.to_numpy()
    edges = np.quantile(w, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.searchsorted(edges, w)
    cell = 2 * x + y
    est = {c: [[] for _ in range(n_bins)] for c in ("C0", "C1", "C2", "C3")}
    init = None
    for rep in range(reps + 1):
        i = np.arange(len(x)) if rep == 0 else rng.integers(0, len(x), len(x))
        xb, yb, wb, gb, bb = x[i], y[i], w[i], g[i], bins[i]
        sp, init = p_y_x0_given_w(wb, cell[i], sigma, bb, n_bins, init=init)
        # oracle: P(Y=1 | X=0, g) estimated in the resample, averaged over P(g | decile)
        pyg = pd.Series(yb[xb == 0]).groupby(gb[xb == 0]).mean()
        for b in range(n_bins):
            m = bb == b
            xs, ys = xb[m], yb[m]
            p_xpy, p_x = np.mean((xs == 0) & (ys == 1)), xs.mean()
            v1 = ys[xs == 0].mean() if (xs == 0).any() else np.nan
            gm = pd.Series(gb[m]).map(pyg).dropna()
            v3 = gm.mean() if len(gm) else np.nan
            if rep == 0:
                continue                                  # the full-sample fit only warms up EM
            est["C0"][b].append(_tp(xs, ys, p_xpy, p_xpy + p_x))
            est["C1"][b].append(_tp(xs, ys, v1, v1) if np.isfinite(v1) else (0.0, 1.0))
            est["C2"][b].append(_tp(xs, ys, sp[b], sp[b]))
            est["C3"][b].append(_tp(xs, ys, v3, v3) if np.isfinite(v3) else (0.0, 1.0))
    theta = {}
    for c, per_bin in est.items():
        theta[c] = []
        for lst in per_bin:
            a = np.array(lst)
            lo = min(1.0, float(np.quantile(a[:, 0], alpha / 2)))
            theta[c].append((lo, max(lo, min(1.0, float(np.quantile(a[:, 1], 1 - alpha / 2))))))
    return theta, edges
