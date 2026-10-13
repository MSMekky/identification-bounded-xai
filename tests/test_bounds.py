import numpy as np
import pandas as pd

from ibx import bounds as bd


def _scm(n, rng, confound):
    """Binary SCM with known PN: Y = D*U1 OR (1-D)*U2 style mechanism, hidden B confounds D."""
    B = rng.uniform(size=n) < 0.5
    D = rng.uniform(size=n) < np.where(B, 0.5 + confound / 2, 0.5 - confound / 2)
    U = rng.uniform(size=n)
    y1 = U < np.where(B, 0.9, 0.6)           # Y under D=1
    y0 = U < np.where(B, 0.2, 0.5)           # Y under D=0 (monotone: y0 implies y1)
    Y = np.where(D, y1, y0)
    pn = np.mean(~y0[D & Y])                  # P(Y_0 = 0 | D=1, Y=1)
    return pd.DataFrame(dict(D=D.astype(int), B=B.astype(int), Y=Y.astype(int))), pn, y0


def test_tian_pearl_contains_true_pn():
    rng = np.random.default_rng(0)
    obs, pn, _ = _scm(400_000, rng, confound=0.6)

    e, _, _ = _scm(400_000, rng, confound=0.0)
    lo, hi = bd.pn_bounds(obs, exp=e)
    assert lo - 0.01 <= pn <= hi + 0.01
    lo0, hi0 = bd.pn_bounds(obs)
    assert lo0 <= lo and hi <= hi0 + 1e-9


def test_bounds_are_ordered_and_in_unit_interval():
    rng = np.random.default_rng(1)
    obs, _, _ = _scm(2_000, rng, confound=0.8)
    e, _, _ = _scm(2_000, rng, confound=0.0)
    for th in (bd.pn_bounds(obs), bd.pn_bounds(obs, exp=e), bd.pn_bounds_boot(obs, exp=e, reps=50)):
        assert 0.0 <= th[0] <= th[1] <= 1.0


def test_claim_side():
    assert bd.claim_side(0.6, 0.9) == 1
    assert bd.claim_side(0.1, 0.4) == 0
    assert bd.claim_side(0.3, 0.7) == -1
