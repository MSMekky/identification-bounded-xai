import numpy as np

from ibx import explainers as ex


class _Linear:
    """Stand-in classifier: P(dim) is linear in the features, so Shapley values are known."""

    def __init__(self, w, b=0.1):
        self.w, self.b = np.asarray(w, float), b

    def predict_proba(self, X):
        p = np.clip(self.b + X @ self.w, 0, 1)
        return np.stack([1 - p, p], axis=1)


def test_shapley_efficiency_and_linear_case():
    r = np.random.default_rng(0)
    bg = r.uniform(0, 0.3, (300, 3))
    X = r.uniform(0, 0.3, (20, 3))
    m = _Linear([0.5, 0.4, 0.6])
    groups = [[0, 1], [2]]
    phi, base = ex.shapley_groups(m, X, bg, groups, "marginal", max_bg=300)
    f = ex.proba(m, X)
    assert np.allclose(phi.sum(1), f - base, atol=1e-9)          # efficiency
    mu = bg.mean(0)
    expected = np.stack([(X[:, :2] - mu[:2]) @ m.w[:2], (X[:, 2] - mu[2]) * m.w[2]], axis=1)
    assert np.allclose(phi, expected, atol=1e-9)                 # interventional SHAP of a linear model


def test_conditional_shapley_efficiency():
    r = np.random.default_rng(1)
    bg = r.uniform(0, 0.3, (300, 3))
    X = r.uniform(0, 0.3, (10, 3))
    m = _Linear([0.5, 0.4, 0.6])
    phi, base = ex.shapley_groups(m, X, bg, [[0, 1], [2]], "conditional", k=30)
    assert np.allclose(phi.sum(1), ex.proba(m, X) - base, atol=1e-9)
