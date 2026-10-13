"""Explainers, scored at the level of feature groups ("players").

Every explainer returns, for each decision, an attribution per group. Groups here are the
polarizer pair {pol_1, pol_2} and, when visible, the proxy sensor {ir_1}. With at most two
players, Shapley values are computed exactly from the value function, so the marginal,
conditional and causal-context variants differ only in how the absent players are filled in.

cc-Shapley (Martin and Haufe, UAI 2026) intervenes on the present players and draws the absent
ones from their interventional distribution. In the light tunnel the polarizers and the source
(hence the proxy) have no causal path between them, so the interventional distribution of the
proxy is its marginal and cc-Shapley coincides with marginal (interventional) Shapley. It is
therefore reported under that name rather than as a separate column.
"""
from itertools import combinations
from math import factorial

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def lgbm(seed=0, leaves=31, n=300):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=leaves, random_state=seed,
                              n_jobs=1, verbose=-1)


def mlp(seed=0):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400,
                                                         random_state=seed, early_stopping=True))


def proba(model, X):
    return model.predict_proba(np.asarray(X, float))[:, 1]


def _shapley_weights(p):
    """Weight of a coalition of size s when adding one player, for p players."""
    return {s: factorial(s) * factorial(p - s - 1) / factorial(p) for s in range(p)}


def shapley_groups(model, X, background, groups, kind="marginal", k=50, max_bg=200, seed=0):
    """Exact group Shapley values of P(dim) for each row of X.

    kind="marginal": absent players drawn from the background marginal (interventional SHAP).
    kind="conditional": absent players drawn from the k background rows nearest to the present
    players' values (standardized), approximating p(absent | present).
    """
    X, bg = np.asarray(X, float), np.asarray(background, float)
    r = np.random.default_rng(seed)
    if len(bg) > max_bg and kind == "marginal":
        bg = bg[r.choice(len(bg), max_bg, replace=False)]
    p, n = len(groups), len(X)
    scale = bg.std(0) + 1e-9
    cache = {}

    def value(S):
        S = tuple(sorted(S))
        if S in cache:
            return cache[S]
        cols = [c for g in S for c in groups[g]]
        if len(S) == p:
            v = proba(model, X)
        elif kind == "marginal" or not cols:
            m = len(bg)
            Z = np.repeat(bg[None, :, :], n, axis=0).reshape(n * m, -1)
            Z[:, cols] = np.repeat(X[:, cols], m, axis=0)
            v = proba(model, Z).reshape(n, m).mean(1)
        else:
            nn = NearestNeighbors(n_neighbors=min(k, len(bg))).fit(bg[:, cols] / scale[cols])
            idx = nn.kneighbors(X[:, cols] / scale[cols], return_distance=False)
            Z = bg[idx.ravel()].copy()
            Z[:, cols] = np.repeat(X[:, cols], idx.shape[1], axis=0)
            v = proba(model, Z).reshape(n, idx.shape[1]).mean(1)
        cache[S] = v
        return v

    w = _shapley_weights(p)
    phi = np.zeros((n, p))
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for s in range(p):
            for S in combinations(others, s):
                phi[:, j] += w[s] * (value(S + (j,)) - value(S))
    return phi, value(())


def lime_groups(model, X, train, groups, n_samples=300, seed=0):
    """LIME-style local linear surrogate; attribution = coefficient x (x - training mean), summed by group."""
    X, train = np.asarray(X, float), np.asarray(train, float)
    r = np.random.default_rng(seed)
    mu, sd = train.mean(0), train.std(0) + 1e-9
    d = X.shape[1]
    width = 0.75 * np.sqrt(d)
    out = np.zeros((len(X), len(groups)))
    for i, x in enumerate(X):
        Z = train[r.integers(0, len(train), (n_samples, d)), np.arange(d)]   # independent marginals
        Z[0] = x
        dist = np.sqrt((((Z - x) / sd) ** 2).sum(1))
        wts = np.exp(-dist ** 2 / width ** 2)
        fit = Ridge(alpha=1.0).fit((Z - mu) / sd, proba(model, Z), sample_weight=wts)
        contrib = fit.coef_ * (x - mu) / sd
        out[i] = [contrib[g].sum() for g in groups]
    return out


def integrated_gradients_groups(model, X, train, groups, steps=32, eps=1e-3):
    """Integrated gradients of P(dim) from the training mean, gradients by central differences."""
    X, train = np.asarray(X, float), np.asarray(train, float)
    base = train.mean(0)
    sd = train.std(0) + 1e-9
    d = X.shape[1]
    total = np.zeros_like(X)
    for a in (np.arange(steps) + 0.5) / steps:
        P = base + a * (X - base)
        for j in range(d):
            h = eps * sd[j]
            up, dn = P.copy(), P.copy()
            up[:, j] += h
            dn[:, j] -= h
            total[:, j] += (proba(model, up) - proba(model, dn)) / (2 * h)
    ig = (X - base) * total / steps
    return np.stack([ig[:, g].sum(1) for g in groups], axis=1)


def rashomon_members(Xtr, ytr, Xva, yva, eps=0.01, seed=0):
    """Fit a pool of model classes and keep those within eps nats of the best validation log-loss."""
    pool = [lgbm(seed + s, leaves) for s in range(3) for leaves in (7, 31, 63)]
    pool += [mlp(seed), mlp(seed + 1),
             RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1)]
    fitted, losses = [], []
    for m in pool:
        m.fit(Xtr, ytr)
        fitted.append(m)
        losses.append(log_loss(yva, np.clip(proba(m, Xva), 1e-6, 1 - 1e-6)))
    best = min(losses)
    keep = [m for m, l in zip(fitted, losses) if l <= best + eps]
    return keep, losses


def read_top_reason(phi, pol=0):
    """Reading 1: the polarizers carry the largest positive push toward 'dim'."""
    others = np.delete(phi, pol, axis=1)
    top_other = others.max(1) if others.shape[1] else np.full(len(phi), -np.inf)
    return ((phi[:, pol] > 0) & (phi[:, pol] >= top_other)).astype(int)


def read_counterfactual(phi, f, pol=0):
    """Reading 2: removing the polarizers' share drops P(dim) below one half."""
    return ((f - phi[:, pol]) < 0.5).astype(int)


def read_ranking(phi, pol=0, other=1):
    """Ranking claim: the polarizers mattered more than the source (absolute attribution)."""
    return (np.abs(phi[:, pol]) > np.abs(phi[:, other])).astype(int)
