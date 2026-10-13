"""WP1 benchmark on the light tunnel: explanation claims scored against identified sets.

Extends the pilot (ibx/pilot.py) with:
  * two claim types for each dim reading taken with near-crossed polarizers:
      threshold  "the polarizers are the reason"        PN_D >= 0.5
      ranking    "the polarizers mattered more than the source"   PN_D > PN_L
    where D is near-crossed polarizers and L is a below-median source intensity;
  * more explainers (marginal and conditional exact Shapley, path-dependent TreeSHAP, LIME,
    integrated gradients on an MLP, a LEWIS-style necessity estimate, a Rashomon ensemble that
    abstains when its members disagree, and a coin flip);
  * abstention-aware scores: method error, overclaim and over-caution;
  * experimental factors: confounding direction and strength, sample size, proxy noise.
"""
import numpy as np
import pandas as pd
import shap

from . import bounds as bd
from . import explainers as ex
from .mechanism import fit as fit_mechanism
from .pilot import CROSSED, DIM_Q, osrct, pn_counterfactual

COND_FEATS = {"C0": ["pol_1", "pol_2"], "C1": ["pol_1", "pol_2", "ir_1"],
              "C2": ["pol_1", "pol_2"], "C3": ["pol_1", "pol_2", "ir_1"]}


def add_proxy_noise(rct, noise, seed=777):
    if noise <= 0:
        return rct.copy()
    r = np.random.default_rng(seed)
    out = rct.copy()
    out["ir_1"] = np.clip(out.ir_1 * (1 + noise * r.standard_normal(len(out))), 0, 65535)
    return out


def prepare(rct, mech):
    from .mechanism import malus
    df = rct.copy()
    df["I"] = mech.intensity(df)
    df["D"] = (malus(df.pol_1, df.pol_2) < CROSSED).astype(int)
    df["B"] = (df.I > df.I.median()).astype(int)
    tau = float(df.ir_3.quantile(DIM_Q))
    df["Y"] = (df.ir_3 < tau).astype(int)
    df["rho"] = np.clip(df.ir_3 / np.maximum(mech.predict(df), 20.0), 0.2, 5.0)
    return df, tau


def pn_source(units, intensity, rho, bright, mech, tau, rng, low_mask, k=40):
    """PN of a below-median source for each unit: share of bright intensities under which the
    reading stops being dim, polarizers held at their actual angles. Zero where the source is
    already bright (the cause is absent)."""
    flips = np.zeros(len(units))
    for _ in range(k):
        cf_I = bright[rng.integers(0, len(bright), len(units))]
        cf = mech.ir3(cf_I, units.pol_1.to_numpy(), units.pol_2.to_numpy(),
                      units.l_31.to_numpy(), units.l_32.to_numpy()) * rho
        flips += cf >= tau
    return np.where(low_mask, flips / k, 0.0)


def side_threshold(lo, hi, t=0.5):
    return 1 if lo >= t else (0 if hi < t else -1)


def side_ranking(d_lo, d_hi, l_lo, l_hi):
    return 1 if d_lo > l_hi else (0 if d_hi < l_lo else -1)


def score(claims, sides, truth, abstain=None):
    """claims: 0/1 per decision; sides: 1, 0 or -1 (ambiguous); truth: 0/1 or -1 (tie)."""
    claims, sides, truth = np.asarray(claims), np.asarray(sides), np.asarray(truth)
    abstain = np.zeros(len(claims), bool) if abstain is None else np.asarray(abstain, bool)
    ident, amb = sides >= 0, sides < 0
    asserted = ~abstain
    ok = truth >= 0
    return dict(
        ambiguous_share=float(amb.mean()),
        abstention_rate=float(abstain.mean()),
        overclaim_rate=float((amb & asserted).mean()),
        over_caution_rate=float((ident & abstain).mean()),
        method_error_identified=float(np.mean(claims[ident & asserted] != sides[ident & asserted]))
        if (ident & asserted).any() else np.nan,
        accuracy_vs_individual_truth=float(np.mean(claims[asserted & ok] == truth[asserted & ok]))
        if (asserted & ok).any() else np.nan,
        theta_agreement_individual=float(np.mean(sides[ident & ok] == truth[ident & ok]))
        if (ident & ok).any() else np.nan,
    )


def run(rct_raw, n_base, strength, seed, proxy_noise=0.0, n_dec=300, proxy_bins=10, boot_reps=200,
        rashomon_eps=0.01):
    rct = add_proxy_noise(rct_raw, proxy_noise)
    mech = fit_mechanism(rct)
    rng = np.random.default_rng(seed)
    df, tau = prepare(rct, mech)
    perm = rng.permutation(len(df))
    base = df.iloc[perm[:n_base]].reset_index(drop=True)
    exp = df.iloc[perm[n_base:]].reset_index(drop=True)
    obs = base[osrct(base, rng, strength)].reset_index(drop=True)
    obs["ir1_bin"] = pd.qcut(obs.ir_1.rank(method="first"), proxy_bins, labels=False)
    obs["Lhat"] = (obs.ir_1 <= obs.ir_1.median()).astype(int)       # proxy-based "dim source"
    parallel = df.loc[df.D == 0, ["pol_1", "pol_2"]].to_numpy()
    bright_I = df.loc[df.B == 1, "I"].to_numpy()

    cand = np.flatnonzero((obs.D == 1) & (obs.Y == 1))
    dec = obs.iloc[rng.choice(cand, size=min(n_dec, len(cand)), replace=False)].copy()
    rho = dec.rho.to_numpy()
    pnD = pn_counterfactual(dec, dec.I.to_numpy(), rho, parallel, mech, tau, rng)
    pnL = pn_source(dec, dec.I.to_numpy(), rho, bright_I, mech, tau, rng, dec.B.to_numpy() == 0)
    truth_thr = (pnD >= 0.5).astype(int)
    truth_rank = np.where(pnD > pnL, 1, np.where(pnD < pnL, 0, -1))

    # identified sets
    th0 = bd.pn_bounds(obs)
    th2 = bd.pn_bounds_boot(obs, exp=exp, reps=boot_reps, seed=seed)
    th1 = {b: bd.pn_bounds_boot(g, ignorable=True, reps=boot_reps, seed=seed) for b, g in obs.groupby("ir1_bin")}
    od = obs[obs.D == 1]
    thL = bd.pn_bounds_boot(pd.DataFrame(dict(D=od.Lhat, Y=od.Y)), ignorable=True, reps=boot_reps, seed=seed)
    I_hat = mech.intensity_from_proxy(dec.ir_1.to_numpy())
    pnD_c3 = pn_counterfactual(dec, I_hat, np.ones(len(dec)), parallel, mech, tau, rng)
    pnL_c3 = pn_source(dec, I_hat, np.ones(len(dec)), bright_I, mech, tau, rng, I_hat <= np.median(df.I))
    bins, lhat = dec.ir1_bin.to_numpy(), dec.Lhat.to_numpy()
    sides_thr = {"C0": np.full(len(dec), side_threshold(*th0)),
                 "C1": np.array([side_threshold(*th1[b]) for b in bins]),
                 "C2": np.full(len(dec), side_threshold(*th2)),
                 "C3": (pnD_c3 >= 0.5).astype(int)}
    sides_rank = {"C1": np.array([side_ranking(*th1[b], *(thL if l else (0.0, 0.0))) for b, l in zip(bins, lhat)]),
                  "C3": np.where(pnD_c3 > pnL_c3, 1, np.where(pnD_c3 < pnL_c3, 0, -1))}

    rows = []
    common = dict(n_base=n_base, n_obs=len(obs), strength=strength, seed=seed, proxy_noise=proxy_noise,
                  n_decisions=len(dec), mean_true_pnD=float(pnD.mean()), share_truth_thr=float(truth_thr.mean()),
                  share_truth_rank=float(np.mean(truth_rank == 1)), mechanism_r2=mech.r2, proxy_r2=mech.proxy_r2,
                  theta_c2_lo=th2[0], theta_c2_hi=th2[1])
    cache = {}
    split = np.random.default_rng(seed + 1).permutation(len(obs))
    cut = int(0.7 * len(obs))
    for cond, feats in COND_FEATS.items():
        X_obs, y = obs[feats].to_numpy(float), obs.Y.to_numpy()
        X = dec[feats].to_numpy(float)
        groups = [[0, 1]] + ([[2]] if len(feats) == 3 else [])
        key = tuple(feats)
        if key not in cache:        # same features -> same models and attributions (only Theta differs)
            tr, va = split[:cut], split[cut:]
            members, _ = ex.rashomon_members(X_obs[tr], y[tr], X_obs[va], y[va], eps=rashomon_eps, seed=seed)
            model = ex.lgbm(seed).fit(X_obs, y)
            net = ex.mlp(seed).fit(X_obs, y)
            f = ex.proba(model, X)
            phi_m, _ = ex.shapley_groups(model, X, X_obs, groups, "marginal", seed=seed)
            phi_c, _ = ex.shapley_groups(model, X, X_obs, groups, "conditional", seed=seed)
            tp = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X)
            tp = tp[1] if isinstance(tp, list) else tp
            phi_p = np.stack([tp[:, g].sum(1) for g in groups], axis=1)
            phi_l = ex.lime_groups(model, X, X_obs, groups, seed=seed)
            phi_ig = ex.integrated_gradients_groups(net, X, X_obs, groups)
            member_phis = [ex.shapley_groups(m, X, X_obs, groups, "marginal", seed=seed)[0] for m in members]
            coin_thr, coin_rank = rng.integers(0, 2, len(dec)), rng.integers(0, 2, len(dec))
            cache[key] = (members, f, phi_m, phi_c, phi_p, phi_l, phi_ig, member_phis, coin_thr, coin_rank)
        members, f, phi_m, phi_c, phi_p, phi_l, phi_ig, member_phis, coin_thr, coin_rank = cache[key]

        # LEWIS-style necessity: PN under ignorability given the visible features (proxy deciles if visible)
        if "ir_1" in feats:
            lew = np.zeros(len(dec))
            for b in np.unique(bins):
                g = obs[obs.ir1_bin == b]
                p1, p0 = g.Y[g.D == 1].mean(), g.Y[g.D == 0].mean() if (g.D == 0).any() else 0.0
                lew[bins == b] = max(0.0, (p1 - p0) / p1) if p1 > 0 else 0.0
        else:
            p1, p0 = obs.Y[obs.D == 1].mean(), obs.Y[obs.D == 0].mean()
            lew = np.full(len(dec), max(0.0, (p1 - p0) / p1))

        thr = {
            "Marginal Shapley (= cc-Shapley here), top-reason": (ex.read_top_reason(phi_m), None),
            "Marginal Shapley (= cc-Shapley here), counterfactual": (ex.read_counterfactual(phi_m, f), None),
            "Conditional Shapley, top-reason": (ex.read_top_reason(phi_c), None),
            "TreeSHAP path-dependent, top-reason": (ex.read_top_reason(phi_p), None),
            "LIME, top-reason": (ex.read_top_reason(phi_l), None),
            "Integrated gradients (MLP), top-reason": (ex.read_top_reason(phi_ig), None),
            "LEWIS-style necessity (visible features)": ((lew >= 0.5).astype(int), None),
        }
        votes = np.stack([ex.read_top_reason(p) for p in member_phis])
        thr["Rashomon ensemble, abstains on disagreement"] = (votes[0], ~(votes == votes[0]).all(0))
        thr["Coin flip"] = (coin_thr, None)
        for name, (c, ab) in thr.items():
            rows.append(dict(common, condition=cond, claim="threshold", method=name,
                             n_rashomon=len(members), **score(c, sides_thr[cond], truth_thr, ab)))
        if cond in sides_rank:
            rk = {
                "Marginal Shapley (= cc-Shapley here)": (ex.read_ranking(phi_m), None),
                "Conditional Shapley": (ex.read_ranking(phi_c), None),
                "TreeSHAP path-dependent": (ex.read_ranking(phi_p), None),
                "LIME": (ex.read_ranking(phi_l), None),
                "Integrated gradients (MLP)": (ex.read_ranking(phi_ig), None),
            }
            rv = np.stack([ex.read_ranking(p) for p in member_phis])
            rk["Rashomon ensemble, abstains on disagreement"] = (rv[0], ~(rv == rv[0]).all(0))
            rk["Coin flip"] = (coin_rank, None)
            for name, (c, ab) in rk.items():
                rows.append(dict(common, condition=cond, claim="ranking", method=name,
                                 n_rashomon=len(members), **score(c, sides_rank[cond], truth_rank, ab)))
    return rows
