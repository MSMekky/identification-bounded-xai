"""Pilot A: score explanation claims against identified sets on the light tunnel.

Claim, for each dim reading taken with near-crossed polarizers (D=1, Y=1):
    "the polarizers are the reason: with near-parallel polarizers this reading would not be dim",
    i.e. PN_D >= 0.5.

Confounding is constructed by observational sampling from randomized data (OSRCT; Gentzel,
Pruthi and Jensen, ICML 2021): rows are kept with a probability that depends on (D, B) only, where
B is the logged source intensity. P(Y | D, B) and every counterfactual stay those of the real system.

Conditions (identifying information available to the auditor):
    C0  observational only; source intensity and all its proxies hidden
    C1  C0 + the proxy ir_1, assumed to deconfound within its deciles
    C2  C0 + randomized data on D
    C3  the fitted mechanism, with intensity recovered from ir_1
Individual truth (T3) comes from the fitted mechanism plus each unit's own residual, with
counterfactual polarizer angles drawn from the randomized rows that have D=0.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from . import bounds as bd
from .mechanism import malus

CROSSED = 0.30      # D = 1 when cos^2(pol_1 - pol_2) < CROSSED
DIM_Q = 0.40        # Y = 1 when ir_3 is below this quantile
FEATS = {"C0": ["pol_1", "pol_2"], "C1": ["pol_1", "pol_2", "ir_1"],
         "C2": ["pol_1", "pol_2"], "C3": ["pol_1", "pol_2", "ir_1"]}


def label(rct, mech):
    df = rct.copy()
    df["I"] = mech.intensity(df)
    df["D"] = (malus(df.pol_1, df.pol_2) < CROSSED).astype(int)
    df["B"] = (df.I > df.I.median()).astype(int)
    tau = float(df.ir_3.quantile(DIM_Q))
    df["Y"] = (df.ir_3 < tau).astype(int)
    pred = np.maximum(mech.predict(df), 20.0)
    df["rho"] = np.clip(df.ir_3 / pred, 0.2, 5.0)        # unit residual, kept fixed in counterfactuals
    return df, tau


def osrct(df, rng, strength):
    """strength > 0 keeps D == B rows with prob (1+|s|)/2; strength < 0 keeps D != B rows."""
    hi, lo = 0.5 + abs(strength) / 2, 0.5 - abs(strength) / 2
    same = (df.D == df.B) if strength > 0 else (df.D != df.B)
    return rng.uniform(size=len(df)) < np.where(same, hi, lo)


def pn_counterfactual(units, intensity, rho, parallel, mech, tau, rng, k=40):
    """Share of near-parallel angle pairs (drawn from randomized D=0 rows) under which Y flips to 0."""
    flips = np.zeros(len(units))
    for _ in range(k):
        pick = parallel[rng.integers(0, len(parallel), len(units))]
        cf = mech.ir3(intensity, pick[:, 0], pick[:, 1], units.l_31.to_numpy(), units.l_32.to_numpy()) * rho
        flips += cf >= tau
    return flips / k


def run(rct, mech, n_base, strength, seed, n_dec=500, proxy_bins=10, boot_reps=200):
    rng = np.random.default_rng(seed)
    df, tau = label(rct, mech)
    perm = rng.permutation(len(df))
    base = df.iloc[perm[:n_base]].reset_index(drop=True)
    exp = df.iloc[perm[n_base:]].reset_index(drop=True)
    obs = base[osrct(base, rng, strength)].reset_index(drop=True)
    obs["ir1_bin"] = pd.qcut(obs.ir_1.rank(method="first"), proxy_bins, labels=False)
    parallel = df.loc[df.D == 0, ["pol_1", "pol_2"]].to_numpy()

    cand = np.flatnonzero((obs.D == 1) & (obs.Y == 1))
    dec = obs.iloc[rng.choice(cand, size=min(n_dec, len(cand)), replace=False)].copy()
    pn_true = pn_counterfactual(dec, dec.I.to_numpy(), dec.rho.to_numpy(), parallel, mech, tau, rng)
    truth = (pn_true >= 0.5).astype(int)

    th0 = bd.pn_bounds(obs)
    th2_plug = bd.pn_bounds(obs, exp=exp)
    th2 = bd.pn_bounds_boot(obs, exp=exp, reps=boot_reps, seed=seed)
    th1 = {b: bd.pn_bounds_boot(g, ignorable=True, reps=boot_reps, seed=seed) for b, g in obs.groupby("ir1_bin")}
    pn_c3 = pn_counterfactual(dec, mech.intensity_from_proxy(dec.ir_1.to_numpy()), np.ones(len(dec)),
                              parallel, mech, tau, rng)
    sides = {"C0": np.full(len(dec), bd.claim_side(*th0)),
             "C1": np.array([bd.claim_side(*th1[b]) for b in dec.ir1_bin]),
             "C2": np.full(len(dec), bd.claim_side(*th2)),
             "C3": (pn_c3 >= 0.5).astype(int)}
    m = pn_true.mean()
    se = pn_true.std(ddof=1) / np.sqrt(len(pn_true))
    cover = {"C0": th0[0] - 2 * se <= m <= th0[1] + 2 * se, "C2": th2[0] - 2 * se <= m <= th2[1] + 2 * se}
    c1 = []
    for b, g in dec.groupby("ir1_bin"):
        mb = pn_true[dec.ir1_bin.to_numpy() == b]
        s = mb.std(ddof=1) / np.sqrt(len(mb)) if len(mb) > 1 else 0.5
        c1.append(th1[b][0] - 2 * s <= mb.mean() <= th1[b][1] + 2 * s)
    cover["C1"] = float(np.mean(c1))

    p1, p0 = obs.Y[obs.D == 1].mean(), obs.Y[obs.D == 0].mean()
    pn_ign = max(0.0, (p1 - p0) / p1)
    rows = []
    for cond, feats in FEATS.items():
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
        model.fit(obs[feats], obs.Y)
        X = dec[feats]
        f = model.predict_proba(X)[:, 1]
        bg = obs[feats].sample(min(500, len(obs)), random_state=seed)
        phi_i = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional",
                                   model_output="probability").shap_values(X)
        phi_p = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X)
        phi_p = phi_p[1] if isinstance(phi_p, list) else phi_p
        pol = [feats.index("pol_1"), feats.index("pol_2")]
        other = [i for i in range(len(feats)) if i not in pol]

        def top_reason(phi):     # reading 1: polarizers carry the largest push toward "dim"
            g_pol = phi[:, pol].sum(1)
            g_oth = phi[:, other].max(1) if other else np.full(len(phi), -np.inf)
            return ((g_pol > 0) & (g_pol >= g_oth)).astype(int)

        claims = {
            "SHAP interventional, top-reason reading": top_reason(phi_i),
            "SHAP interventional, counterfactual reading": ((f - phi_i[:, pol].sum(1)) < 0.5).astype(int),
            "SHAP path-dependent, top-reason reading": top_reason(phi_p),
            "PN assuming no hidden confounding": np.full(len(dec), int(pn_ign >= 0.5)),
        }
        s = sides[cond]
        ident = s >= 0
        for name, c in claims.items():
            rows.append(dict(
                n_base=n_base, n_obs=len(obs), strength=strength, seed=seed, condition=cond, method=name,
                n_decisions=len(dec), ambiguous_share=float(np.mean(~ident)),
                overclaim_rate=float(np.mean((~ident))),
                method_error_identified=float(np.mean(c[ident] != s[ident])) if ident.any() else np.nan,
                accuracy_vs_individual_truth=float(np.mean(c == truth)),
                theta_agreement_individual=float(np.mean(s[ident] == truth[ident])) if ident.any() else np.nan,
                coverage=float(cover[cond]) if cond in cover else np.nan,
                theta_c0_lo=th0[0], theta_c0_hi=th0[1], theta_c2_lo=th2[0], theta_c2_hi=th2[1],
                theta_c2_plugin_lo=th2_plug[0], theta_c2_plugin_hi=th2_plug[1],
                mean_true_pn=float(m), share_truth_pn_ge_half=float(truth.mean()),
                pn_no_confounding=float(pn_ign), p_dim_obs=float(obs.Y.mean()),
                mechanism_r2=mech.r2, proxy_r2=mech.proxy_r2,
                proxy_saturated_share=float((dec.ir_1 >= 65000).mean())))
    return rows
