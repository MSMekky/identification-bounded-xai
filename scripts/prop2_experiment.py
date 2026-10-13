"""Proposition 2 on real data: identification and model multiplicity move independently.

    python scripts/prop2_experiment.py --data data/lt_walks_v1.zip --out results/prop2.csv

Same observational sample, same visible features (polarizer angles and the proxy ir_1):
  factor A (identifying information): proxy assumption (C1) or fitted mechanism (C3);
  factor F (model multiplicity): Rashomon tolerance eps in {0, 0.01, 0.03, 0.1} nats.
Reported per cell: the ambiguous share of the threshold claim (a property of Theta) and two
measures of the explanation set E_eps: the share of decisions on which Rashomon members disagree
about the claim, and the mean range of the polarizers' marginal Shapley value across members.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ibx import bounds as bd, data, explainers as ex  # noqa: E402
from ibx.benchmark import prepare, side_threshold  # noqa: E402
from ibx.mechanism import fit as fit_mechanism  # noqa: E402
from ibx.pilot import osrct, pn_counterfactual  # noqa: E402

warnings.filterwarnings("ignore")


def run(rct, seed, n_base=10000, strength=-0.8, n_dec=300, eps_grid=(0.0, 0.01, 0.03, 0.1)):
    mech = fit_mechanism(rct)
    rng = np.random.default_rng(seed)
    df, tau = prepare(rct, mech)
    perm = rng.permutation(len(df))
    base = df.iloc[perm[:n_base]].reset_index(drop=True)
    obs = base[osrct(base, rng, strength)].reset_index(drop=True)
    obs["ir1_bin"] = pd.qcut(obs.ir_1.rank(method="first"), 10, labels=False)
    parallel = df.loc[df.D == 0, ["pol_1", "pol_2"]].to_numpy()
    cand = np.flatnonzero((obs.D == 1) & (obs.Y == 1))
    dec = obs.iloc[rng.choice(cand, size=min(n_dec, len(cand)), replace=False)].copy()

    th1 = {b: bd.pn_bounds_boot(g, ignorable=True, reps=200, seed=seed) for b, g in obs.groupby("ir1_bin")}
    pn_c3 = pn_counterfactual(dec, mech.intensity_from_proxy(dec.ir_1.to_numpy()), np.ones(len(dec)),
                              parallel, mech, tau, rng)
    sides = {"C1 proxy assumption": np.array([side_threshold(*th1[b]) for b in dec.ir1_bin]),
             "C3 fitted mechanism": (pn_c3 >= 0.5).astype(int)}
    width = {"C1 proxy assumption": float(np.mean([th1[b][1] - th1[b][0] for b in dec.ir1_bin])),
             "C3 fitted mechanism": 0.0}

    feats = ["pol_1", "pol_2", "ir_1"]
    X_obs, y, X = obs[feats].to_numpy(float), obs.Y.to_numpy(), dec[feats].to_numpy(float)
    split = np.random.default_rng(seed + 1).permutation(len(obs))
    cut = int(0.7 * len(obs))
    members_all, losses = ex.rashomon_members(X_obs[split[:cut]], y[split[:cut]], X_obs[split[cut:]],
                                              y[split[cut:]], eps=max(eps_grid), seed=seed)
    # rashomon_members returns only those within max eps; recover their losses for sub-tolerances
    from sklearn.metrics import log_loss
    lv = [log_loss(y[split[cut:]], np.clip(ex.proba(m, X_obs[split[cut:]]), 1e-6, 1 - 1e-6)) for m in members_all]
    best = min(losses)
    groups = [[0, 1], [2]]
    phis = [ex.shapley_groups(m, X, X_obs, groups, "marginal", seed=seed)[0] for m in members_all]
    rows = []
    for eps in eps_grid:
        keep = [i for i, l in enumerate(lv) if l <= best + eps + 1e-12]
        P = np.stack([phis[i][:, 0] for i in keep])
        votes = np.stack([ex.read_top_reason(phis[i]) for i in keep])
        disagree = float(np.mean(~(votes == votes[0]).all(0)))
        for cond, s in sides.items():
            rows.append(dict(seed=seed, A=cond, eps=eps, n_members=len(keep),
                             ambiguous_share=float(np.mean(s < 0)), theta_mean_width=width[cond],
                             members_disagree_share=disagree,
                             phi_pol_mean_range=float(np.mean(P.max(0) - P.min(0)))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rct = data.load_real(a.data)
    rows = []
    for s in map(int, a.seeds.split(",")):
        rows += run(rct, s)
        print("seed", s, file=sys.stderr, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["A", "eps"])[["n_members", "ambiguous_share", "theta_mean_width", "members_disagree_share",
                                   "phi_pol_mean_range"]].mean().round(3).to_string())
