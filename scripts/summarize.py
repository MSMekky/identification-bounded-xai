"""Summarize pilot results: mean over seeds, per source, strength, sample size, condition, method.

    python scripts/summarize.py results/pilot_real.csv results/pilot_sim.csv > results/summary.md
"""
import sys

import pandas as pd

d = pd.concat([pd.read_csv(p) for p in sys.argv[1:]])
keys = ["source", "strength", "n_base", "condition", "method"]
cols = ["ambiguous_share", "method_error_identified", "accuracy_vs_individual_truth",
        "theta_agreement_individual", "coverage"]
g = d.groupby(keys)[cols].mean().round(3).reset_index()
print("# Pilot A results (mean over seeds)\n")
print(g.to_markdown(index=False))
print("\n## Identified sets and truth (mean over seeds)\n")
k2 = ["source", "strength", "n_base"]
c2 = ["n_obs", "theta_c2_lo", "theta_c2_hi", "theta_c2_plugin_lo", "theta_c2_plugin_hi", "mean_true_pn",
      "share_truth_pn_ge_half", "pn_no_confounding", "p_dim_obs", "mechanism_r2", "proxy_saturated_share"]
print(d.drop_duplicates(k2 + ["seed"]).groupby(k2)[c2].mean().round(3).reset_index().to_markdown(index=False))
