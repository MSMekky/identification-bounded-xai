"""Error decomposition for root-cause claims on PetShop (Hardt et al., CLeaR 2024).

    git clone https://github.com/amazon-science/petshop-root-cause-analysis data/petshop
    python scripts/petshop_decomposition.py data/petshop/dataset results/petshop

For each of the 68 injected issues:
  1. Score every node's anomaly at the SLO-violation time against the issue-free period
     (robust z-score of latency mean, p50, p90, p99 and availability).
  2. Identified set. Under the call-graph model, a latency or error fault at node c shows at c
     (when c is measured) and can propagate only to c's callers (its ancestors). c is a
     consistent root cause if every anomalous node lies in {c} plus anc(c), and c itself is
     anomalous or unmeasured. Theta = the consistent nodes. The claim "r is the root cause" is
     identified when Theta = {r}.
  3. Methods rank all nodes: anomaly magnitude (z-score), SHAP on a model of the target's
     latency, and a graph-aware rule that ranks Theta first.
  4. Each Top-k miss is split into method error (Theta = {r}: the data single out r, the method
     did not) and identification-limited (Theta is not {r}).
"""
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")
SCENARIOS = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]
STATS = [("latency", "Average"), ("latency", "p50"), ("latency", "p90"), ("latency", "p99"),
         ("availability", "Average")]


def load(path):
    g = nx.from_pandas_adjacency(pd.read_csv(os.path.join(path, "graph.csv"), index_col=0),
                                 create_using=nx.DiGraph)
    normal = pd.read_csv(os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0)
    issues = []
    for split in ("train", "test"):
        for d in sorted(os.listdir(os.path.join(path, split))):
            if d.startswith("."):
                continue
            m = pd.read_csv(os.path.join(path, split, d, "metrics.csv"), header=[0, 1, 2], index_col=0)
            t = json.load(open(os.path.join(path, split, d, "target.json")))
            issues.append((split, d, m, t))
    return g, normal, issues


def robust_z(x, ref):
    med = np.nanmedian(ref)
    mad = 1.4826 * np.nanmedian(np.abs(ref - med))
    scale = np.nanmax([mad, 0.25 * np.nanstd(ref)])     # floor: quantized metrics have MAD near 0
    if not np.isfinite(scale) or scale <= 0:
        return np.nan
    return (x - med) / scale


def node_scores(nodes, normal, row):
    s = {}
    for n in nodes:
        zs = []
        for metric, stat in STATS:
            c = (n, metric, stat)
            if c in normal.columns and c in row.index and np.isfinite(row[c]):
                z = robust_z(row[c], normal[c].to_numpy(float))
                if np.isfinite(z):
                    zs.append(-z if metric == "availability" else z)   # falling availability = errors
        s[n] = max(zs) if zs else np.nan
    return s


def identified_set(g, scores, thr):
    """Most parsimonious single-fault explanations under upward propagation.

    Candidates: anomalous nodes and unmeasured nodes. For each, count the anomalous nodes it
    cannot explain (those outside itself and its callers). Theta = the candidates with the fewest
    unexplained anomalies. The strict version (zero unexplained) is empty in almost every issue,
    because faults also leave traces downstream (request volume) and in unrelated nodes.
    """
    A = {n for n, v in scores.items() if np.isfinite(v) and v > thr}
    if not A:
        return set(g.nodes), A, None
    cand = [c for c in g.nodes if c in A or not np.isfinite(scores.get(c, np.nan))]
    unexplained = {c: len(A - ({c} | nx.ancestors(g, c))) for c in cand}
    best = min(unexplained.values())
    return {c for c, u in unexplained.items() if u == best}, A, best


def shap_ranking(g, normal, row, target):
    feats = [n for n in g.nodes if n != target[0] and (n, "latency", "Average") in normal.columns]
    cols = [(n, "latency", "Average") for n in feats]
    Xn = normal[cols].to_numpy(float)
    med = np.nanmedian(Xn, axis=0)
    Xn = np.where(np.isfinite(Xn), Xn, med)
    yn = normal[target].to_numpy(float)
    ok = np.isfinite(yn)
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                              verbose=-1).fit(Xn[ok], yn[ok])
    x = np.array([row[c] if c in row.index and np.isfinite(row[c]) else m for c, m in zip(cols, med)])[None]
    phi = shap.TreeExplainer(model, data=Xn[ok][:200], feature_perturbation="interventional").shap_values(x)[0]
    return dict(zip(feats, phi))


def rank_of(r, scores):
    order = sorted([n for n, v in scores.items() if np.isfinite(v)], key=lambda n: -scores[n])
    return order.index(r) + 1 if r in order else np.inf


def main(root, out_dir, thresholds=(3.0, 5.0)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sc in SCENARIOS:
        g, normal, issues = load(os.path.join(root, sc))
        for split, name, m, t in issues:
            target = (t["target"]["node"], t["target"]["metric"], t["target"]["agg"])
            r = t["root_cause"]["node"]
            ts = t["target"]["timestamp"]
            after = m.index[m.index >= ts]
            row = m.loc[after[0] if len(after) else m.index[-1]]
            z = node_scores(g.nodes, normal, row)
            sh = shap_ranking(g, normal, row, target)
            for thr in thresholds:
                theta, A, unexp = identified_set(g, z, thr)
                graph_rule = {n: (1e6 if n in theta else 0) + (z[n] if np.isfinite(z[n]) else -1e3) for n in g.nodes}
                base = dict(scenario=sc, split=split, issue=name, root_cause=r, threshold=thr,
                            root_measured=bool(np.isfinite(z.get(r, np.nan))), root_z=z.get(r, np.nan),
                            n_anomalous=len(A), unexplained_best=unexp, theta_size=len(theta), root_in_theta=r in theta,
                            identified=theta == {r}, theta="; ".join(sorted(theta)) if len(theta) <= 5 else f"{len(theta)} nodes",
                            fault=t["metadata"].get("reproducibility", {}).get("command", ""))
                for meth, sc_ in (("Anomaly magnitude (z-score)", z), ("SHAP on target-latency model", sh),
                                  ("Graph-aware: identified set first", graph_rule)):
                    rk = rank_of(r, sc_)
                    rows.append(dict(base, method=meth, rank=rk, top1=rk <= 1, top3=rk <= 3))
    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "petshop_per_issue.csv", index=False)
    summ = []
    for (thr, meth), g_ in d.groupby(["threshold", "method"]):
        for k in ("top1", "top3"):
            miss = ~g_[k]
            summ.append(dict(threshold=thr, method=meth, k=k, issues=len(g_), recall=g_[k].mean(),
                             misses=int(miss.sum()),
                             misses_identified_method_error=int((miss & g_.identified).sum()),
                             misses_not_identified=int((miss & ~g_.identified).sum()),
                             share_identified=g_.identified.mean()))
    s = pd.DataFrame(summ)
    s.to_csv(out_dir / "petshop_decomposition.csv", index=False)
    return d, s


if __name__ == "__main__":
    d, s = main(sys.argv[1], sys.argv[2])
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
