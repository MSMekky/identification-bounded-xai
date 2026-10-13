"""WP1 light-tunnel benchmark sweep on the real Causal Chamber data.

    python scripts/run_benchmark.py --data data/lt_walks_v1.zip --out results/benchmark_real.csv

Rows are appended as each run finishes, so a partial sweep is still usable.
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ibx import benchmark, data  # noqa: E402


def _job(args):
    warnings.filterwarnings("ignore")
    rct, n, s, noise, seed = args
    return benchmark.run(rct, n, s, seed, proxy_noise=noise)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sizes", default="5000,10000")
    ap.add_argument("--strengths", default="-0.8,-0.4,0.4,0.8")
    ap.add_argument("--noises", default="0,0.25")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rct = data.load_real(a.data)
    grid = [(rct, n, s, z, sd) for n, s, z, sd in product(
        map(int, a.sizes.split(",")), map(float, a.strengths.split(",")),
        map(float, a.noises.split(",")), map(int, a.seeds.split(",")))]
    out = Path(a.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = set(map(tuple, prev[["n_base", "strength", "proxy_noise", "seed"]].drop_duplicates().to_numpy()))
    todo = [g for g in grid if (g[1], g[2], g[3], g[4]) not in done]
    print(f"{len(todo)} of {len(grid)} runs to do", file=sys.stderr, flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(_job, g) for g in todo]
        for fut in as_completed(futs):
            rows = pd.DataFrame(fut.result())
            rows.to_csv(out, mode="a", header=not out.exists(), index=False)
            r = rows.iloc[0]
            print("done", r.n_base, r.strength, r.proxy_noise, r.seed, file=sys.stderr, flush=True)
