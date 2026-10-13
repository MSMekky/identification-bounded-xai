"""Run pilot A on the real chamber data or the calibrated simulator.

    python scripts/run_pilot.py --source real --data data/lt_walks_v1.zip --out results/pilot_real.csv
    python scripts/run_pilot.py --source sim --out results/pilot_sim.csv
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ibx import data, mechanism, pilot  # noqa: E402


def _job(args):
    rct, mech, n, s, seed = args
    return pilot.run(rct, mech, n, s, seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["real", "sim"], required=True)
    ap.add_argument("--data", help="lt_walks_v1.zip or extracted actuators_white.csv (real source)")
    ap.add_argument("--sim-rows", type=int, default=200000)
    ap.add_argument("--sizes", default="2500,5000,10000")
    ap.add_argument("--strengths", default="0.8,-0.8")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rct = data.load_real(a.data) if a.source == "real" else data.simulate(a.sim_rows, np.random.default_rng(12345))
    mech = mechanism.fit(rct)
    print(f"rows={len(rct)} mechanism R2={mech.r2:.4f} proxy R2={mech.proxy_r2:.4f}", file=sys.stderr)
    grid = [(rct, mech, n, s, seed) for n, s, seed in product(
        map(int, a.sizes.split(",")), map(float, a.strengths.split(",")), map(int, a.seeds.split(",")))]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for out in ex.map(_job, grid):
            rows += out
            print(a.source, out[0]["n_base"], out[0]["strength"], out[0]["seed"], file=sys.stderr, flush=True)
    pd.DataFrame(rows).assign(source=a.source).to_csv(a.out, index=False)
