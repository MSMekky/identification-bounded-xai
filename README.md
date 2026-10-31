# Identification-bounded explanations

Scores explanation claims against the set of answers the data can actually support.

An attribution such as SHAP describes a model (T1). It is often read as a claim about how a
decision would respond to a real intervention (T2) or about what caused an outcome in the
world (T3). Those readings are warranted only if the causal quantity behind them is
identified. This repository contains the pilot, the main benchmark (WP1, work package 1 of the research proposal) on the
[Causal Chambers](https://github.com/juangamella/causal-chamber) light tunnel, a test of
Proposition 2 (identification versus model multiplicity: whether models disagree does not tell whether the data can settle a claim), and an error decomposition for
root-cause claims on [PetShop](https://github.com/amazon-science/petshop-root-cause-analysis).

## Design

- **Data.** `lt_walks_v1`, experiment `actuators_white`: 20,000 light-tunnel measurements with
  red, green, blue, both polarizer angles and the LEDs drawn independently and uniformly before
  each measurement, so the rows form a randomized experiment.
- **Claims.** For each dim reading taken with near-crossed polarizers: "the polarizers are the
  reason" (threshold claim, PN >= 0.5) and "the polarizers mattered more than the source"
  (ranking claim).
- **Confounding.** Built by observational sampling from the randomized rows (OSRCT, Gentzel,
  Pruthi and Jensen, ICML 2021): rows are kept with a probability that depends on polarizer
  state and source intensity only, so every counterfactual stays that of the real system.
- **Identified sets.** Tian-Pearl bounds on PN under four conditions: observational only (C0),
  plus a proxy sensor (C1), plus randomized data (C2), the fitted mechanism (C3); bootstrap
  outer intervals, because plug-in bounds under-cover.
- **Truth.** Population truth from the randomized rows; individual truth from a mechanism fitted
  on all rows (Malus' law times source intensity, R squared 0.977), keeping each reading's own
  residual.
- **Explainers.** Exact group Shapley (marginal, which equals cc-Shapley here, and conditional),
  path-dependent TreeSHAP, LIME, integrated gradients on an MLP, LEWIS-style necessity, a
  Rashomon ensemble that abstains on disagreement, and a coin flip. Threshold claims are read
  two ways (top-reason, counterfactual).

## Results

**Benchmark** (`results/benchmark_real.csv`, 40 runs of 300 decisions). Ambiguous share of the
threshold claim: C0 100%, C1 7.7% (23.5% at confounding -0.8), C2 0%, C3 0%. Method error on
identified decisions, C1, clean proxy:

| Method | Confounding -0.8 | Confounding +0.8 |
| --- | --- | --- |
| Marginal Shapley, counterfactual reading | 40.3% | at most 6% |
| Path-dependent TreeSHAP | 29.9% | at most 6% |
| LIME | 16.3% | at most 6% |
| Conditional Shapley | 16.1% | at most 6% |
| Marginal Shapley, top-reason reading | 8.0% | at most 6% |
| Rashomon ensemble | 7.6% | at most 6% |
| Integrated gradients, LEWIS-style PN | 0% | at most 6% |
| Coin flip | 51% | 51% |

The Rashomon ensemble abstains on only about 4% to 6% of ambiguous decisions (threshold claim, C1; 4.1% with a clean proxy, 6.1% with proxy noise 0.25).
**Proposition 2** (`results/prop2.csv`). The ambiguous share is 21.4% under C1 and 0% under C3
at every Rashomon tolerance; Rashomon members disagree on 0%, 1.1%, 8.4% and 56.2% of decisions
at tolerances 0, 0.01, 0.03 and 0.1 nats, identically under both conditions.

**PetShop** (`results/petshop/`, z threshold 5). Of 68 issues, 5 are identified under a
single-fault, upward-propagation model, 20 ambiguous, 13 show no signal at the root and 30
contradict the model. Anomaly ranking and SHAP get none of the 5 identified issues at Top-1;
ranking the identified set first gets all 5, and Top-3 recall rises from 40% (z-score) and 28%
(SHAP) to 60%.

The pilot (`results/summary.md`) reports the first version of these numbers with per-feature
TreeSHAP.

## Run it

```bash
pip install -r requirements.txt
python scripts/fetch_data.py data/        # downloads lt_walks_v1.zip and checks its MD5
python scripts/run_benchmark.py --data data/lt_walks_v1.zip --sizes 10000 --out results/benchmark_real.csv
python scripts/prop2_experiment.py --data data/lt_walks_v1.zip --out results/prop2.csv
git clone https://github.com/amazon-science/petshop-root-cause-analysis data/petshop
python scripts/petshop_decomposition.py data/petshop/dataset results/petshop
python -m pytest -q tests
```

Pilot: `scripts/run_pilot.py` and `scripts/summarize.py` (see their docstrings). The benchmark
takes about 55 minutes on two cores; set `OMP_NUM_THREADS=1`.

## Layout

| Path | Contents |
| --- | --- |
| `ibx/data.py` | Loading the real data (with MD5 check) and the calibrated simulator |
| `ibx/mechanism.py` | Fitting the physical mechanism and the proxy map |
| `ibx/bounds.py` | Tian-Pearl bounds on PN, bootstrap outer intervals, claim classes |
| `ibx/pilot.py` | OSRCT construction, conditions C0 to C3, pilot explainers and scoring |
| `ibx/explainers.py` | Group Shapley (marginal, conditional), LIME, integrated gradients, Rashomon set, reading rules |
| `ibx/benchmark.py` | The WP1 benchmark: conditions, claims, scoring |
| `scripts/` | Data download, sweeps, Proposition 2, PetShop decomposition, summaries |
| `tests/` | Bounds contain the true PN on a known SCM; Shapley efficiency and the linear case |
| `results/` | Per-run results |

## Limits

- Individual truth on real data comes from a fitted mechanism, not from observed
  counterfactuals; population truth comes from the randomized rows.
- The causes are binarised (near-crossed polarizers; source intensity below its median), as in
  LEWIS and Watson et al.
- PetShop identified sets are only as good as the single-fault model, which 30 of 68 issues
  contradict.

## Data and licences

Code: MIT (see `LICENSE`). No data are redistributed here. Causal Chambers: CC BY 4.0, Gamella,
Peters and Buhlmann, "Causal chambers as a real-world physical testbed for AI methodology",
Nature Machine Intelligence (2025). PetShop: CC BY 4.0, Hardt et al., CLeaR 2024.
