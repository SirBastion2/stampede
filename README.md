# STAMPEDE: a wind tunnel for bank runs

STAMPEDE is a GPU agent-based simulator of bank runs. You inject a rumor or a shock, watch it spread through a depositor population, drain the bank's cash, force bond sales, and move the bank's stock. Then the engine searches for the crisis response that keeps the bank open longest and replays the day with that response applied.

It is calibrated to Silicon Valley Bank's March 2023 run and checked, without refitting, against Signature Bank's run the next day. It is a stress-testing and teaching tool, not a forecasting product. Results are reported as distributions over seeded runs (failure rates, time-of-failure windows), never as a single prediction.


---

## Results at a glance

Every number below comes from a file in this repo or from a recorded run on the development machine (RTX 3060 12 GB, Linux). The source is given for each one.

| What | Result | Source |
|---|---|---|
| SVB fit (in sample), share of deposits requested by day-one close | 24.49% simulated vs. 24.9% target (−1.6% relative) | `docs/svb-fit-verification.md` §8 |
| SVB fit (in sample), share requested by next open | 81.85% vs. 81% target (+1.05%) | same |
| **Signature Bank holdout (not fitted)**, share requested by day-one close | **25.98% vs. 24.9% target (+4.3%, band ±35%)** | same |
| Placebo: fully insured community bank | survives (expected: survives) | same |
| Batched calibration, one CMA-ES generation (12 candidates × 8 seeds = 96 simulations of 100,000 depositors, 460 ticks) | 13.8 s, down from about 180 s (about 13×) | `docs/svb-fit-verification.md` §1 |
| GPU vs. CPU reference, 21 plan × seed cells | 361 s GPU vs. 1,099 s CPU (2 workers), 6.1×; 21/21 identical outcomes and trigger minutes; worst endpoint gap 0.0036 percentage points of deposits | `scripts/gpu_parity.py` run output (VF-1, `data/` not committed) |
| Response search | 2,320 simulated days (1,441 minutes each, 50-seed pool), 31 plans simulated out of 321 considered; 111 plans with unsupported predictions withheld; 3 useful plans in the final ranking; 12,043 s wall on 6 workers | `scripts/pipeline.py` run output (`data/` not committed) |
| Test suite (final version) | RTX 3060: 1,056 passed, 2 known baseline failures, 1 skipped, 1 xfailed. CPU-only rerun (`STAMPEDE_OFFLINE=1`, no CUDA): 1,072 passed, 1 known failure, 18 GPU tests skipped, 1 xfailed | final validation run (Nov 8 2026); CPU rerun Nov 12 2026 |
| Code size | about 59,700 lines of Python (about 21,100 of them tests), about 6,100 lines of HTML/TypeScript/JS UI; 27 modules | `wc -l` on the tree |

## How it works

### The panic loop

```
rumor -> attention -> withdrawal requests -> cash drain -> forced bond sales
      -> realized losses -> stock falls -> news -> more attention
```

The policy loop runs against it: a deposit guarantee or emergency lending slows withdrawals, at a cost to the insurance fund.

### Time and randomness

- One tick is one simulated business minute. The calendar models wire cut-offs, lending-facility hours and weekends.
- All randomness comes from seeded NumPy/CuPy Philox streams. LLM outputs are cached in SQLite, keyed by prompt hash and model, so a scenario replays identically. `scripts/determinism_smoke.py` checks this in CI.

### Modules (27)

| Layer | Modules |
|---|---|
| Clock and macro | M00 clock, calendar and event scheduler · M01 macro and rates (Nelson–Siegel curve) |
| Information | M02 multilayer information network (feeds, group chats, news; CSR adjacency) · M03 attention and rumor engine · M04 media and analyst agents |
| Depositors | M05 population and deposit insurance (insured vs. uninsured balances, 7 segments) · M06 decision model · M07 payment and withdrawal channels (wire and branch capacity, queues) |
| Bank | M08 balance sheet and accounting · M09 securities and interest-rate risk · M12 liquidity facilities (FHLB, discount window) · M23 bank management agent |
| Market | M14 per-bank stock: limit order book with five trader types (value, momentum, short sellers and others), ~300 traders per bank |
| Policy | M20 regulator triggers and closure · M22 policy lab |
| Crisis response | M24 crisis comms room · M25 persona panel · M26 ensemble and futures engine |
| Validation and ops | M27 calibration and validation harness · M28 scenarios, presets and replay · M33 observability and determinism · M34 safety guardrails |
| UI | M29 crowd screen · M30 market screen and crisis room |
| LLM agents | M41 VIP depositors · M42 bank CEOs · M44 short sellers and activists |

The full design, including every ADR, lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### Attention update

Each tick, attention decays and then picks up exposure from every information layer:

```
a <- decay(a) + sum over layers of  w_layer * (A_layer @ exposure)
```

`A_layer` is a sparse CSR matrix per layer (row i lists who node i hears). It runs as a CuPy sparse mat-vec on the GPU and a deterministic NumPy CSR mat-vec on the CPU.

### The truth filter

The crisis room has an LLM (Grok, via `xai-sdk`, structured JSON output) draft candidate public statements. Each claim in a statement is extracted as JSON and checked:

- **Numeric claims** (cash, capital, insured share, facility access) are checked in code against the bank's true simulated ledger.
- **Vague wording** is judged by an LLM against the same ledger.
- **One false claim rejects the whole statement.**

This is a hard constraint, not a penalty. If the optimizer could trade a little dishonesty for fewer withdrawals, it would learn to lie. See `agents/crisis_room.py`.

### Response search

The engine scores crisis-response plans, such as pledging bonds in advance, borrowing deposits from big banks, a credit line or a guarantee, each triggered at a cash or withdrawal threshold. It runs them across many seeded versions of the same day. The UI's 10 / 25 / 50 selector sets how many simulated days each plan is scored on.

Of 321 plans considered, 31 were simulated directly. The rest were worked out from simulated trajectories or knocked out (193 too early, 11 fail, 3 not applicable). Plans whose cash prediction depended on funding the engine had not simulated were withheld from ranking rather than guessed.

In the final run, the winning plan is bonds pledged in advance, then deposits from big banks once withdrawal requests reach 20% of deposits. It ranks first in 5 of 7 changed-assumption tests, and the closest margin is about $516M (0.3% of deposits).

## Calibration and validation (M27)

- **Targets (E1, SVB, March 9–10 2023):** about $42B of $175B in deposits requested on day one (requested fraction 0.249), about 81% requested by the next open, and about 24% actually settled on day one.
- **Objective:** weighted squared log-error between simulated and target fractions, averaged over 8 shared seeds per candidate. Soft decision gates (a sigmoid around each depositor's breaking point) are annealed toward hard decisions so the objective stays smooth early in the search.
- **Search:** stage 1 is a batched sweep that brackets the tipping point. Stage 2 is CMA-ES (`cma`) inside that bracket, with restarts. The behavioral vector has 17 coordinates. An identifiability sweep (`calibration/diagnostics/identifiability-sweep.md`) showed that only three move the targets enough to be fit: worry bias, mobile share of uninsured deposits, and the decision-to-request rate. Those three are fit, and the rest stay at documented priors.
- **Persisted fit** (`m27-e1-svb37-20261102`, seed 3060): worry bias −1.737, mobile share 0.9635, decision-to-request rate 0.00210/min, capacity multiplier 13.41.
- **Holdout (E5, Signature, March 10 2023):** NYDFS reports $18.6B withdrawn on day one (20% of deposits), plus about 1,000 pending wires worth $4.6B. That gives a requested fraction of at least 0.249. Signature is never in the objective. The model gives 0.2598.
- **Every parameter is tagged in `params/registry.csv`** (514 rows) as VERIFIED, SOURCED, FITTED or ASSUMED, with its source.

## Engineering challenges

1. **Batched calibration on one 12 GB GPU.** Each CMA-ES generation is 96 full simulations (12 candidates × 8 seeds) of 100,000 depositors over 460 ticks. They run as one batched array program (`engine/gpu_sim.py`, `engine/gpu_full.py`) instead of 96 processes. Attention, contagion, the market quote and payments became CuPy array math, with fused `ElementwiseKernel`s for the depositor step. That cut a generation from about 180 s to 13.8 s. CUDA-event profiling then broke down the 26.65 ms tick (rumor spread 8.2 ms, depositor step 6.8 ms GPU, about 10 ms of unwrapped host code), and a per-tick host sync was removed from the depositor step. One persisted fit used 1,221 of 11,960 MiB of GPU memory.
2. **CPU/GPU path agreement.** The GPU path has to give the same answer as the reference CPU engine. It is checked cell by cell (`scripts/gpu_parity.py`) on matching outcomes, trigger minutes within 1 minute, and requested deposits within 0.01 percentage points at the end and at every sample. All 21 cells agree. A kernel-only diagnostic narrowed the remaining 0.0036-point gap to float32 paths outside the depositor kernel, so the fused kernels were not the cause. The market rules and rumor classification share one definition across both paths.
3. **Finding and fixing a fit that was not fitting anything.** Diagnostics showed every candidate pinned at the uninsured-deposit ceiling (0.9238) by minute 30. The cause: in soft-gate mode every depositor had a nonzero request on tick 1, so contagion jumped to 1. The fix was a decide-once rule (SVB-28). An earlier fit had a near-zero objective (0.000124) only because the mobile-share parameter set the next-open number directly. It was thrown out and redone. `docs/svb-fit-verification.md` records each step.
4. **Deterministic replay with LLM agents.** Grok agents (by design count, 15 VIP depositors, 2 bank CEOs and about 7 media, analyst and short-seller agents) act inside a simulation that has to replay byte-for-byte. Seeded Philox streams and a SQLite LLM cache make that possible, and the default demo runs fully offline (`STAMPEDE_OFFLINE=1`).
5. **Honest optimization.** The truth filter above, plus withholding 111 plans whose predictions the engine couldn't support, so the search can only win on truthful statements and simulated evidence.
6. **Testing a stochastic system.** Over 1,000 offline pytest tests, including a few Hypothesis property tests, golden fingerprints, determinism smoke tests, CPU/GPU agreement checks and an architecture-doc gate in CI.

## Tech stack

Python 3.11+ (developed on 3.12), NumPy, SciPy, CuPy (CUDA 12), Numba, `cma`, Pydantic v2, FastAPI + Uvicorn + WebSockets (msgpack frames), React + TypeScript + Vite, `xai-sdk` (Grok), SQLite, pytest + Hypothesis, ruff, mypy (strict on the core packages).

**Hardware:** NVIDIA RTX 3060 12 GB, Linux, CuPy 14.2 (`cupy-cuda12x`). The CPU fallback runs everywhere, but it does not reproduce the GPU timings.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,server]'
# optional GPU path (match your CUDA driver):
pip install cupy-cuda12x

# offline tests (no API key needed)
STAMPEDE_OFFLINE=1 python -m pytest -q

# live demo server (calibrated modules; GPU if available)
STAMPEDE_MODE=real STAMPEDE_BACKEND=gpu PYTHONPATH=. python -u -m ui.server
# then open http://localhost:8000/ui
```

| Variable | Default | Meaning |
|---|---|---|
| `STAMPEDE_OFFLINE` | `1` | No paid LLM calls; cached or rule-based agents |
| `STAMPEDE_MODE` | `fake` | `real` runs the calibrated engine; `fake` plays a scripted arc |
| `STAMPEDE_BACKEND` | `cpu` | `gpu` uses CuPy when installed |
| `STAMPEDE_PORT` | `8000` | Server port |

Calibration (long; GPU recommended):

```bash
python -m calibration.run_fit --gpu-full --seed 3060 --population 12 --eval-minutes 1440 --holdout-minutes 780
python scripts/check_gpu_sim_agreement.py   # CPU/GPU agreement
python scripts/benchmark_gpu_sim.py         # timings
```

Live LLM agents need `XAI_API_KEY` in your environment. It is never stored in the repo.

## Project structure

```
agents/        LLM agents: crisis room + truth filter, VIP, CEO, media, shorts, guardrails, LLM cache
calibration/   M27: CMA-ES fit, holdout, diagnostics, registry updates
contracts/     Pydantic event and state schemas shared by every module
engine/        simulation core: clock, network, attention, depositors, bank, liquidity, market, policy, futures, GPU paths
params/        registry.csv: every parameter with status and source
scenarios/     presets (SVB-like, Signature-like, fictional banks)
scripts/       run, benchmark, parity, search and rescore tools
tests/         pytest + Hypothesis suite
ui/            FastAPI server + React crowd and market screens
docs/          ARCHITECTURE.md (single source of truth), fit verification, live demo, local LLM, research notes
```

## Limitations

- **Calibrated on one event.** Three behavioral parameters are fit to SVB, and Signature is a single holdout point. That is evidence the dynamics transfer, not proof. The Signature target is a lower bound, since requests that were never submitted aren't counted.
- **The settled-day-one target is not fully reached.** Stage 2 plateaus at 22.0% settled against a 24% target (still within the ±25% band, but `reachable: false`).
- **The stock is reported, not fitted.** SVB's −60% move on March 9 followed a loss announcement that the stock path doesn't model, so the stock-price term is weighted 0.
- **Scale.** The design budget is 1,000,000 depositors (Tier 2). Calibration and the timings above are measured at 100,000 depositors × 96 parallel simulations. A full-detail 1M-depositor tick has not been benchmarked.
- **Messaging effects are assumptions.** The effects of statements and rate rises are engine assumptions, not fitted causal estimates. Many registry rows are ASSUMED and labeled as such.
- **Two known test failures** (an old M03 fingerprint and a non-bank CUDA price path) also reproduce on the earlier baseline; they are known issues, not regressions.
- **Not a prediction product.** Banks in the demo are fictional. Price paths are not calibrated market forecasts.

## Author

Designed and built by **Sebastian Villalba**: architecture, simulation engine, GPU paths, calibration, LLM agents, UI and countermeasures.

Historical data: FDIC, Federal Reserve and NYDFS reports on the March 2023 failures of SVB and Signature Bank (sources listed per parameter in `params/registry.csv`).
