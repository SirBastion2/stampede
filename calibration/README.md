# M27 calibration (GPU fit)

Long runs use `python -m calibration.run_fit`. CI and quick checks use `calibration.fit.fit_behavioral_parameters` with the fake engine (default).

The E1 fit scores **requested** withdrawals (`WithdrawalRequested`) with payment-channel caps and bank closure **off**. E1 already counts requests ($42B withdrawn plus $100B scheduled). Soft gates (a sigmoid around each M06 breaking point) are annealed toward hard decisions. Each candidate is the mean loss over 8 shared seeds. The search is three identifiable log-space parameters (worry bias, mobile share, h; identifiability sweep in `calibration/diagnostics/`); the other coordinates stay at their ASSUMED values. Stage 1 is a batched sweep that brackets the tipping point; stage 2 is CMA-ES inside that bracket with restarts. Best-so-far is written as the run goes; there is no wall-clock cap. `validate_fitted_parameters` replays SVB with caps and closure **on**. Default non-calibration engine paths are unchanged.

## Environment

| Variable | Default | Role |
| :---- | :---- | :---- |
| `STAMPEDE_CALIBRATION_ENGINE` | `fake` in tests; `run_fit` sets `real` | Real M02–M09 evals vs fake tick harness |
| `STAMPEDE_CALIB_EVAL_MINUTES` | registry / `--eval-minutes` | Simulated minutes per candidate |
| `STAMPEDE_CALIB_DEPOSITORS` | Tier-0 registry count (100k) | Override depositor count for smoke runs |
| `STAMPEDE_BACKEND` | `cpu` | `gpu` enables M05 CuPy balance solve plus M02/M03/M06 CuPy paths when CUDA is visible |
| `STAMPEDE_CALIB_GPU_SIM` | off | `1` selects the batched depositor eval in `engine/gpu_sim.py`. Requires `STAMPEDE_BACKEND=gpu`. `--gpu-sim` sets this. |
| `STAMPEDE_CALIB_GPU_FULL` | off | `1` moves attention, market, and payments onto the same array kernels. Requires `--gpu` and `--gpu-sim`. `--gpu-full` sets all three. |

## `run_fit` flags

- `--workers N` (default `1` without `--gpu`, `2` with `--gpu`): spawn-pool candidate evaluators per generation. On a 12-core CPU-only box use `--workers 10`; with `--gpu` keep `N` small (2–3) so workers share the RTX 3060. Results match `workers=1` for the same seed. With `--gpu-sim` the generation is one in-process batch, so pass `--workers 1`.
- `--gpu` (recommended on the RTX 3060): sets `STAMPEDE_BACKEND=gpu` for M05 balance solve and engine CuPy modules. Logs `device=cuda:0 …` when active; falls back to NumPy on CPU-only hosts.
- `--gpu-sim`: sets `STAMPEDE_CALIB_GPU_SIM=1`. Both `--gpu` and `--gpu-sim` are required. `--gpu-sim` alone logs a warning and stays on the legacy per-depositor engine. With the flags set and no CUDA device, the same path runs on NumPy.
- `--gpu-full`: sets `STAMPEDE_CALIB_GPU_FULL=1` and implies `--gpu` plus `--gpu-sim`. Attention, contagion, the market quote, and payments run as CuPy (or NumPy) array math. Objective, parameters, and the flag-off engine stay the same.
- `--resume`: continue from `calibration/runs/<run-id>/checkpoint.json`.
- `--write-registry` (off by default, SVB-49): also write the fitted rows into `params/registry.csv` at the end. Without it a run only proposes rows: `summary.json` lists `persist_command` (`python -m calibration.persist_fit --run-dir <run>`), and persisting needs explicit maintainer approval. `fitted_rows_written` is true only when a row was updated or appended.
- The validation board is written to `calibration/runs/<run-id>/validation-board-<run-id>.md` (SVB-55), not into the tracked `docs/` tree.

Launch a full-scale fit on the RTX 3060 with the batched path (no time cap; checkpoints keep the best vector):

```bash
python -m calibration.run_fit \
  --seed 3060 \
  --extra-seeds 3061,3062,3063 \
  --population 12 \
  --eval-minutes 390 \
  --holdout-minutes 780 \
  --checkpoint-every 2 \
  --gpu-full \
  --workers 1
```

`--gpu-full` is the 3060 path. `engine/gpu_sim.py` plus `engine/gpu_full.py` move attention, depositor contagion, the market quote, and payments onto CuPy (NumPy when no device). Ledger and securities stay on their `step` methods. E1 day-1 and two-day fractions may differ from the legacy engine by float reduction order, up to the ASSUMED row `M27 GPU sim E1 relative tolerance` (1e-6).

On the 3060, compare one full-scale candidate and time the paths:

```bash
python scripts/check_gpu_sim_agreement.py
python scripts/benchmark_gpu_sim.py
```

Agreement prints `max_rel_error` and exits 0 when that error is within 1e-6 and the closed flags match. The benchmark prints `legacy_engine_s`, `cpu_numpy_sim_s`, and `gpu_sim_s` (`unavailable` when CUDA is missing). Smoke either script with `--minutes` and `--depositors`. The cloud VM that added this path has no GPU: reduced-size NumPy parity is what CI runs. Full-scale timing and CuPy agreement were not executed there.

Artifacts land under `calibration/runs/<run-id>/` (`calibration.log`, checkpoints, `summary.json`). FITTED M06 rows are written once per run when a seed pass completes.

## Non-bank re-score (`run_nonbank_fit --score-only`, SVB-38c)

`--score-only <run>/converged.json` does not fit. It loads that run's vector and seed (also `summary.json` or `best_vector.json`), re-measures the equity objective at the source's `eval_minutes_fit`, scores crypto and rates at `--crypto-holdout-minutes` / `--rates-holdout-minutes` (both required), and by default also scores them at the source run's own holdout horizons as a baseline (`--baseline-holdout-minutes N` overrides, `0` skips). Fit-only flags (`--generations`, `--population`, `--patience`, `--extra-seeds`, `--resume`, `--checkpoint-every`, `--improvement-epsilon`, `--holdout-minutes`) are an error. A behavioral vector needs `--gpu-full` (the fake engine ignores it), and a vector of the wrong length is refused.

It writes only `summary.json` and `calibration.log` in a new run dir and refuses the source dir or an existing summary. The summary has `term_status` (evaluated, unset, excluded with the reason, or weight 0 per term), per-term losses, `crypto_profile` / `rates_profile` (per-day open, close, low, volume, new lows, `share_of_drawdown_after_day1`, `covers_horizon`, `clock`, event counts per day), `comparison` against the source's reported numbers, wall time per holdout path with a fixed-plus-per-minute `timing_model`, device, CuPy-pool and peak host memory, and `fitted_rows_written: false` with `persist_command: null`. It has no `best_vector` key, so `persist_fit` can't read the directory.

## Diagnostics

Every CLI calibration run writes `/calibration/diagnostics/<run-id>/`:

- `parameter_sweeps.json` — best vector, mean, and samples per generation
- `withdrawn_share.json` — share of depositors who withdrew
- `bind_rates.json` — how often payment caps (`queued_remaining > 0`) and closure bind
- `seed_variance.json` — objective mean and variance across seeds
- `parameters_at_bounds.json` — vector components sitting on 0 or 1
- `sigma_history.json` — CMA-ES step-size history
- `summary.json` — short roll-up

The fit calls `calibration.diagnostics.record_generation(run_dir, generation, objective, vector, **data)` with per-generation fields. `python -m calibration.diagnostics.summary [run-id]` prints that roll-up.

## Constraints

Unrelated inputs (`I blinked`, `It's raining in Princeton`, `ajjfalk`) use the one shared `classify_rumor_market` rule: category `irrelevant`, rumor-box message `Not market-related: no effect on the simulation.`, and an identical event log. A market event about another company (for example `GameStop stock fell`) may still move crowd and price. Daily deposit requests must stay within the ASSUMED `Non-bank daily deposit-request bound` (0.1% of deposits per day). Violations add `nonbank_request_bound_penalty` in `calibration/nonbank.py` and fail validation.
