# SVB fit verification (SVB-10 / SVB-23)

**Current fit: §8.** This file is the development log of the calibration, in order. §1–§7 record earlier fits and the fixes between them (the `SVB-nn` labels are internal change IDs); the fit the README reports, and the one persisted in `params/registry.csv`, is the one in §8. The §1 timings are the per-generation benchmark the README cites.

All runs: RTX 3060 12 GB, Linux.

## 1. Timings (main `ea012af`)

**Setup:** `python -m calibration.run_fit --gpu-full --seed 3060 --population 12 --generations 3 --patience 0`, with 12 candidates × 8 seeds = 96 lanes, 100,000 depositors, a 460-tick schedule (390 × 1 min + 70 × 15 min), and eval_minutes 1440.

- **Stage 1: 13.8 s per generation,** against about 180 s originally, so about 13× faster.
- **The 3-generation run took 320.6 s,** including setup and about 4 minutes of stage 2.

**Tick sections** (CUDA events; 26.65 ms wall per tick, including instrumentation):

| Section | GPU ms/tick | Host ms/tick |
|---|---|---|
| Rumor spread (`_spread_posted_rumors`, host M03) | 8.18 | 8.18 |
| Depositors (`step_depositor_batch`) | 6.84 | 2.64 |
| Headline flags (`_pending_host_flags`, host sync) | 0.05 | 3.55 |
| Hub shock | 0.77 | 1.22 |
| Attention | 0.41 | 0.40 |
| Price / news | 0.09 | 0.44 |
| Other unwrapped host code | n/a | 10.06 |

SVB-28 removes the per-tick `mobile_share.get()` host sync from the depositor section.

**Gather bench** (`tools/svb_gather_bench.py --depositors 100000 --seeds 8 --lanes-per-seed 12`):

| Degree | (A) expanded int64 | (B) per-seed 2-D int32 | (C) per-seed CSR SpMM (main) |
|---|---|---|---|
| 4 (registry M06 contact degree) | 41.0 ms | 8.1 ms (5.0×) | 12.7 ms (3.2×) |
| 20 | 206.5 ms | 44.5 ms (4.6×) | 54.6 ms (3.8×) |

Switching to (B) would save about 1–2 ms per tick. It's optional.

## 2. What the measurements showed, and what fixed it

| Finding | Fix |
|---|---|
| **Requests by day-one close sat at the uninsured ceiling (0.9238)** for every candidate, so next-open was constant and only the stock ratio moved the loss | SVB-24 added the mobile-share cap; SVB-26 fit only price impact and mobile share |
| **After SVB-27, requests still hit the cap (0.812) by minute 30** for every seed and vector. The cause: in soft mode every depositor had a nonzero amount on tick 1, so `already` was true for everyone and the contagion share jumped to 1; small per-tick probabilities then compounded. | **SVB-28:** each depositor decides once. Hard gates request once; soft gates ratchet a decided probability. |
| **The saturation probe's "low contact weight / feedback" run didn't lower those coordinates** (its printout shows `vec[1,12] = 0.5 0.5`) | `tools/svb28_probe.py` sets the coordinates directly |
| **SVB-16's `dt` scaling reached the fit** (forcing dt = 1 cut gen-0 overnight requests 6.4×), but couldn't move the objective, because nothing was left to request overnight | Moot after SVB-28 |
| **Signature (E5) printed 0.239979,** the SVB settled number | SVB-26 / SVB-27d: Signature compares *requested* day one with 0.249 |
| **Next-open 0.9237 against 0.81 "passed"** | SVB-26: ±10% relative band |
| **`best_vector.json` lagged the best objective** | SVB-26: written from the global best |
| **The non-bank days terms were constants:** E16's 33 days needed 47,520 sim minutes and E13's 1 day needed 1,440, but the runs were 390 minutes | **SVB-29:** a trading-day clock (390 minutes per day; E16 = 23 trading days), with unreachable days targets excluded and reported |
| **`persist_fit` printed 1.24192 / 0.877877, but wrote 1.18464 / 0.878938** | **SVB-31:** print the cells actually written, and refuse to persist the same run twice |

## 3. Seed test

**Setup:** 4 near-optimal vectors, evaluated on two disjoint 4-seed sets.

**Result:** a constant offset between the sets of 0.00105–0.00109, the same ranking in both, and a 0.00309 spread between the vectors.

**Decision: 4 seeds.** Re-check after SVB-28: with saturation gone, the spread between vectors will grow, which makes the test more informative.

## 4. Identifiability

SVB-26 fit price impact (index 14) and mobile share (index 15); contact weight, feedback gain, and dispersion stayed at their priors.

That was right under saturation, since only end-of-day numbers moved. **After SVB-28, contact weight and feedback gain may become visible to the end-of-day targets.** Rerun the sweep, plus `tools/svb28_probe.py`, before choosing the refit set.

## 5. Persisted fit (pre-SVB-28): `m27-e1-svb26-20261101`, main `936ea4b`, persisted in `e7ed9c3`

**Setup:**
- Seed 3060.
- CMA-ES over the two identifiable coordinates.
- 4 shared seeds.
- A 1,440-minute stage-1 schedule, and a 780-minute holdout.
- Wall time 869 s; GPU memory 1,221 of 11,960 MiB.

| Target | Observed | Expected | Band | Result |
|---|---|---|---|---|
| Requests by next open (caps off) | 0.8119 | 0.81 | ±10% relative | pass |
| Settled day one (caps on) | 0.239979 | 0.24 | ±25% | pass |
| Placebo insured community bank | survives | survives | — | pass |
| Signature (E5), requested day one | rerun needed after SVB-28 | 0.249 | ±35% | — |

**Persisted values:**

| Row | Value |
|---|---|
| `M14 price impact (ASSUMED)` | 1.18464 (normalized 0.559023) |
| `Depositor mobile share of uninsured deposits (FIT)` | 0.878938 (normalized 0.498490) |
| `M07 capacity multiplier (FIT)` | 4.20575 (stage 2 re-solved, about 8.7 min) |
| Other coordinates | priors (0.5) |
| Stage-1 objective | 0.000124 |

**Reading it:** two free coordinates and two stage-1 targets, so a near-zero objective is expected. Under the leak, next-open was simply 0.879 × 0.9238 = 0.812: the mobile share set it directly, whatever the dynamics did. That's why the fit has to be redone after SVB-28.

**Non-bank fit `nonbank-svb26-c-20261101`:** objective 7.4732 (was 30.8861); rates holdout loss 43.53. It was fit with E15 unset and with unreachable days terms (fixed by SVB-27c and SVB-29), so it has to be redone too.

## 6. Post-SVB-27 checks

| Check | Result |
|---|---|
| Coupling at 460 ticks (`tests/engine/test_svb18_coupling_full_schedule.py`, `b0967b6`) | **XPASS:** no coupling, so SVB-18b isn't needed |
| Saturation probe (`satprobe_after.txt`) | Flat 0.812 everywhere, but the low vectors weren't lowered. The mechanism is shown by `tests/engine/test_svb28_decide_once.py`. |
| QLoRA smoke run | 5 steps, loss 4.69, peak GPU 8,845 MiB, adapter saved |

## 7. Redo, in order

1. **Merge SVB-28,** apply `svb28_withdrawal_decision_mode.json`, and run the full suite on the 3060. Expect some golden values to need refreshing.
2. **Run `tools/svb28_probe.py`.** Next-open should now differ across the probes, with a spread above 0.01. Compare with `--legacy`.
3. **Rerun the identifiability sweep** and choose the fit set (§4).
4. **SVB refit:** `python -m calibration.run_fit --gpu-full --seed 3060 --population 12 --generations 40 --patience 30 --eval-minutes 1440 --holdout-minutes 780 --run-id m27-e1-svb28-<date>`, then `python -m calibration.persist_fit --run-dir calibration/runs/m27-e1-svb28-<date> --stage2`. Persist each run once (SVB-31 enforces it).
5. **Merge SVB-29,** apply `svb29_nonbank_trading_days.json`, then the non-bank refit: `python -m calibration.run_nonbank_fit --gpu-full --seed 3060 --population 12 --eval-minutes 390 --generations 40 --patience 8 --rates-holdout-minutes 8970 --run-id nonbank-svb29-<date>`, then `python -m calibration.persist_fit --run-dir calibration/runs/nonbank-svb29-<date> --kind nonbank`.
6. **Record the new results** in §5, then run the drills: board, diagnostics, recovery test, and the GPU invariance hash.

## 8. Persisted fit: m27-e1-svb37-20261102 (SVB-40, 2026-11-02)

Saved with `python -m calibration.persist_fit --run-dir calibration/runs/m27-e1-svb37-20261102 --stage2` (GPU full, RTX 3060, seed 3060) after SVB-40a/b. Stage 2 now solves before the fitted rows are written.

| Row | Before | Persisted |
|---|---|---|
| M06 worry bias | -2.4 (ASSUMED) | -1.73736 (FITTED) |
| Depositor mobile share of uninsured deposits | 0.878938 | 0.963547 |
| M07 decision-to-request rate (h) | 0.00166667 | 0.00210008 |
| M07 capacity multiplier | 4.20575 | 13.414 |

Stage 2 report: target 0.24, reachable **false**, plateau 0.22023, multiplier 13.414 (capacity is non-binding at or above it).

Board (all-0.5 vector against the saved registry, seed 3060, eval 1440 min):

| Target | Observed | Expected | Passed |
|---|---|---|---|
| E1-requested-day1 | 0.244923 | 0.249 | yes |
| E1-next-open | 0.818517 | 0.81 | yes |
| E1-settled-day1 | 0.21846 | 0.24 (±25%) | yes |
| E5-day1 | 0.259811 | 0.249 | yes |
| placebo | 1 | 1 | yes |
