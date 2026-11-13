# STAMPEDE: Master Architecture (living document)

| Field | Value |
| :---- | :---- |
| Version | 1.105.3 |
| Repo location | /docs/ARCHITECTURE.md (this file) |
| Author / doc owner | Sebastian Villalba |
| Code areas | L1–L8 (§10.1) |

**Summary: a wind tunnel for bank runs. A large simulated depositor population (100,000 in the calibrated runs; the engine is designed to scale toward a million), about 25 Grok agents (VIP depositors, bank CEOs, a regulator, journalists, short sellers), banks with real balance sheets, and a live stock market for each bank. You start a rumor or a rate shock, watch the crisis spread on the dashboard, and find the true message or policy that stops it.**

# 0\. How this document works (update protocol)

This file is the single source of truth. Code follows the document; when they disagree, fix one of them in the same pull request (PR).  
Every PR that adds or changes behavior must update:

1. Its row in the module registry (§4).  
2. Its module spec (§5).  
3. Contracts (§6), if any event or schema changes.  
4. The parameter registry (§8.1), if a number changes. Every parameter has a source or is marked FIT or ASSUMED.  
5. The change log (§16), with one line.  
6. An ADR (§16.2) if a design decision was made. ADR stands for architecture decision record: one entry per choice, with the reason.

**Rules:**

* **Status values: Implemented, or CUT. Never delete** a section; mark it CUT and give the reason.  
* **Versioning:** bump the minor version (+0.1) for a new module or feature, and the patch version (+0.0.1) for clarifications.   
* **Code areas:** each section belongs to one code area (L1–L8, §10.1).  
* Read §4–§8 before every change. A change that would break a contract or a parameter source is redesigned first.  
* **Accuracy rule:** no mechanism ships without an evidence entry in §8 (VERIFIED, TO VERIFY, or ASSUMED). This table answers "why do you believe that?" for every mechanism.

# 1\. Product summary

**Users:**

* A bank's crisis team rehearsing a run before it happens.  
* Regulators testing policy.  
* Students learning how financial panics work.

**What it does:**

1. Simulates panic spreading through depositors, banks, and bank stocks.  
2. Lets a user inject rumors, rate shocks, and policies.  
3. Generates crisis statements, filters out any that aren't true, and runs every remaining option as a parallel future.  
4. Shows outcomes as ranges, including the odds of failure, never as one prediction.

**How it proves itself: it's fit on Silicon Valley Bank (SVB) in 2023, then tested without refitting on one held-out bank, Signature (§8.2).**  
**Demo spine:**

1. Calm system.  
2. The user injects a rumor or shock.  
3. Panic and losses spread.  
4. The crisis room finds a fix.  
5. Time rewinds and the run replays with the fix applied.

# 2\. System overview

## 2.1 Layers

`┌──────────────────────────────── SCENARIO + POLICY LAB (M22, M28) ────────────────────────────────┐`  
`│                                                                                                  │`  
`│ MACRO & RATES (M01) ── yield curve, Fed path, CRE prices, unemployment                           │`  
`│     │                                                                                            │`  
`│     ▼                                                                                            │`  
`│ INFORMATION (M02–M04) ── feeds, group chats, news, analysts, attention, rumor mutation ◄───────┐ │`  
`│     │                                                                                          │ │`  
`│     ▼                                                                                          │ │`  
`│ DEPOSITORS (M05–M07) ── beliefs, payroll, insurance                                            │ │`  
`│     │                                                                                          │ │`  
`│     ▼                                                                                          │ │`  
`│ BANKS (M08, M09, M12, M23) ── balance sheets, securities, liquidity, management                │ │`  
`│     │                                                                                          │ │`  
`│     ▼                                                                                          │ │`  
`│ MARKETS (M14) ── per-bank stocks, short sellers (M44) ─────────────────────────────────────────┘ │`  
`│     │                                                                                            │`  
`│     ▼                                                                                            │`  
`│ SYSTEM (M20) ── regulators and closure triggers                                                  │`  
`│                                                                                                  │`  
`│ CRISIS ROOM (M24–M26) ── messages, truth filter, persona panel, 16 parallel futures              │`  
`└──────────────────────────────────────────────────────────────────────────────────────────────────┘`

## 2.2 Feedback loops the model must reproduce

| Loop | Chain | Evidence |
| :---- | :---- | :---- |
| L1 Panic loop | Rumor → attention → withdrawals → cash drain → forced bond sales → realized losses → stock drop → news → more attention | SVB, Signature (§8.2) |
| L4 Policy loop | Guarantee or emergency lending → withdrawals slow → insurance-fund cost and moral hazard rise | March 12, 2023 actions |

## 2.3 Time and execution model

* **Clock (M00):** one tick is one simulated minute during business hours. Overnight and weekend periods are skipped, but events still fire during them (statements and closures).  
* **Calendar:** business days, wire cutoff times, lending-facility hours, and weekends, which is when the FDIC resolves failed banks.  
* **Execution:**  
  * Agent and graph updates run as CUDA kernels on the GPU (structure-of-arrays layout).  
  * Bank accounting and order books run on the CPU in the same process.  
  * Grok calls are asynchronous and cached.  
* **Determinism:** each scenario has a seed. The same seed, inputs, and cached LLM outputs always produce the same run, which keeps the demo replayable.  
* **Futures:** the crisis room and ensemble runs use a coarser setup: 5-minute ticks and 250,000 depositors per future. Only the chosen future replays at full detail.

# 3\. World model

| Entity | Count (by tier) | Key state | Module |
| :---- | :---- | :---- | :---- |
| Depositors | 100k (Tier 0\) → 1M (Tier 2), split across banks by size | Segment, balance, insured amount, belief, threshold, attention, needs, second-bank flag | M05–M06 |
| Banks | 1 (Tier 0\) → 2 (Tier 2\) | Balance sheet, securities, loans, capital, liquidity ladder, stock | M08, M09, M12 |
| Information nodes | Same people as depositors, plus \~5 media and analyst agents | Followers, group memberships, credibility | M02–M04 |
| Traders | \~300 per bank stock | Type, cash, inventory, signals | M14 |
| Regulators | One regulator role (as an agent) | Triggers, tools, insurance-fund balance | M20, M22 |
| Grok agents | \~25 (15 VIPs, 2 CEOs, 1 regulator role, \~7 media and short sellers) | Persona, memory, tools, decision rule extracted for futures | M04, M22, M41, M42, M44 |

# 4\. Module registry

Retained modules: this registry lists the 27 modules in the current scope (M00–M09, M12, M14, M20, M22–M30, M33, M34, M41, M42, M44). Module IDs keep their original numbers, so gaps in the sequence are expected. The tier sets when a module is needed (§11). The work-stream build order is in §10.

| ID | Module | Tier | Stream | Status | Evidence |
| :---- | :---- | :---- | :---- | :---- | :---- |
| M00 | Clock, calendar, event scheduler | 0 | L1 | Implemented | Design |
| M01 | Macro and rates (yield curve, Fed path, property prices) | 1 | L2 | Implemented | Design (ADR-019) |
| M02 | Multilayer information network | 0 | L1 | Implemented | Cookson et al.; ADR-027 |
| M03 | Attention and rumor engine (with mutation) | 1 | L1 | Implemented | Cookson et al. (ADR-028) |
| M04 | Media and analyst agents | 2 | L5 | Implemented | NYCB coverage |
| M05 | Depositor population and insurance | 0 | L1 | Implemented | SVB, Signature; ADR-065 |
| M06 | Depositor decision model | 0 | L1 | Implemented | Kazinnik; Cookson (ADR-029, ADR-067, ADR-081, ADR-082, ADR-084, ADR-088) |
| M07 | Payments and withdrawal channels | 1 | L2 | Implemented | Signature wire backlog |
| M08 | Bank balance sheet and accounting | 0 | L2 | Implemented | SVB filings |
| M09 | Securities and interest-rate risk | 0 | L2 | Implemented | SVB \$1.8B loss |
| M12 | Liquidity and funding facilities | 1 | L2 | Implemented | FHLB March 2023; Signature |
| M14 | Per-bank stock: order book and traders | 1 | L3 | Implemented | Cookson et al. (ADR-023, ADR-068) |
| M20 | Regulators and supervision triggers | 1 | L4 | Implemented | SVB closure |
| M22 | Policy lab | 2 | L4 | Implemented | March 12, 2023 |
| M23 | Bank management agent | 1 | L5 | Implemented | SVB March 8 |
| M24 | Crisis comms room | 1 | L5 | Implemented | Kazinnik |
| M25 | Persona panel (Grok) | 1 | L5 | Implemented | Kazinnik |
| M26 | Ensemble and futures engine | 1 | L1 | Implemented | Design (ADR-032) |
| M27 | Calibration and validation harness | 1 | L8 | Implemented | §8.2 (ADR-085, ADR-086, ADR-087, ADR-045, ADR-050, ADR-051, ADR-052, ADR-053, ADR-054, ADR-056, ADR-058, ADR-059, ADR-060, ADR-061, ADR-062, ADR-063, ADR-066, ADR-067, ADR-068, ADR-069, ADR-083, ADR-084, ADR-089) |
| M28 | Scenario builder, presets, replay | 1 | L8 | Implemented | Design |
| M29 | UI: crowd screen | 0 | L6 | Implemented | Design (ADR-042, ADR-044, ADR-135, ADR-136, ADR-137, ADR-138, ADR-139, ADR-140, ADR-141, ADR-142) |
| M30 | UI: market screen and crisis room | 1 | L7 | Implemented | Design |
| M33 | Observability, determinism, demo mode | 0 | L8 | Implemented | Design (ADR-026) |
| M34 | Safety and content guardrails (/agents/guardrails.py) | 0 | L5 | Implemented | Design (ADR-055) |
| M41 | Named VIP depositor agents (Grok) | 2 | L5 | Implemented | Design (ADR-040) |
| M42 | Bank CEO agents (Grok) | 2 | L5 | Implemented | Design (ADR-039) |
| M44 | Short-seller and activist agents | 3 | L5 | Implemented | ASSUMED (ADR-038) |

# 5\. Module specifications

Each spec gives the module's purpose, state, how it updates, its inputs and outputs, its parameters (see §8.1), and how it's validated.

## Core engine

### M00 Clock, calendar, event scheduler (Tier 0 · L1)

**Purpose:** advance time and fire scheduled and triggered events in a fixed order.  
**State:** sim time, calendar flags (business hours, wire window, facility hours, weekend), and a priority queue of events.  
**Update order within each tick:**

1. Macro.  
2. Information.  
3. Depositors.  
4. Payments.  
5. Banks.  
6. Markets.  
7. System and regulators.  
8. UI snapshot.

**Outputs:** Tick and every scheduled event (§6). Calendar numbers come from §8.1 (M00 rows in `params/registry.csv`).

### M01 Macro and rates (Tier 1 · L2)

**Purpose:** set the external economy.  
**State:** Treasury yield curve (3 months to 30 years), policy rate, a commercial-real-estate price index split into office, multifamily, and retail, and unemployment.  
**Update:** scenario-driven paths, such as "replay the 2022 rate hikes," with optional shocks. `MacroRates.step(t)` in `/engine/macro/rates.py` reads `Scenario.rate_path` as a piecewise-linear policy-rate schedule (flat outside the knots; the registry default when the path is empty) and clamps it to the registry floor and cap. The curve is Nelson–Siegel on the registry maturity buckets: beta0, beta2, and lambda come from the registry, and beta1 = policy rate − beta0 so the zero-tenor yield equals the policy rate (ADR-019). Unemployment and the three CRE indexes start at registry baselines. Each shock applies once, on the first tick with t at or after its `at`, in list order. Kind `unemployment` sets `params.rate`. Kind `property` multiplies `params.sector` (`office`, `multifamily`, `retail`, or `all`; default `office`) by `params.factor`. Other kinds are ignored. CRE indexes never fall below the registry floor.  
**Outputs: RatesChanged (`curve`, `policy_rate`), which feeds securities values (M09).** Property prices and unemployment stay on the module. §6 does not put them on the event, and `bank_id` is null because the path is economy-wide (ADR-019).  
**Validation:** the same scenario and seed publish the same events. A 2022-like rising policy path makes the M09 mark −D·Δy + ½·C·Δy² more negative at each step for the registry duration and convexity, so bond losses rise steadily. M01 does not check assets = liabilities + equity; that invariant is M08.

### M02 Multilayer information network (Tier 0 · L1)

**Purpose:** decide who sees what.  
**Three graph layers:**

* **A public feed:** a follower graph with heavy-tailed hub accounts.  
* **Private group chats:** founder groups and VC portfolio chats.  
* **Professional ties:** treasurers and advisors.

**State:** each layer is a CSR adjacency (`indptr`, `indices`, `data`) on the active array backend, plus a per-node credibility score and a per-node channel weight. Depositor indexes match M05. Media and analyst nodes are appended after the depositors.  
**Update:** `InformationNetwork.spread(exposure)` returns the sum over layers of layer channel weight times A times (node channel weight elementwise times exposure). The public layer is directed: row i lists the accounts i follows, and follower count is the column sum. Group chats and professional ties are stored in both directions. `NetworkModule.step(t)` builds the layers on the first tick and publishes nothing (§6, §19).

**Construction (ADR-027):** the array backend is CuPy when a CUDA device is available and NumPy otherwise. Draws use a seeded Generator on that backend, and never the backend's global RNG: CuPy's Generator has no `permutation`, so the advisor sample ranks one uniform draw per index instead (SVB-68). Each stream is deterministic for a fixed seed. The GPU stream is selected only by `select_backend` and is not required to match the CPU stream. The public feed is batched preferential attachment. Media nodes sit in the seed clique so they become hubs, then indexes are remapped so depositors stay aligned with M05. Group chats are a sparse stochastic block model on founder, VC, and other, with registry homophily. Professional ties are bipartite: advisors are the media nodes plus a registry-sized sample of founder and VC nodes, and each treasurer draws a registry number of advisors. The scheduler loads segments by calling M05 (`sample_population`, or all retail when the preset name contains "community", matching the M05 placebo) because the tick order runs M02 before M05. Degrees, homophily, credibility, channel weights, and the population tier are ASSUMED rows read through `params/loader.py`.

**Outputs:** none. Credibility, channel weights, and the three CSR layers stay on the module for M03.  
**Validation:** the same seed rebuilds the same CSR on the CPU path. Follower counts are heavy-tailed. Founder and VC blocks are recoverable from group-chat neighbors. Every professional edge joins one treasurer and one advisor. One million depositor nodes across the three layers build in under 30 seconds.

### M03 Attention and rumor engine (Tier 1 · L1)

**Purpose:** move panic based on attention, not tone. This follows Cookson et al., who found that attention predicts run losses while negative sentiment doesn't amplify them (§8.2).  
**Files:** /engine/agents/attention.py. `AttentionEngine.step(t)` is the scheduler entry.  
**State:** rumor objects (claim, source, credibility, reach, mutation lineage, notes) and per-node attention. The half-life is the Attention memory window (4 hours, E10).  
**Update:** each tick applies `a ← decay(a) + Σ_layer w · A_layer · exposure` (§18). `decay` is `0.5 ** (tick_minutes / half_life_minutes)` with tick length from the registry. Exposure is a rumor impulse at the posting node and at reposters, plus an impulse at that bank's most-followed node for HeadlinePublished, ThesisPublished, LiquidityDrawn, TradingHalt, StatementIssued, BankClosed, and VIPDecision. Those impulses are attention, not tone (ADR-001, ADR-028).  

* Reposts are draws with probability clip(rumor credibility × node credibility × attention × scale, 0, 1).  
* About 1% of reposts go to the mutator (`M03 rumor mutator sample fraction`), capped by `M03 rumor mutator max calls per rumor`. Each accepted variant is one RumorMutated.  
* Grok suggestions are cached and validated. The mutator flag defaults off; a rejected or missing suggestion uses the offline rewrite (ADR-028).  
* An AnalystNote multiplies that bank's rumor credibilities by `M03 context note credibility factor`.  
* Flagged amplifiers, all off by default (ADR-093). The poster role arrives as `RumorPosted.source_node` (the UI passes the role id). `M03 role exposure multiplier enabled` scales the post impulse by `M03 role exposure multipliers[role]`; `M03 role source node enabled` posts a role from `M03 role source nodes[role]` (fallback-graph ids; an id not in the installed graph keeps the hub). Off, the arithmetic is skipped and logs are byte-identical. The flags change this CPU engine only; `AttentionEngine` refuses to start with a flag on when `STAMPEDE_CALIB_GPU_SIM` or `STAMPEDE_CALIB_GPU_FULL` is 1, because `gpu_sim`/`gpu_full` do not read them.  
* `M03 rumor strength impulse enabled` (off) also scales the post impulse by `M03 rumor strength impulse multipliers[strength]`, where strength is the offline lexicon strength of the claim (`agents/rumor_lexicon.lexicon_tone_and_strength`, the same rule M34 classification uses). This is the one flagged exception to ADR-001; see ADR-094.  
* `M03 exaggeration feedback enabled` (off): after an accepted mutation, that rumor's repost exposure is multiplied by `min(1 + M03 exaggeration feedback gain · exaggeration, M03 exaggeration feedback cap)`, using the highest accepted `exaggeration_score` so far (ADR-095).  
* `M03 uses M02 network` (off): `build_engine` hands the real M02 module to M03; on its first step M03 installs M02's public/group/professional CSR layers and node credibility instead of the 16-node fallback graph (`m02_layers`), and roles map to M02 nodes by segment (`M02_ROLE_SEGMENTS`: media for journalist/VIP/regulator/short-seller/CEO/top shareholder, vc, founder, treasurer, least-followed retail for regular user and anonymous chat) when the role-node flag is on. Changes every real-mode log (ADR-103).  
* `M03 cure rumor enabled` (off; live on via `M29 live cure rumor`): a `RumorPosted` whose claim has positive lexicon tone (`claim_tone`) is not spread; after the tick's update the bank's node attention is scaled by `1 - M03 cure attention cut x credibility`. The second flagged exception to ADR-001 (ADR-105).  

**Backend:** `resolve_backend()` selects CuPy sparse matvec when a CUDA device is available and NumPy CSR matvec otherwise. GPU helpers are not imported on the CPU path. CPU draws use NumPy Philox, so a fixed seed repeats.  
**Layers:** CSR layers from M02 via `set_layers`. Until M02 supplies them, `fallback_layers` builds a deterministic public star, group ring, and professional ties. The node `hub-public-1` is the hub, matching the M28 stand-in source. Unmapped source nodes post as that hub.  
**Outputs:** AttentionTick (bank\_id, level, delta\_4h) once per bank. `level` is the mean node attention. `delta_4h` subtracts the level one memory window earlier, or 0 when that window is not full. RumorMutated (rumor\_id, claim, exaggeration\_score) is one event per accepted variant. §6 payloads are unchanged.  
**Functions (§19):** `resolve_backend`, `csr_matvec`, `decay_multiplier`, `update_attention`, `repost_probability`, `select_mutator_indices`, `mutate_claim`, `AttentionEngine.step(t)`.  
**Validation:** the half-life matches the memory-window parameter; a hub post outruns a small-account post; the same seed repeats on the CPU path.

### M04 Media and analyst agents (Tier 2 · L5)

**Purpose:** turn events into coverage.  
**Behavior:**

* Journalists publish when attention or price moves cross thresholds.  
* Analysts issue notes when losses surprise their expectations.  
* Grok writes the headlines.

**Outputs:** HeadlinePublished and AnalystNote (text, cited\_event\_ids, bank\_id). Both feed back into M03 and M14. Notes are text and cite the events that triggered them.

**Real mode (`/agents/media.py`):** `MediaModule.step(t)` watches AttentionTick, PriceTick, and LossRealized against registry thresholds (TBD rows fall back to ASSUMED). Crossings queue headline jobs; Grok copy uses `/agents/llm_cache.py` when online and rule templates when `STAMPEDE_OFFLINE` is on (ADR-032). Analyst notes cite triggering `cause_id`s.

**§19 API (`/agents/media.py`):**

* `load_media_thresholds(path=None) -> MediaThresholds` — reads M04 rows in §8.1; TBD cells use ASSUMED rows.  
* `attention_crossed(previous, level, threshold) -> bool`  
* `price_move_crossed(open_price, price, threshold_fraction) -> bool`  
* `loss_surprises(amount, threshold_amount) -> bool`  
* `default_headline_provider(cache=None) -> HeadlineProvider` — Grok via `/agents/llm_cache.py` when online; rule templates offline.  
* `default_analyst_provider(cache=None) -> AnalystProvider`  
* `validate_headline_payload(payload, *, cited_event_ids) -> HeadlinePayload`  
* `validate_analyst_payload(payload, *, cited_event_ids) -> AnalystPayload`  
* `MediaModule.step(t) -> list[BaseModel]` — queues coverage on threshold crossings; publishes within the latency ASSUMED row.

## People

### M05 Depositor population and insurance (Tier 0 · L1)

**Purpose:** a realistic customer base for each bank.  
**Segments:**

* Startup operating accounts, with payroll dates.  
* VC funds and advisors.  
* Wealthy founders.  
* Corporate treasurers.  
* Mostly insured retail customers.  
* Local governments, with collateralized deposits.  
* Crypto firms (Signature-like banks only).

**Balances:** heavy-tailed. The distribution is fit so the bank's insured share and top-depositor concentration match its preset (for example, SVB's top 10 depositors held \$13.3B). Mass transfer in that fit is one float64 kernel on the array module already in use (ADR-065).  
**Insurance:** calculated per depositor, per bank, per ownership category, capped at \$250,000.  
**Flags: has a second bank.**

### M06 Depositor decision model (Tier 0 · L1)

**Purpose:** decide whether and how much each depositor withdraws.  
**Rule each tick:** worry \= f(attention, share of contacts who already withdrew, belief that the bank is insolvent, stock move, advisor directive, payroll urgency, insured share). A depositor withdraws when worry passes their threshold, with the amount depending on segment and insurance gap. A depositor never requests more than `Depositor mobile share of uninsured deposits (FIT)` times its initial uninsured balance (default 0.88; 1.0 is the previous uncapped path).  
**Decides once (SVB-28):** each depositor decides at most once. With hard gates, a depositor who clears both gates and hasn't withdrawn requests its segment fraction of the uninsured gap, then never again. With soft gates (calibration), a per-depositor decided probability ratchets at `d_new = decided + (1 − exp(−dt·h)) · max(0, p_act − decided)` and only the increase is requested, against the initial uninsured balance; contagion reads that probability. That closed form composes across sub-steps when `p_act` is held constant. Every amount path (`DecisionModule.step(t, dt)`, `_evaluate`, fused `svb28_once_amount`, NumPy fallback, lean stage-1) passes the real per-tick `dt` and the same `h`. Lean applies the same-tick M41 VIP hub shock as the event path (M41 before M03). `h` is `M07 decision-to-request rate (FIT)` (prior 1/600 per minute; bounds 1/5000–1/20). Large `h` recovers the old `max(decided, p_act)` step (ADR-084). `M06 withdrawal decision mode (ASSUMED)` is `once`; `repeat` is the legacy per-tick take (ADR-076).  
**Awareness (SVB-33):** each depositor has a private awareness in \[0, 1\], starting at `aware0` (default 0, same as `decided`). Before the logits each tick, `aware += (1 − aware) · (1 − exp(−dt · (r_media·attention + r_wom·(w1/w1_prior)·share_decided + r_news·max(0, news))))`. Feature 0 is `attention · aware`, not the broadcast lane attention. Soft once-mode sizes the request as `mobile_share · initial_uninsured · gap_scale · amount_scale` (the hard-gate path still uses `segment_fraction`). Media rate stays at its registry prior (not fitted). WOM and news rates are ASSUMED. Fit index 16 is `h` (ADR-084). When gpu-sim `dt` is an integer greater than 1, the run expands to one-minute clock times so once-mode requests stay timestep-invariant (ADR-082).  
**Contagion type:** complex contagion, meaning a depositor needs several worried contacts before acting, and big hub accounts carry extra weight.  
**Messages:** bank statements shift worry by segment, using the persona panel's output (M25).  
**Validation:** SVB's first day (§8.2).

**Real mode (`/engine/agents/decide.py`):** `DecisionModule.step(t)` keeps a private copy of M05 depositor arrays and publishes one `WithdrawalRequested` per depositor who clears both gates: worry above their threshold, and a weighted count of worried contacts at least k. Worry is a sigmoid of the §18.2 weighted features. Insolvency belief is a damped Bayesian update from attention and the stock return. Hub accounts use the ASSUMED hub weight. Statements shift the worry logit and the threshold for the named segments (M25 stand-in until the persona panel is live). `guarantee_deposits` scales the uninsured gap; `cap_withdrawals` scales the amount. The mobile-share cap is applied at the end of the amount chain (VIP forced `p_act=1` included). An explicit `VIPDecision` to withdraw is published at the stated amount. The insured-community preset publishes nothing (placebo). The array backend is CuPy when a CUDA device is available and NumPy otherwise, and a fixed seed is deterministic on the CPU path (ADR-029). k and the weights are ASSUMED registry rows; the TBD k row is not read. `M06 worry bias (ASSUMED)` loads when Status is ASSUMED or FITTED so a persisted stage-1 fit can rebuild the engine for `persist_fit --stage2`; every other M06 row still requires ASSUMED (ADR-088). The SVB tolerance remains M27's fit.

**Fake mode (`/mocks/l1_engine.py`, `/mocks/demo_arc.py`):** the default scripted demo does not read the real kernel or the FIT behavioral row. After a rumor, depositor headcount follows an ASSUMED logistic over `M29 UI live crisis sim minutes`: calm, then a wavering band, then an accelerating share. A strong rumor ends in the 30–60% band; a weak rumor ends much lower. The operating-cash gauge reaches empty about 65% of the way through that window. Dollar withdrawals track the gauge, then an ASSUMED hole lets the book close. Statements and guarantee or cap levers scale only the remaining headcount, so the share never steps backward. Live poster credibility is an ASSUMED map (ADR-047).

### M07 Payments and withdrawal channels (Tier 1 · L2)

**Purpose:** withdrawals take real time.  
**Channels:** app or online, wire, ACH transfer, and branch. Each has its own capacity per tick, cutoff time, and daily limit. App, wire, and ACH capacities are scaled by `M07 capacity multiplier (FIT)` (default 1.0).  
**Effects:** wire queues build up. After the cutoff, pending wires carry over to the next business day.  
**Validation:** Signature had about 1,000 wires worth \$4.6B still pending when the bank reopened (§8.2).

## Banks

### M08 Bank balance sheet and accounting (Tier 0 · L2)

**Assets:**

* Cash and reserves.  
* **Securities:** available-for-sale (AFS) and held-to-maturity (HTM).  
* Loans, by type.  
* Other assets.

**Liabilities:**

* Deposits, by segment and insured status.  
* FHLB advances (Federal Home Loan Bank loans).  
* Brokered deposits.  
* Other borrowings.

**Equity and capital:** CET1 capital, which can optionally include or exclude unrealized AFS losses (a regulatory toggle).  
**Update:** double-entry postings for every withdrawal, asset sale, loss, and draw. An invariant test checks that assets equal liabilities plus equity every tick.  
**Inputs:** each preset is built from public filings, using fictional names with real-like profiles.

### M09 Securities and interest-rate risk (Tier 0 · L2)

**State:** bond holdings by maturity bucket and duration, with book value and market value.  
**Update:** market value moves with the yield curve (M01).

* Unrealized losses grow as rates rise.  
* Selling turns losses into realized losses, which hit capital.  
* Selling held-to-maturity securities can force the rest of that portfolio to be reclassified (a toggle).

**Validation:** an SVB-like preset shows its unrealized losses when replaying the 2022 rate hikes. SVB realized a \$1.8B loss on March 8, 2023\.

### M12 Liquidity and funding facilities (Tier 1 · L2)

The ladder, from fastest cash to slowest:

1. Cash on hand.  
2. Repo.  
3. **FHLB advances:** business hours only, against pledged collateral, with haircuts and a daily limit.  
4. **The Fed's discount window:** only against collateral pledged in advance, with an operational delay if it wasn't.  
5. An emergency program, if the policy lab enacts one (for example, lending against bonds at face value).  
6. Brokered deposits.  
7. Asset sales (M09).

**Each source has:** time to cash, capacity, haircut, cost, and a stigma effect on attention.  
**Outputs:** LiquidityDrawn(source, amount, eta).

### M23 Bank management agent (Tier 1 · L5)

**Purpose:** the bank acts, and its actions can backfire.  
**Actions:** sell assets, announce a capital raise, issue statements, call its top clients, borrow from facilities, cut dividends.  
**Rules:**

* Each action creates an event that M03 and M14 react to.  
* A capital raise succeeds with a probability set by the stock price, attention, and how big the gap is. A failed raise becomes a headline.

**Example:** at SVB, the March 8 combination of a securities loss and a capital raise was read by uninsured depositors as distress (§8.2).

## Markets

### M14 Per-bank stock: order book and traders (Tier 1 · L3)

**Structure:** one limit order book per bank, with a volatility-halt rule.  
**Trader agents:**

* Market makers, who widen spreads as volatility rises.  
* Momentum traders.  
* Value traders, who estimate solvency from public numbers (capital, unrealized losses).  
* Attention traders, who react to attention over the past 4 hours, per Cookson et al.  
* Short sellers, limited by borrowing availability and cost.

**Outputs:** PriceTick and TradingHalt. The price also feeds into depositor worry (M06) and capital-raise odds (M23).

**Fake mode (`/mocks/l3_markets.py`):** the print is the same logistic plus a small attention discount, as a level rather than a per-tick compound, so the line falls with the crowd. The halt rule is still an 8% move inside the 5-print window (ADR-047). Lean and `--gpu-full` price impact (`price_impact_move` / `apply_price_impact` in `/engine/gpu_full_io.py`) move the print toward `impact * cum_share * attention_factor` at the registry cap; recovery is allowed at the same cap. `attention_factor = clip(attention_level / attention_ref, 0, 1)` with ASSUMED `M14 attention reference level` = 1.0 (no peak attention is exposed). The gpu-sim / fake halt countdown reads SOURCED `M14 trading halt minutes` = 5 (ADR-068). Calibration quotes (`step_market_group` and `MarketFake` whenever the calibration engine or `--gpu-sim` is on) pass `scripted=` from `calibration_uses_scripted_drawdown()` in `/engine/gpu_sim.py` (the M27 registry row; default on unless the cell is `no`). Demo `price_quote` and non-calibration `MarketFake` ticks stay scripted unless that path is on (ADR-086, ADR-087).

**Real mode (`/engine/markets/book.py`):** one book for the scenario bank. `match_price_time(side, price, quantity, time_key)` is the Numba price-time matcher (earlier order sets the trade price). `market_maker_half_spread(volatility, base, multiplier)` widens quotes as trailing absolute returns rise. `public_fair(...)` is the value traders' solvency mark from CET1, duration × the policy-rate increase, and attention over the past 4 hours (E10 premium). `halt_triggered(times, prices, t, move_frac, window_min)` fires when the print moves more than X% within Y minutes, up or down. `OrderBook.step(t)` submits the five trader types and publishes PriceTick, plus TradingHalt when the rule trips. X and Y are ASSUMED rows because §18.4 does not set them (ADR-023). `TradingHalt.duration` reads SOURCED `M14 trading halt minutes` = 5 (SEC LULD), the same row as the gpu-sim and fake halts (ADR-074). The TBD halt rows stay in the registry and are not read.

## System

### M20 Regulators and supervision triggers (Tier 1 · L4)

**State:** supervisory status for each bank, and closure triggers.  
**Closure:** the bank closes when it can't pay its obligations, or its capital falls below a threshold.

* SVB-style failures close on a business day.  
* Signature-style failures close over a weekend.

**Outputs:** BankClosed and SupervisoryAction (bank\_id, action). SupervisoryAction records a status change that is not a closure. No other module consumes it.

### M22 Policy lab (Tier 2 · L4)

Each lever records its timing and its costs: Deposit Insurance Fund losses, Fed exposure, and a moral-hazard meter (how much it encourages future risk-taking).  
**Levers:**

* Guarantee all deposits at named banks.  
* Create an emergency lending program (lend against bonds at face value).  
* Raise the insurance limit.  
* Cap or delay withdrawals.  
* Have big banks deposit money into a troubled bank. That's the First Republic analog: \$30B in 2023\.  
* Suspend trading.  
* Issue a joint official statement.

**Real mode (`/engine/policy/levers.py`):** `PolicyLabModule.step(t)` reads `Scenario.policies` (lever, `at`, scope) and publishes one `PolicyEnacted` per row when `t` reaches `at`. Lever names are normalized with the same aliases as the Wave A fake. `cost_model` carries modeled DIF loss, Fed exposure, and a moral-hazard score from `MoralHazardMeter` (breadth and frequency of guarantees). Dollar numerators use ASSUMED registry fractions and the verified First Republic \$30B peer-deposit row. `explain_policy_options` is the regulator advisor (§7): offline rule text by default, SQLite-cached Grok JSON when live. The advisor never enacts policy.

## Crisis room and engine

### M24 Crisis comms room (Tier 1 · L5)

**Pipeline:**

1. The crisis writer drafts 32 statements across different strategies.  
2. The truth filter checks every claim against the claims ledger, the bank's true simulated numbers (cash, capital, insured share, facility access). Any false claim gets the statement rejected, with the reason shown.  
3. The persona panel (M25) scores how each depositor segment would react.  
4. The futures engine (M26) runs every surviving statement forward.  
5. Statements are ranked by deposits kept, the stock's drawdown, and whether the bank survives.

**Timing and channel:** each statement can go out at a chosen time, through a press release, an X post, a CEO video, or direct calls and texts to the top clients.

**Real mode (`/agents/crisis_room.py`):** vague-claim review goes through `/agents/llm_cache.py`. `STAMPEDE_LLM=local` uses the shared local backend; a reply that fails validation falls back to the deterministic truth-filter rule (ADR-057).

### M25 Persona panel (Tier 1 · L5)

**Purpose:** estimate how each depositor type reacts to each statement.  
**Method:** about 20 depositor archetypes per bank type. Each candidate statement gets one batched Grok call, which returns JSON: change in worry, change in threshold, and the share who would act now or wait.  
**Caching:** by statement hash and archetype version.  
**Grounding:** Kazinnik's study found that direct, personalized reassurances reduce withdrawal intent, and the panel's output agrees with that direction (§8.2).

### M26 Ensemble and futures engine (Tier 1 · L1)

**Purpose:** turn single runs into ranges.  
**Batching:** runs N futures as a batch on the GPU, with different seeds, perturbed parameters, or different interventions. A typical run is 16 futures × 250k depositors × 5-minute ticks.  
**Outputs:** failure probability, deposit-outflow bands, price bands, and time-to-failure distributions.  
**Display: the futures fan (M30).**

**Real mode (`/engine/futures/batch.py`):** `FuturesRunner.step(t)` launches once, on the first tick after `attach`. It freezes the latest bus attention, stock return, directive (`DirectiveIssued` or `VIPDecision` with action `withdraw`), policy levers, and `StatementIssued` ids, then runs the forward book. No other module is mutated, and no Grok call is made. The only published event is `FutureCompleted` (§6.1 unchanged).

Each future is a tensor row. `draw_population` builds a seeded lognormal book (Philox via `SeedSequence`) scaled to the SVB-like deposit total, with insured amounts at the statutory cap times the registry ownership-category count. Hubs are the top balance fraction and count as extra contacts. `step_m06_kernel` is the batched §18.2 rule, shape `[futures × agents]`: insolvency belief updates in logit space from attention and a negative stock return; worry is a sigmoid of the registry weights on attention, lagged hub-weighted contact-withdrawal share, belief, stock return, directive, payroll need, and uninsured gap; a depositor withdraws the segment fraction of the uninsured balance when worry exceeds the shifted threshold and the lagged hub-weighted worried-contact count is at least k. Ticks are 5 minutes. One sim-day is the §9 length of about 78 ticks. Failure is the first tick whose cumulative withdrawals reach the opening cash fraction; `t_fail` is elapsed minutes. The price path moves with outflow, attention, and a Philox shock. `run_batch` reduces each row to deposits kept, a price-path band, cost, and failure, then the cross-future failure probability, outflow band, price band, and time-to-failure list.

`run_serial_reference` is the main-engine path: the same formulas, one depositor at a time. One future from `run_batch` matches it on the same seed within the registry tolerance. Future 0 of a wider batch matches a one-future run (draws do not depend on batch width). With no statements, future 0 is unperturbed and later futures perturb the worry weights. With statements, every future shares the baseline seed and differs by the reduced worry and threshold shift. A guarantee, or an insured community preset, zeros the uninsured gap. Array math uses CuPy when a CUDA device is visible and NumPy otherwise. Device transfer is the only GPU-only code (ADR-032). Weights, k, cash, price sensitivities, and the other kernel inputs are ASSUMED rows; the ADR-004 batch shape rows stay as they are. The §18.2 k row that is still `TBD` is not read.

### M27 Calibration and validation harness (Tier 1 · L8)

**Fit:** about 10–15 behavioral parameters, using SVB only.  
**Test without refitting: Signature.**  
**Placebo checks:**

* An insured community bank doesn't run.

**Golden tests in CI:** each preset must stay within its tolerance band (§13).

**§19 API (`/calibration/fit.py`):**

* `load_calibration_targets(path=None) -> CalibrationTargets` — E1/E5 fractions from §8.1, including E1 requested-by-day-one-close (0.249).  
* `fit_behavioral_parameters(seed, *, path=None, scenario=None) -> FitResult` — two-stage search on SVB only: a batched sweep brackets the tipping point for the CMA-ES start (`x0`, `sigma0`) only, then `cma.CMAEvolutionStrategy` runs in identifiable log space with `_theta_bounds` only (fake engine by default; `STAMPEDE_CALIBRATION_ENGINE=real` uses M02–M09). Stage-1 coordinates are worry bias, mobile share, and the decision-to-request rate (`7,15,16`; ADR-084, ADR-085). Price impact stays at its registry value, and the E1 stock ratio is reported, not fitted (stock weight 0; ADR-085). Contact weight and feedback gain stay at their registry priors but now move day-1 requests through awareness. The media awareness rate stays at its registry prior. Checkpoints pickle `es.pkl`; IPOP doubles the population on `es.stop()` stagnation. The fit scores requested withdrawals by day-one close (0.249) and by next open (0.81) with payment caps and closure off, uses `fixed_softness()`, and averages eight shared seeds. `best_vector.json` is always the global best, including a final write after the last generation.  
* `validate_fitted_parameters(vector, seed, *, path=None) -> CapsOnValidation` — SVB replay with payment caps and closure on (hard decisions).  
* `run_identifiability_sweep(...)` — one-at-a-time requested-outflow sweep of the 12-vector; writes `calibration/diagnostics`.  
* `calibration/run_fit.py` — long GPU runs with checkpoints, `--workers` parallel candidate eval, optional `--gpu` (`STAMPEDE_BACKEND`), and FITTED registry output. `--extra-seeds` defaults empty so one seed pass runs; CMA-ES restarts only on stagnation (ADR-060, ADR-061). The flag remains for extra passes. No wall-clock cap; best-so-far is saved as it goes. After stage 1, `_run` calls `fit_payment_capacity` once at `global_best` and writes `M07 capacity multiplier (FIT)`. `summary.json` and `converged.json` record `capacity_multiplier`, `fitted_rows`, and `fitted_rows_written`, and are written after `write_fitted_registry`. `python -m calibration.persist_fit --run-dir <run>` persists a finished run's rows from `best_vector.json` (`--stage2` re-solves and writes the capacity multiplier with the same `apply_run_fit_engine` setup as `run_fit._run --gpu-full`: `STAMPEDE_CALIBRATION_ENGINE=real`, `STAMPEDE_OFFLINE=1`, `STAMPEDE_BACKEND=gpu`, `STAMPEDE_CALIB_GPU_SIM=1`, `STAMPEDE_CALIB_GPU_FULL=1`; CUDA missing falls back as `run_fit` does and the path is printed; `--kind nonbank` writes the non-bank row) (ADR-071, ADR-089).  
* `fit_payment_capacity(...)` — stage 2 (ADR-066): one batched `simulate_logs` call, caps on, 16 log-spaced multipliers from 0.05 to 20 × the shared seeds at `global_best`; monotone log interpolation to settled day-one 0.24; one refine if the bracket is wider than 1.2×; stop within ±0.002.  
* `write_fitted_registry(vector, *, run_id, seed, path=None)` — writes FITTED rows for the identifiable coordinates only, through `calibration/registry_patch.py::patch_registry_rows`. Only the named rows change; every other byte of `params/registry.csv` (line endings, quoting, extra cells) is kept, so `git diff` shows just those rows. `write_payment_capacity` uses the same patcher. `python -m calibration.registry_patch apply <spec.json>` applies row updates from `calibration/registry_updates/` (ADR-070).  
* `evaluate_population_objectives(..., workers=N)` — deterministic spawn-pool candidate eval (§19).  
* `configure_calibration_gpu(enable) -> str` — sets `STAMPEDE_BACKEND` when requested; returns M06 backend label.  
* `run_holdout(vector, seed, *, path=None) -> HoldoutResult` — Signature-like crypto-heavy preset (E5) plus placebo community bank. Uses the FITTED vector and the SVB-fitted capacity multiplier held fixed (not re-solved).  
* `build_validation_board(fit, holdout, *, path=None) -> list[ValidationBoardRow]` — §13 tolerances from the registry; includes `E1-requested-day1`. `format_validation_board_markdown` and `summary.json` also print the E1 stock ratio (observed vs 0.40).  
* `run_calibration_harness(seed, *, path=None)` — fit, holdout, and board in one call.  
* `CalibrationModule.step(t) -> list[BaseModel]` — tracks outflow metrics from `PaymentSettled`; publishes nothing.  
* `engine/gpu_sim.py` — opt-in when `STAMPEDE_BACKEND=gpu` and `STAMPEDE_CALIB_GPU_SIM=1` (`--gpu-sim`). Depositor worry, belief, contagion, and amounts are array math (CuPy, or NumPy when no device). Rumor, attention, market quote, payments, ledger, and securities stay on the existing `step` methods. `--gpu-full` (`STAMPEDE_CALIB_GPU_FULL=1`) also runs attention, the market quote, and payments as array kernels (`/engine/gpu_full.py`, `/engine/gpu_sim_kernels.py`). A generation is one candidate×seed batch on that axis (12×8 on the 3060 command). Packed static state is cached per seed across generations; each generation updates only the per-lane parameter rows (ADR-061). Stage 1 (`_evaluate_population_gpu_sim`, `objective_e1`, `bracket_tipping_point`) scores `simulate_requested_fractions` — day-one requests, next-open requests, and the day-one stock ratio — with attention, decisions, and price feedback on the device, no per-lane engines or events, and no M07/M08/M09/M20. Validation and holdouts keep `simulate_logs`. Both quote paths honor `M27 calibration uses scripted drawdown` through `calibration_uses_scripted_drawdown()` (ADR-062, ADR-086, ADR-087). CPU `MarketFake` reads that helper whenever `STAMPEDE_CALIBRATION_ENGINE=real`, not only when `gpu_sim_enabled()`. Stage 2 calls `simulate_logs` with caps on at `global_best`, 16 log-spaced `capacity_multipliers` × the shared seeds (ADR-066). Per-depositor arrays (belief, balance, worry, attention, exposure, amounts, thresholds) are float32; money accumulators stay float64. Each graph layer is one float32 CSR (`scipy.sparse` on NumPy, `cupyx.scipy.sparse` on CuPy) and spread is one `(A @ X.T).T` per layer (ADR-063). M05 refine stays on host NumPy at large n and is sampled once per seed (ADR-059). The same kernel batches the SVB preset and the non-bank equity, crypto, and rates presets. Agreement with the CPU engine, for outflow fractions and for price paths, is the ASSUMED E1 relative tolerance (ADR-051, ADR-052, ADR-056, ADR-058, ADR-059, ADR-061, ADR-062, ADR-063).

**Non-bank price paths (`/calibration/nonbank.py`):** a separate harness. It does not change the SVB, Signature, or placebo scenarios and it does not write FITTED rows. Three presets — equity-company, crypto, and rates — each carry an explicit `Rumor` (credibility and reach from the registry) and an `event-type` note. `classify_rumor_market` labels the claim and does not replace that credibility. The fit minimizes weighted squared log-error of peak-to-trough, days-to-bottom, and volume multiple on the equity preset (E13, Meta's February 3, 2022 session). Crypto (E15) and the rates/macro index path (E16) are holdouts and are not refit. GameStop in January 2021 (E14) is documented; its point shape statistics stay UNSET. The harness reads `PriceTick` only. In fake mode a three-component vector scales those read statistics. UNSET point targets drop out of the loss. The post-SVB launch command is in §20.3.

**§19 API (`/calibration/nonbank.py`):**

* `load_nonbank_targets(path=None) -> NonbankTargets` — E13–E16 point targets; UNSET cells are `None`.  
* `build_nonbank_scenario(event_type, seed) -> Scenario` — explicit credibility, reach, and event-type tag.  
* `summarize_price_path(...) -> PricePathStats` — peak-to-trough, days to the trough, volume versus the first print.  
* `measure_price_path(scenario, steps) -> PricePathStats` — current engine, `PriceTick` only.  
* `shape_loss(stats, target) -> float` — weighted squared log-error; skips UNSET targets.  
* `shape_loss_terms(stats, target, *, weights) -> dict[str, float | None]` — per-statistic terms (`peak_to_trough`, `days_to_bottom`, `volume_multiple`). An UNSET target is None; the set terms sum to `shape_loss`.  
* `fit_nonbank_parameters(seed) -> FitResult` — CMA-ES-style search on the equity preset. The default measures one path and scales it with a 3-vector. With the real engine and `engine_candidates`, each sample is a behavioral vector and the loss is that path's E13 shape.  
* `run_nonbank_holdout(vector, seed, *, steps=None, rates_steps=None, crypto_steps=None)` — crypto and rates, no refit. A 3-vector scales the read paths. A behavioral vector is applied on those scenarios. `crypto_steps` / `rates_steps` lengthen E15 / E16 so `reachable_targets` keeps multi-day days-to-bottom terms.  
* `run_nonbank_harness(seed)` — fit, then holdout. The flag-off harness is the 3-vector scale fit.  
* `python -m calibration.run_nonbank_fit` — CLI. `--gpu`, `--gpu-sim`, `--gpu-full`, and `--workers` match `calibration.run_fit`, including `--run-id`, `--runs-dir`, `--extra-seeds`, `--patience`, `--checkpoint-every`, `--resume`, `--holdout-minutes`, `--rates-holdout-minutes`, and `--crypto-holdout-minutes`. `--crypto-holdout-minutes` is the E15 step count (default: the shared holdout horizon). The summary records `eval_minutes_holdout_crypto` and `eval_minutes_holdout_rates`. Behavioral vectors re-measure the best equity path once and write `equity_stats`, `equity_terms`, `equity_request_penalty`, `targets_used`, `crypto_terms`, and `rates_terms`; a 3-vector scale run sets the terms to null. With `--gpu` and `--gpu-sim`, each generation batches the equity scenario on `engine/gpu_sim.py` and the crypto and rates holdouts use that kernel. `--gpu-full` moves attention, market, and payments onto that array device. The run writes per-generation `checkpoint.json` and `converged.json` under `calibration/runs/<run-id>/` and does not write FITTED rows (ADR-052, ADR-056, ADR-089).
* `calibration.diagnostics.record_generation(run_dir, generation, objective, vector, **data)` — the fit calls this with per-generation sigma, samples, withdrawn share, and bind rates. Each CLI run writes `/calibration/diagnostics/<run-id>/` (parameter sweeps, withdrawn share, payment-cap and closure bind rates, seed variance, parameters at bounds, CMA-ES step-size history).
* `python -m calibration.diagnostics.summary [run-id]` — print that roll-up.
* `validate_exact_zero_invariance` (`/calibration/invariance.py`) — unrelated text (`I blinked`, `It's raining in Princeton`, `ajjfalk`) is category `irrelevant` on the shared classifier and leaves the event log identical.
* `nonbank_request_bound_penalty` / `validate_nonbank_bound` (`/calibration/nonbank.py`) — daily deposit requests may not exceed the ASSUMED 0.1% of deposits per day; violations add a penalty and fail validation. Fear-sounding market text may still move crowd and price inside that bound (ADR-053).

### M28 Scenario builder, presets, and replay (Tier 1 · L8)

**Presets:** see §12.  
**Scenario controls:**

* Which bank.  
* The rumor and who posts it.  
* A rate path.  
* A property-price shock.  
* The time of day and day of week.  
* Policies.

**Replay:** scrub the timeline, branch at any event ("what if they'd said X on March 8?"), and compare branches side by side.

**§19 API (`/scenarios/`):**

* `list_presets() -> list[dict]` — preset menu from §12.1 registry rows.  
* `bank_for_preset(preset) -> Bank` — opening balance sheet for the preset.  
* `build_scenario(controls: ScenarioControls) -> Scenario` — assemble rate path, shocks, rumors, and policies.  
* `event_log_hash(events) -> str` — deterministic SHA-256 for replay checks (M33).  
* `snapshot_at(scenario, log, branch_index) -> RunSnapshot` — capture state for branching.  
* `branch_scenario(snapshot, overrides) -> Scenario` — fork with new controls.  
* `compare_branches(left, right) -> BranchCompare` — side-by-side divergence summary.  
* `ScenarioModule.step(t) -> list[BaseModel]` — publishes `RumorPosted` once per scenario rumor; consumes `Tick`.

## Interface and plumbing

### M29 UI: crowd screen (Tier 0 · L6)

**Main view:** community clusters of orbs. Each orb is one block of depositors, not one dot per person.  
**Encoding:** color is the block's majority state (calm, wavering, withdrew). Uninsured blocks are larger and capped. A few orbs change state per tick, and each orb eases its color. Cluster tint darkens toward red as the share of withdrew orbs rises.  
**Panels:** phase pills (calm, rumor, run, response, outcome), cash on hand, time to failure, a visible rumor field (Enter to post) with poster chip, and a legend (`1 orb = N people`, live calm / wavering / withdrew counts, depositor total). Outcome is the end of the registry crisis window, not the first empty cash-buffer reading.

**Files:** `/ui/crowd/` (`orbs.py`, `layout.py`, `frames.py`, `feed.py`, React shell).  
**§19 API (`/ui/crowd/`):**

* `build_layout(seed, point_count, community_count) -> CrowdLayout` — community-grouped xy layout (computed once).  
* `block_size(total, orb_count) -> int` — `N = round(total / orb count)`, at least 1.  
* `majority_state(calm, wavering, withdrew) -> int` — mode of one block; ties break toward withdrew.  
* `aggregate_orb_states(...) -> list[int]` — one majority color per orb. Length is the orb budget, never the population.  
* `styled_orb_radius(uninsured, balance_norm) -> float` — pixel radius from the registry, capped.  
* `build_crowd_frame(crowd, t, layout?) -> CrowdFrame` — maps the M29 gauges onto that orb budget.  
* `encode_crowd_frame` / `decode_crowd_frame` — msgpack wire blob (float32 `positions_xy`, uint8 `state` per orb).  
* `CrowdFeed.attach(engine)` — subscribes to `Tick` and `PaymentSettled`; `snapshot(crowd, t)` returns a `CrowdFrame`.  
* `snapshot_from_engine(engine, t?) -> CrowdFrame` — one-shot snapshot after a fake run.
* `CrowdFake.queue_live_rumor(...)` — queues a moderated UI rumor for the next M29 tick (`RumorPosted`).
* `classify_rumor_market(text, poster=None)` (`/agents/rumor_input.py`) — one shared rule for the UI and calibration labels. Classes are `bank_rumor`, `market_event`, `macro_event`, `irrelevant`, and `unsafe`. Non-market text is `irrelevant`, returns `Not market-related: no effect on the simulation.`, and does not publish `RumorPosted`. Live credibility is computed in code from the poster role and specificity features; roles and their ASSUMED bases are the `M29 fake poster credibility (ASSUMED)` row (`rumor_roles()`: regular user, anonymous group chat, VIP, treasury desk, short-seller, journalist, founder circle, bank CEO/CFO, top shareholder, VC partner, regulator). The bases order documented market-pull cases; a price move is not a credibility value, so none is SOURCED. `credibility_breakdown()` returns the parts; the M34 rumor judge (`/agents/rumor_judge.py`) scores specificity, plausibility, claim type, and call to action; code adds `specificity·w` (model specificity held within the ASSUMED 0.25 of lexicon hits/4) and `(plausibility − 0.5)·2·0.1`, and a judged call to act or run claim raises fear strength. The model never writes either number. Offline, after a timeout, or on any invalid reply the judge is the lexicon, which reproduces 77db0e2 exactly (ADR-092). The result is clamped to 0.05–0.9. The rumor box rejects text over 280 characters, merges repeats of the same class, and allows at most one new rumor per 30 seconds. Calibration fit scenarios keep explicit `Rumor.credibility` / `reach` (ADR-053, ADR-055).
* `POST /api/rumor/interpret` — moderation plus classification JSON for fake-mode UI playback scaling. It also returns `role` and `credibility_breakdown` (role base, specificity bonus, model shift, raw, floor, cap, final credibility) so viewers can see why a rumor got its credibility. `GET /api/rumor/roles` lists the role table. `POST /api/rumor` returns the same breakdown (ADR-091).
* `ui.live.pacing` — registry-backed wall-clock pacing (`PacingClock.advance`, `base_sim_minutes_per_real_second`, `client_stream_fps`); speed scales sim rate only, not tick outcomes (ADR-042).
* `LiveSession.advance_wall` / `skip_to_outcome` — deterministic engine steps for the unified live server (`ui.server`).

**Shell:** fake and live modes both draw the registry orb budget (see §8.1), never one point per depositor. For 1,000,000 people at 2,048 orbs, N is 488. The React app defaults to bundled `fake_frame.json` and a 2D canvas; deck.gl is opt-in via `VITE_CROWD_DECK=1`. The canvas reads the msgpack float32 blobs as floats. Fake mode plays a looping scripted spread in real time after rumor submit: positions stay on the seed layout and the wave fills orbs gradually. Live mode uses `ui.server` (`/ws/crowd`, msgpack at `M29 UI client stream fps`) with `POST /api/rumor`, `POST /api/control` (pause/resume, 1×–8× speed, reset, skip to the crisis horizon), client interpolation of orb state and panels between frames (positions stay on the layout), and `STAMPEDE_MODE` / `STAMPEDE_BACKEND` flags (ADR-041, ADR-042, ADR-044). The crowd page also mounts the M30 market and crisis room so one URL shows both monitors (ADR-046). Live fake headcount, wavering, and the cash gauge come from `mocks/demo_arc.py` (ADR-047). With `STAMPEDE_MODE=real` (the `scripts/run_live.sh` default since SVB-50) the server builds the modules in `M29 live engine modules` (the calibrated M02/M03/M05–M09 set plus the M14 order book; all others fake), and the crowd reads engine state: withdrew = M06 `already_withdrew`, wavering = M06 worry ≥ `M29 live crowd wavering worry threshold`, cash gauge = M08 cash plus securities over opening; the market screen shows the M14 order book's `PriceTick` (ADR-098). A dropped socket shows a reconnecting banner and retries; playback pause and speed follow `GET /health` so a reload or a second tab matches the server. Calm is a circle, wavering a square, withdrew a triangle. Reduced motion is a control and follows `prefers-reduced-motion`.

**Search replay layout (ADR-135):** `ui/live/stampede_ui.html::renderWallChrome` puts the group summary, description, selector, and wrapping legend in normal DOM rows around the canvas, with a separate footer for navigation and replay controls. `drawGroups` fits day headers to their combined measured width and budgets sidebar label text, including its ellipsis, separately from the right-hand count; `fitCanvasText` preserves that budget. The sidebar fits its row count to the available canvas height. `drawTile` reserves width for the computed-plan marker. The finale groups labels whose measured bounds would overlap, including triggers that land on different rounded pixels, and separates the window descriptions from the opening/next-open labels. `selectTab` resets page scroll so the next beat's headline is visible beneath the sticky toolbar. These are presentation changes only: the selector still chooses 10/25/50 versions of the day and the saved plans, ranking, and simulation paths are unchanged. `tools/check_wall_alignment.py` previews a local page against saved server data with every non-GET request blocked; it checks actual canvas glyph bounds and DOM rows at both desktop widths through the group replay and finale.

**Prediction scope (VF-4, ADR-136):** `ui/live/useful.py::predict` raises `CashPathRequired` for a cash condition after an earlier action, including bonds pledged before the day. The no-response cash curve is valid only before the first intervention; it cannot establish a later funding-dependent cash condition. `rescore` preserves such policies as `unverified_plans` with status `needs-simulation`, excluding them from useful rankings, sensitivity winners and replay groups. No survival, trigger-time or margin value is claimed for them. The live cash gauge, trigger checks, regulator rule and parameter values stay unchanged. The live vault and end cards label its gross book measure as cash plus securities, including pledged bonds, rather than spendable cash. What helps discloses the withheld count and lists the policies separately. Independent shortcut checks exclude generated paths; old holdout evidence is retained as `previous_holdout` rather than attributed to a different winner.

**First-page vault refresh (ADR-137):** The live `drawVault` uses the plan wall's spaced-dot concentric bands and simple gold centre. Its central percentage is total deposits requested, labelled “asked”; gold remains the unchanged gross book liquid-asset fraction. A deterministic screen-density subsample of the existing depositor sample supplies the dots and their engine-reported requested fractions. Rings no longer encode wealth or geography, superseding that presentation detail of ADR-117; resizing changes display density only. The amount, historical comparison, warning and legend sit outside the canvas. The neutral first-beat headline is preserved; the selected pick's label appears only when the user runs that pick. The first-page vault has no ranking badge or enclosing border in any live mode; those belong to ranked plan tiles. Canvas accessibility text exposes the same requested percentage. Simulation state, data and ranking are unchanged.

**Tipping refill and operator controls (VF-5, ADR-138):** `scripts/pipeline.py --tip-strengths` selects named rumor strengths. `_tipping_runs` checks saved series keys across the option files and schedules only missing strength/seed pairs. `--reuse` skips the main search; `--messaging-plans 0` skips messaging. Appended series are restricted to each file's seed count. Replenishing coverage is separate from a full search and must be followed by coverage and rescoring checks for every option. In the live UI, `#wallRun` is operator-only and asks for confirmation before launching the full pipeline. `warningLine` returns no forecast once the current run ends. This adds no simulation event or numeric model parameter; VF-4's cash-path exclusions remain in effect.

**Early-day tile wording (VF-6, ADR-139):** `groupsTally` counts each plan's `early` classifications among the selected groups; `applyGroupRanking` exposes the count as client-side `groups_early`. `tileDays` explicitly names early-intervention days when they account for the remainder, or reports all-useful and mixed cases. The subtitle is separate from the tile's open-bank count: early intervention concerns weaker rumors that would have died out. Saved data, per-group classifications, ranking and simulation outcomes stay unchanged.

**Viewer-facing wording (UI-6, ADR-140):** the main journey keeps three beats and one short, consistent plan label (`shortLabel`). Beat 1 gives a neutral question, sampled-customer legend and current-run result; expected ranges and response-tool cards expand on demand. Presenter view abbreviates the performance badge, leaving timing/buffer details in its tooltip and operator view. The default results (`renderWhy`) show the leading plan, explain pledging as arranging to borrow against bonds, distinguish requested deposits from settled withdrawals, and define open as reaching the next morning. `windowText` reads the saved threat/deadline minutes rather than legacy window metadata; the deadline explicitly belongs to the unprepared bank. The close-call margin, assumption sensitivity, computed provenance and excluded-plan count stay visible. Alternatives, full scoring rules, validation evidence and withheld plans remain expandable. Saved results, thresholds, ranking and engine calculations are unchanged. `runPlanLive` accepts calculated plans through the same reset/preset API as the existing pick chip; the obsolete computed-plan refusal is removed because those steps are supported by `LiveSession`. `startPlanLive` holds polling during reset/load and increments a client load sequence so an older state response cannot clear the new plan label. `tools/check_ui_copy.py` verifies the rendered copy and read-only request wiring; `tools/check_wall_alignment.py` verifies canvas placement.

**Manual saved replay and rendering (UI-7, ADR-141):** All 10/25/50-day views load paused. `wallToggle`, `wallRestart` and `wallSetSpeed` supply Play/Pause, Reset and measured 1×/2×/4× controls; Reset and a count change return to the paused start at 1×. Leaving the replay pauses it. Replay keys and controls are local and cannot start the bank simulation; live responses remain explicit on What helps. A request sequence prevents an older count response replacing the latest choice. The separate saved-replay clock/progress replace the bank clock/controls on this screen. Paused canvases redraw only when needed.

`groupCells` reads the saved 30-minute samples directly through `seriesAt`, without expanding intermediate arrays. `groupsTally` caches fixed counts; `groupStandings` indexes elimination times and handles backward seeks. `miniCard` caches bounded, pixel-precision thumbnails of the original framed vault artwork; large cards keep the original 180 depositor dots and share orbit geometry via `tileOrbits`. Numerical series, policy classifications and ranks stay unchanged. Elimination cards keep the original reason colours (early, unavailable, failed). Each group's layout reserves a departing card's existing slot during its shrink/fade, then grows surviving cards inside disjoint new grid slots. It does not slide cards through one another. Grids and partial rows are centred in the day panel, and shrinking vaults stay centred in their red cards. Names, graphics and footers occupy separate regions; final vaults share a baseline, and tile subtitles wrap. `tools/check_replay.py` checks controls, source-data equivalence and placement; `tools/check_replay_motion.py` checks actual painted cards for overlap through elimination at all counts/speeds. The alignment checker tags actual canvas frames so paused snapshots do not mix two resize frames. No simulation events, saved-file schemas or model parameters change.

**Viewer-facing words and reveal (UI-8, ADR-142):** `shortLabel` spells out withdrawal-request triggers and joins full plan steps as sentences. Beat 2 keeps the neutral question until its saved replay completes; Reset hides the answer, and beat 3 shows it. `howItWorks` explains the final requested share and each useful plan's mechanism. The final sidebar reports open-day counts for useful plans; during playback it shows at most eight rows and a remaining-plan count. `spare` converts deposit-share margins to dollars using the registry-backed `deposits_usd` from `/api/search`, with a share-only fallback. Withheld-plan counts remain accessible in the evidence fold. Beat 1 uses calibrated/what-if run badges, asked-to-withdraw wording, and an out-of-cash shortfall. Live playback controls are hidden on beats 2 and 3. `tools/live_walkthrough.py` checks the three beats, asynchronous preset badges and both live outcomes, with pipeline launch requests blocked. Original vault artwork, numerical series, policy ranks and engine behavior remain unchanged.

### M30 UI: market screen and crisis room (Tier 1 · L7)

**Market:** price chart with event markers, order-book depth bars, and a halt banner.  
**Crisis room:** the list of candidate statements (accepted ✓ or rejected ✗ with the reason) and the futures fan. Rows show a short plain-language label for known statement ids (`statementLabel`); an unknown id is shown unchanged.  
**Shell (Wave B):** `/ui/market` React scaffold plus `ui.market.snapshot` frames from the fake engine (`MarketScreenFrame` in §6.2). The unified live server at `ui.server` streams `/ws/market` (msgpack, registry stream rate) from the same engine as M29. The same market components render under the crowd page (ADR-046). The price chart labels price in dollars and simulated minutes, and keeps the vertical axis anchored at zero. `ui.market.server` re-exports that app for tests. Fake URL `?fake=1` renders without GPU. A dropped socket shows a reconnecting banner and does not swap in the static fake frame.

### M33 Observability, determinism, and demo mode (Tier 0 · L8)

**Includes:**

* Seeded runs.  
* An event log with replay.  
* Cached LLM outputs for the scripted demo path.  
* An offline mode that works with Wi-Fi off.  
* A metrics panel (tick time, GPU memory, LLM latency).

**§19 API (`/ops/observability.py`, `/ops/rng.py`, `/ops/event_log.py`, `/ops/llm_cache.py`):**

* `load_observability_config(path=None) -> ObservabilityConfig` — reads M33 rows in §8.1 / `params/registry.csv`.  
* `demo_mode_enabled(*, config=None) -> bool`; `offline_flag_default(*, config=None) -> bool`; `offline_ready(cache, *, config=None) -> bool`.  
* `philox_uniform(seed, count, *, offset=0, prefer_gpu=True) -> tuple[float, ...]` — Philox4x32-10 on the **GPU path** when CuPy is importable; **CPU fallback** otherwise (§18.0).  
* `event_log_hash(events) -> str`; `write_event_log_parquet(events, path)`; `read_event_log_parquet(path)`.  
* `LlmCache` — SQLite get/put keyed by prompt hash and model (§7.1).  
* `MetricsSnapshot` — tick time, GPU memory (0 without CuPy), LLM cache latency, first Philox sample.  
* `ObservabilityModule.step(t) -> list[BaseModel]` — updates metrics and log hash each tick; publishes nothing.

### M34 Safety and content guardrails (Tier 0 · L5)

**Files:** /agents/guardrails.py, /agents/rumor_input.py, /agents/rumor_inbox.py (stream L5).  
**Rules:**

* Fictional bank names and tickers only. A blocklist rejects real bank names in the rumor box.  
* Rumor moderation (no slurs, no real people).  
* The truth filter is a hard rule.  
* The product never forecasts a real institution.  
* An optional live-attention mode shows aggregate attention only and is off by default.  
* Live rumor credibility is computed in code (poster + specificity). The model never sets the number. Caps, merge, and the 280-character limit are in `/agents/rumor_inbox.py` (ADR-055).

**§19 API (`/agents/guardrails.py`):**

* `load_guardrail_config(path=None) -> GuardrailConfig` — reads M34 rows in §8.1 / `params/registry.csv`.  
* `bank_name_allowed(name, *, config=None) -> bool`  
* `ticker_allowed(ticker, *, config=None) -> bool`  
* `claim_allowed(claim, *, config=None) -> bool` — rumor-box gate (fictional institutions only).  
* `statement_allowed(text, *, config=None) -> bool`  
* `moderate_rumor(text, *, config=None) -> ModerationResult` — blocklist plus slur and real-person checks.  
* `live_attention_enabled(*, config=None) -> bool`  
* `GuardrailModule.step(t) -> list[BaseModel]` — on the first tick, records blocked scenario rumor IDs; publishes nothing.

## Additional modules (M41, M42, M44)

Each spec below covers what the module does, its outputs, and its evidence status. Build specs are in §18.

## Grok-driven agents

### M41 Named VIP depositor agents (Tier 2 · L5)

The top 15 depositors in a scenario are Grok agents with a persona, a short memory, and tools: check their exposure, move funds, post, or message their group.  
They decide every 30 simulated minutes or when triggered. Each decision is then reduced to simple numbers (a threshold and sensitivities), so the futures engine can replay it without new calls.  
**Outputs:** DirectiveIssued, VIPDecision.

**Fake mode:** the scripted-demo withdrawal is an ASSUMED fraction of opening cash so one VIP move does not empty the gauge. Real amounts are unchanged.

**Real mode (`/agents/vip.py`):** `VIPModule.step(t)` batches the registry count of named fictional personas, ten per call, through `/agents/llm_cache.py`. Offline mode defaults on (`STAMPEDE_OFFLINE=1`) and uses a deterministic rule: insured-community presets stay; other presets withdraw or, for a VC persona, message the portfolio group. The Founder of Kestrel Labs withdrawal uses the ASSUMED example amount so the feed line is `Founder of Kestrel Labs moved $40M`. Every reply is validated with pydantic. `reduce_decision` sets a threshold and sensitivities so `replay_reduced_rule` returns the same action on the same features; `rules_for_futures` is the record M26 replays, with no new Grok call. Memory keeps the last five tool results. The scheduler subscribes to Tick only (§19). A decision also runs when `note_trigger` is set, or when `on_tick` is handed an AttentionTick that crosses the ASSUMED attention gate. The §7.1 gate starts at the registry semaphore size, adds one after a successful batch, and halves on HTTP 429. Live Grok uses xai-sdk only when offline is off and `XAI_API_KEY` is set (ADR-032).

**§19 API (`/agents/vip.py`):**

* `load_vip_config(path=None) -> VipConfig` — reads M41 rows in §8.1 / `params/registry.csv`.  
* `persona_cards(count=None) -> tuple[PersonaCard, ...]`  
* `check_exposure(card, context) -> ExposureView`  
* `move_funds(card, amount) -> MoveResult`  
* `post(card, text) -> PostResult`  
* `message_group(card, text) -> GroupMessage`  
* `validate_vip_batch(payload, *, expected_ids=None) -> VipBatchReply`  
* `reduce_decision(decision, features, *, sensitivities, margin) -> ReducedRule`  
* `replay_reduced_rule(rule, features) -> str`  
* `format_feed_line(display_name, action, amount) -> str`  
* `rules_for_futures(rules) -> list[dict]` — M26 replay records.  
* `VIPModule.step(t)` publishes `DirectiveIssued` and `VIPDecision`.

### M42 Bank CEO agents (Tier 2 · L5)

One Grok strategist per bank chooses actions through M23's action menu. It can only take actions its balance sheet allows, and every action is logged with a rationale.  
**Outputs:** CEOAction. DepositRateChanged when the rate is affordable.

**Real mode (`/agents/ceo.py`):** `CEOModule.step(t)` reads the book by replaying M08 postings and acts when an event hits the bank and M23's policy (or the ASSUMED attention, slide, or loss trigger) says so. Grok tool calls go through `/agents/llm_cache.py` and `map_tool_call` keeps only `MANAGEMENT_ACTIONS` from M23. `ledger_allows_action` refuses a sale larger than securities or a borrow larger than haircut collateral. `ledger_allows_deposit_rate` refuses a rate outside the ASSUMED bounds or whose segment interest exceeds cash. Refusals are written to the per-bank CEO log and are not published. An accepted action publishes `CEOAction`; an affordable rate also publishes `DepositRateChanged`. `offline_ceo_decision` is the deterministic fallback: it calls `load_management_policy` and `capital_raise_success_probability` and does not roll the raise (M23 still owns that draw). `reduce_ceo_decision` stores a threshold, sensitivities, and the action; `replay_reduced_rule` replays them with no new Grok call (ADR-009). Live Grok stays off unless `STAMPEDE_OFFLINE` is off, `XAI_API_KEY` is set, and the ASSUMED live-call flag is true (ADR-032). FAKE stays the default.

### M44 Short-seller and activist agents (Tier 3 · L5)

**Purpose:** read public metrics and, when a bank's vulnerability score is high, publish a thesis. That spikes attention (M03) and adds short positions in M14.  
**Files:** /agents/shorts.py. The SQLite LLM cache and `STAMPEDE_OFFLINE` fallback live in /agents/llm_cache.py (§7.1). Offline is the default.  
**State:** the latest public metrics, the vulnerability score, whether a thesis has already gone out, the validated thesis text, and the short order recorded for M14.  
**Update:** each tick, score attention × uninsured share, and add the capital-stress bump when CET1 is below its reference. At or above the threshold, ask for thesis text through the cache. The simulation sets `reach`. Publish `ThesisPublished` once. Record a short order; do not write `/engine/markets/book.py`. M14 already consumes `ThesisPublished` (§6.1).  
**Parameters:** `Short-seller vulnerability publish threshold` stays TBD/UNSET. The code reads `M44 assumed vulnerability publish threshold` and the other M44 ASSUMED rows in `/params/registry.csv` (ADR-022).  
**Validation:** a published thesis raises attention and short interest. Real mode is `ModuleMode.REAL` for M44; fake stays the default.  
**Outputs:** ThesisPublished.

**§19 API (`/agents/shorts.py`):**

* `load_short_params(path=None) -> ShortParams` — reads M44 rows through `params/loader.py`.  
* `vulnerability_score(metrics, params=None) -> float` — public-metric score. Grok does not set it.  
* `metrics_from_events(events, preset, params=None) -> PublicMetrics` — latest attention, CET1, and price already on the bus.  
* `thesis_text(metrics, *, seed, bank_name, score, params=None, cache=None) -> str` — cached Grok text, or the deterministic offline template. Guardrails reject a non-fictional reply.  
* `thesis_reach(seed, params=None) -> int` — simulation-owned `reach`.  
* `short_order_for(metrics, *, author_id, bank_id, params=None) -> ShortOrder` — order handed to M14.  
* `ShortSellerModule.step(t) -> list[BaseModel]` — consumes Tick; publishes `ThesisPublished` once the score is at or above the threshold.

# 6\. Contracts: events and schemas

These are frozen. Any change requires an ADR plus a contract-test update. Field types are pseudo-schemas, so each work stream can implement them in its own language.

## 6.1 Event bus (every event carries t, seed, bank\_id?, cause\_id?)

| Event | Payload | Published by | Consumed by |
| :---- | :---- | :---- | :---- |
| Tick | t | M00 | All |
| RatesChanged | curve\[\], policy\_rate | M01 | M09, M14 |
| RumorPosted | rumor\_id, claim, source\_node, credibility | M28, UI | M03 |
| AttentionTick | bank\_id, level, delta\_4h | M03 | M06, M14, M04 |
| HeadlinePublished | text, reach, bank\_id | M04 | M03, M14 |
| WithdrawalRequested | depositor\_id, amount, channel | M06 | M07 |
| PaymentSettled | amount, channel, queued\_remaining | M07 | M08, UI |
| LiquidityDrawn | source, amount, eta | M12 | M08, M03 |
| AssetSold | class, amount, price\_impact | M08 | M09 |
| LossRealized | amount, kind (securities, credit) | M09 | M08, M04 |
| CapitalChanged | cet1\_ratio | M08 | M20, M14 |
| PriceTick | bank\_id, price, volume | M14 | M06, M23, M04, UI |
| TradingHalt | bank\_id, reason, duration | M14 | M03, UI |
| StatementIssued | stmt\_id, channel, segments\[\] | M23/M24 | M06, M03 |
| PolicyEnacted | lever, scope, cost\_model | M22 | M06, M12, M20 |
| BankClosed | bank\_id, reason | M20 | M03 |
| FutureCompleted | future\_id, metrics{} | M26 | M24, UI |
| DepositRateChanged | bank\_id, segment, rate | M42 | M06, M08 |
| DirectiveIssued | vc\_id, portfolio\_ids\[\], action | M41 | M06 |
| VIPDecision | agent\_id, action, amount, rationale | M41 | M06, M07, M03 |
| CEOAction | bank\_id, action, params, rationale | M42 | M23 |
| ThesisPublished | author\_id, bank\_id, reach | M44 | M03, M14 |
| RumorMutated | rumor\_id, claim, exaggeration\_score | M03 | — |
| AnalystNote | text, cited\_event\_ids, bank\_id | M04 | M03, M14 |
| SupervisoryAction | bank\_id, action | M20 | — |

## 6.2 Core schemas

| `Schema` | `Fields` |
| :---- | :---- |
| `Bank` | `id, name (fictional), ticker, preset; assets { cash, reserves, securities[], loans[], other }; liabilities { deposits_by_segment{insured, uninsured}, fhlb, brokered, other }; capital { cet1, aoci_toggle }; ladder[]; stock_ref` |
| `DepositorArrays (structure of arrays on the GPU, one row per depositor)` | `bank_idx, segment, balance, insured, belief, threshold, attention, needs_flag, has_second_bank, state` |
| `Security` | `class, maturity_bucket, duration, book_value, market_value, htm` |
| `Rumor` | `id, claim, lineage[], credibility, reach, notes[]` |
| `Statement` | `id, text, claims[], channel, verdict, reasons[]` |
| `ClaimsLedger` | `bank_id, cash, capital_ratio, insured_share, facility_capacity, as_of` |
| `Scenario` | `preset, seed, rate_path, shocks[], rumors[], policies[], fidelity` |
| `FutureResult` | `future_id, failed, t_fail?, deposits_kept, price_path_band, cost` |
| `CrowdPanels` | `cash_gauge, time_to_failure_ticks?, tipping_point, rumor, state_ribbon, calm_count, wavering_count, withdrew_count` |
| `CrowdFrame` | `t, seed, bank_id?, point_count, positions_xy[], state[], balance_norm[], uninsured_mask[], panels` |
| `MarketScreenFrame` | `module, t, bank_id?, prices[], volumes[], depth[], halt?, markers[], statements[], futures[]` |

## 6.3 Saved search analysis metadata (VF-4)

These additive fields describe analysis coverage, not simulation events:

- `unverified_plans`: records with `plan`, `label`, `steps`, `computed`, `status` (`needs-simulation`) and `reason`; deliberately no predicted survival, response time, margin or rank.
- `considered.unverified`: number of policies withheld from ranking; `considered.simulated` and `worked_out` count eligible policies, while `simulated_total` retains the number of source policies with independent runs. Eligible simulated + worked-out + unverified equals all considered plans.
- `prediction_scope`: description of the cash-path limitation.
- `previous_holdout`: preserved independent evidence for an earlier plan; `holdout` is null unless its `plan` matches the selected winner.

`plans`, `useful`, `useful_all`, replay groups and sensitivity winners contain only eligible policies. `pick` is null if no useful eligible policy exists. `tests/test_cash_path_validation.py` checks this coverage contract and input preservation.

`GET /api/search` additionally returns `deposits_usd` (UI-8, ADR-142), the opening deposit total from `engine.bank.ledger.registry_deposits_total()`. This response-only field converts share margins to display dollars without a live session; it does not alter saved files or simulation events. Contract coverage is in `tests/test_ui8_viewer_words.py`.

# 7\. Runtime AI agents (Grok)

**Principle:** Grok writes, judges language, and plays personas. The simulation computes every number. No agent can change a number in the state; agents only create events through the same contracts as everything else.

| Agent | Job | Input → output | Call budget | Guardrail |
| :---- | :---- | :---- | :---- | :---- |
| Persona panel (M25) | Reactions of \~20 depositor types to a statement | Statement → JSON of worry and threshold changes per archetype | 1 batched call per statement (32 per crisis room) | Cached by hash; replies checked against a schema |
| Rumor mutator (M03) | Rewrites a sample of reposts | Claim → variants, plus an exaggeration score | ≤ 20 per rumor | Fictional names only |
| Headline writer (M04) | Headlines from events | Event bundle → headline | ≤ 1 per threshold crossing | Must cite event IDs |
| Crisis writer (M24) | Candidate statements | Ledger plus state → 32 statements across strategies | 1–2 calls | Must list its claims explicitly |
| Truth filter (M24) | Accept or reject claims | Statement claims plus the ledger → verdicts | Numeric checks run in code; Grok handles vague wording | Hard reject on any false claim |
| Narrator | Plain-language explanations | State diff → one sentence | ≤ 1 per 10 seconds | Never states a number not taken from state |
| Analyst agent (M04) | Notes | Surprise events → note | On trigger | Text only |
| Regulator advisor (M22) | Explains policy options and costs | Lever menu plus state → explanation | On request | Never decides; the user picks |

**Models:**

* A fast, cheap tool-calling model for the persona panel and mutator: Grok 4.1 Fast.  
* The flagship Grok 4.7 for the crisis writer and narrator.  
* Confirm current model names and prices in the xAI console.

**Credits:** SuperGrok doesn't include API credits, so budget them separately and set a spend cap.

## 7.1 Running as many Grok agents at once as possible (runtime)

**Population in the main timeline:**

| Agents | Count |
| :---- | :---- |
| VIP depositors (M41) | 15 |
| Bank CEOs (M42) | 2 |
| Regulator advisor (M22) | 1 |
| Journalists, analysts, and short sellers (M04, M44) | \~7 |
| Persona panel, crisis writer, truth filter, narrator | On demand |

That's about 25 concurrent Grok agents. Everyone else (the million depositors and the traders) is rule-based on the GPU.  
**Cadence:** VIP agents decide every 30 simulated minutes or on a trigger. CEOs act when events hit their bank. Media agents act when thresholds are crossed.  
**Concurrency control:**

* asyncio with an adaptive semaphore. Start with 32 requests in flight. Add one after each success; halve the count when the API reports a rate limit (429).  
* Batch 10 VIP agents per call using structured JSON output, and validate every reply with pydantic.  
* Cache in SQLite, keyed by prompt hash and model.
* **Local backend:** `STAMPEDE_LLM=offline` (default), `grok`, or `local`. Local routes M03, M04, M22 advisor, M24 vague filter, M25, the M34 rumor judge, M41, M42, and M44 through `/agents/llm_cache.py` → `/agents/local_llm.py` to an OpenAI-compatible HTTP server (`STAMPEDE_LOCAL_LLM_URL`, default `http://127.0.0.1:8080/v1`). Default model id `Meta-Llama-3.1-8B-Instruct-Q4_K_M` (Llama 3.1 8B Instruct Q4\_K\_M, ~5 GB on an RTX 3060); any model name works via `STAMPEDE_LOCAL_LLM_MODEL`. Weights directory is the ASSUMED registry row, override `STAMPEDE_LOCAL_LLM_WEIGHTS` (repo-relative default `models/gguf`). Real inference is GPU-only (`scripts/serve_local_llm.sh` refuses CPU). Unit tests use `FakeLocalBackend` on CPU and never open the host weights path. A failed validation retries once, then the agent uses its rule path (ADR-043, ADR-057).

**Cost estimate:**

* A one-day scenario makes about 2,000 calls at roughly 1k input and 200 output tokens each.  
* At Grok 4.1 Fast list prices of \$0.20 per million input tokens and \$0.50 per million output tokens, that's about \$0.60 per scenario. Verify current pricing in the console.

**Futures stay free:**

* Each LLM decision in the main timeline is reduced to simple numbers: a threshold, sensitivities, and the action chosen.  
* The 16 futures (M26) replay those numbers on the GPU without making any new calls.  
* Only the winning future is re-run with live agents, and only when time allows.

**Fallback:** if the API is slow or down, agents fall back to their last reduced decision rule, and the demo continues offline.

# 8\. Parameters, evidence, and validation targets

## 8.1 Parameter registry (excerpt; the full table lives in /params/registry.csv)

| Parameter | Value | Status | Source |
| :---- | :---- | :---- | :---- |
| Insurance limit | \$250,000 per depositor, per bank, per ownership category | VERIFIED (statutory) | FDIC |
| SVB-like preset: deposits | \$175B (year-end 2022\) | VERIFIED | E1 |
| SVB-like: uninsured share | \>93% | VERIFIED | E2 |
| SVB-like: top-10 depositor balance | \$13.3B | VERIFIED | E3 |
| Signature-like: first-day outflow | 20% of deposits | VERIFIED | E5 |
| Attention memory window | 4 hours | VERIFIED (design follows) | E10 |
| Calendar business open / close hour | 9 / 17 | UNSET | §2.3 (M00) |
| Calendar wire cutoff hour | 17 | UNSET | §2.3 (M00) |
| Calendar facility open / close hour | 9 / 17 | UNSET | §2.3 (M00) |
| Calendar sim epoch minute (Monday 09:00) | 540 | UNSET | §2.3 (M00) |
| Behavioral thresholds and weights (≈10–15) | — | FIT on SVB only | M27 |
| M34 blocklist bank names | silicon valley bank; signature bank; first republic; silvergate; chase; jpmorgan; jp morgan; bank of america; wells fargo; citibank; goldman sachs; morgan stanley; hsbc; truist; capital one | VERIFIED (design) | §5 M34 |
| M34 blocklist tickers | svb; sbny; sivb; frb | VERIFIED (design) | §5 M34 |
| M34 fuzzy match max edit distance | 2 | ASSUMED | §5 M34 |
| M34 live attention mode default | false | UNSET | §5 M34 |
| M34 rumor max characters / cooldown / credibility floor-cap / model shift / specificity weight | 280; 30s; 0.05–0.9; ±0.1; 0.2 | ASSUMED | §5 M34 (ADR-055) |
| M34 rumor judge model / live flag / timeout / max specificity deviation / call-to-action levels / run-claim minimum strength | grok-4.1-fast; true; 4 s; 0.25; 1; medium | ASSUMED | §5 M34 (ADR-092) |
| M29 UI fake-mode render points | 2048 | ASSUMED | §5 M29 |
| M29 layout community count | 8 | ASSUMED | §5 M29 |
| M29 UI live crisis sim minutes | 120 | ASSUMED | §5 M29 |
| M29 UI live playback target seconds | 105 | ASSUMED | §5 M29 (~90–120 s at 1×) |
| 106 | The live demo runs the depositor model the SVB fit calibrated, over SVB's timeline. The fit uses soft decide-once decisions (behavioral noise 0.06); M06 defaults to hard decisions (softness 0), where everyone whose gates pass requests at once, which drained the live bank within hours. Soft requests are fractional per depositor per tick, so the CPU engine aggregates them per (segment, channel) per tick: totals equal the kernel's and the event log stays small (a handful of requests per tick); M07's limits are bank-wide and M08 ignores depositor ids. The crowd's withdrew count stays depositors with decided >= 0.5. SVB: ~24% of deposits requested on day one ($42B), ~82% by the next open, closure the next morning (Barr testimony). Known gap: the gpu-full array path (the fit's) gives 0.245 by day-one close where gpu-sim and the CPU engine give 0.189; next open agrees (~0.81). That is the open CPU/GPU parity issue (P1-9, diag-02), not this change. | Sebastian: no one-hour wipe-out after a rumor; use SVB's real pace as the basis. diag-08 confirmed the hard-mode and CPU-emission findings on his machine. |
| 107 | Role washout fix (approved by Sebastian, Nov 3). M06 clips the bank attention level to [0, 1] (`decide.py` AttentionTick branch) and M03's level passes 1 within minutes for any credibility, so after that every rumor drove the same curve (sandbox: 0.38 vs 0.8 gave 16.7% vs 17% by day-one close; D2 alone still 84%). With the flag, the clip is `min(1, c / reference)`: rumors at or above the SVB fit rumor's credibility (0.8) keep the fitted clip at 1 and follow the fitted curve; weaker posters reach proportionally less attention. Positive-tone (cure) posts do not count. M06 reads rumor credibility from `bus.log` with its own cursor, so `consumes` is unchanged. Registry flag off: calibration and seeded logs do not move; refused under GPU calibration when on in the registry. | The preset picker needs `frozen-branches` to be visibly weaker than `lost-bonds` (rule: about half by day-one close). |
| 108 | The live price is display pacing, not a market model. It follows the engine's own run (cumulative requested deposits) and reaches SVB's day-one -60% (E1 stock target 0.40, reported not fitted since SVB-36) exactly when the simulated requests reach SVB's day-one level (0.249); it never rises and stops at -60% (SVB's shares were halted the next morning). It is labelled as not calibrated in `/health` and, with label L-1, on screen. Calibration and fake mode do not use it. | Sebastian chose it (Nov 3) over the raw order book (fell to about 9 within an hour) and over no price. |
| 109 | Every calibration lane spreads its own rumors (SVB-66). The seed-shared spread of SVB-02 and SVB-18a assumed reposts depend only on the network, the rumor and the seed; `_draw_reposts` reads the lane's attention, so every candidate in a generation got reposts driven by its seed's first lane. Single-lane runs (validation, the board) are unchanged; batched runs now equal their solo runs exactly on NumPy. Cost: one host spread per lane per tick (about 0.06 ms each on the 16-node graph). | The SVB-02 parity test failed once the svb37 rows (4ed6ad9) put withdrawals inside its window; diag-11 showed the gap appears only after a lane's first withdrawal. |
| 110 | One model on every path (SVB-65). The SVB-06 attention feedback and the SVB-25 price impact were written into the GPU kernels only, so the CPU module engine (the live server, the CPU calibration engine) and the per-lane gpu-sim path ran a model the fit never calibrated: at the persisted fit, day-one requested 0.189 there against 0.245 on gpu-full and the lean path. The CPU modules now reach the same functions through `step_module`; the GPU kernels are unchanged, so the fit and its board are unchanged. Live depositors still ignore the stock price (SVB-58) and the live M14 is still the order book, so only the attention feedback reaches the live run. Seeded and default engines keep the old path. | diag-11 and an independent NumPy check: CPU 0.1887 = gpu-sim 0.1887 against gpu-full 0.2447 (NumPy) and 0.2449 (CuPy); with SVB-65 the CPU engine gives 0.2447. The price-parity tests had failed since SVB-25 (6630806), not SVB-08. | Scope (diag-14 v2, corrected by PA-C, Nov 4): one model while a rumor drives the run. With no rumor the paths differ: the CPU calibration engine's scripted agents (an M41 VIP withdrawal at minute 30; M04, M44 and M42 reacting to the bank's first asset sales at minute 303) start a run overnight on the CPU path only (0.765 requested by the next open against 0.004 on every GPU path), and the GPU paths have a small no-rumor leak of their own (0.0015 by the close; the CPU with those agents off requests 0). The agents are not simply missing on the GPU: with them off, the CPU's rumor-driven SVB-fit run moves from 0.245 to 0.459 by the close while the GPU stays at 0.245, so the GPU reproduces their effect once a rumor drives the run. Kept as is until after Nov 13 (Sebastian, Nov 4): the fit and the live demo are rumor-driven and agree on every path. |
| 111 | The live demo's countermeasure is a policy, not a sentence: the preset table's insured chip enacts the deposit guarantee M06 already models (`PolicyEnacted guarantee_deposits`, uninsured gap x 0) at its minute; depositors who already requested stay withdrawn. Calming text typed by a viewer keeps SVB-59's weak cure, and only the chip guarantees. At the paced price floor trading halts for the rest of the run (display only; live depositors ignore the price). The live window is 1441 minutes so the 1440 closure is shown. | Sebastian (Nov 3-4): a full guarantee, the Mar 12 2023 backstop that stopped SVB's run; the chip's sentence reads as neutral to the lexicon and spread as panic (diag-11). The flat floor for half the playback read as a bug; SVB's stock was halted before the Mar 10 open. diag-11 v2: the closure lands after step 1441. |
| 112 | Live playback is clamped to the engine, not to the wall clock: the screens play computed frames and never pass the newest one, so an engine slower than 1x slows playback (a 105 s run takes about 107 s on the 3060 desktop) instead of jumping or falling out of step. The head start (2 s) only absorbs step jitter, and it also bounds how far the engine leads, so a mid-run post takes effect about 2 s of playback (about 30 simulated minutes) after the minute on screen. | diag-11 v2: up to 106.9 s of compute for 105 s of playback. A fixed-rate design needed a 12 s head start, which would put a mid-run insured chip about 165 simulated minutes (12 s) behind the screen; testing countermeasures is the point, so the click has to land close to when it is made. Design call (Nov 4); the head start is a registry row. |
| 113 | Honesty labels (PA-C). Only the 2023 replay is fitted: the lost-bonds preset with no response and no typed rumor; every other run is labelled a scenario, including lost-bonds with a response. The server supplies the badges and the limits screen's copy (`/api/labels`, `/api/limits`, `run` in `/health`) so the screens render them at the event. Viewers pick preset rumors: `M29 live free-text rumors` (off) gates `POST /api/rumor` and the socket rumor command, after moderation so a blocked name keeps its reason. | Sebastian, Nov 4: a stress-test and what-if tool, not a forecaster; no real bank names and no forecasting words on screen; the fitted replay reaches 26% by the close and 82% by the next open against about 25% and 81% in 2023. |
| 114 | Countermeasures (PA-A): the deposit guarantee, an emergency credit line and a withdrawal limit, in that order, each enacted as the engine's existing `PolicyEnacted` lever at minute 240 (or mid-run at the engine's minute): `guarantee_deposits` (M06 uninsured gap x 0, the insured chip), `emergency_lending` (M12's emergency pool: 5% of deposits, 1% a minute, 80% of each minute's shortfall) and `cap_withdrawals` (M06 cuts each request to `M06 cap withdrawal scale`, 0.3; the live amount scale is 1). No new model terms or registry rows; the sizes are assumed and every run with a response is a scenario. Live real mode only. | Sebastian, Nov 4: compare responses on the same preset and seed; the credit line and the limit use the existing levers as they are, labelled scenario. Depositors do not see the credit line, so it changes the bank's cash, not the run. |
| 115 | The multi-seed table runs offline (PA-B): `scripts/batch_table.py` runs every preset x response x seed of the live window (lost-bonds and frozen-branches with no response, the guarantee, the credit line and the limit, plus stadium; seeds 1-10: 90 runs) in parallel processes and writes one file the server serves (`/api/batch`, `/api/batch/run`). Each run keeps a series every 30 minutes, so the no-response run of the same preset and seed is PA-A's comparison. Start districts are not covered (not built). | Sebastian, Nov 4: show the spread over seeds as proof the results are not hard-coded; two live engines would not keep 1x, so the comparison comes from the file. |
| 116 | The four-screen UI is driven by the live engine (UI-1): every screen polls `GET /api/ui/state`, a snapshot at the playback minute (the buffered frame's numbers, so a screen never shows what the engine computed ahead), and the page's renderers read the same `sim` shapes the browser mock filled. The dots show the engine's withdrawn / worried / calm shares; the engine has no map, so a dot's position carries no meaning, and the mock's districts, VIPs, peer banks, automatic backstop and attract loop are hidden. Viewers pick preset rumors and responses as chips; the recap's story and chart use only engine numbers. `?mock=1` keeps the browser mock, marked as such. | Sebastian, Nov 4: wire the engine to the mockup; honesty labels and the limits screen on screen; no free text; one source of truth for every monitor. |
| 117 | The main screen is the vault (UI-2): a fixed sample of the engine's real depositors (the 400 biggest with uninsured money, 6,600 more, 3,000 fully insured; from the seed) around the bank's cash, each coloured by its own share of deposits requested (which sums to the engine's total), with one particle per $100M of real requests and borrowed cash flying into the vault; the rings orbit like moons (inner faster), decoration only. The city map is gone in engine mode because the engine has no geography and its segments run almost in lockstep. Hovering a response shows a card whose measured parts come from the ten-seed batch and whose public cost and moral hazard come from M22's assumed cost model, marked as such; a Fed loan is exposure, not a loss, so it is stated, not added to the public-cost bar. The stocks screen is one whole-window axis with deposits requested per 30 minutes. | Sebastian, Nov 5: a visual that is striking but honest; pros and cons like a game; the vault turning slowly with the outer rings orbiting like Saturn's moons. |
| 118 | The engine searches response plans for the 2023 replay instead of leaving the choice to the viewer (UI-3a). About 90 plans (each response alone at 11 times, pairs at 4 times, all three) are scored on what the engine measures over the same seeds: the bank still open at the next open first, then deposits lost by then; M22's assumed public cost is only a tie-break, because it charges a guarantee whether or not the bank survives. Successive halving (1 seed, the best third on 3, the best third of those on 10) and a check of the top 3 on 10 held-out seeds, so the winner is not a lucky pick. Decision deadlines: the latest minute each response alone keeps the bank open in 9 of 10 seeds (binary search). The tournament screen replays these saved runs; all searches run offline. | Sebastian, Nov 5: the engine should find what would have saved the bank, batches playing out and losing plans eliminated; pick plans by money lost, not a seed. |
| 119 | The search is shown as a shrinking wall of vaults, without rounds (UI-3b). Every plan runs on the same 10 seeds; replaying the day, a plan leaves the wall the moment its result is certain to be worse than the pick: its mean deposits requested already exceed what the pick loses all day (requests only add up), or its bank has closed on enough seeds; the rest are ranked out at the next open. Survivors grow into the freed space; large tiles carry orbiting dots coloured by the engine's crowd counts (a fixed random share per state). A separate page shows why each plan lost, day by day. The pick is checked on 10 held-out seeds. `--earliest` bounds the first response minute, because a guarantee at the opening bell trivially wins. | Sebastian, Nov 5: no rounds; eliminate whatever is already certain to be a bad outcome; enlarge the survivors; a separate screen for why. |
| 120 | A guarantee is searched as a trigger rule, not a clock time (UI-3d): a regulator acts on what the bank reports (withdrawals past a share of deposits, or cash below a share of its opening), so a plan is "guarantee once withdrawals pass X%", checked every minute and acting the minute after the level is crossed. The same 10 seeds are also run with no rumor (the stadium preset): a rule that would fire on one of those normal days is out at once, as a guarantee for a healthy bank. The pick is the rule that saves the bank with the least deposits lost among those that never fire on a normal day. The action window runs from when the run stood out (withdrawals past twice the most a normal day reaches) to the decision deadline (the latest clock time a guarantee still keeps the bank open in 9 of 10 seeds). Whether a guarantee is merited is a legal judgment the engine does not make; it shows when the evidence was there and when it was too late. | Sebastian, Nov 5: a guarantee as early as possible is unrealistic; act when the risk merits it. |
| 121 | A guarantee is justified only once failure is likely (UI-3e). The bond rumor is also run at 9 strengths (believability 0.2 to 0.9, weighted equally) on the same 10 seeds with no response; for each withdrawal level, of the runs that reached it, how many failed? The point of likely failure is the lowest level where at least 9 in 10 did (and at least 10 runs got there). A guarantee rule that fires below it (judged from the exact share requested when it fired) is out as too early, alongside rules that fire on a normal day; the pick is the first justified rule that still saves the bank. The action window opens there and closes at the decision deadline. The level depends on the mix of strengths, and the screens say so. | Sebastian, Nov 5: the earliest rule wins on money lost but guaranteeing after 0.1% left makes no sense; act when failure is at least 90% likely. |
| 122 | The emergency credit line is on the 2023 BTFP terms (CM-2). Behind `M12 emergency line on BTFP terms` (default off; the live server opts in through `M29 live emergency line on BTFP terms`), the live M12 (`LiquidityFake`) lends up to the bank's opening securities at par (M08's SVB-like row, 67.8% of deposits; no new number), covers the whole shortfall (`M12 BTFP line shortfall share` = 1.0, SOURCED) at most `M12 BTFP line ramp fraction of deposits per minute` (0.01, ASSUMED), and once the line is on measures the shortfall against M08's own opening cash instead of the old 0.1% slice. Before the line fires nothing changes, so the 2023 replay is unchanged. M20 still folds the old book (a known gap, after Nov 13). | Sebastian, Nov 5 (option B): the credit line the search tries is the policy that was actually offered in 2023 (the Fed's BTFP lent up to par value of collateral, no haircut), not an assumed 5% line. PA-A's line was measured on the 0.1% slice, so it was spent early and the bank sold its bonds anyway. |
| 123 | The countermeasures' values follow the sources (CM-1). `M22 DIF loss uninsured fraction` = 0.104 (SOURCED: the FDIC's 2023 loss on protecting SVB's uninsured depositors, $15.7B on $151.6B). The guarantee's gap scale stays 0 (Martin, Puri & Ufier: a temporary guarantee works like insurance; the row stays ASSUMED because M06 loads only ASSUMED rows). The moral-hazard bases stay ASSUMED, with only their ordering supported (Demirgüç-Kunt & Detragiache). The cards' public stat is money at risk (the uninsured deposits a guarantee covers once it lands; the Fed's measured lending for the credit line), with the expected loss in words: the fund loses only if the bank still fails. | Sebastian, Nov 5: source the countermeasures; a loss the FDIC takes on a failed bank is not a cost of a guarantee that keeps the bank open. |
| 124 | Viewers choose how many runs per plan to include, not the response (UI-4). The search runs once on a pool of 50 seeds (`--runs 10,25,50`) and writes one result per option; each option uses the first N seeds, the same for every plan (never a chosen subset). The held-out days move to seeds 1001-1010, outside the pool, and the deadlines stay on seeds 1-10. The wall and the Why page share a "Runs per plan" selector (synced across windows) and say how many runs each plan used. The main screen runs the engine's pick; the three responses stay as read-only cards. | Sebastian, Nov 5: the viewer shouldn't pick a response, since the simulations are batched, but can choose how many to run, up to a limit (50). |
| 125 | GPU-1 (sandbox test, opt-in): the offline runs can be batched with the depositors on the GPU. `--gpu` on `search_responses.py` and `batch_table.py` steps many live sessions together (`ui/live/gpu_batch.py::step_sessions_batched`): each lane keeps its own session logic and every module except M06 on the CPU, while M06 runs as one `step_depositor_batch` (the calibration kernel; CuPy if a device is visible). M06's arrays are copied back on sampling minutes. Not for the fitted replay or any shipped number until `scripts/gpu_parity.py` passes on the GPU. | Sebastian, Nov 5: shift everything onto the GPU as a sandbox test; the CPU search takes about 4 hours. |
| 126 | CuPy is the default for every array path (CPY-1). `engine/xp.py` is one switch: `STAMPEDE_BACKEND` now defaults to `gpu` (it was `cpu`), and `cpu` is the NumPy reference, which now also forces M02, M03 and M06 to NumPy on a machine with a device (before, they took CuPy whenever it was visible). A GPU path runs only with CuPy and a device; otherwise it falls back to NumPy, and `STAMPEDE_REQUIRE_GPU=1` (`--require-gpu`) makes the fallback an error. The offline runs batch by default when the GPU paths run (`--cpu` for the reference), with 6 workers sharing the one 12 GB RTX 3060. Payments, the ledger, bonds, liquidity, the regulator and the order book stay on the CPU: scalar bookkeeping that CuPy would only slow. | Sebastian, Nov 5: change everything to CuPy; he tests on the desktop. |
| 128 | Plans are ranked by how useful they are to the bank (UF-1), not by a single winner. `scripts/rescore_useful.py` rescores the search's files in place (originals kept as `*.raw.json`), with no new simulations: until a trigger plan acts, its run is the no-response run, so each plan's day at each of the 9 rumor strengths follows from the search's no-response strength runs (`ui/live/useful.py`). A day is lost when depositors have asked for more than the bank can pay by the next open (`engine/system/can_pay.py`: cash + 2023 borrowing + bonds sellable near book value, 25.5% of deposits; with the Fed's 2023 program on in time, 77.6%). Knocked out: plans that act on more than 1 in 10 rumors that fizzle (too early), the withdrawal limit (not open to a US bank), and plans that save fewer than 9 in 10 dangerous days. The rest are ranked: dangerous days saved, harmless days acted on, escalation, acting between the threat (5% asked) and the point of no return, acting later. Behind `M20 closes on what the bank can pay` (live opt-in), the live regulator closes on the same rule. ADRs 125-126 are applied in this tree; ADR-127 is held by the withdrawn CM-3 change. | Sebastian, Nov 5: a rule that acts at once isn't useful to the bank; keep the wall, knock out the useless plans, and sort the rest by how useful they are. The old regulator test closed the bank once it had sold about $0.7B of bonds (its 0.4% slice), so the wall and the live run disagreed. |
| 129 | The bank's balance-sheet tools join the plans (UF-2), computed from the runs, and the wall shows groups. Bonds pledged at the discount window in advance (`prepared`, minute 0; the bank can pay about 65.3% of deposits: cash + bonds at market value x an assumed 0.95 margin + repo and FHLB) and deposits from big banks (`big-bank-deposits`, $30B as 11 banks gave First Republic on Mar 16 2023, about 17.1 points, if they arrive before the bank runs short) change what the bank can pay, not what depositors do, so `ui/live/useful.py` works out 13 such plans from the no-response runs (marked ≈; the live engine doesn't simulate them and won't run them from the wall). Ranking prefers what the bank controls (the bank, then other banks, then regulators) after dangerous days saved and harmless days acted on, and flags close calls. Each search file gains `groups` (one per seed: every plan's own day and its result in that version of the day) and the wall draws them as rounded squares of small vaults that grow as others drop out, with a tally across groups; the selector picks 10, 25 or 50 groups. | Sebastian, Nov 5: banks have more reactions than the three tools; add the ones that need no new simulations first; batch the simulations in groups with the vaults inside. |
| 130 | The automatic loop (AU-1..7). One job (`scripts/pipeline.py`, started by the wall's Run the search button through `POST /api/pipeline/run`, progress at `/api/pipeline/status`) simulates the pool (skipped when the registry, the plans and the options are unchanged), fills in the strengths where the rumor tips, simulates messaging on the best ladders (a report: statements and rate rises change behaviour, so they can't be worked out), and rescores every file. The rescoring writes its own plans (`ui/live/plangen.py`: every tool at every trigger, ladders with the bigger step later, each with and without bonds pledged in advance, about 316) and works them out from the no-response runs; it checks the shortcut on every simulated run, reranks under 7 versions of the assumptions, finds the tipping point, stores the early-warning data, and writes one plain sentence per plan from its numbers. The live engine runs the bank's tools and messages (`presets.TOOLS`: pledged bonds, big-bank deposits, a statement, an 8% deposit rate), so any plan on the wall plays in the main vault. GPU-1 and CPY-1 were applied before this change (ADR-125, ADR-126). | Sebastian, Nov 5: STAMPEDE is a lot of automatic things, so stick with that and implement the automatic loop from where we are. |
| 131 | The demo is three beats with one headline (UI-5). The nav is 1 A rumor hits (the vault), 2 The engine tries the plans (the wall), 3 What helps (the old Why page, trimmed); Recap and Peers are operator-only, and each beat has a button to the next. The Bank and Stocks screens are removed with their code and the Open screens menu (`?screen=bank|stocks` opens the main screen); this supersedes ADR-117's whole-window stocks screen. One headline sits under the top bar. On beats 2 and 3 it gives the answer, written only from the search's numbers (the #1 plan's label and its share of dangerous days saved) and tagged ≈ when the plan is worked out rather than simulated. On beat 1 and its recap it is a neutral sentence, so the answer is revealed with the wall (option B: a presenter-led demo). What helps shows the #1 plan with its close call, the best plan of each other kind, why the rest are out (counted across every plan considered: `considered.knocked`), when the bank has to act, and the checks; the full table and the method are folded under details (open in operator view). On screen, groups and seeds are versions of the day and strengths are weaker and stronger rumors. | Sebastian, Nov 5-6: the Bank and Stocks tabs mislead; the UI got too complicated for a first-time viewer, so three beats, plain words and one headline; the answer waits for beat 2 (Nov 6, option B). |
| 132 | GPU parity is judged in the units the search uses (VF-1). `scripts/gpu_parity.py` passes a run when the outcome is the same, every response fires on both paths within 1 minute, and deposits requested agree within 1e-4 of deposits (0.01 points, about $17.5M) at the end and at every 10-minute sample; the relative gap is printed for information only. This replaces GPU-1's 1e-4 relative or 1e-6 absolute, which failed 8 of 21 runs on the 3060 (Nov 6) while every outcome and trigger minute agreed. The gap is float32 rounding, not a defect: the same batched code with the kernel in float64 matches the CPU path to 2e-10 relative (independent NumPy check); its sign is random by seed and the same for every plan on a seed; and it is largest where the bank stays open with most depositors still deciding, which is also where a run asks little, so a relative test divides the same dollars by a small number (worst $6.4M, 0.0036 points, on an 8% run). The search ranks on outcomes and trigger minutes (compared exactly) and on margins of 0.3 points (the close call) and 0.9 points (the checks); 0.01 points is 30x below the smallest and 2.5x above the worst gap seen anywhere in the day. The 3060's CuPy gap is about 45x NumPy float32's on the same code; `--unfused` (no fused CuPy kernels) measures how much of that the fused float32 arithmetic makes, and a float64 kernel stays an option if a GPU result ever needs bit-level agreement. With CPY-1 the search batches on the GPU by default, so the 236 runs added by AU's pipeline (Nov 6) came from this path and the 2,330 earlier runs from the CPU; under this rule the files may mix the two. Supersedes ADR-125's pass rule; ADR-125's condition (no shipped number from the GPU path until parity passes) now reads against this rule. | Sebastian, Nov 6: explain the 8 failures and fix them or justify a tolerance, not loosen it to pass. |
| 133 | The CuPy fused kernels are not the source of the 3060's extra float32 gap (VF-2). On the 3060 (Nov 6), `scripts/gpu_parity.py --unfused` (no `cupy.fuse` or `ElementwiseKernel` paths) gave the same gap as the fused run on every one of 21 runs, to 1e-9 of deposits; both pass ADR-132 21 of 21 (worst 0.0036 points at the end, 0.004 at a sample). The gap stays float32 rounding (ADR-132's evidence), about 45x NumPy float32's on the same code; the remaining candidate is the float32 GPU paths of M02, M03 and M05, which ran on NumPy float64 in the sandbox experiment. `--kernel-only` (`STAMPEDE_BACKEND=cpu` on the batched side, the kernel on `--xp cupy`) separates them. Diagnostic only: ADR-132's rule and the shipped numbers do not depend on its result. | Grok's VF-1 validation (Nov 6): unfused does not materially reduce the gap. |
| 134 | The live screen names every lever in plain words and says who acts (VF-3). `LiveSession._lever_words` takes each lever's label from the plan steps (`presets.STEPS`): the bank's own tools (bonds pledged in advance, deposits from big banks, a statement, a rate rise) read as Copperleaf acting ("Copperleaf: Bonds pledged in advance", a bank card with the step's note), and the regulators' responses as announced; an unknown lever is shown as words, never its code name. The early warning on the vault says its similar days are days with no response ("sank the bank with no response"), so it doesn't read as a forecast for a run that has a plan on. | The Nov 7 live walkthrough: the engine's pick showed `pledge_collateral` on the vault, in the crisis card and the headlines, and the warning said 50 of 50 similar days sank the bank while the plan that saves it ran. |
| M29 UI client stream fps | 30 | ASSUMED | §5 M29 |
| UI WebSocket stream rate | 10 Hz | UNSET | §20.3 (M29) |
| M03 sim tick minutes | 1 | ASSUMED | §2.3 one tick is one simulated minute |
| M03 role exposure multiplier enabled / multipliers; role source node enabled / nodes | false; 11-role map; false; 11-role map | ASSUMED | ADR-093 (CPU engine only) |
| M03 rumor strength impulse enabled / multipliers | false; low 0.8, medium 1, high 1.25 | ASSUMED | ADR-094 (CPU engine only) |
| M03 exaggeration feedback enabled / gain / cap | false; 1; 1.5 | ASSUMED | ADR-095 (CPU engine only) |

M29 draws that render-point budget as orbs. Block size is `round(population / orb count)` (1,000,000 / 2,048 = 488). Base radius, uninsured extra, radius cap, orbs per tick, uninsured fraction, and the bundled demo population are ASSUMED rows in `/params/registry.csv` (ADR-044). Live crisis minutes, playback target seconds, and client stream fps are ASSUMED rows in the same file (ADR-042). This excerpt does not repeat them.

M01's Nelson–Siegel inputs (lambda, beta0, beta2, maturity buckets), policy-rate default and bounds, CRE and unemployment baselines, and the representative bond duration and convexity used for the §18 loss check are ASSUMED rows in `/params/registry.csv` (module M01). This excerpt does not repeat them.

M26 reads the ADR-004 rows for the batch shape (16 futures, 250k depositors, 5-minute ticks). Worry weights, contagion k, the cash fraction, price-path sensitivities, and the other kernel inputs are ASSUMED rows in `/params/registry.csv` (module M26). This excerpt does not repeat them. The forward run uses the §9 sim-day length of about 78 ticks.

M14's halt threshold (X%) and window (Y minutes) are unspecified in §18.4; the original registry rows stay TBD. Real mode reads ASSUMED rows `M14 halt price move threshold (X%)` = 8 and `M14 halt window (Y minutes)` = 5 (the Wave A stand-in). The real book, the gpu-sim, and the fake halt countdown all read SOURCED `M14 trading halt minutes` = 5 (SEC LULD; SVB-25). `M14 trading halt duration` (15) is DEPRECATED and unread (ADR-074). Lean/gpu-full price impact is the ASSUMED `M14 price impact (ASSUMED)` coordinate (default 1.13, bounds 0.3–4) so full saturation at the mobile-share default (0.813 requested) with attention_factor 1 is about a −60% stock. Other order-book coefficients are ASSUMED M14 rows in the same file (ADR-023, ADR-068).

M02's attachment degree, seed size, batch size, group-chat degree and homophily, advisor count, professional degree, media-node count, population tier, credibility terms, and channel weights are ASSUMED rows in `/params/registry.csv` (module M02). This excerpt does not repeat them (ADR-027).

M03's half-life is the Attention memory window above (E10), not a second row. Tick length, exposure weights, fallback channel weights, the repost scale, the mutator sample fraction, the call cap, the context-note factor, and the offline exaggeration are ASSUMED rows in `/params/registry.csv` (module M03, ADR-028). The older "about 1%" and "≤20" wording rows stay UNSET because they are not numeric. This excerpt does not repeat the assumed rows.

M06's complex-contagion k stays TBD on `Depositor complex contagion minimum worried contacts (k)` (UNSET). Real mode reads the ASSUMED row `M06 complex contagion k (ASSUMED)` and does not hard-code k. Worry weights, the hub weight, segment withdrawal fractions, and the statement shift are ASSUMED M06 rows in `/params/registry.csv` (ADR-029). `M06 worry bias (ASSUMED)` is a stage-1 fit coordinate and may be FITTED after persist (ADR-088). This excerpt does not repeat them. Fake demo pacing (end share, logistic midpoint and steepness, cash-empty fraction, closure-hole fractions, poster credibility, price drawdown) is a separate set of ASSUMED rows (ADR-047). Those rows are not the FIT behavioral weights and do not replace VERIFIED outflow totals.

M44 reads `M44 assumed vulnerability publish threshold` (ASSUMED). The older row `Short-seller vulnerability publish threshold` stays TBD/UNSET. The capital-stress bump, reference CET1, community uninsured share, thesis reach, short-order quantity, and thesis model are further ASSUMED rows in `/params/registry.csv` (ADR-038). This excerpt does not repeat them.

M42's attention trigger, stock-slide trigger, deposit-rate level, cap, floor, segment, segment balance fraction, borrow haircut, sale fraction, sensitivity weights, fleet size, cooldown, Grok model, and live-call flag are ASSUMED rows in `/params/registry.csv` (module M42, ADR-039). This excerpt does not repeat them. Menu actions stay M23's `MANAGEMENT_ACTIONS`.

M41's agent count, batch size, memory depth, and initial semaphore stay on the existing UNSET rows (15, 10 per call, 5 events, 32 in flight). The cadence prose row stays UNSET. Real mode reads ASSUMED rows for the 30-minute interval, the attention trigger, the withdrawal fraction, the Kestrel Labs example amount ($40M), the Grok model, reduced-rule sensitivities and margin, the tail-rank fraction, and the baseline belief (ADR-040). This excerpt does not repeat them.

The local LLM default model, OpenAI base URL, concurrency semaphore, request timeout, and weights directory are ASSUMED rows in `/params/registry.csv` (ADR-043, ADR-057). This excerpt does not repeat them.

Non-bank price-path targets (E13–E16) and the fake-mode knobs for `/calibration/nonbank.py` are rows in `/params/registry.csv` (ADR-050). Verified point statistics are the Meta session drop, its one-day length, its volume multiple, and the S&P 500 February–March 2020 drop and day count. GameStop and LUNA point shape statistics that could not be pinned to one number stay UNSET and are left out of the loss. Credibility, reach, the rate path, and the post-read scale are ASSUMED. This excerpt does not repeat those rows. The SVB rows that are fit on SVB only are unchanged. The GPU non-bank path reuses the behavioral-dimension row and the M27 GPU sim E1 relative tolerance. It adds no registry rows and it does not write FITTED cells (ADR-052). The ASSUMED row `Non-bank daily deposit-request bound` is 0.001 (0.1% of deposits per day); its penalty weight is a separate ASSUMED row (ADR-053). This excerpt does not repeat those rows.

M27's batched candidate eval reads the ASSUMED rows `M27 GPU sim E1 relative tolerance` (1e-6) and `M27 GPU sim array budget bytes` (3e9). This excerpt does not repeat them (ADR-051, ADR-061). The E1 requested-outflow fit also reads ASSUMED rows for the identifiable indices (price impact and mobile share), eight shared seeds, softness anneal, weak log-space prior, bracket grid, log/bias bounds, and CMA-ES restarts (ADR-054, ADR-067, ADR-069). Contact weight, feedback gain, and dispersion stay at their priors. Stage-1 fit indices are 7, 15, 16 (worry bias, mobile share, decision-to-request rate); price impact stays at its registry value and the E1 stock ratio is reported, not fitted (ADR-085). The media awareness rate stays at its registry prior (ADR-084). E1 and E5 target cells are unchanged except ASSUMED `Calibration E1 requested day-one target` = 0.249 (SVB-34; same weight and relative error as next-open). `Validation target: requests by next open` is ±10% relative and is also the board band for `E1-requested-day1`. M07 app, wire, and ACH capacities are scaled by `M07 capacity multiplier (FIT)` (default 1.0; FITTED by stage 2). This excerpt does not repeat that row (ADR-066). The ASSUMED row `Depositor mobile share of uninsured deposits (FIT)` is 0.88 (bounds 0.30–1.00); 1.0 reproduces the previous uncapped amounts. The FIT row `M07 decision-to-request rate (FIT)` is 1/600 per minute (bounds 1/5000–1/20). This excerpt does not repeat the min/max companion rows.

## 8.2 Validation targets (verified)

| ID | Event | Fact used | How it's used |
| :---- | :---- | :---- | :---- |
| E1 | SVB, March 9–10, 2023 | \$42B withdrawn March 9, \$100B scheduled for March 10, together 81% of \$175B in deposits; about 0.249 requested by day-one close | Fit |
| E2 | SVB depositor base | Uninsured deposits were over 93% of deposits | Input |
| E3 | SVB concentration | Top 10 depositors held \$13.3B | Input |
| E4 | SVB trigger | March 8: a \$1.8B securities loss plus a capital-raise plan, read by uninsured depositors as distress. The bank ended March 9 with negative cash of nearly \$1B. | Input and timing |
| E5 | Signature, March 10, 2023 | \$18.6B withdrawn in hours (20% of deposits); a cash deficit of nearly \$4B; about 1,000 wires worth \$4.6B pending for Monday | Test |
| E7 | System policy | March 12: all deposits guaranteed at SVB and Signature, and an emergency lending program created. March 16: big banks place \$30B at First Republic. | Policy-lab timing |
| E10 | Social media mechanism | Attention over the past 4 hours predicts hourly stock losses; negative sentiment doesn't amplify run risk; banks with high pre-run Twitter exposure lost 4.3 percentage points more | Design and test (M03, M14) |
| E11 | Messaging | Personalized, strong reassurances reduce withdrawal intent | Persona-panel direction check |
| E12 | Speed and monitoring | 2023 runs were unprecedented in speed. Social-media signals alone produce false positives and need bank-specific calibration. | Framing; reason for simulating |
| E13 | Meta (Facebook) earnings day, February 3, 2022 | Close 323.00 to 237.76 (26.390% of the prior close), one trading session, volume 188,119,900 versus 58,458,300 the prior session | Equity-company fit (price path) |
| E14 | GameStop, January 2021 | Intraday high $483 on January 28; $17.25 at the start of the month; January 25 volume more than 175 million versus a 29.8 million 30-day average; more than 80% off the intraday peak by February 2. Peak-to-trough, days to bottom, and the volume multiple as point targets are UNSET | Documented equity episode; not a fit preset |
| E15 | Terra LUNA, May 2022 | $87 on May 5 to less than $0.00005 on May 13; April 5 peak $119.18; NBER: ecosystem value to zero within three days. Point peak-to-trough, days, and volume multiple are UNSET | Crypto holdout |
| E16 | S&P 500, February 19–March 23, 2020 | Close 3,386.15 to 2,237.40, −33.9% in 33 calendar days (weekends included). Volume multiple is UNSET. Using this index path as the rates-shock target is ASSUMED | Rates/macro holdout |

## 8.3 Assumptions to confirm

* Current Grok model names and prices (§7).  
* Desktop RTX 3060, 12 GB VRAM (§9).

## 8.4 Non-bank price-path targets

The loss compares a simulated `PriceTick` path to these point statistics. A row marked UNSET is not a number the fit may invent. Sources for the tags are also in §17.

| ID | Statistic | Value | Status | Source |
| :---- | :---- | :---- | :---- | :---- |
| E13 | Meta close-to-close drop, Feb 2–3 2022 | 0.2639009287925697 = (323.00 − 237.76) / 323.00 | VERIFIED | [Yahoo Finance META history](https://finance.yahoo.com/quote/META/history/?filter=history&frequency=1d&includeAdjustedClose=true&interval=1d&period1=1509408000&period2=1667088000) |
| E13 | Days to the earnings-session trough | 1 trading day | VERIFIED | [CNBC, Feb 3 2022](https://www.cnbc.com/2022/02/03/facebook-shares-plummet-22percent-after-reporting-weak-guidance.html) |
| E13 | Volume versus the prior session | 3.218018656033446 = 188,119,900 / 58,458,300 | VERIFIED | Same Yahoo history page |
| E14 | GameStop intraday high, Jan 28 2021 | $483 | VERIFIED | [Wikipedia, GameStop short squeeze](https://en.wikipedia.org/wiki/GameStop_short_squeeze) |
| E14 | GameStop price at the start of January 2021 | $17.25 | VERIFIED | Same Wikipedia page |
| E14 | GameStop Jan 25 volume versus 30-day average | more than 175 million shares versus 29.8 million | VERIFIED | Same Wikipedia page (Dow Jones market data) |
| E14 | Decline from the intraday peak by Feb 2 2021 | more than 80% | VERIFIED | Same Wikipedia page |
| E14 | Peak-to-trough fraction, days to bottom, volume multiple | UNSET | UNSET | The page gives bounds, not a trough price, a trough date, or a point volume multiple |
| E15 | LUNA price on May 5 2022 | $87 | VERIFIED | [arXiv:2207.13914](https://arxiv.org/html/2207.13914v3) |
| E15 | LUNA price on May 13 2022 | less than $0.00005 | VERIFIED | Same arXiv page |
| E15 | LUNA peak on April 5 2022 | $119.18 | VERIFIED | [NBER w31160](https://www.nber.org/system/files/working_papers/w31160/w31160.pdf) |
| E15 | Ecosystem value | to zero within three days | VERIFIED | Same NBER paper |
| E15 | Peak-to-trough fraction | 0.9999996 | SOURCED | Derived: 1 − 0.00005/119.18 from the April 5 peak and the May 13 price (a lower bound, since the May 13 price is an inequality) |
| E15 | Days to bottom, volume multiple | UNSET | UNSET | The three-day NBER window disagrees with the May 5–13 arXiv span. Neither paper states a crash-day volume multiple |
| E16 | S&P 500 peak close, Feb 19 2020 | 3,386.15 | VERIFIED | [Yardeni appendix 15.4](https://www.yardenibook.com/pub/15.4appendix.pdf) |
| E16 | S&P 500 trough close, Mar 23 2020 | 2,237.40 | VERIFIED | Same appendix |
| E16 | Peak-to-trough | 33.9% (0.339) | VERIFIED | Same appendix, printed loss |
| E16 | Days to trough | 33 calendar days, weekends and holidays included | VERIFIED | Same appendix footnote |
| E16 | Volume multiple | UNSET | UNSET | The appendix does not report volume |
| E16 | Map the index path onto the simulated stock | ASSUMED | ASSUMED | The harness still reads the scenario bank's `PriceTick` path |
| — | Daily deposit-request bound on a non-bank market event | 0.001 of deposits per day | ASSUMED | A GameStop-like shock may nudge the bank stock; deposit requests stay inside this bound |

# 9\. Performance budget (desktop RTX 3060, 12 GB VRAM)

| Item | Budget |
| :---- | :---- |
| 1M depositors × \~64 bytes | ≈ 64 MB |
| 3 graph layers × \~20M edges × 8 bytes | ≈ 0.5 GB |
| Full-detail tick (1M depositors, 2 banks) | Budget: \< 5 ms (GPU) plus \< 3 ms (CPU accounting) |
| 16 futures × 250k depositors × 5-minute ticks × 1 sim-day (\~78 ticks) | Budget: \< 5 s, batched on the GPU |
| Crisis room end to end (32 statements) | Budget: \< 20 s: about 10 s of LLM time (parallel, cached) plus about 5 s of futures |
| UI frame rate | 30 fps with 1M points (deck.gl), with level of detail when zoomed out |

Batched M27 candidate eval uses the same 12 GB card. The working set of one candidate×seed group is capped by the ASSUMED row `M27 GPU sim array budget bytes` (3 GB) so 256 lanes × 100k depositors stay in one group (float32 per-depositor arrays; ADR-051, ADR-058, ADR-061, ADR-063).

# 10\. Development workflow

## 10.1 Code areas

| Area | Folders | Modules |
| :---- | :---- | :---- |
| L1 Engine | /engine/core, /engine/agents, /engine/futures | M00, M02, M03, M05, M06, M26 |
| L2 Banks and liquidity | /engine/bank (except management.py), /engine/liquidity, /engine/macro | M01, M07–M09, M12 |
| L3 Markets | /engine/markets | M14 |
| L4 System and policy | /engine/system, /engine/policy | M20, M22 |
| L5 Grok agents | /agents, plus /engine/bank/management.py (M23 only) | M04, M23–M25, M34, M41, M42, M44 |
| L6 Crowd UI | /ui/crowd | M29 |
| L7 Market and crisis-room UI | /ui/market | M30 |
| L8 Calibration and reliability | /calibration, /scenarios, /ops | M27, M28, M33 |

M23's rule core lives at /engine/bank/management.py (area L5) so it can post into the ledger; L2 covers every other file under /engine/bank (ADR-016).

**Shared files: /contracts, /params/registry.csv, dependency manifests, and /docs/ARCHITECTURE.md. Changes to them go with the code that needs them.**

## 10.2 Build order

Modules were built in dependency order: contracts and mocks first, then foundations (M00, M01, M05, M08, M09, M33, M34), then the core loop (M02, M03, M06, M07, M12, M14, M23–M28), then system depth (M04, M20, M22, M41, M42, M44), then calibration, validation and the live UI. Tiers are in §11.

## 10.3 Gates (all must pass before merging to main)

1. Type checking and linting.  
2. Unit tests for the changed modules.  
3. **Contract tests:** every event matches its schema.  
4. **Accounting invariant:** assets equal liabilities plus equity on every tick of a 1-day run.  
5. **Golden scenarios:** each preset stays inside its tolerance (§13).  
6. **Performance:** tick time and futures time stay within the budget (§9).  
7. **Demo smoke test:** a headless run of the scripted 2-minute path.  
8. Screenshot diff of the dashboard.  
9. **Doc check:** CI fails if a PR changes /engine or /contracts without changing /docs/ARCHITECTURE.md. The root ARCHITECTURE.md file is only a pointer and does not satisfy this gate.

## 10.4 Change cadence

Each change is small, lands on its own branch with its tests, and updates this document (§0) when files, functions, contracts or parameters change.

## 10.5 Repository rules

1. **Read first:** read §4–§8 of /docs/ARCHITECTURE.md before each change.  
2. **No hard-coded numbers:** read every parameter from /params/registry.csv. A new parameter needs a row with a source or a FIT or ASSUMED status.  
3. **Events only:** never mutate another module's state. Publish events through /contracts.  
4. **Grok outputs are suggestions:** they never write numbers into the simulation state. Every LLM reply is validated against a schema and cached.  
5. **Fictional names only:** every bank name passes the M34 blocklist.  
6. **Tests with every change:** each behavior change adds or updates a test and updates the module spec in §5.  
7. **Determinism:** tests always run with fixed seeds.

## 10.6 Build plan

CUT: superseded by §10.2.

# 11\. Feature catalog and tiers

| Tier | Features |
| :---- | :---- |
| 0: Core | One bank (SVB-like preset); 100k depositors; insurance; securities and rate losses; cash and repo on the ladder; crowd screen; rumor box with "who posts it"; deterministic demo mode |
| 1: The loop | Attention engine and rumor mutation; withdrawal channels and wire queues; FHLB and discount window; per-bank stock with order book and halts; management agent; crisis room with truth filter, persona panel, and 16 futures; SVB fit |
| 2: The system | 2 banks (SVB-like and a held-out Signature-like bank); media and analyst agents (M04); policy lab with and without regulator action; Signature test; 1M depositors; VIP and CEO agents (M41, M42) |
| 3: Depth | Short sellers (M44) |

**Feature flags: multibank, live\_attention, vip\_agents, ceo\_agents, shorts. Each flag is off until its golden test passes.**

# 12\. Scenarios and demo scripts

## 12.1 Presets (all fictional banks with real-like profiles)

| Preset | Built to reproduce |
| :---- | :---- |
| Tech-concentrated bank, rate shock and rumor | E1–E4 |
| Crypto-heavy follower bank | E5 |
| Insured community bank | Placebo: no run |

## 12.2 Two-minute demo

| Time | What happens |
| :---- | :---- |
| 0:00 | Hook (the SVB number) |
| 0:15 | The user picks who posts the rumor |
| 0:30 | The run starts: red spreads, cash drains, the stock slides, trading halts |
| 1:05 | The crisis room tests 32 statements in parallel |
| 1:25 | Rewind and replay: the run stalls |
| 1:40 | Mechanism and validation number |
| 1:55 | Close |

# 13\. Validation status board

| Target | Metric | Tolerance | Result | Status |
| :---- | :---- | :---- | :---- | :---- |
| E1 SVB day 1 | Outflow as % of deposits | ±25% relative | — | ☐ |
| E1 requested day one | Requests by day-one close | ±10% | — | ☐ |
| E1 SVB 2-day / requests by next open | Cumulative requests, including scheduled | ±10% | — | ☐ |
| E5 Signature | First-day outflow % (no refit) | ±35% | — | ☐ |
| Placebo | Insured community bank survives | Yes or no | — | ☐ |

Tolerances are deliberately loose. The claim is "reproduces the shape and ordering of 2023–24 events," not "forecasts."

E13–E16 are non-bank price-path targets (§8.4). They are not rows on this SVB board.

# 14\. Visual system (red and green, colorblind-safe)

* **Palette:** a bluish green for calm, rising, or available; gray for wavering; an orange-leaning red for withdrew, falling, or exhausted. These two hues stay distinct for the most common types of color blindness.  
* **Never color alone:** pair every color with ▲/▼ arrows, ✓/✗ icons, brightness (withdrawn dots are brighter), or a text label.  
* **Calm default:** the screen is muted before any shock, so the first red stands out.  
* **Fictional branding:** every screen shows a "FICTIONAL" tag next to the bank name.

# 15\. Risks and descoping order

| Risk | Mitigation |
| :---- | :---- |
| Scope explosion (27 modules) | Tiers and feature flags; Tier 0–1 is a complete demo on its own |
| Overfitting or false precision | Fit on SVB only; held-out tests; ranges, never point forecasts; loose tolerances stated publicly |
| LLM latency or cost | Batched calls, caching, a spend cap, and template fallbacks |
| GPU memory | Reduce depositors per future; split by bank |
| Audiences can't follow the complexity | The front of the demo stays a simple red/green story |
| Legal or ethical concerns | Fictional names, a hard truth filter, no real-bank forecasts |

**Descoping order:**

1. Simplify the order book.

The crisis room is never cut.

# 16\. Change log and decisions

## 16.1 Change log

| Version | Change |
| :---- | :---- |
| v1.0 | Initial architecture |
| v1.0.1 | Add `params/loader.py` to load `/params/registry.csv` (§21 tree lists only `registry.csv`). |
| v1.0.2 | Clarified M06 and M34 paths, gate 9, three event rows, the 27-module wave count, M23 ownership, and §19 event edges (ADR-012–017). |
| v1.0.3 | M00 real clock at `/engine/core/clock.py`; calendar parameters in §8.1 and `params/registry.csv`; real mode wired for M00 only (ADR-018). |
| v1.0.4 | M05 Wave B: `/engine/agents/population.py` real mode; tail targets use lognormal+Pareto seed plus iterative mass transfer (§18.2 bisection targets both insured share and top-10). |
| v1.1.0 | M01 real mode in `/engine/macro/rates.py` (Nelson–Siegel curve, piecewise policy path, one-shot shocks). `RatesChanged` unchanged. M01 parameters added to `/params/registry.csv`. ADR-019. |
| v1.1.1 | M08 real ledger at `/engine/bank/ledger.py` (Wave B): double-entry postings, CET1 with AOCI toggle, invariant with `pending_realized_loss` until `LossRealized`; opening cash/securities slices match fake fold_book until M28 preset `Bank` objects land; `ModuleMode.REAL` wired for M08 in `engine/registry.py`. |
| v1.1.2 | M09 real `SecuritiesValuer` in `/engine/bank/securities.py` (`price_change_fraction`, `mark_buckets`, `realize_sale`, `BondBucket`); HTM reclassify on sale defaults on (ADR-020). |
| v1.1.3 | M34 real guardrails in `/agents/guardrails.py`; registry rows for blocklist and fuzzy distance; `build_engine` wires `ModuleMode.REAL` for M34 only. |
| v1.1.4 | M07 real payment processor in `/engine/bank/payments.py` (app daily limit, FIFO wire queue with per-tick capacity, ACH batch at business close, branch drip); ASSUMED channel limits in `/params/registry.csv`; `ModuleMode.REAL` wired for M07 in `engine/registry.py`. |
| v1.1.5 | M12 real liquidity ladder in `/engine/liquidity/ladder.py` (`FundingSource`, `CollateralLedger`, fastest-feasible draw); `ModuleMode.REAL` wired for M12 in `engine/registry.py`. |
| v1.1.6 | M23 real bank management rule core in `/engine/bank/management.py`; policy thresholds stay TBD in CSV with ASSUMED fallbacks (ADR-022); `build_engine` wires `ModuleMode.REAL` for M23. |
| v1.1.7 | M14 real order book at `/engine/markets/book.py` (Numba price-time match, five trader types, volatility halt). Halt X%/Y minutes and halt duration are ASSUMED registry rows. `ModuleMode.REAL` wired for M14. ADR-023. |
| v1.2.0 | M29 crowd UI shell in `/ui/crowd/` (`CrowdFrame` / `CrowdPanels`, CPU layout, msgpack frames, React panels); fake-mode render point count in `params/registry.csv`. ADR-024. |
| v1.2.1 | M30 UI shell at `/ui/market`: `MarketScreenFrame` (§6.2), fake-engine snapshot feed, optional WebSocket server; `MarketScreenFake` adds depth and chart markers. ADR-025. |
| v1.3.0 | M33 real observability in `/ops/` (Philox RNG with GPU path and CPU fallback, Parquet event logs, SQLite LLM cache, offline smoke); `ModuleMode.REAL` wired for M33 in `engine/registry.py`; M33 steps after M30 so the event-log hash includes the full tick. ADR-026. |
| v1.4.0 | M02 real multilayer network at `/engine/agents/network.py`: batched preferential-attachment public feed, founder/VC stochastic block model, bipartite treasurer–advisor ties, CSR on CuPy when CUDA is available and NumPy otherwise. `ModuleMode.REAL` wired for M02; FAKE stays default. ASSUMED rows in `params/registry.csv`. ADR-027. |
| v1.5.0 | M03 real attention and rumor engine in `/engine/agents/attention.py` (NumPy/CuPy backend, half-life, hub spread, offline mutator). §6 payloads unchanged. `ModuleMode.REAL` wired for M03. ADR-028. |
| v1.6.0 | M06 real depositor decision kernel in `/engine/agents/decide.py` (NumPy on CPU, CuPy when CUDA is available). Complex-contagion k is an ASSUMED registry row; the TBD row stays UNSET. `ModuleMode.REAL` wired for M06; FAKE stays the default. ADR-029. |
| v1.7.0 | M25 real persona panel in `/agents/personas.py` with SQLite LLM cache (`/agents/llm_cache.py`), rule-based offline fallback, and `ModuleMode.REAL` in `engine/registry.py`. M25 rule and catalog parameters added to `/params/registry.csv`. ADR-030. |
| v1.8.0 | M27 real calibration harness in `/calibration/fit.py` (CPU CMA-ES on fake-engine evals, E1 fit, E5 holdout, §13 board); ASSUMED rows for optimizer knobs in `params/registry.csv`; `ModuleMode.REAL` wired for M27 in `engine/registry.py`. ADR-031. |
| v1.8.1 | `engine/modes.py` lists every module wired to `ModuleMode.REAL` (M00, M01, M02, M03, M05, M06, M07, M08, M09, M12, M14, M23, M25, M27, M33, M34). |
| v1.9.0 | M26 real ensemble at `/engine/futures/batch.py` (`FuturesRunner.step`, `run_batch`, `step_m06_kernel`, `run_serial_reference`). NumPy on CPU, CuPy only when CUDA is visible. `FutureCompleted` unchanged. ASSUMED kernel rows in `/params/registry.csv`. `ModuleMode.REAL` wired for M26; FAKE stays the default. ADR-032. |
| v1.10.0 | M24 real crisis-room pipeline in `/agents/crisis_room.py` (asyncio writer → JSON claims → in-code numeric truth filter → cached vague filter via `/agents/llm_cache.py`); `ModuleMode.REAL` wired for M24 in `engine/registry.py`. ADR-033. |
| v1.11.0 | M28 real scenario builder in `/scenarios/` (preset `Bank` objects, builder, replay hash, branch compare); `ModuleMode.REAL` wired for M28 in `engine/registry.py`. ADR-034. |
| v1.11.1 | `engine/modes.py` lists every module wired to `ModuleMode.REAL` (M00, M01, M02, M03, M05, M06, M07, M08, M09, M12, M14, M23, M24, M25, M26, M27, M28, M33, M34). ADR-031 notes that M06 real mode is wired. |
| v1.12.0 | M22 real policy lab in `/engine/policy/levers.py` (scheduled levers, cost model, moral-hazard meter, offline regulator advisor); ASSUMED M22 rows in `params/registry.csv`; `ModuleMode.REAL` wired for M22 in `engine/registry.py`. ADR-035. |
| v1.13.0 | M20 real supervision and closure engine in `/engine/system/regulators.py` (`fold_book` triggers, business-day vs weekend timing); ASSUMED threshold rows in `params/registry.csv`; `ModuleMode.REAL` wired for M20 in `engine/registry.py`. ADR-036. |
| v1.14.0 | M04 real media and analyst agents in `/agents/media.py` (threshold crossings, cited events, SQLite LLM cache with offline templates); ASSUMED threshold rows in `params/registry.csv`; `ModuleMode.REAL` wired for M04 in `engine/registry.py`. ADR-037. |
| v1.15.0 | M44 real short-seller and activist agents in `/agents/shorts.py` (vulnerability score, offline Grok thesis cache, short order recorded for M14). ASSUMED registry rows for the unset threshold. `ModuleMode.REAL` wired for M44; fake stays default. ADR-038. |
| v1.16.0 | M42 real bank CEO agents in `/agents/ceo.py` (Grok tool calls mapped onto M23's menu, ledger refusal, per-bank log, offline rule fallback via `/agents/llm_cache.py`). ASSUMED rows in `params/registry.csv`. `ModuleMode.REAL` wired for M42; FAKE stays default. ADR-039. |
| v1.17.0 | M41 real VIP agents in `/agents/vip.py` (persona cards, four tools, batches of 10, pydantic schema, threshold-and-sensitivity reduction for M26). Grok calls go through `/agents/llm_cache.py`; offline rule fallback is the default. `ModuleMode.REAL` wired for M41 in `engine/registry.py`. ASSUMED rows in `params/registry.csv`. ADR-040. |
| v1.17.1 | `engine/modes.py` lists every module wired to `ModuleMode.REAL` (M00, M01, M02, M03, M04, M05, M06, M07, M08, M09, M12, M14, M20, M22, M23, M24, M25, M26, M27, M28, M33, M34, M41, M42, M44). |
| v1.18.0 | Unified live FastAPI at `ui.server` (crowd + market WebSockets, rumor POST/WS, start/pause/reset); `CrowdFake.queue_live_rumor`; `scripts/run_live.sh`; offline E2E test. ADR-041. |
| v1.19.0 | M29 orbs are fixed blocks (`round(population / render points)`, 488 people at 1M / 2,048). Majority color, capped uninsured radius, a few orbs ease per tick, cluster tint follows the red share. Live msgpack floats decode as float32. M30 fake monitor matches the run-phase comp. ADR-044. |
| v1.20.0 | Real-time live spread pacing (`ui.live.pacing`, wall-clock loop, 1×–8× speed, skip to outcome, 30 fps stream with client backpressure); fake-mode scripted spread on orb frames; client interpolation keeps positions on the layout. Offline pacing tests. ADR-042. |
| v1.21.0 | Local OpenAI-compatible LLM backend in `/agents/llm_cache.py` (`STAMPEDE_LLM=offline\|grok\|local`, `STAMPEDE_LOCAL_LLM_URL`, `STAMPEDE_LOCAL_LLM_MODEL`). Grok agents call `/v1/chat/completions` with the server chat template. ASSUMED local concurrency and timeout rows in `params/registry.csv`. `docs/local-llm.md` and `scripts/serve_local_llm.sh`. ADR-043. |
| v1.22.0 | M27 GPU calibration: real M02–M09 evals (`STAMPEDE_CALIBRATION_ENGINE=real`), `decision_params_from_vector`, `calibration/run_fit.py` checkpoints and FITTED registry output; CuPy RNG/permutation fixes in M02/M03. ADR-045. |
| v1.23.0 | Demo shell: crowd page embeds the market monitor; visible rumor field; reconnecting banner; playback follows `/health`; skip runs to the crisis horizon; Chase and other real names join the M34 blocklist (short names match as whole words); chart axes anchored at $0. README two-minute demo script. ADR-046. |
| v1.23.1 | Shared rumor market classifier in `/agents/rumor_input.py` (UI rumor box + calibration labels). Non-market free text has no simulation effect; market fear sets `RumorPosted.credibility` from lexicon strength. `POST /api/rumor/interpret` for fake-mode crowd playback. Calibration `measure_outflow` unchanged. ADR-047. |
| v1.24.0 | Fake default run paces withdrawals, cash, and price on an ASSUMED logistic over the crisis horizon so a strong rumor shows a slow S-curve and a weak rumor stays smaller. Crisis-room ids map to short labels. Real engine params and FIT rows are unchanged. ADR-048. |
| v1.24.1 | M27 `calibration/run_fit.py` adds `--workers` spawn-pool candidate eval and `--gpu` (`STAMPEDE_BACKEND`); M05 caches `sample_population`, vectorizes balance solve on CPU, and adds CuPy balance solve when `STAMPEDE_BACKEND=gpu`; per-candidate progress logs; `calibration/README.md`. ADR-049. |
| v1.25.0 | M27 non-bank price-path harness in `/calibration/nonbank.py` (equity fit on E13, crypto and rates holdouts E15–E16, GameStop E14 documented). UNSET point targets are omitted from the loss. SVB scenarios and FITTED rows are unchanged. ADR-050. |
| v1.26.0 | M27 opt-in batched GPU candidate eval in `/engine/gpu_sim.py` (`--gpu-sim`, `STAMPEDE_CALIB_GPU_SIM=1`, requires `STAMPEDE_BACKEND=gpu`). Default engine path unchanged. NumPy parity test and `scripts/check_gpu_sim_agreement.py`, `scripts/benchmark_gpu_sim.py`. ASSUMED tolerance and array-budget rows. ADR-051. |
| v1.27.0 | M27 non-bank CLI accepts `--gpu`, `--gpu-sim`, and `--workers`, plus run id, checkpoints, patience, and extra seeds. `--gpu` and `--gpu-sim` batch the E13 equity search and the E15/E16 holdouts on `/engine/gpu_sim.py` within the E1 relative tolerance. The flag-off fake path is unchanged and writes no FITTED rows. ADR-052. |
| v1.28.0 | M27 diagnostics under `/calibration/diagnostics` (sweeps, withdrawn share, cap/closure binds, seed variance, bounds, CMA-ES sigma). Exact-zero invariance for unrelated text via the shared rumor classifier. ASSUMED 0.1%/day non-bank deposit-request bound with a penalty in `/calibration/nonbank.py`. ADR-053. |
| v1.29.0 | M27 E1 fit scores requested withdrawals with payment caps and closure off, anneals soft M06 gates, averages eight shared seeds, and searches four identifiable log-space parameters after a sensitivity sweep (`calibration/diagnostics`). Two stages: bracket sweep, then CMA-ES with restarts. Validation entry `validate_fitted_parameters` keeps caps on. Mean-of-vector amount scale removed. Default engine path unchanged. ADR-054. |
| v1.30.0 | Live rumor credibility is computed in code from the poster selector and specificity features. The model may shift it by at most ±0.1. The value is clamped to 0.05–0.9. Shared `classify_rumor_market` adds input classes; repeats merge; one new rumor per 30 seconds; 280-character cap. Calibration scenario hashes stay the same. ADR-055. |
| v1.31.0 | M27 `--gpu-full` runs attention, contagion, market, and payments as CuPy array kernels (NumPy fallback) for the SVB E1 fit and the non-bank price-path fit. Objective, parameters, and flag-off engine paths are unchanged. NumPy parity uses the E1 relative tolerance. ADR-056. |
| v1.32.0 | Local LLM settings and fake backend in `/agents/local_llm.py`. Configurable weights dir (`STAMPEDE_LOCAL_LLM_WEIGHTS`); GPU-only serve script; M24 vague filter on the shared local path; validation failures fall back to the rule path. ADR-057. |
| v1.33.0 | M27 `--gpu-full` batches every candidate×seed lane on one array axis, skips unused M02/M29 work, fuses depositor elementwise ops, and keeps channel totals on device until one host copy. Default non-calibration paths unchanged. NumPy lane-parity test and `scripts/benchmark_gpu_full.py`. ADR-058. |
| v1.33.1 | M27 `--gpu-full` CSR matvec stays on the CuPy device (cupyx sparse or ``xp.add.at``). Detects CuPy via the module name, not ``type(xp).__module__``. Fake-xp test rejects implicit ``np.asarray``. ADR-058. |
| v1.33.2 | M27 fused depositor helpers call ``xp.clip`` / ``xp.maximum`` via module ``_XP`` so ``cupy.fuse`` tracers never use ``.clip``. ADR-058. |
| v1.33.3 | M05 large-n refine stays on host NumPy; `--gpu-full` samples the population once per seed and prints setup vs step times. ADR-059. |
| v1.33.4 | M27 `write_fitted_registry` drops extra CSV cells (`DictReader` restkey) so FITTED writes round-trip; `run_fit` still writes `converged.json` after the registry. |
| v1.34.0 | M27 `fit_behavioral_parameters` uses `cma.CMAEvolutionStrategy`; the bracket supplies only `x0`/`sigma0`; `_project_theta` uses `_theta_bounds` only; `es.pkl` restores RNG and covariance; IPOP doubles popsize on stagnation (ADR-060). |
| v1.35.0 | M27 GPU sim caches packed static state per seed across generations and updates only per-lane parameter rows. `M27 GPU sim array budget bytes` is 3e9 so one lane group holds 256 × 100k in float64. `run_fit --extra-seeds` defaults empty; CMA-ES restarts stay on stagnation (ADR-061). |
| v1.36.0 | M27 stage 1 scores `simulate_requested_fractions` (attention, decisions, price feedback on device; no per-lane engines or events; no M07/M08/M09/M20). Validation and holdouts keep `simulate_logs`. NumPy parity within 1e-6. ADR-062. |
| v1.37.0 | M27 lean and `--gpu-full` paths keep per-depositor arrays in float32 and spread each layer with one batched SpMM (`scipy.sparse` / `cupyx.scipy.sparse`, `(A @ X.T).T`). Money accumulators stay float64. NumPy spread rel. err. < 1e-5; requests < 1e-4. ADR-063. |
| v1.37.1 | `params/loader.py` caches parsed registry rows per resolved path and invalidates on `st_mtime_ns` or `st_size`. `get_parameter` uses a name→row index. `load_registry` returns a fresh list. ADR-064. |
| v1.37.2 | M05 `_transfer_mass_array` stays on the active array module (NumPy/CuPy) with a deterministic sort/cumsum path and no host copy. ADR-065. |
| v1.38.0 | M27 stage 2 runs after stage 1: one batched `simulate_logs` grid of 16 log-spaced `M07 capacity multiplier (FIT)` values (default 1.0), monotone log interpolation to settled day-one 0.24, one refine if the bracket is wider than 1.2×. M07 multiplies app, wire, and ACH capacities by that row. Lean `simulate_requested_fractions` is unchanged. ADR-066. |
| v1.39.0 | M06 amount chain caps cumulative requests at `Depositor mobile share of uninsured deposits (FIT)` times initial uninsured (default 0.88, bounds 0.30–1.00). The cap is the last amount step in the fused elementwise kernel, the unfused fallback, the NumPy lean path, and `evaluate_decisions`. 1.0 matches the previous amounts. The 16-vector adds a `mobile_share` CMA-ES coordinate with those registry bounds (ADR-067). |
| v1.40.0 | Lean/gpu-full price impact moves toward `impact * cum_share * attention_factor` at the cap (recovery allowed). `M14 price impact (ASSUMED)` default 1.13 (bounds 0.3–4). ASSUMED `M14 attention reference level` = 1.0. SOURCED `M14 trading halt minutes` = 5; `ASSUMED_HALT_MINUTES` is a registry alias. ADR-068. |
| v1.41.0 | M27 stage-1 identifiable set is price impact and mobile share only (`14,15`). Signature holdout runs the crypto-heavy preset with the SVB-fitted capacity held fixed. Next-open board band is ±10% relative. `best_vector.json` is always the global best. ADR-069. |
| v1.42.0 | M27 registry writes patch only the changed rows (`calibration/registry_patch.py`); line endings, quoting, and other rows are kept byte for byte. ADR-070. |
| v1.43.0 | M27 `run_fit` records `capacity_multiplier` and the rows it wrote. The non-bank fit writes its vector to its own row, `Non-bank E13 fitted behavioral vector (normalized)`, never the shared M06/M14 rows. `calibration/persist_fit.py` persists finished runs. ADR-071. |
| v1.44.0 | M27 crypto (E15) holdout has a set peak-to-trough target (0.9999996, SOURCED). A non-bank holdout with no set targets reports a null loss and status `no targets`, never 0.0. ADR-072. |
| v1.45.0 | M27 Signature (E5) holdout compares *requested* day one (caps on) against SOURCED `Validation target value: E5 Signature first-day requested fraction` = 0.249. Settled stays in the JSON for reference. ADR-073. |
| v1.46.0 | M14 real book reads SOURCED `M14 trading halt minutes` = 5; `M14 trading halt duration` (15) is DEPRECATED. One halt length everywhere. ADR-074. |
| v1.47.0 | `training/qlora_train.py`: offline 4-bit QLoRA for Llama 3.1 8B Instruct on the 3060, under `$STAMPEDE_TRAIN_ROOT` (default `./training_runs`); refuses to start while the GPU is busy. §20.4. ADR-075. |
| v1.48.0 | M06 depositors decide once: hard gates request the segment fraction once; soft gates ratchet a decided probability and request only its increase; contagion reads it. All amount paths (fused CUDA kernel, NumPy fallback, CPU `_evaluate`, `DecisionModule`) follow `M06 withdrawal decision mode (ASSUMED)` = `once`. The per-tick `mobile_share.get()` host sync is gone. `tools/svb28_probe.py`. ADR-076. |
| v1.49.0 | Non-bank days-to-bottom use trading days: `Non-bank sim minutes per day` = 390 and E16 reads SOURCED `Non-bank E16 SPX trading days to bottom` = 23. A days target the run can't reach is excluded from the loss and listed (`crypto_excluded` / `rates_excluded`). `--rates-holdout-minutes 8970` evaluates E16's days term. ADR-077. |
| v1.50.0 | QLoRA finds the base model with no flags (`hf/` then `models/`, or `$STAMPEDE_QLORA_MODEL`). `training/export_jsonl.py` exports local SQLite prompt/response text to JSONL. §20.4 has the full-run command. ADR-078. |
| v1.51.0 | `persist_fit` prints exactly the cells it writes (`fitted_registry_updates`, computed before the write) and refuses to persist a run whose rows already carry that run's Source (`--force` to override). ADR-079. |
| v1.52.0 | `test_gpu_sim_cli_batches_equity_and_holdouts` accepts `array module=cpu` or `gpu`, so it passes on CPU CI and on the 3060. ADR-080. |
| v1.53.0 | M06 per-depositor awareness: feature 0 is `attention · aware`; the hazard uses media, WOM scaled by `w1/w1_prior`, and feedback news. Soft once-mode sizes requests as `mobile_share · initial_uninsured · gap_scale · amount_scale`. Fit set is `7,14,15,16` on a 17-vector. `tools/svb28_probe.py` checks day-1 deltas. ADR-081. |
| v1.53.1 | M06 `aware0` (default 0) is the start value on all four decision paths; a reused kernel scratch is filled from it when `aware is None`. gpu-sim integer-minute coarse ticks expand to one-minute clock times (ADR-082). Awareness rate defaults come from the registry via `awareness_rate_defaults()`. |
| v1.54.0 | Stage-1 E1 objective includes requested-by-day-one-close (0.249, same relative error and weight as next-open). Board row `E1-requested-day1`; summary.json and the board markdown print the E1 stock ratio. Signature (E5) stays a holdout. ADR-083. |
| v1.55.0 | Soft once-mode ratchets at a finite decision-to-request rate `h` (`d_new = decided + (1-exp(-dt·h))·max(0, p_act-decided)`). `h` is FIT index 16 on the 17-vector (prior 1/600 per minute); media awareness rate stays at its registry prior. Stage 2 and the Signature holdout are unchanged. ADR-084. |
| v1.55.1 | Finite-h ratchet paths agree: `DecisionModule.step` threads `dt`; the closed form `1-exp(-dt·h)` is used in `_evaluate`, the fused kernel, and the NumPy fallback (schedule-invariant when `p_act` is constant); lean applies the same-tick M41 VIP hub shock; composed attention `tanh(rumor+news)` is the M06 level on every lane so event / lean / float32 match at the original tolerances. `tests/engine/test_svb35b_paths.py`. |
| v1.56.0 | E1 stock ratio is reported, not fitted: `Calibration E1 day-1 stock objective weight` = 0 and `_objective_from_fractions` skips the term at weight 0. Fit set is `7,15,16` (price impact left). `M14 price impact (ASSUMED)` keeps 1.18464 and is relabeled ASSUMED (carried from the pre-SVB-28 svb26 fit). Board and summary.json still print the stock ratio. ADR-085. |
| v1.56.1 | `simulate_logs` market quotes honor `M27 calibration uses scripted drawdown` via `calibration_uses_scripted_drawdown()` (same helper as `simulate_requested_fractions`). `price_quote` / `price_quote_batch` still default `scripted=True` for the demo. ADR-086. |
| v1.56.2 | `MarketFake` honors the M27 scripted-drawdown row on the CPU real-engine path, not only when `gpu_sim_enabled()`. Demo ticks stay scripted. ADR-087. |
| v1.56.3 | M06 `load_decision_params` accepts FITTED as well as ASSUMED for `M06 worry bias (ASSUMED)` so `persist_fit --stage2` can rebuild the engine after stage 1 writes FITTED; other M06 rows stay ASSUMED-only. ADR-088. |
| v1.57.0 | Non-bank summary splits crypto and rates holdout horizons (`eval_minutes_holdout_crypto` / `eval_minutes_holdout_rates`). `--crypto-holdout-minutes` lengthens E15 so multi-day targets stay reachable. Behavioral runs record per-statistic `shape_loss_terms` (UNSET is null). ADR-089. |
| v1.57.1 | `persist_fit --stage2` calls `apply_run_fit_engine` (the `run_fit._run` helper) with `--gpu-full` so the capacity solve uses the real engine, gpu-sim, and `calibration_uses_scripted_drawdown()`. CUDA missing falls back as `run_fit` does; the path is printed. ADR-090. |
| v1.58.0 | Rumor poster roles: the poster row grows from 5 to 11 roles (regular user, anonymous group chat, VIP, treasury desk, short-seller, journalist, founder circle, bank CEO/CFO, top shareholder, VC partner, regulator), all ASSUMED; the old five keep their values, so offline results and calibration hashes are unchanged. `classify_rumor_market` carries a `CredibilityBreakdown`; interpret, submit, and `GET /api/rumor/roles` expose it; the UI shows it under the rumor. Fake-mode playback scales effect size by credibility over the cap (1 at the cap, so the default VC role plays as before). ADR-091. |
| v1.59.0 | M34 rumor judge (`/agents/rumor_judge.py`): one bounded JSON call per new rumor text (registry model, 4 s timeout, per-text cache) returns specificity, plausibility, claim type, and call to action. It replaces the separate ±0.1 shift call, which passed `rumor-shift` as the model name. Code combines the scores with the role base; strength rises for judged calls to act and run claims. Injection-looking text skips the model; invalid replies, errors (including 429), and timeouts use the lexicon. The server judges in a worker thread before taking the session lock. Offline results and calibration hashes are unchanged. ADR-092. |
| v1.60.0 | M03 role reach (flags off): `RumorAmplifierParams` / `load_rumor_amplifiers` read four new ASSUMED M03 rows. With `M03 role exposure multiplier enabled` the post impulse is credibility × `M03 rumor exposure` × the role multiplier; with `M03 role source node enabled` a role id posts from its mapped fallback node (public hub, professional hub `node-2`, or `small-account`). Off, logs equal b674fa7 (96-run fingerprint test). §6 payloads unchanged. CPU only; a flag on plus GPU calibration env raises. ADR-093. |
| v1.61.0 | M03 claim strength impulse (flag off): with `M03 rumor strength impulse enabled` the post impulse also scales by the claim's offline lexicon strength (low 0.8, medium 1, high 1.25). The tone/strength rule moved into `agents/rumor_lexicon.lexicon_tone_and_strength`, shared by M34 classification and M03; offline classification is unchanged (196-case golden table). ADR-094. |
| v1.62.0 | M03 exaggeration feedback (flag off): `LiveRumor.exaggeration` keeps the highest accepted mutation score; with the flag on, reposts add `min(1 + gain·score, cap)` × the usual exposure (gain 1, cap 1.5). `RumorMutated` payload unchanged. ADR-095. |
| v1.63.0 | M28 `compare_branches` finds the first divergence by canonical event JSON (type and payload, the same rows `event_log_hash` hashes), not by event type alone, so `first_divergence_index is None` exactly when the two log hashes are equal. Seeded logs are unchanged; only the comparison report changes. ADR-096. |
| v1.64.0 | `calibration.run_fit` no longer writes FITTED rows by default: `--write-registry` opts in; otherwise `summary.json` carries `persist_command` for `persist_fit`, as the non-bank harness already did. `fitted_rows_written` counts `updated`/`appended` only (it counted `unchanged`). Engine, registry, and seeded logs unchanged. ADR-097. |
| v1.65.0 | Live crowd and price from the real engine (SVB-50): real mode builds `live_engine_modes()` with `default_mode=FAKE` (`build_engine(default_mode=REAL)` fails in the stream loader on M09, so real mode never started), `CrowdFake.follow_engine` skips the scripted arc, and `LiveSession.step_once` copies M06/M08 state onto the crowd; `ui.server` steps through `step_once`; `/health` reports `crowd_source`. `run_live.sh` defaults to real. Fake mode and seeded logs unchanged. ADR-098. |
| v1.66.0 | Non-bank depositor rows (SVB-52, D3): `engine/depositor_rows.py` holds a `ContextVar` row map; inside `nonbank_depositor_rows()` the M06 bias, mobile-share, request-rate and M07 capacity loaders (and the gpu_sim capacity read) use four ASSUMED non-bank rows at the pre-SVB-fit values. `calibration/nonbank.py` activates it around every engine run and gpu-sim batch. Outside it nothing changes, so the SVB fit and seeded hashes are untouched. ADR-099. |
| v1.67.0 | M07 stop-after-closure flag (SVB-53, D4), off: with `M07 stop settlement after closure` on, M07 publishes no `PaymentSettled` after this bank's `BankClosed` and freezes its queues. It reads the closure from the bus log, so `consumes` and the §6 consumer table are unchanged. CPU only; refused under GPU calibration. Off, logs are byte-identical. ADR-100. |
| v1.68.0 | `build_engine(default_mode=REAL)` builds for every caller (SVB-54, E1): it means REAL for each module in `engine.registry.REAL_MODULE_IDS` and fake for the rest. Before, the stream loader got REAL for M09 and the UI modules and raised (P0-3). Explicitly requesting a module without a real implementation still raises. Fake mode and seeded logs unchanged. ADR-101. |
| v1.69.0 | `calibration.run_fit` writes `validation-board-<run-id>.md` into the run directory, not `docs/` (SVB-55, E2); `make ui-build` and the README give the frontend build order (`ui/market` before `ui/crowd`, E4). No engine change. |
| v1.70.0 | SVB-like M08 opening book (SVB-56): SOURCED rows from SVB's FY2022 10-K (cash 7.97%, AFS fair value + HTM amortized cost 67.81% of deposits). Used when `M08 SVB-like opening book enabled` is on (off: calibration and seeded logs unchanged) or inside `svb_like_opening_book()`, which the live server's real mode opens when `M29 live SVB-like opening book` is true. ADR-102. |
| v1.71.0 | D2 wiring (SVB-57), off: with `M03 uses M02 network` M03 runs on M02's depositor network (installed on its first step via `NetworkModule.ensure_network`), with M02 node credibility and segment-based role nodes. Off: unchanged. On: every real-mode log and real-M03 hash changes; seeded default-mode hashes do not (M03 is fake there). CPU only. ADR-103. |
| v1.72.0 | Live pacing, option B (SVB-58, item 6): `engine/live_overrides.py` lets the live server switch engine flags on for its own engine only. `M29 live depositors ignore stock price` turns on `M06 depositors ignore stock price` (off in the registry), so the real M14 order book shows the price but no longer feeds depositor worry. Calibration and seeded runs unchanged. ADR-104. |
| v1.73.0 | Cure rumor (SVB-59, item 9), flags off; the live server turns both on (`M29 live cure rumor`): M03 cuts the bank's attention by `cut x credibility` instead of spreading a positive-tone post, and M06 reads positive-tone `RumorPosted` from the bus log (consumes unchanged) and applies the statement worry shifts scaled by `factor x credibility`. No redeposit and no direct price lift. ADR-105. |
| v1.74.0 | Soft decide-once on the CPU engine and the live run on the calibrated model (SVB-61). M06 in soft mode (softness > 0) now publishes each tick's request increments, aggregated per (segment, channel) (`agg:<segment>` ids), as the gpu-sim kernel counts them; before, only depositors with decided >= 0.5 published and the CPU engine requested ~0 at the fit's softness (diag-08). Hard mode (softness 0, seeded default) unchanged. The live server's real mode sets M06 softness to `Calibration behavioral noise temperature` (the fit's) and spans `M29 live real crisis sim minutes` (1440: SVB's day one plus the night). ADR-106. |
| v1.75.0 | Attention ceiling by rumor credibility (SVB-62), flag off; the live server turns it on (`M29 live attention ceiling by credibility`). M06 clips the bank attention level at `min(1, c / 0.8)` for the strongest live panic rumor (c = its credibility, 0.8 = the SVB fit rumor's) instead of at 1. Off: unchanged. ADR-107. |
| v1.76.0 | Live price paced to SVB's day-one drop (SVB-63): in real mode with `M29 live price paced to SVB day-one drop`, the market frame's price series is `opening x (1 - (1 - E1 stock target) x min(1, requested / E1 requested target))`; `/health` reports `price_source`. The M14 order book still runs underneath. ADR-108. |
| v1.77.0 | Live closure hold (SVB-64): `RegulatorFake.earliest_closure_minute` (default None: unchanged). The live server sets it from `M29 live earliest closure minute` (1440, the next open); a closure decision before then is held (`held_reason`, one `watch` action) and re-checked each tick. Calibration scenarios never set it. |
| v1.78.0 | Per-lane rumor spread (SVB-66). M03 reposts read the lane's own attention, so `step_attention_group` and the stage-1 lean path keep one rumor state per lane (one `AttentionEngine` per lane, identical at t=0) instead of copying the seed's first lane. A lane's results no longer depend on the other candidates in its batch. `STAMPEDE_CALIB_SPREAD_PER_LANE` and `STAMPEDE_CALIB_PER_SEED_SYNC` are gone. ADR-109. |
| v1.79.0 | The CPU engine runs the fitted feedback (SVB-65). `engine/fitted_feedback.py::step_module` advances one module of one engine through the functions the GPU kernels call: after M03, `apply_composed_attention` (depositors and the fake M14 read `tanh(mean attention + news)`); at M14, `step_market_group` (news from the tick's requested share x `M03 feedback gain`; the fake quote moves toward `price impact x cumulative requested share`; a real book steps as before and only updates news). `Engine.run`, the live `step_engine` and the per-lane gpu-sim path use it. On for `STAMPEDE_CALIBRATION_ENGINE=real` and for the live real-mode engine (`M29 live fitted feedback`); off elsewhere. CPU calibration call sites install the lane's gain, dispersion and impact (`apply_lane_feedback`). ADR-110. |
| v1.80.0 | M02's advisor sample is seeded on CuPy too (SVB-68). `_permutation` ranks one uniform draw per index from the seeded Generator when the Generator has no `permutation` (CuPy's), instead of calling the backend's global RNG (`cupy.random.permutation`), which no seed reached: on a CUDA machine every build sampled different advisors, so the professional layer and SVB-57's role nodes changed between two runs of one seed. The CPU stream keeps NumPy's `permutation` and is byte-identical. Nothing outside the flag-gated M03 path reads the professional layer. |
| v1.81.0 | Live preset table, deposit guarantee chip, halt at the price floor, 1441-minute window (P-A part 1). `ui/live/presets.py` holds the four posts diag-09/11 validated (lost-bonds vc-partner 0.90, frozen-branches ui-user 0.65, lost-bonds+insured with the regulator's chip at minute 240, stadium posting nothing); `LiveSession.load_preset` schedules them by minute and `enact_guarantee` adds the chip mid-run at the engine's minute; `/api/presets`, `/api/preset`, `/api/guarantee` and socket commands `preset`/`guarantee`. With `M29 live insured chip guarantee` the chip publishes `PolicyEnacted(guarantee_deposits)` (M06 scales the uninsured gap by 0) instead of a rumor. With `M29 live trading halt at price floor` the paced price stops printing at the floor and the market frame carries the halt and a halt marker. `M29 live real crisis sim minutes` 1441. Live only. ADR-111. |
| v1.82.0 | Buffered live playback (P-A2). `ui/live/playback.py::PlaybackBuffer`; with `M29 live buffered playback` the server's lifespan starts `LiveSession.start_producer`, a thread that steps the engine into the buffer (only while the run plays, up to the live window, at most `M29 live playback head start seconds` of 1x playback plus 2 minutes ahead); `advance_wall` then moves playback through computed frames, starting once the head start is buffered and never passing the newest one, and `crowd_frame`/`market_frame` return the frame at the playback minute. Engine entry points share one re-entrant lock. Rumor and guarantee replies carry `takes_effect_at_minute` and `screen_minute`; `/health` reports `playback_tick` and `buffered`. Without the producer (tests, diagnostics, fake mode) nothing changes. ADR-112. |
| v1.83.0 | The fakes read only new events (SVB-69): `mocks.support.resume_fold` resumes folds on the same list, M33's replay hash feeds a running sha256 with the same digest, and M30 extends its series, so a live step stays near 30 ms instead of growing to 108 ms over a run. Output unchanged. ADR-110 scope note: with no rumor the CPU calibration engine runs scripted agent fakes the GPU kernels lack. |
| v1.84.0 | Honesty labels and the limits screen (PA-C): `ui/live/labels.py` (badges, the 2023 replay rule, the approved copy); `LiveSession.run_label`; `/api/labels`, `/api/limits`, `run` in `/health`; `M29 live free-text rumors` (off) refuses typed rumors unless an operator allows them. ADR-110's scope note corrected. ADR-113. |
| v1.85.0 | Countermeasures (PA-A): `ui/live/presets.py::RESPONSES` (guarantee, credit-line, withdrawal-limit) and `response_post`; `LiveSession.load_preset(name, response)` adds the response at minute 240, `enact_response(name)` mid-run; `response_minutes`; market statements `credit-line` / `withdrawal-limit`; `POST /api/response`, `response` on `POST /api/preset`, responses in `/api/presets`, socket command `response`. ADR-114. |
| v1.86.0 | Multi-seed batch table (PA-B): `ui/live/batch.py` (grid, `run_one`, `summarize`, `run_grid` with spawn workers, `write_batch`/`load_batch`; default file `data/live_batch.json` or `STAMPEDE_BATCH_FILE`, gitignored), `scripts/batch_table.py`, `GET /api/batch` (summary) and `GET /api/batch/run?preset=&seed=&response=` (one run's series). ADR-115. |
| v1.87.0 | UI wired to the engine (UI-1): `ui/live/stampede_ui.html` (the mockup, with an engine adapter: polling, shares on dots, per-minute price bars, timeline events, outcomes, preset and response chips, the fitted / scenario badge, the limits panel, the ten-seed table and the without-a-response comparison on the recap); `LiveSession.ui_state` and `_ui_scan` (incremental totals and timeline; `BufferedFrame.info`); `GET /api/ui/state`, `GET /ui`. ADR-116. |
| v1.88.0 | The vault, response cards, whole-day stock chart (UI-2): `LiveSession._vault_sample` / `vault_layout` / `_vault_fracs` (each frame's shares in `BufferedFrame.info`), `money.borrowed_usd` / `emergency_usd`, `GET /api/ui/depositors`; `ui/live/cards.py` (cards in `/api/presets`); preset `expect` lines; batch rows record `borrowed_usd` / `emergency_usd`; the page's vault scene, orbit, cards, counter (share of deposits requested) and day chart; the mock's window sync is off in engine mode. ADR-117. |
| v1.89.0 | Response search and timing (UI-3a): `ui/live/search.py` (plans, scoring, `tournament`, `deadline`, `run_search`, `write_search` / `load_search`; default `data/live_search.json` or `STAMPEDE_SEARCH_FILE`), `scripts/search_responses.py`, `GET /api/search`, `GET /api/search/run`; `LiveSession.load_preset(..., response_minute, plan)`; the batch runs each response at 11:00 AM, 1:30 PM, 4:00 PM and 1:00 AM for the bond rumor (180 runs; rows record `response_minute` and `response_enacted_minute`); cards read the 1:30 PM entries. Minute 240 is 1:30 PM: served copy that said 1:00 PM is corrected, and the guarantee preset's note no longer claims 2023's timing. ADR-118. |
| v1.90.0 | Search screens (UI-3b): `ui/live/search.py::run_full_search` and `elimination` (every plan on 10 seeds, elimination minute and reason per plan, pick, held-out check, deadlines; runs record crowd shares every 10 minutes); `all_plans(earliest)`; `scripts/search_responses.py` (full mode by default, `--earliest`, `--rounds`); `POST /api/preset` takes a `plan`; the page's Search tab (the wall, finale) and Why tab (the table), `?screen=search` and `?screen=why`; clicking a plan runs it live. ADR-119. |
| v1.91.0 | The 2023 replay is a reference on the search screen, not a candidate (UI-3c): it no longer competes on the wall (no duplicate tile), it is pinned with a dashed red border as "THE 2023 REPLAY · no response", with "For comparison: how the run went" and the real 2023 marks (25% by 4 PM, 81% by the next open); the finale and the Why page name it the same way ("The 2023 replay (no response)"), the Why page tags it as the baseline, and the header counts only the plans the engine chose between. |
| v1.92.0 | Trigger rules and the action window (UI-3d): `LiveSession` takes plan steps `(response, ("requested" | "cash", level))` and fires them the minute after the level is crossed (`trigger_minutes`); `ui/live/search.py::trigger_plans` (32), `false_alarms`, `action_window`, `run_trigger_search` (the default for `scripts/search_responses.py`; `--clock` for clock times); runs record what fired when; reasons use one decimal under 10%. Screens: tiles label their number "asked" (deposits asked for; gold is cash), the finale draws the action window on the day's axis, the Why page shows the window and when each rule fired. ADR-120. |
| v1.93.0 | Failure-likely test (UI-3e): `LiveSession.load_preset(..., credibility)` and `trigger_requested`; `ui/live/search.py::STRENGTHS`, `likely_failure`, `too_early`, `first_cross` (crossings interpolated between 10-minute samples); `run_trigger_search` runs the strengths and records `likely_failure`; the action window opens at failure likely. Screens: the finale's band opens at "failure likely" (a tick marks when the run stood out); the Why page states how it is measured. ADR-121. |
| v1.94.0 | Credit line on BTFP terms (CM-2): `engine/liquidity/terms.py` (`EmergencyTerms`, `ASSUMED_TERMS`, `emergency_terms`); `mocks/l2_banks.py::LiquidityFake` reads the terms at attach and, with the line on, measures the shortfall from M08's opening cash; four registry rows (`calibration/registry_updates/cm2_btfp_credit_line.json`); `ui/live/session.py` live opt-in; the credit-line note and card text (measured Fed lending, closure line only when runs close); `scripts/diag19_credit_line.py`. ADR-122. |
| v1.95.0 | Sourced countermeasure values (CM-1): `calibration/registry_updates/cm1_sourced_values.json` (rows 261 and 303 Source only, row 306 = 0.104 SOURCED); `ui/live/cards.py` public stat "Public money at risk" with the expected loss; the withdrawal limit's note and cons (until the next open; expecting a limit makes people rush first, Li et al. 2021); `ui/live/labels.py` limits text (the engine tries the responses; insured depositors never run; 2023 terms). ADR-123. |
| v1.96.0 | Runs-per-plan selector (UI-4): `ui/live/search.py::run_trigger_options`, `_trigger_result`, `RUN_OPTIONS`, `POOL_HOLDOUT_SEEDS`, `option_file`, `write_search_options`, `available_run_options`, `load_search(runs=)`; `scripts/search_responses.py --runs`; `GET /api/search?runs=` with `run_options`; `PlanStep.minute` takes a trigger (a two-item list) so the wall's click and the pick post trigger plans; page: the selector, the days strip, short tile labels, the engine's-pick chip, read-only response cards; `tools/ui4_preview.py` (placeholder preview). ADR-124. |
| v1.97.0 | GPU-1 batched runs (sandbox test): `ui/live/gpu_batch.py` (`step_sessions_batched`, `run_plans_batched`, `run_batch_cells_batched`, `default_group_size`, `HOST_LANE_CAP`); `search.Runner(batched=)`, `gpu_search_enabled`, `_run_batch`; `batch.run_grid(gpu=)`; `--gpu` on `search_responses.py` and `batch_table.py`; `scripts/gpu_parity.py` (CPU against batched, run by run, with timing). ADR-125. |
| v1.98.0 | CuPy by default (CPY-1): `engine/xp.py` (`backend_preference`, `gpu_active`, `array_module`, `require_gpu`, `backend_report`, `format_report`); M06 `active_backend_name`, M03 `resolve_backend` (cached per preference), M02 `select_backend`, M05 `prefer_population_gpu`, `gpu_sim_enabled`, `gpu_full_enabled` and `ui/live/config.prefer_gpu` read it; `search.gpu_search_enabled` and `batch.run_grid(gpu=None)` batch by default on a GPU; `--cpu` / `--require-gpu` on `search_responses.py` and `batch_table.py`; `gpu_parity.py` compares the NumPy reference with CuPy end to end; `/health` reports each path's backend; `run_live.sh` defaults to `gpu`; `default_workers` (6) and `vram_mb` (peak use per batch, printed by `gpu_parity.py`). ADR-126. |
| v1.100.0 | Useful plans (UF-1): `ui/live/useful.py` (`predict`, `rescore`), `engine/system/can_pay.py` (`can_pay_fractions`, `capacity`), `scripts/rescore_useful.py`; `RegulatorFake._can_pay_reason` behind `M20 closes on what the bank can pay` (live opt-in `M29 live regulator closes on what the bank can pay`); registry rows in `calibration/registry_updates/uf1_can_pay.json`; the page: knocked-out tiles say why, useful plans carry rank badges, the finale shows the top three and the window from threat to point of no return, the Why page explains the rules. ADR-128. |
| v1.101.0 | Bank tools and groups (UF-2): `can_pay_fractions` gains `prepared` and `big_banks` (rows in `calibration/registry_updates/uf2_bank_tools.json`); `ui/live/useful.py` `computed_plans`, `_synth_series`, `FAMRANK`, `_groups`, close calls in the reasons; the page: `drawGroups`, `groupCells`, `groupsTally`, `applyGroupRanking`, the Groups selector, ≈ on computed vaults (not runnable live), the finale's consistency counts and close-call note. ADR-129. |
| v1.102.0 | The automatic loop: `scripts/pipeline.py` (`--force`, and `--reuse` to rank runs made before the input key), `/api/pipeline/run` and `/api/pipeline/status`, the Run the search button; `ui/live/plangen.py`; `useful.rescore(generated=, wall_size=, wall_useful=, checks=)`, `day_path`, `_checks`, `_sensitivity`, `_tipping`, `_warning`, `_summary`, strengths read from the data; `presets.TOOLS` and `STEPS`, the session publishing statements and rate rises; `LiquidityFake` lends against pledged bonds and takes big-bank deposits; M08 doesn't sell pledged bonds; `RegulatorFake` counts the bank's tools; the early-warning line on the main vault; the Why page's summary of the search and plain-sentence reasons. ADR-130. |
| v1.103.0 | Three beats and one headline (UI-5): `ui/live/stampede_ui.html` nav (crowd, search, why; recap and banks operator-only), `#headline` with `headlineText` / `renderHeadline` / `planWords` and `REVEAL_TABS` (the answer on search and why; `selectTab` rerenders it), `renderWhy` rewritten with `whyDetails`, `alsoUseful`, `planKind`, `knockCounts`, `bindWhy`, `wrapFill` on the wall's finale, `TABKEYS` m/p/w/c, short labels say Fed lending; removed the Bank and Stocks sections and `drawChart`, `chartBars`, `legend`, `updateMarketDom`, `drawWatch`, `updateBankScreen`, `drawFlow`, `tapeTick`, `drawDayChart`, the headlines list and the Open screens menu. `ui/live/useful.py`: `considered.knocked` (early, not-us, fails) and the tipping sentence in plain words. `tests/test_ui5_three_beats.py`; page needles in UI-2, UI-4 and UF-2 tests follow the new words. ADR-131. |
| v1.104.0 | Validation fixes (VF-1): `scripts/gpu_parity.py` (`TOL` = 1e-4 of deposits, `compare` on outcome, triggers, the end and every sample; `--reuse-cpu` reads the NumPy reference from an earlier output and its worker count; `--unfused` sets `STAMPEDE_CALIB_UNFUSED_LOGITS` and `_PART2` on the GPU side; the output adds `gaps`, `tolerance_kind`, `unfused`, `workers`); `scripts/rescore_useful.py::summary_line` counts every useful plan (`considered.useful`) and says how many are on the wall, and `scripts/pipeline.py` prints it for each file; `ui/live/useful.py` documents that `useful` is the wall's useful plans (at most `wall_useful`, 16) and `useful_all` all of them; `ui/live/stampede_ui.html::usefulTotal` for the wall's title, the finale's "and N more" and What helps. `tests/test_vf1_parity_counts.py`. ADR-132. |
| v1.104.1 | VF-2: `scripts/gpu_parity.py` `--kernel-only` and `batched_backend` (the batched side's `STAMPEDE_BACKEND`), the output's `kernel_only`, and the docstring records the `--unfused` result; the docstring also records that the fused kernels are ruled out, with the 3060 validation. `tests/test_vf2_kernel_only.py`. ADR-133. |
| v1.104.2 | VF-3: `ui/live/session.py::LiveSession._lever_words` (label, `bank` or `regulator`, note) and the timeline's response events carry `by` and `note`; `ui/live/stampede_ui.html::engineEvent` words the bank's own tools as Copperleaf acting and the early warning as compared with no response. `tests/test_vf3_live_words.py`. ADR-134. |
| v1.104.3 | UI alignment: `renderWallChrome`, `fitCanvasText`, width-aware group headers and sidebar labels, computed-marker space in `drawTile`, height-aware sidebar rows, wrapping DOM header/legend and separate footer controls; grouped trigger markers and separate time-label rows in `drawFinale`; `selectTab` resets scroll. `tools/check_wall_alignment.py` measures the 10/25/50 replay at 1600 and 1280 pixels wide without starting runs. ADR-135. |
| v1.104.4 | VF-4: `CashPathRequired` prevents post-action cash predictions from reusing the no-response curve; rescoring retains unverified policies separately, limits ranking to eligible plans, counts independent evidence correctly, and preserves old holdouts without relabeling them. What helps discloses the withheld policies; the vault and end cards label gross book liquid assets explicitly. `tests/test_cash_path_validation.py`. ADR-136. |
| v1.104.6 | Remove the first-page vault border and rank badge, including during a selected-plan run, so the live replay cannot be mistaken for a ranked recommendation. ADR-137 clarification. |
| v1.104.5 | Refresh the first-page `vaultGeometry`, `dotColor`, `drawVault` and `engineDom` to match the supplied plan-tile reference: spaced sampled depositor rings, flat gold centre with requested percentage, selected-plan caption and badge, external warning/legend and accessible percentage. ADR-137. |
| v1.104.7 | VF-5: `scripts/pipeline.py` `--tip-strengths` and `_tipping_runs` skipping runs already on disk; `ui/live/stampede_ui.html` `#wallRun` operator-only (`opx`) with a confirmation, `warningLine` empty once `sim.over`. `tests/test_vf5_tipping_fill.py`. ADR-138. |
| v1.104.8 | After the VF-5 refill, group finale trigger captions by their measured text bounds rather than an identical rounded pixel. Keeps the near-coincident 20% response labels readable at 1280 pixels / 25 versions. Simulation values and marker times are unchanged. ADR-135 clarification. |
| v1.104.9 | VF-6: `ui/live/stampede_ui.html` `groupsTally` counts each plan's early days, `applyGroupRanking` stores `groups_early`, `tileDays` words the tile. `tests/test_vf6_tile_days.py`. ADR-139. |

| v1.104.10 | UI-6: simplify viewer-facing copy across the three beats, define pledging and outcome units, retain visible uncertainty and expandable evidence. Calculated plans can use the existing live reset/preset path from results. `tools/check_ui_copy.py`; updated UI wording checks. ADR-140. |

| v1.105.0 | UI-7: manual saved-replay controls for 10/25/50 days, cached original vault cards and shared orbit geometry, disjoint elimination transitions, wrapped captions and aligned plots. Direct sample interpolation and cached standings; late-response protection. Browser control/data/layout/motion checks. ADR-141. |

| v1.105.1 | UI-7: preserve the original elimination-reason colours in cached departure cards, including early/unavailable cases. ADR-141. |
| v1.105.2 | UI-8: `ui/live/stampede_ui.html` (`shortLabel` sentences, `renderHeadline` reveal on `wall.phase === 'final'`, `howItWorks`, `spare`, the panel's `keep`/`openOf`/`moreIn`, `#wallDescription`, the legend, the cash card, beat-specific `.clockbox` CSS); `ui/server.py` `/api/search` adds `deposits_usd`; `ui/live/labels.py` badges. Tests: `tests/test_ui8_viewer_words.py`; `test_ui5_three_beats`, `test_vf6_tile_days` follow UI-8, and `test_vf1_parity_counts`, `test_au_automatic`, `test_uf1_useful`, `test_uf2_bank_tools`, `test_ui4_runs_selector`, `test_ui2_vault`, `test_ui1_state` follow UI-7's words. ADR-142. |
| v1.105.3 | Public-repo cleanup: §10 rewritten as the development workflow (code areas, build order, gates, repository rules); process notes removed; QLoRA scripts read `$STAMPEDE_TRAIN_ROOT` (default `./training_runs`); local LLM weights default to `models/gguf`; rumor-role research notes moved to `docs/research/`; `tools/check_ui_copy.py` replaces the old copy checker name; presenter view replaces the old view name. No simulation, contract or parameter-value changes beyond the two path defaults. |

## 16.2 Decision log (ADRs)

| ADR | Decision | Reason |
| :---- | :---- | :---- |
| 001 | Panic is driven by attention, not sentiment | Cookson et al. (E10) |
| 002 | The truth filter is a hard rule, not one input to a score | Ethics; an optimizer would otherwise learn to lie |
| 003 | Fictional banks only | Defamation and market-manipulation risk |
| 004 | Futures run at 250k depositors and 5-minute ticks; only the winner replays at full detail | GPU budget (§9) |
| 005 | Fit on SVB only; test on Signature | Prevents overfitting and gives an honest validation story |
| 006 | Results are always shown as ranges | Social-media signals alone produce false positives (E12) |
| 007 | Colorblind-safe red and green, with redundant cues | About 1 in 12 men has red-green color blindness |
| 008 | Stack: Python, CuPy, Numba, FastAPI, React, deck.gl, uPlot, xai-sdk | Covers GPU kernels and speed |
| 009 | LLM agents act only in the main timeline; their decisions are reduced to simple rules for the futures | 16 futures would otherwise multiply cost and latency 16× |
| 010 | CUT: superseded by the §10 development workflow. | Not applicable to the public repository. |
| 011 | Each module has one build spec in §18 | Clear scope means fewer merge conflicts |
| 012 | M06's code path is /engine/agents/decide.py (CuPy RawKernel). M34's code path is /agents/guardrails.py. | The engine module is Python, like every other engine file and like the scaffold; §18.0 runs CUDA through RawKernel. M34 is stream L5, whose folder is /agents, and it had no path. |
| 013 | Gate 9 requires an edit to /docs/ARCHITECTURE.md. The root ARCHITECTURE.md pointer does not count. | That is the path CI checks. The root file only points at the real document. |
| 014 | §6.1 rows: RumorMutated (rumor\_id, claim, exaggeration\_score; M03; no further consumer), AnalystNote (text, cited\_event\_ids, bank\_id; M04; M03, M14), SupervisoryAction (bank\_id, action; M20; no further consumer). bank\_id stays optional on every event. StatementIssued.segments is a list. | Payloads follow §5 and §7. §5 already sends analyst notes to M03 and M14. No module spec names a consumer for the other two. The header marks bank\_id as optional. |
| 015 | The build order (§10.2) covers 27 retained modules, not 59. | §4, §10.2, §11, and §15 all describe the same 27 modules. |
| 016 | M23 stays on stream L5 and stays at /engine/bank/management.py. | §4 and §10.1 assign L5. The rule core sits beside the ledger it posts to, so the file does not move under /agents. |
| 017 | §19 matches §5 and §6.1 on these edges: M04 consumes PriceTick; M03 and M14 consume AnalystNote; M29 publishes RumorPosted and consumes PaymentSettled; M30 consumes PriceTick, TradingHalt, and FutureCompleted. | Price thresholds are in §5 and §18. §5 sends analyst notes to M03 and M14. §6.1 already named the UI publisher and consumers; M29 is the rumor box and cash gauge, and M30 is the market screen. |
| 018 | M00 real mode skips overnight and weekend minutes for business ticks but still drains the scheduled heap across those gaps; `Tick.t` is simulated minute-from-epoch, not the engine step index (the Wave A fake used the index). | Matches §2.3; keeps fakes unchanged while the real clock owns calendar time. |
| 019 | M01 turns the single `Scenario.rate_path` into a policy rate and sets Nelson–Siegel slope to beta1 = policy rate − beta0. beta0, beta2, lambda, maturity buckets, and the CRE and unemployment baselines are ASSUMED rows in `/params/registry.csv`. Each shock applies once. `RatesChanged` stays `curve` and `policy_rate` (`bank_id` null). CRE prices and unemployment stay on the module. The §18 hike check uses the M09 mark −D·Δy + ½·C·Δy² on that curve. M01 does not enforce assets = liabilities + equity. | §6.2 has one rate path and §6.1 freezes the payload, which §5 does not extend. The Nelson–Siegel short-rate limit is beta0 + beta1, so every bucket yield rises with the policy path and a 2022-like hike raises bond losses steadily (§18). The fake multiplies a property factor on every later tick; real mode applies each shock once. Assets = liabilities + equity is M08 and gate 4; §5 M01 does not state it. |
| 020 | M09 HTM reclassify-on-sale toggle defaults to on when not set on `SecuritiesValuer`. | §5 names the toggle but not a registry row; SVB-like runs need AFS marking after an HTM sale. |
| 021 | M34 blocklist content lives in `params/registry.csv`; fuzzy distance defaults to 2 edits on multi-word bank names. | Keeps §10.5 “no hard-coded numbers” while matching the Wave A fake blocklist; §19 API spelled out in §5 M34. |
| 022 | M23 policy thresholds remain `TBD` in `params/registry.csv` until calibration; `load_management_policy()` uses ASSUMED cash borrow at 0 and capital-raise CET1 at 0.10 when cells are TBD. | §18.3 names the rules but not numeric thresholds; Wave C must not invent silent registry values. |
| 023 | M14 real mode reads ASSUMED halt rows: X% = 8, Y = 5 minutes, and `TradingHalt.duration` = 15 minutes. The TBD rows stay. Quotes move toward public fair value (CET1, duration × Δpolicy rate, E10 4.3 percentage-point attention premium). The halt is two-sided. Real mode does not consume PolicyEnacted. One book per scenario bank. | §18.4 leaves X and Y unspecified. The Wave A stand-in already used an 8% drawdown, a 5-tick lookback, and a 15-minute halt. §19 consumes AnalystNote and does not consume PolicyEnacted. §6 freezes duration as its own field, so it is an ASSUMED row rather than a silent constant. |
| 024 | M29 shell renders a registry-sized CPU point subset (default 2048) with layout fixed per seed; full 1M deck.gl path stays acceptance-target only. Wire frames use msgpack with float32 positions and uint8 state; React fake mode uses 2D canvas unless `VITE_CROWD_DECK=1`. | Delivers L6 UI without CuPy while matching §18.7 binary-frame shape; defers WebSocket server integration to the shared FastAPI wave. |
| 025 | M30 UI frames are pydantic `MarketScreenFrame` payloads (JSON or msgpack), built from fake `M30` with crisis verdicts read from `M24` at snapshot time (rejected statements are not bus events). | Matches §5 crisis-room display without new event types; keeps the Wave A fake log unchanged. |
| 026 | M33 lives under `/ops/` (stream L8). Philox uses CuPy when present; otherwise CPU Philox4x32-10 so CI stays GPU-free. Event-log hash matches the Wave A fake; Parquet via pyarrow; demo/offline defaults and cache keys are ASSUMED rows in `params/registry.csv`. M33 runs last in `TICK_ORDER` (after M30) so each hash covers the full tick. | §10.1 folder map; §18.0 stack; acceptance is offline smoke without network. |
| 027 | M02 stores the three layers as CSR adjacency on CuPy when a CUDA device is available and on NumPy otherwise. Draws use a seeded Generator on that backend. The CPU stream is deterministic for a fixed seed. The GPU stream is selected only in `select_backend` and is not required to match the CPU stream. Preferential attachment is batched: each batch links to the degree bag built so far, and both endpoints are written back. Media nodes occupy the seed, then ids are remapped so depositor indexes match M05. Group chats are an undirected stochastic block model (founder, VC, other). Professional ties are undirected and bipartite (treasurers to media plus a sample of founder and VC nodes). `spread` is the sum over layers of layer weight times A times (node channel weight elementwise times exposure). M02 publishes no events. The scheduler calls M05 for segments because the tick order runs M02 before M05. Counts and weights are ASSUMED registry rows. | §18 requires CSR on the GPU and CuPy, while CPU CI has no CUDA device and §2.3 requires a fixed seed to repeat. §6 gives M02 no event. §19 says M02 calls M05 and does not publish. |
| 028 | M03 real mode uses a backend switch: CuPy sparse CSR matvec only when a CUDA device is available; otherwise NumPy CSR matvec, which is what CI runs. Stored attention stays on the host. CPU draws use NumPy Philox so a fixed seed repeats. The half-life is the Attention memory window. `AttentionTick.level` is the mean attention of that bank's nodes, one event per bank per tick. `delta_4h` subtracts the level one window earlier, or 0 when the window is not full. Repost probability is clip(rumor credibility × node credibility × attention × scale, 0, 1). The mutator samples the ASSUMED fraction of reposts, capped by the ASSUMED call budget. Grok stays off unless its ASSUMED flag is set; a rejected or missing suggestion uses a cached offline rewrite. AnalystNote multiplies that bank's rumor credibility by the context factor. Other consumed events add a flat exposure at the bank's most-followed node and do not encode tone. Until M02 supplies CSR layers, M03 builds a deterministic fallback graph and maps `hub-public-1` to the hub. Unmapped source nodes post as that hub. | §18 requires a CuPy sparse matvec and CPU CI must stay green. §6 payloads stay frozen. E10's 4-hour window is the half-life the module's test checks. ADR-001 keeps tone out of the update. The existing "about 1%" and "≤20" cells are not numeric, so the operational values are ASSUMED rows. |
| 029 | M06 decision math is one array kernel selected at runtime: CuPy if a CUDA device is available, NumPy otherwise. The GPU entry point is isolated and CPU CI never calls it. Complex-contagion k is the ASSUMED row `M06 complex contagion k (ASSUMED)`; the TBD row `Depositor complex contagion minimum worried contacts (k)` stays UNSET. Worry weights and segment amounts are ASSUMED rows, not literals. FAKE remains the default mode. | ADR-012 names a CuPy RawKernel, which cannot run in CPU CI. A shared array implementation keeps a fixed seed deterministic on the CPU path and keeps `make ci` green. k is still TBD for M27, so the working value is marked ASSUMED instead of hard-coded. |
| 030 | M25 Grok calls go through `/agents/llm_cache.py` (SQLite, prompt hash + model). Offline mode defaults on (`STAMPEDE_OFFLINE=1`) with a registry-backed rule scorer so CI needs no network; live Grok uses xai-sdk only when offline is off and `XAI_API_KEY` is set. | Matches §7 caching and fallback; E11 direction check uses personalized vs generic worry deltas from the rule path in tests. |
| 031 | M27 uses a NumPy evolution-strategy fit (CMA-ES-shaped) on short fake-engine runs; behavioral dimension and optimizer limits are ASSUMED registry rows. Holdout and placebo use preset scenarios; no network or Grok in the harness. | §18.6 requires CMA-ES and 10–15 parameters. M06 real mode is wired; the harness still evaluates short fake-engine runs so CPU CI stays green without the `cma` package. |
| 032 | M26 batches the §18.2 M06 rule in `/engine/futures/batch.py` as tensors `[futures × agents]`. Array math uses CuPy when a CUDA device is visible and NumPy otherwise; CUDA copies stay in `_gpu_to_device` and `_gpu_to_host`. Philox streams keep one future independent of batch width. `run_serial_reference` is the main-engine match path. One sim-day is the §9 count of about 78 five-minute ticks. Kernel numbers the spec does not set are ASSUMED rows. The TBD contagion-k row is left unread. `build_engine` keeps FAKE as the default and installs `FuturesRunner` only for M26 REAL. `FutureCompleted` is unchanged. | §18.6 and ADR-004 fix the batch and the 5-minute tick. M06's RawKernel file is still a stub, so the futures module cannot call it. §9's under-5-second budget is a GPU target; CPU CI has to stay deterministic without CUDA. §6.1 already freezes the event. |
| 033 | M24 real mode uses a deterministic offline crisis writer (32 registry statements), hard numeric checks against `ClaimsLedger`, and vague-claim review through `agents/llm_cache.py` (SQLite, `STAMPEDE_OFFLINE` default on). The published winner stays `stmt-crisis-1` to match the Wave A fake. | §7 requires cache/offline and no network in tests; ADR-002 keeps the truth filter a hard reject. |
| 034 | M28 preset `Bank` objects are built from registry deposit totals and M05 segment weights; opening CET1 and default rumor poster come from ASSUMED M28 rows. Replay hash matches the M33 fake JSON canonicalization. | Ships preset balance sheets for M08 without hard-coded dollars; keeps rumor posting aligned with M34 guardrails and deterministic CI. |
| 035 | M22 `PolicyEnacted.cost_model` uses ASSUMED DIF and Fed exposure fractions on the SVB-like deposit total and uninsured share; moral hazard aggregates guarantee breadth and frequency. The regulator advisor uses the shared SQLite LLM cache with an offline template fallback; only `Scenario.policies` schedules levers in the engine. | §5 names three cost buckets but not formulas; E7 supplies the \$30B peer deposit. §7 requires advisor text without auto-enactment. |
| 036 | M20 real mode shares `closure_reason` / `watch_recommended` with `RegulatorFake` via `fold_book` on the bus log. CET1 floor, opening CET1, and liquidity cutoffs are ASSUMED registry rows. SVB-like presets publish `BankClosed` during business hours; crypto-heavy presets defer to weekend (`closure-pending` `SupervisoryAction` until then). Community placebo skips closure. | §5 timing policy; §18.5 acceptance uses no policy and `BankClosed`; keeps fake/real parity on the default panic scenario. |
| 037 | M04 thresholds stay `TBD` in `params/registry.csv`; `load_media_thresholds()` reads ASSUMED rows for attention, price move, and loss surprise. Headlines and analyst notes cite triggering `cause_id`s. Grok copy goes through `/agents/llm_cache.py`; offline defaults to rule templates aligned with `MediaFake`. Price-move headlines are real-only (§19 consumes `PriceTick`). | §18.1 leaves numerics open; Wave D must not hard-code silent constants. §7 caching and CI offline policy match M25 (ADR-030). |
| 038 | M44 scores vulnerability as attention × uninsured share, plus an ASSUMED capital-stress bump when CET1 is below an ASSUMED reference. The code reads `M44 assumed vulnerability publish threshold` and leaves `Short-seller vulnerability publish threshold` TBD/UNSET. Thesis text is a validated Grok suggestion behind the SQLite cache; `STAMPEDE_OFFLINE` defaults on and uses a deterministic template. `reach` and the short-order size are simulation numbers. `ThesisPublished` is the M14 handoff; M44 does not write `/engine/markets/book.py`. Real mode is `ModuleMode.REAL` for M44; fake stays the default. | §18.4 names the threshold but §8.1 left it TBD. §6.1 freezes ThesisPublished at author_id, bank_id, and reach, so the thesis text stays off the bus. §10.5 forbids hard-coded numbers and forbids mutating another stream's state. §7 caches LLM replies and falls back offline. |
| 039 | M42 lives in `/agents/ceo.py` and does not edit `/engine/bank/management.py`. Tool calls map onto `MANAGEMENT_ACTIONS`. Affordability replays M08 postings: a sale above securities, a borrow above haircut collateral, or a deposit rate whose segment interest exceeds cash is refused and logged, not published. The CEO reports `capital_raise_success_probability` and leaves the Bernoulli draw to M23. Grok goes through `/agents/llm_cache.py`. Offline defaults on with `offline_ceo_decision`. The reduced rule stores a threshold, sensitivities, and the action for futures. FAKE stays the default; `build_engine` wires `ModuleMode.REAL` for M42. Thresholds are ASSUMED registry rows. One strategist runs per scenario bank; fleet size 2 is the two-bank population in §7.1. | `management.py` belongs to area L5, so the strategist stays in its own file and reuses the policy helpers. §18 requires the ledger to reject unaffordable actions and a per-bank log. §7 forbids live calls on the offline path. |
| 040 | M41 Grok calls go through `/agents/llm_cache.py` (SQLite, prompt hash + model). Offline mode defaults on (`STAMPEDE_OFFLINE=1`) with a registry-backed rule so tests make no network call; live Grok uses xai-sdk only when offline is off and `XAI_API_KEY` is set. The scheduler subscribes to Tick only. `note_trigger`, or an AttentionTick passed into `on_tick`, is the §7.1 trigger. Each decision reduces to a threshold and sensitivities that `replay_reduced_rule` reproduces; `rules_for_futures` is the M26 handoff while `/engine/futures/batch.py` is still a stub. A VC `message_group` publishes `DirectiveIssued` with action `withdraw`. The head balances partition the E3 top-10 total. The feed acceptance line uses the ASSUMED $40M example amount for Founder of Kestrel Labs. | §6 already names DirectiveIssued and VIPDecision and does not add a trigger event. §7.1 caches by prompt hash and falls back offline. §18.2 asks for a schema check and a reduced rule that reproduces the decision. |
| 041 | UI rumors are moderated in the API with `moderate_rumor` (M34) before `CrowdFake.queue_live_rumor` publishes `RumorPosted` on the next engine tick. One `ui.server` process owns a single `LiveSession` (`build_engine`, default FAKE; `STAMPEDE_MODE=real` and `STAMPEDE_BACKEND=gpu` select REAL modules and CuPy when available). Crowd and market WebSockets read the same engine at the registry stream rate. | §19 names M29 as the rumor-box publisher; §6.1 payloads stay frozen. Defers a dedicated `RumorRejected` bus event—rejections are API/WebSocket responses only. |
| 042 | Live playback uses a wall-clock loop in `ui.server`: `PacingClock` converts real seconds to whole engine ticks at `crisis_sim_minutes / playback_target_seconds × speed` (registry rows). Speed and frame rate change delivery only; `step_engine` outcomes are unchanged. WebSocket sends at `M29 UI client stream fps` with per-client drop-on-backpressure. React interpolates crowd orb state and panels (and market price) between snapshots and leaves positions on the seed layout; fake mode uses a deterministic cluster-wave script that fills those orbs gradually. | Matches Plague Inc.-style UX without new bus events; keeps §10.5 numbers in `params/registry.csv`. Orb aggregation stays ADR-044. |
| 043 | `STAMPEDE_LLM` selects `offline` (default), `grok` (xai-sdk), or `local` (OpenAI-compatible HTTP). Local defaults: `Meta-Llama-3.1-8B-Instruct-Q4_K_M` and `http://127.0.0.1:8080/v1` from ASSUMED registry rows; overrides via `STAMPEDE_LOCAL_LLM_MODEL` and `STAMPEDE_LOCAL_LLM_URL`. JSON and text completions use `/v1/chat/completions` with `messages` only (server applies the chat template). A threading semaphore caps parallel local calls per the ASSUMED concurrency row. Pydantic validation failures retry once, then agents use existing deterministic offline rules and increment `local_llm_validation_failures`. M03, M04, M22 advisor, M25, M41, M42, and M44 route through `/agents/llm_cache.py`. | §7.1 requires cache, fallback, and concurrency control; CI stays offline with stub HTTP tests. Desktop demo targets Llama 3.1 8B Instruct Q4\_K\_M on an RTX 3060 via `scripts/serve_local_llm.sh`. |
| 044 | The crowd screen draws a registry orb budget (default 2,048), never one point per depositor and never a 1M-item frame. `N = round(population / orb count)` (488 at 1M). Orb color is the block majority. Uninsured radius is larger and capped. The client eases a registry count of orbs per tick. Cluster tint darkens red with the withdrew share. Msgpack float32 positions and balances are decoded as floats. `CrowdFrame` fields are unchanged. Interpolation and the scripted spread reuse those fields and do not move orb positions. | A byte-wise float decode stacked the live cloud into one circle, and a full population upload cannot hit 30 fps. §6.2 already stores counts on `CrowdPanels` and per-point state on `CrowdFrame`, so the orb budget reuses those fields. |
| 045 | GPU calibration (`STAMPEDE_CALIBRATION_ENGINE=real`) evaluates candidates with real M02–M09 via `build_engine`; the 12-vector maps onto M06 ASSUMED rows through `decision_params_from_vector` and `calibration_amount_scale`. `calibration/run_fit.py` checkpoints each generation, supports multi-seed restarts, and writes FITTED registry rows plus a §13 board markdown artifact. CPU CI keeps the fake-engine path as the default. | §18.6 targets CMA-ES on SVB with Signature holdout; ADR-027/029 require CuPy paths without breaking CPU determinism. |
| 046 | The live demo is one page: `/ui/crowd` mounts the M30 components beside the crowd. `LiveSession.skip_to_outcome` steps until `M29 UI live crisis sim minutes`. The phase pill reaches Outcome only in the last tenth of that window, so an empty operating-cash buffer stays on Run while depositors are still wavering. Orb state keeps catching up a few orbs per frame until it matches the server, including after a skip. WebSocket loss shows a reconnecting banner and retries; it does not substitute the bundled fake frame. `/health` publishes pause, speed, the crisis horizon, and the demo population so a second tab or a reload matches the server. Additional real-bank names live in the M34 blocklist cell. A one-word entry of at most six letters matches only as a whole word, so "chase" does not fuzzy-match "share". | A viewer should see crowd and market together, post a rumor without hunting for a hidden field, and recover when the API process restarts. §6 payloads stay the same. |
| 047 | Free-text rumor classification lives in `/agents/rumor_input.py` (`classify_rumor_market`). The UI applies it after M34 moderation; calibration references the same function via `calibration/rumor_labels.py` for documentation and tests only—fit `Rumor` objects keep explicit credibility and reach, and `measure_outflow` is unchanged. Non-market text does not publish `RumorPosted`. Live `RumorPosted.credibility` is `max(lexicon credibility, poster credibility from registry)`. | One lexicon for live demos and calibration labeling without retro-fitting M27 vectors or changing §6 payloads. |
| 048 | The default FAKE demo paces depositor headcount, the cash gauge, and the stock with `mocks/demo_arc.py` over `M29 UI live crisis sim minutes`. The curve is an ASSUMED logistic: little outflow at 10% of the run, a steep middle, and about 30–60% withdrawn by the end of a strong rumor. A weaker credibility ends much lower. The gauge is empty at about 65% of the horizon for a full-strength rumor. Dollar withdrawals follow that gauge; a later ASSUMED hole is what lets the book post a watch and then close. Real M06, M08, and M14 parameters, and FIT or VERIFIED registry rows, are not used for this curve. Crisis-room rows map known statement ids to short labels and show the id when it is unknown. | A two-minute demo needs a visible calm-to-run arc. The old fake tape emptied a 0.1% cash buffer in a few ticks and froze headcount near 615 people, so the crowd stayed green and the price sat on the floor. §6 payloads stay the same. |
| 049 | M27 long fits parallelize candidate eval with a spawn `ProcessPoolExecutor` (`--workers`, default 1). OMP/MKL/torch threads are capped at 1 per worker. Failed evals log and retry once, then take a penalty objective. `--gpu` sets `STAMPEDE_BACKEND=gpu`; M02/M03/M06 still auto-select CuPy when CUDA is visible. M05 caches `sample_population` and vectorizes balance solve on CPU with an optional CuPy path when `STAMPEDE_BACKEND=gpu`. | §18.6 GPU calibration was single-process and CPU-bound on population + tick orchestration; parallel eval uses idle cores without changing CMA-ES sampling order. |
| 050 | Non-bank calibration is a separate harness in `/calibration/nonbank.py`. Three scenarios carry explicit `Rumor` credibility and reach plus an event-type note. The harness runs the current engine and scores only `PriceTick` peak-to-trough, days-to-bottom, and volume multiple. The equity preset (Meta, February 3, 2022, E13) is the fit. Crypto (LUNA, E15) and rates (S&P 500, February–March 2020, E16) are holdouts. GameStop in January 2021 (E14) is recorded, and any shape statistic that is not a verified point stays UNSET and is omitted from the loss. In fake mode a 3-vector scales those read statistics; it does not write FITTED rows and does not change the SVB, Signature, or placebo scenarios. `classify_rumor_market` labels claims and does not replace the explicit credibility. `--gpu` and `--workers` wait until `calibration.run_fit` has them. | The SVB fit must stay valid on the current engine. Non-bank events need their own price-path loss and public targets, and unverified points must not be filled in. |
| 051 | M27 candidate eval has an opt-in batched path in `/engine/gpu_sim.py`. It runs only when `STAMPEDE_BACKEND=gpu` and `STAMPEDE_CALIB_GPU_SIM=1` (`--gpu` and `--gpu-sim`). The flag-off path is the existing engine and is unchanged. The depositor kernel uses the same belief update, sigmoid, and feature formula as `evaluate_decisions`. Candidates are batched and split into groups under the ASSUMED array-budget row. Each tick publishes one `WithdrawalRequested` per channel; M07 settles the sum. Rumor posting, attention, the M14 fake `price_quote`, payments, the ledger, securities, liquidity, and closure stay on their existing `step` methods. M02 is built once and reused because M03 still uses the fallback graph (ADR-028). M27, M30, and M33 are skipped because they do not publish outflow inputs. Attention RNG stays inside `AttentionEngine`. The depositor kernel draws nothing. E1 day-1 and two-day fractions may differ from the per-depositor sum by float reduction order, up to the ASSUMED relative tolerance. | The serial path emits one event per withdrawing depositor and later modules rescan that log, so one full-scale candidate takes about 18 minutes while the GPU stays idle. Depositor state is the wide part of the tick. Scalar market rules stay on the shared `step` methods so the fit trains the same model. |
| 052 | The non-bank CLI takes the same run controls as `calibration.run_fit` (`--gpu`, `--gpu-sim`, `--workers`, run id, runs dir, checkpoints, patience, extra seeds, holdout minutes). With no new flags the fake 3-vector scale fit is unchanged: one read of the price path, no directory under `calibration/runs/`, no FITTED rows. With `--gpu` and `--gpu-sim` (`STAMPEDE_BACKEND=gpu`, `STAMPEDE_CALIB_GPU_SIM=1`, real engine) each generation scores the equity scenario as one in-process batch through `/engine/gpu_sim.py`. The search vector is the existing behavioral vector, because that is what the depositor kernel applies. The loss is still the E13 price-path shape. Crypto (E15) and rates (E16) are holdouts on that same kernel and are not refit. Price, volume, and outflow fractions stay within the ASSUMED E1 relative tolerance of the per-candidate CPU engine. CuPy is used when a device is visible; NumPy is the same kernel when it is not. `--workers` parallelizes the legacy per-candidate engine when the batch flag is off and does not split a GPU batch. The run writes `checkpoint.json` and `converged.json` and does not write FITTED rows. | ADR-050's 3-vector is a fake-mode stand-in and does not re-run the engine per sample. A full real-engine population at the 390-minute horizon is days on one core. ADR-051 already batches depositor state for any scenario. The non-bank presets use that kernel, so one generation is one batch on an RTX 3060. FITTED M06 rows stay on the SVB harness. |
| 053 | Calibration diagnostics live under `/calibration/diagnostics`. The fit calls `record_generation` with per-generation data; the CLI writes sweeps, withdrawn share, cap/closure binds, seed variance, parameters at bounds, and CMA-ES sigma. Unrelated text is category `irrelevant` on the one shared `classify_rumor_market` lexicon, shows `Not market-related: no effect on the simulation.`, and must leave the event log identical. A non-bank market event may nudge bank stocks, but daily deposit requests stay within the ASSUMED 0.1% of deposits bound; violations add a penalty in `/calibration/nonbank.py` and fail validation. The SVB objective and CMA-ES loop stay in `/calibration/fit.py`. | The playbook needs inspectable fits and two hard constraints without forking the rumor rule or rewriting the SVB search. |
| 054 | The M27 SVB (E1) fit scores `WithdrawalRequested` with payment-channel caps and bank closure off, because E1 already counts requests ($42B withdrawn plus $100B scheduled) and settled outflow is capacity-capped (about 6% vs the 24% day-1 target). Soft sigmoid gates around the M06 worry and contagion thresholds are annealed toward hard. The loss is the mean over eight shared seeds. A sensitivity sweep on requested outflow picks four identifiable coordinates; the fit searches those in log space with a weak L2 prior and flags a parameter that sits on its own bound. Runtime clamps stay on user input. Stage 1 is a batched sweep that brackets the tipping point; stage 2 is CMA-ES inside that bracket with restarts. `--gpu` / `--gpu-sim` still use the array-module path (NumPy default, CuPy on GPU). `validate_fitted_parameters` is the caps-on check. The mean-of-vector `_param_scale` collapse is gone. Fake-engine mocks, the shared rumor/market rule, registry targets, and default non-calibration paths stay the same. The fit writes best-so-far and has no wall-clock cap. | Settled PaymentSettled made every candidate score 8.63537 once the channel and closure caps bound. Fitting demand, not the queue, is what E1 measures. |
| 055 | Live rumor credibility is computed in `/agents/rumor_input.py` from the poster map and specificity features. A live model may add at most the ASSUMED ±0.1 shift and never writes the number. The result is clamped to the ASSUMED 0.05–0.9 rows. `classify_rumor_market` is still the only classifier for the UI and for calibration labels; fit `Rumor` objects keep their explicit credibility. Input classes are `bank_rumor`, `market_event`, `macro_event`, `irrelevant`, and `unsafe`. `/agents/rumor_inbox.py` merges repeats of one class, allows one new rumor per 30 seconds, and rejects text over 280 characters. | Playbook §0.25: a rumor must not inject its own credibility. Same-seed calibration logs stay valid. |
| 056 | `--gpu-full` (`STAMPEDE_CALIB_GPU_FULL=1`) is an opt-in on top of `--gpu` and `--gpu-sim`. It runs attention, contagion, the market quote, and payments as the same NumPy/CuPy kernels for the SVB E1 fit and the non-bank equity/crypto/rates fits. CuPy is used when a device is visible; NumPy is the fallback and the CI parity path. The E1 relative tolerance still bounds outflow fractions and price paths against the `--gpu-sim` numbers for the same seed. Objective, searchable parameters, and default non-calibration engine paths are unchanged. Real 3060 runs use this flag; CPU is for testing numbers. | `--gpu-sim` still called Python `step` for M03/M07/M14, so a full candidate left the 3060 idle on those ticks. The fit must train the same model. |
| 057 | Extends ADR-043. Settings and the HTTP/fake client live in `/agents/local_llm.py`. `STAMPEDE_LLM` still defaults to `offline`. Local is model-agnostic via `STAMPEDE_LOCAL_LLM_MODEL`. Weights are `STAMPEDE_LOCAL_LLM_WEIGHTS` or the ASSUMED registry row; Python used by tests does not embed the host path. `scripts/serve_local_llm.sh` requires an NVIDIA GPU and `-ngl 99`. Unit tests install `FakeLocalBackend` (CPU, no weights). M24 joins M03, M04, M22, M25, M41, M42, and M44 on this path. A failed schema check retries once, then the agent uses its rule and increments `local_llm_validation_failures`. | Desktop demo needs Llama 3.1 8B Instruct Q4 on the RTX 3060 without calling xAI; CI must stay CPU-only and never open host GGUFs. |
| 058 | `--gpu-full` / `--gpu-sim` evaluate a generation as one candidate×seed batch on the existing candidates axis (`simulate_logs(..., seeds=)`). Depositor scratch is preallocated; channel totals stay on device until one copy; CSR and public-signal ops are fused/vectorized. M02 and M29 are skipped because M03 still uses the fallback graph and M29 only posts `ui_rumor`. Default engine paths and searchable parameters are unchanged. Float64 remains the parity dtype; depositor chunks still honor the 6 GB ASSUMED budget on a 12 GB card. | A 3060 generation was 12×8 serial sims (candidates=1 in the log), ~56 s each, GPU-idle on launch overhead and host `.get()`. One batched step keeps results inside the E1 relative tolerance. |
| 059 | M05 large-n balance refine stays on host NumPy (same sequential donor loop as `_transfer_mass_array`). A device array copies once per transfer, not per donor. `--gpu-full` / `--gpu-sim` sample the population once per seed and reuse the process cache across candidate lanes and generations. M01 is scalar; M02/M29 stay skipped. | Per-donor `float(balances[i])` on CuPy stalled 100k setup (GPU idle, one CPU core at 100%). Refine is one-time setup and must match the CPU book. |
| 060 | Stage-2 of the M27 SVB fit is `cma.CMAEvolutionStrategy` (`ask` / batched `evaluate_population_objectives` / `tell`). The bracket sweep sets only `x0` and `sigma0`; `_project_theta` clamps to `_theta_bounds` and never takes a bracket `box`. Each checkpoint pickles `es.pkl` beside `GenerationCheckpoint`. Resume unpickles that object so the RNG and covariance continue. IPOP doubles `popsize` only when `es.stop()` reports stagnation, up to the registry restart count. `fixed_softness()` is unchanged. | The NumPy mean-of-top-half / `sigma *= 0.97` stand-in could not learn the correlated ridge and a resume re-seeded generation 0. ADR-031's "no `cma` package" reason is retired; §18.0 already lists `cma`. |
| 061 | `--gpu-sim` / `--gpu-full` cache device-resident packed static state per seed across CMA-ES generations and rewrite only the per-lane parameter rows. Same-seed results stay bit-identical to a fresh pack. `M27 GPU sim array budget bytes` is 3e9 so 256 lanes × 100k depositors fit in one group in float64 (~11 MB/lane). `run_fit --extra-seeds` defaults empty; extra seed passes stay opt-in. Restarts are CMA-ES IPOP on stagnation (ADR-060). | Rebuilding static and population arrays every generation, splitting lanes under a 6 GB cap, and repeating the fit for `--extra-seeds` 3061,3062 multiplied evaluations by at least 4×. The eight shared seeds are already inside the objective. |
| 062 | Stage 1 of the M27 SVB fit (`_evaluate_population_gpu_sim`, `objective_e1`, `bracket_tipping_point`) calls `simulate_requested_fractions` in `/engine/gpu_sim.py`. The lean path keeps depositor, attention, news, and price state on the array device, accumulates window totals there, and copies once. It builds no per-lane `Engine`, publishes no events, and does not step M07, M08, M09, or M20. The three columns are requests by day-one close, requests by next open, and last-over-first `PriceTick` before close, including the SVB-06 feedback terms and `dt` handling. Validation and holdouts keep `simulate_logs`. NumPy parity is 1e-6 against those event scores. | With caps off, stage 1 only needs attention, decisions, and price feedback. Building a full engine per lane and scanning Pydantic logs dominated a generation. |
| 063 | `--gpu-sim` / `--gpu-full` and `simulate_requested_fractions` store per-depositor arrays (belief, balance, worry, attention, exposure, amounts, thresholds) as float32. Window request totals, channel sums, and prices stay float64. Each information layer is built once as a float32 CSR (`scipy.sparse` on NumPy, `cupyx.scipy.sparse` on CuPy). Spread is `(A @ X.T).T` for all lanes, one call per layer. The per-lane `xp.add.at` scatter is gone. NumPy spread vs the old float64 product stays under 1e-5 relative; requested fractions stay under 1e-4. Default non-calibration paths are unchanged. | Float64 on a 3060 is about 1/64 the throughput of float32 and doubles traffic. `add.at` launched one atomic scatter per lane and was not gather-only. |
| 064 | `params/loader.py` caches parsed `RegistryRow` lists and a name→row index per resolved path. The cache is reused only while that file's `st_mtime_ns` and `st_size` still match. `load_registry` copies the cached list so callers cannot mutate it. `get_parameter` reads the index. Public signatures and `RegistryError` texts are unchanged. | A gpu-full generation called `get_parameter` about 55k times and spent about half its wall time re-parsing `params/registry.csv`. Calibration writes FITTED rows and must be visible on the next lookup. |
| 065 | `_transfer_mass_array` runs entirely on the array module already in use (CuPy on GPU, NumPy on CPU). Takes follow the sequential donor recurrence via prefix `cumprod`/`cumsum` of constrained-set affine maps; writes use a packed-key `argsort` plus segmented `cumsum` (no `add.at`, no `.get()`). Float64 math and outputs match the old host loop to 1e-12 relative on CPU. Other functions are unchanged. | A 12-candidate gpu-full generation called the sequential loop 1,056 times (~22 s inside the function, plus host `argsort` and CuPy `.get()`). Per-donor `float(balances[i])` and host copies stall the 3060. |
| 066 | Stage 2 of the M27 SVB fit writes `M07 capacity multiplier (FIT)` (default 1.0). `run_fit` calls `fit_payment_capacity` once after stage 1 at `global_best`. The solve is one batched `simulate_logs` (caps on, shared seeds): 16 log-spaced multipliers from 0.05 to 20, monotone interpolation in log-multiplier to settled day-one 0.24, one refine if the bracketing pair is wider than 1.2×, stop within ±0.002. M07 multiplies app daily, wire per tick, and ACH batch capacities by the row in `load_payment_params`. `simulate_logs` accepts per-lane `capacity_multipliers`. The lean stage-1 path is unchanged. The unused `M07 payment-channel capacity` row is gone. | The old row was never read. Brentq on dollars (xtol 2e-12, up to 200 full sims) never ran from `run_fit`. Settled day-one was therefore the assumed $25M cap. |
| 067 | Each depositor's uninsured balance has a mobile share. After the existing amount chain, `amount = min(amount, max(0, gap_balance − (1 − mobile_share) × initial_uninsured))`. The same formula is in the fused elementwise kernel, the unfused fallback (`STAMPEDE_CALIB_UNFUSED_PART2=1`), the NumPy lean/`--gpu-full` path, and `evaluate_decisions`. VIP forced `p_act=1` is capped. `mobile_share=1.0` is a no-op and matches the previous amounts. The ASSUMED FIT row defaults to 0.88 (bounds 0.30–1.00). The 16-vector exposes it as a per-lane CMA-ES coordinate; a short pre-SVB-24 vector stays uncapped (1.0). | Expected-value amounts with any `p_act>0` drained the whole uninsured share geometrically, so every candidate requested ~0.9238 of deposits by minute 30 and the 0.81 next-open target was unfittable. |
| 068 | `price_impact_move` / `apply_price_impact` move the print toward `target = impact * cum_share * attention_factor` at `±cap * dt`. `attention_factor = clip(attention_level / attention_ref, 0, 1)` with ASSUMED `M14 attention reference level` = 1.0 (no peak attention is exposed). `M14 price impact (ASSUMED)` is 1.13 so 0.813 requested at attention_factor 1 is about ln(0.4); companion min/max rows are 0.3 and 4. The gpu-sim / fake halt reads SOURCED `M14 trading halt minutes` = 5; `ASSUMED_HALT_MINUTES` is a live alias. The real book still reads `M14 trading halt duration`. | After SVB-14 removed the scripted drop, flow impact stopped once requests saturated (~30 min) and the day-one stock sat near 0.93 instead of 0.40. Level targeting keeps the decline going at the same cap. SEC LULD is a 5-minute pause. |
| 069 | Stage-1 identifiable coordinates are registry indices 14 and 15 (price impact, mobile share). Contact weight, feedback gain, and breaking-point dispersion stay at their priors. Signature holdout runs the crypto-heavy preset with the FITTED vector and the SVB-fitted `M07 capacity multiplier (FIT)` held fixed. `Validation target: requests by next open` is ±10% relative (`expected*(1±rel)`). `best_vector.json` is written from the global best at every checkpoint and once at the end of the fit / `run_fit`. | End-of-day targets cannot see speed-only parameters once requests saturate. The old 5-index holdout reused the SVB settled-day-one number (same measurement or a re-solved capacity). ±25% let 0.9237 vs 0.81 pass. Checkpoint writes used a later seed or last generation, so the file lagged the reported best. |
| 070 | M27 writes to `params/registry.csv` go through `patch_registry_rows`: the file is split into logical CSV records, only records whose Parameter matches are rewritten, new rows are appended with the file's own line ending, and nothing else changes. Row updates that aren't a fit output are JSON specs in `calibration/registry_updates/`, applied with `python -m calibration.registry_patch apply`. | `csv.DictWriter` rewrote all ~890 rows, changing line endings and quoting, so a two-value fit showed as a whole-file diff and couldn't be reviewed. |
| 071 | The non-bank fit persists its normalized 16-dim vector to one FITTED row, `Non-bank E13 fitted behavioral vector (normalized)` (semicolon-separated; Source = run id, seed, objective). It never writes the M06/M14 rows. This supersedes ADR-050/052's "writes no FITTED rows" for that one row. `run_fit` records `capacity_multiplier` in its JSON. | Both fits share the 16-dim layout but fit different data; writing the non-bank vector into the shared rows would overwrite the SVB-fitted price impact and mobile share. The stage-2 multiplier wasn't recorded anywhere, so it couldn't be persisted without re-solving. |
| 072 | E15 peak-to-trough is SOURCED at 0.9999996, derived from the VERIFIED April 5 2022 peak (119.18) and May 13 2022 price (below 0.00005). Days to bottom and volume stay UNSET. `run_nonbank_fit` reports a holdout with no set targets as `null` with status `no targets` (`ShapeTargets.has_targets`). | All three E15 rows were UNSET, so `shape_loss` summed over nothing and the crypto holdout printed 0.0, which read as a perfect fit. |
| 073 | `run_holdout` measures Signature's requested outflow by day-one close with caps on, and the board's E5 row compares it with the new target row (0.249 = 0.20 × (18.6 + 4.6)/18.6: withdrawn plus pending wires, a lower bound). Without that row the board falls back to settled. | Settled day one is bound by the shared `M07 capacity multiplier (FIT)`, so every scenario above capacity printed 0.239979, the SVB number. Requests are behavior and move with the vector. |
| 074 | The real order book's halt length reads `M14 trading halt minutes` (SOURCED, 5: the SEC Limit Up-Limit Down pause), like the gpu-sim and fake paths. `M14 trading halt duration` is DEPRECATED and kept for history. Supersedes ADR-023's 15-minute duration. | Two rows gave two halt lengths: the calibration and fast paths paused 5 minutes and the real book 15, so the demo's market screen and the fitted model disagreed. |
| 075 | QLoRA adapters train from HF-format Llama 3.1 8B Instruct weights in 4-bit NF4 (double quantization, bf16 compute) with LoRA on all attention and MLP projections, offline only, in a separate venv under `$STAMPEDE_TRAIN_ROOT` (default `./training_runs`). The trainer checks `nvidia-smi` and won't share the card with a running fit. Converting the adapter for the llama.cpp server is a manual step. | The 3060 has 12 GB: 4-bit base weights plus LoRA and gradient checkpointing fit at 1,024 tokens. Fits and training can't share the card without slowing or failing both. |
| 076 | Each depositor decides at most once. Hard gates: act and not already withdrawn → request `segment_fraction × gap` once. Soft gates: `decided ← max(decided, p_act)`, request `(Δdecided) × segment_fraction × initial uninsured × gap_scale × amount_scale`, contagion share reads `decided`. VIP amounts and the mobile-share cap are unchanged. `repeat` keeps the legacy take for direct-call tests (kernel and `_evaluate` default `decide_once=False`; the simulation reads the registry). `gpu_full.py` has no amount path, so it doesn't change. | The per-tick take re-rolled every minute: a calm depositor's small soft-gate probability compounded, and every depositor had `amount > 0` on tick 1, so `already` was true for everyone and the contagion share jumped to 1. Every book hit the mobile-share cap by minute 30 for every vector, so next-open couldn't inform the fit. |
| 077 | One simulated trading session (390 minutes) is one day for every non-bank event. E13 is 1 trading day; E16 is 23 trading days (derived from the verified 2020-02-19 peak and 2020-03-23 trough; the 33-calendar-day row stays for reference). `reachable_targets` drops a days target longer than `steps / minutes_per_day` from the fit and holdout losses and reports it. The rates holdout can run longer (`rates_steps`, `--rates-holdout-minutes`). | At 1,440 minutes per day, E16's 33 days needed 47,520 sim minutes and E13's 1 day needed 1,440, but both runs were 390 minutes, so the days terms were constants no vector could reduce. That dominated rates_loss (43.5) and put a floor under the E13 fit. |
| 078 | QLoRA's default base model is the first existing of `$STAMPEDE_QLORA_MODEL`, `$STAMPEDE_TRAIN_ROOT/hf/Meta-Llama-3.1-8B-Instruct`, `$STAMPEDE_TRAIN_ROOT/models/Meta-Llama-3.1-8B-Instruct`. Training data comes from `training/export_jsonl.py`: read-only SQLite, explicit table and columns, hash-only prompt columns refused, exact duplicates dropped, a deterministic 5% eval split. Preferred sources are local-model outputs that passed validation and rule-path outputs. | The weights were in `hf/`, so the default path failed without `--model`. `LlmCache` is keyed by prompt hash, so a table may not hold prompt text at all; guessing columns would train on hashes. Some providers' terms forbid training on their outputs. |
| 079 | A fitted vector is normalized against the registry at fit time: physical = registry value × scale(coordinate). `fitted_registry_updates` computes the cells before any write; `write_fitted_registry(values=...)` writes them; `persist_fit` prints them. Persisting the same run twice is refused, because after the first write the fitted values are the new centers and the same vector would compound them. Re-applying an old `best_vector.json` after persisting has the same problem; a refit starts from the persisted values (all coordinates 0.5). | The printout recomputed after the write and showed 1.24192 / 0.877877 (= 1.18464 × 1.0484 and 0.878938 × 0.9988) while the CSV held 1.18464 / 0.878938. |
| 080 | Tests that exercise the gpu-sim batching check the batching and accept whichever array module the machine resolves (`cpu` or `gpu`). Tests that need a specific backend force it explicitly. | The non-bank CLI test asserted `array module=cpu` and failed on the 3060, where CuPy resolves first. |
| 081 | Awareness is per depositor; attention alone cannot decide everyone in three minutes. Each depositor has `aware` in \[0, 1\], reset to 0 at start. Before logits, `aware += (1-aware)·(1-exp(-dt·(r_media·attention + r_wom·(w1/w1_prior)·share_decided + r_news·max(0, news))))`. Feature 0 is `attention · aware`. Soft once-mode requests `mobile_share · initial_uninsured · gap_scale · amount_scale`; hard gates keep `segment_fraction`. The max ratchet is unchanged. Fit indices are 7, 14, 15, 16 (bias, impact, mobile share, media rate) on a 17-vector. Contact weight and feedback gain stay at their priors and now move day-1 requests through WOM and news. | Broadcast attention saturated by t3 (0.62, 0.88, 0.99, 1.00), worry and the decided ratchet followed, and day-1 requests equalled next-open. Contact weight could not add to saturated worry, feedback news entered only through clipped attention, and worry bias was a cliff. `segment_fraction · uninsured` capped requests at 0.6707, below the 0.81 target. |
| 082 | When gpu-sim `dt` is an integer greater than 1, the tick list expands to one-minute clock times so attention, M06, and the lagged scripted price all see the same sequence as `dt = 1`. The kernel keeps closed-form `dt` for repeat-mode take and for fractional steps. `aware0` defaults to 0. | Awareness `1-exp(-dt·hazard)` already matches across timesteps. Once-mode `p_act` is a per-tick level, not a rate, so a 5-minute tick misses the steep scripted drawdown and the decided cascade has 12 generations instead of 60. |
| 083 | Fit the requested-by-day-one-close level so day-one timing is identified. Stage 1 adds ASSUMED `Calibration E1 requested day-one target` = 0.249 (relative log-error, same weight as next-open) evaluated at the SVB-29 trading-session close (`Non-bank sim minutes per day`). Stage 2 (settled 0.24) and the Signature (E5) holdout are unchanged. The board gains `E1-requested-day1`; the log and `summary.json` print the E1 stock ratio. Fit set stays `7,14,15,16` on a 17-vector. | Next-open 0.81 and settled 0.24 were both met while day-one requests equalled the two-day total (0.8093 vs E5's 0.249). Bias and media rate saturated day one because the objective had no term for requested-by-close. |
| 084 | Soft once-mode uses a finite-rate ratchet `d_new = decided + (1-exp(-dt·h))·max(0, p_act-decided)` in `_evaluate`, `_once_amount_fallback`, the fused `svb28_once_amount` kernel, and `DecisionModule.step(t, dt)`. That closed form composes exactly across sub-steps when `p_act` is held constant. `h` is FIT row `M07 decision-to-request rate (FIT)` (prior 1/600 per minute, bounds 1/5000–1/20), read registry-first with an optional override. Large `h` equals the old `max(decided, p_act)` step. Lean stage-1 applies the same-tick M41 VIP hub shock as the event path. Stage-1 fit indices stay `7,14,15,16` on a 17-vector; index 16 is `h` and the media awareness rate stays at its registry prior. Stage 2 and the Signature holdout are unchanged. | The instant ratchet turned a decision into a full request in one tick, so every fitted vector stepped from ~0.003 to the request ceiling in 30–40 minutes. Day-one-close vs next-open could only separate if that cliff landed in a ~5-minute window; worry-bias 0.005 moved day-one from 0.23 to 0.74, and CMA-ES could not hold it. Media rate (old idx16) did not move timing at fitted values. Per-tick exponential smoothing disagrees across tick schedules unless `p_act` is held constant or every path uses the same `dt` and same-tick shocks. |
| 085 | Report the E1 day-one stock ratio; do not fit it | In the calibration stock path the price falls only through withdrawal price impact (no scripted drawdown, SVB-14). Measured at SVB-35 on the NumPy lean path: with 0.15–0.28 requested by close, the close ratio is 0.71–0.81 even at the impact upper bound 4, and SVB-34 reached 0.41 only with 0.46 requested by close. So 0.249 requested by close and a 0.40 close cannot both hold. SVB's real −60% on Mar 9 2023 followed the Mar 8 loss and capital-raise announcement, which this path does not model. Price impact (index 14) leaves the fit set with the term; its value is carried and labeled ASSUMED |
| 086 | `step_market_group` and the gpu-sim `MarketFake` quote pass `scripted=` from `calibration_uses_scripted_drawdown()` (the same lookup as `simulate_requested_fractions`). `price_quote` / `price_quote_batch` still default `scripted=True` so the demo logistic stays on when the M27 row is not `no`. | The validation board's `simulate_logs` path omitted `scripted=` (default True), so the old 0.62/120 min demo drop ran in calibration. Same vector scored 0.430 requested-by-day-one-close on the board vs 0.245 on the fit path. |
| 087 | `MarketFake` passes `scripted=calibration_uses_scripted_drawdown()` when `gpu_sim_enabled()` or `STAMPEDE_CALIBRATION_ENGINE=real`. Demo `price_quote` and non-calibration `MarketFake` ticks keep `scripted=True`. | ADR-086 gated the fake quote on `--gpu-sim` only, so the CPU non-bank path still applied the demo drop while the GPU-flag path did not (peak-to-trough 0.019748 vs 0.019775, above 1e-6). Always reading the helper would also agree, but the registry cell is `no` and would turn off the demo logistic. |
| 088 | `load_decision_params` loads `M06 worry bias (ASSUMED)` when Status is ASSUMED or FITTED. Every other M06 row still requires ASSUMED. persist_fit still writes FITTED. | After stage 1, persist_fit marks the worry-bias row FITTED. `_assumed` then refused that row, so `persist_fit --stage2` could not rebuild the engine (`fit_payment_capacity` → gpu_sim → load_decision_params). The registry key keeps the `(ASSUMED)` suffix; the fitted value must still load. |
| 089 | Non-bank `summary.json` reports crypto and rates holdout horizons separately. `--crypto-holdout-minutes` (like `--rates-holdout-minutes`) is the E15 step count and is passed to `reachable_targets`. Behavioral vectors re-measure the best equity path once and write per-statistic shape-loss terms; a 3-vector scale run sets those terms to null. `shape_loss` is unchanged. | One `eval_minutes_holdout` hid a longer rates or crypto holdout. E15 days-to-bottom was dropped whenever the shared horizon was a single session. |
| 090 | `persist_fit --stage2` calls `apply_run_fit_engine` (shared with `run_fit._run`) at `--gpu-full`: `STAMPEDE_CALIBRATION_ENGINE=real`, `STAMPEDE_OFFLINE=1`, `STAMPEDE_BACKEND=gpu`, `STAMPEDE_CALIB_GPU_SIM=1`, `STAMPEDE_CALIB_GPU_FULL=1`. Missing CUDA falls back as `run_fit` does (NumPy, backend still `gpu`) and the resolved path is printed. | Without that setup the capacity solve used the fake/CPU engine, so `MarketFake` kept the demo `scripted=True` drop instead of `calibration_uses_scripted_drawdown()` (registry cell `no`). Board settled-day-one was 0.1177 vs the fit's 0.2185 (target 0.24), and the run took ~9 minutes on CPU. |
| 091 | Poster roles live in the existing `M29 fake poster credibility (ASSUMED)` row as `id:value` pairs in weakest-to-strongest order; `rumor_roles()` parses it and `mocks/demo_arc.poster_credibility` reads the same row. Bases are ordinal ASSUMED values from documented cases (`docs/research/rumor-roles.md`). `CredibilityBreakdown` is computed with the same arithmetic as before (`base + hits·w/4 + shift`, clamped). The role id still becomes `source_node`; unmapped ids post as the hub, so engine reach is unchanged. | Viewers pick who posts a rumor (user, VIP, CEO, ...), and the number must be explainable. Documented price moves show market pull but mix reach, timing, and bank fragility, so they rank roles rather than set a value. Same-seed logs and the old five roles must not move. |
| 092 | The rumor judge returns scores, not credibility. Credibility = role base + `specificity·4·w/4` + `(plausibility − 0.5)·2·shift max`, clamped; a lexicon judgment (hits/4, plausibility 0.5) gives the 77db0e2 arithmetic for any registry weight. Model specificity is held within the ASSUMED deviation of lexicon specificity, so the model's total effect on credibility is at most 0.25·0.2 + 0.1 = 0.15. Strength changes only for model judgments of fear rumors: a call to act adds the ASSUMED levels and a solvency, liquidity, or regulatory claim is at least the ASSUMED minimum; it never lowers. The market gate (`affects_simulation`) stays lexicon-only. The cache stores the raw reply and re-validates on read; offline never reads it. The prompt JSON-escapes the text and writes `<` and `>` as `\u003c` / `\u003e`. | Sebastian asked for judged strength. Live text must not set its own credibility, offline runs and seeded logs must stay deterministic, and a slow model must not stall the live stream. |
| 093 | Rumor role reach in M03 uses the role id already carried as `RumorPosted.source_node`, so §6 payloads and `event_log_hash` inputs do not change. Two independent flags, both off: a per-role post-impulse multiplier (reach outside the sim graph: media pickup, follower counts) and a role → node map on the fallback graph (option a: `hub-public-1` public broadcast, `node-2` professional/advisor hub, `small-account`). Wiring M02 into M03 (option b) is not done: it would change every real-mode log and the SVB seeded run. Multipliers apply to the post impulse only, never to reposts. With both flags on, reach is counted twice (graph position and multiplier); turn on one. The batched GPU kernels do not mirror the flags, so the CPU engine refuses them under GPU calibration. | Viewers pick a poster role; documented market pull is mostly reach (Cookson et al.: attention, not tone), which credibility alone cannot express. The SVB fit and seeded hashes must not move. |
| 094 | ADR-001 (attention, not tone) stays the rule. The only exception is a flag, off by default: M03 may scale a rumor's post impulse by the claim's lexicon strength. M03 recomputes strength from `RumorPosted.claim` with the shared offline rule, so a replayed log gives the same impulse and §6 payloads do not change. Live judged strength (SVB-40) does not reach M03; carrying it would need a `RumorPosted` field (a §6 change, not made). Multipliers are kept near 1 (medium = 1) because Cookson et al. find negative sentiment does not amplify run risk and lexicon strength counts fear words. A CEO-reassurance dampening rule was not added: the cited cases (Credit Suisse 2022, SVB 2023) show reassurance failing, and M03 already adds attention for any post whatever its tone. Reassurance dampening is already modelled on the StatementIssued path: M06 worry with the M25 reassurance deltas (generic −0.05, personalized −0.12, E11). | Sebastian asked for judged strength to reach the engine. The SVB fit, seeded logs, and ADR-001's evidence must not be overridden by default. |
| 095 | Mutation exaggeration may feed back into repost exposure behind a flag, off by default. The factor applies to reposts only (not the original post, not credibility, not repost probability), uses the highest accepted score so far, and is capped. The offline mutator always scores `M03 offline mutator exaggeration` (0.25), so the feedback also works offline and stays deterministic. The sign follows the finding that false and novel news spreads farther and faster (Vosoughi, Roy and Aral, Science 2018); the size is a placeholder. | RumorMutated carried a score that nothing used. Calibration and seeded logs must not move by default. |
| 096 | Branch comparison uses the canonical event rows that the log hash uses. A type-only comparison called two logs identical when they had the same event sequence but different amounts or credibilities (`test_branch_compare_finds_divergence` failed at 6c57b2d: two seeds gave the same 57-event type sequence). | The rewind/branch view must report a divergence whenever the logs differ, and the report must agree with the hashes it prints next to it. |
| 097 | A calibration run proposes; a person persists. `run_fit` writes the registry only with `--write-registry`; the default run leaves `params/registry.csv` untouched and records the `persist_fit` command. A row patched to the value it already had is not reported as written. | Decided: no persisted vector without explicit maintainer approval. At 6c57b2d `run_fit` wrote FITTED rows at the end of every run (the SVB-34 run wrote 4 rows into the dev checkout), and the summary flag reported unchanged rows as written. |
| 098 | The viewer-facing live screen runs the engine, not the scripted demo curve. Real mode uses the same real modules as calibration (`real_calibration_modes`) plus the M14 order book, listed in one ASSUMED registry row; the crowd counts are M06 state and the gauge is M08 liquid assets, so what is shown is what the engine computed. The scripted arc stays as `STAMPEDE_MODE=fake`. Known gaps, not hidden: M03 still runs on the 16-node fallback graph (M02 is not wired into M03), the M08 opening book holds 0.5% of deposits in cash plus securities, and the live run is near-total within about 20 minutes. | Honesty: no embellishment. At 6c57b2d the crowd followed `mocks/demo_arc` whenever a rumor was posted, even in real mode. |
| 099 | Non-bank presets get their own depositor defaults through a context-scoped row map, not new loader parameters. The four loaders that read the SVB-fitted rows resolve the row name through `engine.depositor_rows.resolve`; the non-bank harness wraps its CPU engine build/run and its gpu-sim batch in `nonbank_depositor_rows()`, so CPU and batched GPU read the same rows. Non-bank fit vectors stay offsets around these defaults, so a non-bank vector no longer changes meaning when the SVB fit is persisted. | At 77db0e2 the non-bank equity scenario requested 0.79 of deposits/day (penalty ~621,976) because it inherited the SVB fit (4ed6ad9). Sebastian approved D3 with the pre-fit values. |
| 100 | Settlement after closure is a flag, default off. Real bank closures stop payments until a receiver or bridge bank reopens (SVB: closed Fri Mar 10 2023, bridge bank Mon Mar 13), but turning it on can move settled totals after a closure, and the batched GPU payments kernel does not mirror it, so it is off for calibration and refused under GPU calibration. M07 learns of the closure from `bus.log` with its own cursor instead of subscribing, so the event-consumer table does not change. | Leftover settlements after closure (e.g. 436 on day 1 of the crypto profile) looked like a bug; Sebastian approved D4 as a default-off flag. |
| 101 | `default_mode=REAL` is rewritten at the top of `build_engine` into an explicit REAL map over `REAL_MODULE_IDS` (explicit `modes` entries win) with `default_mode=FAKE`, so the existing per-module swap logic handles every real module and the loader never sees REAL for a module it cannot build. | P0-3: real mode crashed at startup for every caller; SVB-50 only worked around it in the live server. |
| 102 | The bank's opening liquidity is a flagged, sourced change. M08 is in the calibrated module set and M20 closes on M08's CET1, so a larger opening book can move closure timing (not day-one requests or settlement: M06 reads no ledger events and M07 settles regardless of cash). It stays off for calibration; the live demo opts in through a context so only live ledgers change. | Sebastian decided the demo bank starts with more cash; at 0.5% of deposits the live gauge emptied and the bank closed by about minute 10. |
| 103 | M02 is wired into M03 by reference, not by a new event: `build_engine` sets `M03.network_source` to the real M02 module whenever both are real, and M03 uses it only when its flag is on, installing the network before reading any event of its first step (so a rumor posted at tick 0 is not dropped). M02's CSR rows already list the nodes a node hears, matching M03. | P0-2: M03 never saw M02's graph. Off by default because it changes every real-mode log and would need a refit before any calibration use. |
| 104 | The live run's speed came from a feedback loop the SVB fit never saw: real M14 price crash -> M06 stock-return feature -> more requests. The fix keeps the calibrated depositor model and the real order-book price on screen, and cuts only that link, live-only. Engine flags stay off in the registry; `live_flags()` (a ContextVar) enables them inside the live server's real-mode build; a registry-on flag is refused under GPU calibration. | Item 6: a VC rumor requested every eligible deposit by minute 20; changing only the rumor settings would delay that cliff, not soften it (sandbox: 3,990 at minute 20 with the calibrated market vs 49,995 with the order book). |
| 105 | A calming rumor is the second flagged exception to ADR-001 (attention, not tone; the first is ADR-094). Design (approved Nov 3): any role may post one and its effect scales with the role's credibility; it is weaker than a panic rumor (cut 0.5, worry factor 0.5, both ASSUMED) because reassurance mostly failed in the cited cases; depositors who already withdrew stay withdrawn; the price moves only through attention and withdrawals. M06 learns of cures from `bus.log` with its own cursor, so the §6 consumer table does not change. Registry flags off, so calibration and seeded logs do not move; GPU kernels do not mirror them and a registry-on flag is refused under GPU calibration. | Item 9: a positive post used to add attention like a panic rumor, and nothing calmed the crowd. |
| 135 | Reserve separate space for replay text and controls; measure complete canvas labels, including ellipses and computed markers, before drawing next to counts. Day headers shorten only when their combined measured width needs it. Finale labels sharing a trigger mark are grouped; timeline descriptions occupy a separate row. Navigation starts at the new beat's top. | The 10/25/50 replay showed sidebar label/count collisions, a Day 10/no-response overlap, an unwrapped description beneath the selector, a legend sharing space with controls, and overprinted finale markers. Normal DOM wrapping and measured canvas budgets keep the saved-data presentation legible without changing simulations. |
| 136 | Preserve the live cash gauge and regulator rule. Do not infer a cash condition after an earlier intervention from the no-response cash path; keep the policy as needing simulation and exclude it from ranked recommendations until supported. Generated paths are not independent simulation evidence, and a holdout validates only its named plan. | The former saved winner predicted peer deposits at minute 364.5, but its actual cash condition never fired and the bank closed at the next open. A request-trigger control fired and survived. Silently changing the condition, forcing a response time or suppressing closure would conceal the mismatch. |
| 137 | Use the plan-wall vault visual on the live first page, driven by real engine requested fractions. Bound display density with a deterministic subsample, remove the dense disc/dial, and move explanatory text outside the rings. | The user supplied a plan-vault reference and identified the old first-page vault as outdated. The new visual changes presentation only, retains the neutral first beat, and labels the centre as requested deposits rather than cash remaining. |
| 138 | A rerun must not drop tipping runs, and a press must not replace the saved runs (VF-5). On Nov 6-7 a full resimulation ("Run the search", pipeline with no options) rewrote every result file with the default 10 tipping versions, so the 25- and 50-version files lost the 0.26-0.29 strength runs for versions 11-50 (series 2,530 to 2,360 in the 50 file) and their threat point fell back to 5% asked (about 1:39 PM) while the 10-version file kept 20% (about 3:18 PM). `scripts/pipeline.py --tip-strengths` simulates named strengths and, like every tipping stage now, only the (strength, version) runs no file already has; `--reuse --tip-strengths 0.26,0.27,0.28,0.29 --tip-seeds 50 --messaging-plans 0` restores the 160 runs without a new search. "Run the search" is operator-only and asks first. Rescoring the Nov 6 fully tipped runs with VF-4's predictor in an independent rerun gives every file the same answer: 3 useful plans (bonds pledged + big-bank deposits once withdrawals pass 20% of deposits #1, open 50 of 50, close call 0.3 points, #1 in 5 of 7 assumption variants at 50), the threat at 20% asked (3:18 PM) and no return past 25.5% (3:57 PM). The early warning is a forecast, so it is not shown once the run is over. | Review of the prototype build (`77fbdad`, Nov 7). |
| 139 | A wall tile names the days a plan stepped in early (VF-6). Each version of the day scores a plan as useful, early (it would have stepped in on a weaker rumor that died out), not open to a US bank, or failing. The tile said only "useful on 36 of 50 days", which read as failing on 14 next to OPEN 50/50 and a headline saying every time. It now says "useful on all N days", "stepped in early on E of N days" when every other day was early, or "useful on U of N days · early on E". Wording only: the ranking, counts and data are unchanged. With the Nov 7 refill at 50 versions, all three useful plans read "stepped in early on 14 of 50 days", the #1 adding "#1 on 36". | Review of the VF-5 build (`57662aa`, Nov 7). |

| 140 | Give the viewer a short story with explicit terms (UI-6). Use the same plan names throughout; define pledged bonds, requested deposits and next-morning survival. Keep the unprepared-bank deadline, computed provenance, close-call margin, sensitivity and withheld count visible. Put full methodology and alternatives behind accessible details. The live results button shares the already supported preset path, rather than refusing calculated plans. No scoring, model, saved data, contracts or numeric parameters change. | Sebastian asked for technical text that a viewer without bank-liquidity knowledge can understand, with less changing text across screens. |

| 141 | Preserve the original vault artwork while optimizing saved playback (UI-7). All counts wait for Play and have local reset/speed controls. Cache original framed miniatures, interpolate saved samples directly and reuse common geometry. Departing cards keep their slots until their shrink/fade ends; survivors grow within disjoint replacement slots. Separate text and plot regions, and verify actual moving frames rather than settled screenshots alone. No numerical or ranking changes. | Sebastian requested smoother 10/25/50-day elimination, explicit playback controls and screenshots, then rejected a preview with stacked, redesigned miniatures. |
| 142 | Viewer-facing words (UI-8). One vocabulary: the engine's "requested" is "withdrawal requests" on screen, and plans read as sentences ("Bonds pledged in advance, then deposits from other banks once withdrawal requests reach 20% of deposits"; tight labels join with +). Beat 2 reveals the answer: the headline keeps the neutral question during the replay and shows the best plan once it finishes (beat 3 keeps it; Reset hides it again). The finale tile says how each plan works instead of the early-day count ("Customers still ask for 82%; the bank has the money to pay"; "The guarantee stops the run at 20%, but needs the government"), and the tile's own OPEN n/n is the only open count. The side panel ends on the plans that kept the bank open and a count of the rest, and lists at most 8 plans while the replay plays. The grid's one-line description says what a red border means. Margins are stated in dollars from `deposits_usd`, which `/api/search` now carries from the registry, so no engine is needed ("about $516M (0.3% of deposits)"). Withheld plans are explained in the evidence fold, not the main card. Beat 1's live controls are hidden on beats 2 and 3, which have their own replay controls. Beat 1 says "Out of cash" rather than a negative balance, "Cash and bonds on hand", and "asked to withdraw". The run badges are "Calibrated to a real 2023 bank run" and "What-if on the same calibrated model" (no real bank names on screen, as the honesty test requires). Eight page tests that still quoted pre-UI-7 copy follow the current words. Wording and layout only: the model, data and ranking are unchanged. | Sebastian, Nov 7: the text is still too convoluted for a first-time viewer; check for too many visuals or explanations. |

# 17\. Sources

| Tag | Source |
| :---- | :---- |
| E1 | CNBC, Barr testimony on SVB withdrawals: [https\://www\.cnbc.com/2023/03/28/svb-customers-tried-to-pull-nearly-all-deposits-in-two-days-barr-says.html](https://www.cnbc.com/2023/03/28/svb-customers-tried-to-pull-nearly-all-deposits-in-two-days-barr-says.html) |
| E2, E4 | Barr testimony coverage: [https\://business.mammothtimes.com/mammothtimes/news/read/43262824/silicon\_valley\_bank](https://business.mammothtimes.com/mammothtimes/news/read/43262824/silicon_valley_bank); House Financial Services Committee fact sheet: [https\://www\.angelcapitalassociation.org/data/Documents/Public%20Policy/FSC\_Fact\_Sheet\_on\_Silicon\_Valley\_Bank\_Failure\_032223.pdf](https://www.angelcapitalassociation.org/data/Documents/Public%20Policy/FSC_Fact_Sheet_on_Silicon_Valley_Bank_Failure_032223.pdf) |
| E3 | AP / Investment Executive (Gruenberg testimony): [https\://www\.investmentexecutive.com/?p=12394](https://www.investmentexecutive.com/?p=12394) |
| E5 | NYDFS statement on Signature: [https\://www\.dfs.ny.gov/reports\_and\_publications/statements/st20230530](https://www.dfs.ny.gov/reports_and_publications/statements/st20230530) |
| E7 | House Financial Services timeline: [https\://democrats-financialservices.house.gov/UploadedFiles/HHRG-118-BA20-20230517-SD002.pdf](https://democrats-financialservices.house.gov/UploadedFiles/HHRG-118-BA20-20230517-SD002.pdf) |
| E10 | Cookson et al., "Social Media as a Bank Run Catalyst": [https\://papers.ssrn.com/abstract=4422754](https://papers.ssrn.com/abstract=4422754) |
| E11 | Kazinnik, "Bank Run, Interrupted": [https\://digitaleconomy.stanford.edu/publication/bank-run-interrupted-modeling-deposit-withdrawals-with-generative-ai](https://digitaleconomy.stanford.edu/publication/bank-run-interrupted-modeling-deposit-withdrawals-with-generative-ai) |
| E12 | FSB, 2023 bank turmoil report: [https\://fsb.org/uploads/R231024.pdf](https://fsb.org/uploads/R231024.pdf); International Finance (2026) on social-media signals: [https\://metatoc.com/papers/191584-social-media-monitoring-to-predict-bank-runs-signals-and-limitations](https://metatoc.com/papers/191584-social-media-monitoring-to-predict-bank-runs-signals-and-limitations) |
| E13 | Yahoo Finance META daily history (closes 323.00 and 237.76, volumes 58,458,300 and 188,119,900): [https\://finance\.yahoo\.com/quote/META/history/](https://finance.yahoo.com/quote/META/history/?filter=history&frequency=1d&includeAdjustedClose=true&interval=1d&period1=1509408000&period2=1667088000); CNBC on the one-day drop: [https\://www\.cnbc\.com/2022/02/03/facebook-shares-plummet-22percent-after-reporting-weak-guidance\.html](https://www.cnbc.com/2022/02/03/facebook-shares-plummet-22percent-after-reporting-weak-guidance.html) |
| E14 | GameStop short squeeze (intraday high $483, start-of-month $17.25, January 25 volume, February 2 decline): [https\://en\.wikipedia\.org/wiki/GameStop\_short\_squeeze](https://en.wikipedia.org/wiki/GameStop_short_squeeze) |
| E15 | Briola et al. on Terra-Luna prices: [https\://arxiv\.org/html/2207\.13914v3](https://arxiv.org/html/2207.13914v3); Liu, Makarov, and Schoar, NBER w31160: [https\://www\.nber\.org/system/files/working\_papers/w31160/w31160\.pdf](https://www.nber.org/system/files/working_papers/w31160/w31160.pdf) |
| E16 | Yardeni, appendix 15.4, S&P 500 peak 3,386.15 to trough 2,237.40, −33.9% in 33 days: [https\://www\.yardenibook\.com/pub/15\.4appendix\.pdf](https://www.yardenibook.com/pub/15.4appendix.pdf) |
| Model reference | Rajan, "Social Contagion and Bank Runs: An Agent-Based Model with LLM Depositors": [https\://www\.arxiv.org/abs/2602.15066](https://www.arxiv.org/abs/2602.15066) |

# 18\. Implementation guide: how each module is built (one spec per module)

## 18.0 Stack (locked in ADR-008)

| Layer | Choice | Notes |
| :---- | :---- | :---- |
| Engine | Python 3.11 \+ CuPy | GPU arrays and custom CUDA kernels via RawKernel. PyTorch is the fallback if the CuPy install fails. |
| CPU hot loops | Numba | Order books and accounting |
| Schemas | pydantic | Mirrors /contracts |
| Server | FastAPI \+ WebSockets | Binary msgpack frames at 10 Hz |
| Grok | xai-sdk (Python) | Responses API with structured outputs |
| Cache | SQLite | LLM cache and event index |
| Logs | Parquet | Event logs |
| Calibration | cma or Optuna |  |
| Tests | pytest \+ Hypothesis | Hypothesis generates random event sequences for property tests |
| UI | React \+ deck.gl \+ uPlot | Point clouds and fast charts |

**Spec format: each spec lists the module's files, its implementation, its tests, and its acceptance condition.**

## 18.1 Engine and information

### M00 Clock, calendar, scheduler

**Files:** /engine/core/clock.py  
**Implementation:**

* A heap event queue ordered by time, then phase, then sequence number.  
* A calendar table of business hours, wire windows, facility hours, and weekends.  
* run(until) calls each module's step(t) in phase order.

**Tests:** tie ordering is stable, and weekend events still fire.  
**Acceptance:** an empty simulated week runs in under 1 second.

### M01 Macro and rates

**Files:** /engine/macro/rates.py  
**Implementation:**

* `MacroRates.step(t)` publishes one RatesChanged per tick.  
* A Nelson–Siegel yield curve (level, slope, curvature) mapped onto maturity buckets. Slope is beta1 = policy rate − beta0 (ADR-019).  
* Scenario paths as piecewise-linear schedules.  
* Three property-price series with shock events applied once.

**Tests:** a 2022-like rate-hike path raises bond losses steadily (−D·Δy + ½·C·Δy²).  
**Acceptance:** the curve streams to the UI (`RatesChanged.curve` each tick).

### M02 Multilayer information network

**Files:** /engine/agents/network.py  
**Implementation:**

* The public feed uses preferential attachment, so follower counts are heavy-tailed.  
* Group chats use a stochastic block model with founder and VC communities.  
* Professional ties are a bipartite graph between treasurers and advisors.  
* Each layer is stored in compressed sparse row (CSR) format.  
* Array code uses CuPy when a CUDA device is available and NumPy otherwise (ADR-027). The CPU path is deterministic for a fixed seed. The GPU path is isolated in `select_backend`.

**Tests:** the follower-count distribution has a heavy tail, and the communities can be recovered.  
**Acceptance:** 1M nodes across 3 layers build in under 30 seconds.

### M03 Attention and rumor engine

**Files:** /engine/agents/attention.py  
**Implementation:**

* Each tick, a ← decay(a) \+ Σ\_layer w · A\_layer · exposure, computed with CuPy sparse matrix-vector multiplication.  
* Reposts are random draws on the GPU, weighted by rumor credibility.  
* About 1% of reposts go to the mutator.  
* Context notes multiply credibility down.  
* Backend switch (ADR-028): the same update runs as a CuPy sparse matvec when CUDA is available and as a NumPy CSR matvec otherwise. GPU code stays isolated. The CPU path is deterministic for a fixed seed.  
* M02 CSR layers are the input. When they are absent, a deterministic fallback graph is used so AttentionTick still fires.

**Functions (§19):** `resolve_backend`; `csr_matvec`; `decay_multiplier`; `update_attention`; `repost_probability`; `select_mutator_indices`; `mutate_claim`; `AttentionEngine.step(t)`.  
**Tests:** attention's half-life matches its parameter, and a post from a hub account outruns one from a small account. The CPU path repeats for a fixed seed.  
**Acceptance:** AttentionTick fires for each bank.

### M04 Media and analyst agents

**Files:** /agents/media.py  
**Implementation:**

* About 5 agents with thresholds on attention, price moves, and losses.  
* A crossed threshold queues a headline job, and the event IDs it cites are required.

**Tests:** no headline appears without a triggering event.  
**Acceptance:** headlines arrive within 2 simulated minutes of a crossing.

## 18.2 People

### M05 Depositor population and insurance

**Files:** /engine/agents/population.py  
**Implementation:**

* Segments are sampled from each preset's shares.  
* Balances are lognormal with a Pareto tail. Bisection solves for the tail so the preset's insured share and top-10 concentration match. `_transfer_mass_array` is a float64 NumPy/CuPy kernel (prefix remaining, deterministic sort/cumsum scatter, no atomics, no host copy; ADR-065). Large-n refine still walks its outer loop on host NumPy (ADR-059). `sample_population` is cached per seed in-process.  
* Insured amount \= min(balance, \$250k × ownership categories).

**Tests:** insured share within ±1 point, and the top-10 total within ±5%.  
**Acceptance:** the SVB-like preset matches E2 and E3.

### M06 Depositor decision model

**Files:** /engine/agents/decide.py (array kernel: CuPy when CUDA is available, NumPy otherwise; ADR-029)  
**Implementation:**

* worry \= σ(w · \[attention × aware, share of contacts who withdrew, insolvency belief, stock return, directive, payroll urgency, uninsured gap\]). Awareness starts at `aware0` (default 0) and follows the SVB-33 hazard (ADR-081). Soft once-mode ratchets decided at finite rate `h` with the closed-form `1-exp(-dt·h)` on every amount path (ADR-084).  
* The insolvency belief updates from public signals (Bayesian update).  
* A depositor withdraws when worry exceeds their threshold and at least k worried contacts, weighted by hub size. The amount depends on segment and the uninsured gap, then is capped so cumulative requests cannot exceed `Depositor mobile share of uninsured deposits (FIT)` times the initial uninsured balance (ADR-067).  
* The same array kernel runs on CuPy if a CUDA device is visible and on NumPy otherwise. CPU results for a fixed seed do not depend on CuPy. k is the ASSUMED row `M06 complex contagion k (ASSUMED)`; the TBD row stays UNSET. gpu-sim integer-minute coarse ticks expand to one-minute clock times (ADR-082).

**Tests:** zero attention produces no run, and withdrawal requires at least k worried contacts. CPU results match for a fixed seed.  
**Acceptance:** the SVB fit passes its tolerance (M27).

### M07 Payments and withdrawal channels

**Files:** /engine/bank/payments.py  
**Implementation:**

* App transfers are instant up to a daily limit.  
* Wires queue first-in, first-out, with a capacity per tick.  
* ACH settles in batches. Branch withdrawals are slow.  
* After the cutoff, queued wires roll over to the next business day.  
* App, wire, and ACH capacities are multiplied by `M07 capacity multiplier (FIT)` (default 1.0).

**Functions (§19):** `ChannelLimits`; `load_payment_params()`; `PaymentState`; `normalize_channel(channel)`; `enqueue_withdrawal(state, amount, channel, limits, flags)`; `settle_tick(state, flags, limits)`; `PaymentsModule.step(t)` publishes `PaymentSettled`.

**Tests:** capacity limits hold, and queued wires survive the weekend.  
**Acceptance:** the Signature-like preset shows a wire backlog at close (E5).

### M41 Named VIP depositor agents

**Files:** /agents/vip.py  
**Implementation:**

* Persona cards plus a 5-event memory and four tools: check exposure, move funds, post, message group.  
* Agents are batched 10 per call with a JSON schema.  
* Each decision is then reduced to a threshold and sensitivity numbers for the futures engine (§7.1).

**Functions (§19):** `load_vip_config`; `persona_cards`; `check_exposure`; `move_funds`; `post`; `message_group`; `validate_vip_batch`; `reduce_decision`; `replay_reduced_rule`; `format_feed_line`; `rules_for_futures`; `VIPModule.step(t)` publishes `DirectiveIssued` and `VIPDecision`.

**Tests:** every reply validates against the schema, and the reduced rule reproduces the decision.  
**Acceptance:** the VIP feed panel is live ("Founder of Kestrel Labs moved \$40M").

## 18.3 Banks and liquidity

### M08 Bank balance sheet and accounting

**Files:** /engine/bank/ledger.py  
**Implementation:**

* A double-entry ledger, so every event posts both debits and credits.  
* CET1 capital is computed with the AOCI toggle (whether unrealized bond losses count against capital).  
* An invariant check runs every tick.

**Tests:** Hypothesis property tests confirm random event sequences always keep the books balanced.  
**Acceptance:** the invariant is never violated in golden runs.

### M09 Securities and interest-rate risk

**Files:** /engine/bank/securities.py  
**Implementation:**

* Bond buckets track book value, coupon, duration, and convexity.  
* Price change ≈ −D·Δy \+ ½·C·Δy², where D is duration, C is convexity, and Δy is the change in yield.  
* A sale realizes the gap between market and book value. An HTM sale can reclassify the rest of that portfolio (toggle).

**Functions (§19):** `BondBucket`; `price_change_fraction(duration, convexity, delta_y)`; `mark_buckets(buckets, delta_y)`; `realize_sale(buckets, asset_class, book_amount, *, htm_reclassify)`; `SecuritiesValuer.step(t)` publishes `LossRealized` on underwater sales.

**Tests:** a \+100 bp rate move on a 6-year-duration bond is about −6%.  
**Acceptance:** the SVB-like preset shows its losses on the 2022 rate path.

### M12 Liquidity and funding facilities

**Files:** /engine/liquidity/ladder.py  
**Implementation:**

* Each funding source has a capacity, haircut, time to cash, operating hours, collateral pool, and stigma.  
* The allocator fills requests from the fastest feasible source first.  
* Collateral is tracked so nothing gets pledged twice.

**Functions (§19):** `FundingSource`; `CollateralLedger`; `default_funding_sources(deposits)`; `cash_shortfall(...)`; `next_liquidity_draw(...)`; `LiquidityLadderModule.step(t)` publishes `LiquidityDrawn`.

**Tests:** the discount window is unavailable without pre-pledged collateral, and FHLB is unavailable after hours.  
**Acceptance:** the ladder panel is live.

### M23 Bank management agent (rule core)

**Files:** /engine/bank/management.py (owned by L5, not by the L2 folder owner; ADR-016)  
**Implementation:**

* **Policy rules:** borrow when cash falls below a threshold, raise capital when capital falls below a threshold, cut the dividend.  
* Success models for capital raises.

**Functions (§19):** `load_management_policy()`; `capital_raise_success_probability(stock_return, attention, capital_gap)`; `roll_capital_raise_success(seed, bank_id, salt, probability)`; `statement_template(preset, action, succeeded)`; `ManagementModule.step(t)` publishes `StatementIssued` and logs actions.

**Tests:** the SVB-like March 8 sequence can be reproduced.  
**Acceptance:** actions are logged.

### M42 Bank CEO agents

**Files:** /agents/ceo.py  
**Implementation:** Grok tool calls map onto M23's action menu, and the ledger rejects any action the bank can't afford.  
**Functions (§19):** `load_ceo_policy()`; `ledger_state_from_events(events)`; `ledger_allows_action(state, action, params)`; `ledger_allows_deposit_rate(state, rate)`; `map_tool_call(payload)`; `offline_ceo_decision(snapshot)`; `reduce_ceo_decision(decision)`; `replay_reduced_rule(rule, ...)`; `choose_ceo_action(snapshot)`; `CEOModule.step(t)` publishes `DepositRateChanged` and `CEOAction` and appends the per-bank CEO log.  
**Tests:** an impossible action is refused.  
**Acceptance:** a per-bank CEO log exists.

## 18.4 Markets

### M14 Per-bank stock: order book and traders

**Files:** /engine/markets/book.py  
**Implementation:**

* Price-time-priority order matching, compiled with Numba.  
* Each trader type submits orders every tick based on its signals.  
* Trading halts when the price moves more than X% within Y minutes.

**Functions (§19):** `match_price_time(side, price, quantity, time_key)`; `market_maker_half_spread(volatility, base, multiplier)`; `public_fair(opening_price, opening_cet1, cet1, opening_rate, policy_rate, duration, attention, delta_4h, news, premium)`; `halt_triggered(times, prices, t, move_frac, window_min)`; `OrderBook.step(t)` publishes `PriceTick` and, when the halt trips, `TradingHalt`.

**Tests:** market makers widen their spreads as volatility rises, and the halt fires.  
**Acceptance:** PriceTick streams.

### M44 Short-seller and activist agents

**Files:** /agents/shorts.py, /agents/llm_cache.py  
**Implementation:** a vulnerability score from public metrics. Above a threshold, the agent publishes a thesis (written by Grok) and adds short orders in M14.  
**Functions (§19):** `load_short_params`; `vulnerability_score(metrics)`; `metrics_from_events(events, preset)`; `thesis_text(...)` (Grok via the SQLite cache; `STAMPEDE_OFFLINE` defaults on and uses a deterministic template); `thesis_reach(seed)`; `short_order_for(...)`; `ShortSellerModule.step(t)` publishes `ThesisPublished` once the score is at or above the ASSUMED threshold and records a short order for M14.  
**Tests:** a thesis raises attention and short interest.  
**Acceptance:** behind a flag (`ModuleMode.REAL`; fake remains the default).

## 18.5 System and policy

### M20 Regulators and supervision triggers

**Files:** /engine/system/regulators.py  
**Implementation:** a rule engine for closures (cash runs out after every feasible source, or capital falls below its threshold), with a timing policy for business-day versus weekend closures.  
**Tests:** with no policy action, the SVB-like bank closes on day 2\.  
**Acceptance:** BankClosed fires.

### M22 Policy lab

**Files:** /engine/policy/levers.py  
**Implementation:** each lever is an event generator with a schedule, a scope, an effect hook, and a cost model. A moral-hazard meter tracks how broad and frequent guarantees are.  
**Tests:** a guarantee stops withdrawals within its scope.  
**Acceptance:** the lab UI shows costs.

## 18.6 Crisis room, futures, and analysis

### M24 Crisis comms room

**Files:** /agents/crisis\_room.py, /agents/llm\_cache.py  
**Implementation:** an asyncio pipeline:

1. The writer drafts statements (offline templates when `STAMPEDE_OFFLINE` is on).  
2. Claims are extracted as JSON.  
3. Numeric claims are checked against the claims ledger in code.  
4. Vague wording goes through the SQLite LLM cache (Grok or local when `STAMPEDE_LLM` is set; rule fallback on validation failure).  
5. M25/M26 integration stays on the event bus; M24 publishes the ranked winner after `FutureCompleted`.  
6. Results are ranked (winner `stmt-crisis-1` in demo parity with the fake).

**Tests:** a statement claiming a guarantee that doesn't exist is rejected.  
**Acceptance:** a run takes under 20 seconds with a warm cache.

### M25 Persona panel

**Files:** /agents/personas.py  
**Implementation:** about 20 archetype cards per bank type. One batched call per statement returns, for each archetype, the change in worry, the change in threshold, and the share who would act now. Replies are validated and cached.  
**Tests:** E11 direction: personalized reassurances beat generic ones.  
**Acceptance:** results return in under 8 seconds.

### M26 Ensemble and futures engine

**Files:** /engine/futures/batch.py  
**Implementation:** tensors shaped \[futures × agents\], running batched versions of the M06 kernels with parameters that vary by future. Ticks are 5 minutes, and each future reduces to a set of metrics. The kernel is `step_m06_kernel` (CuPy when CUDA is visible, NumPy otherwise). `run_serial_reference` is the unbatched main-engine path. `run_batch` returns failure probability, outflow and price bands, and the time-to-failure list. `FuturesRunner.step(t)` publishes one `FutureCompleted` per future.  
**Functions (§19):** `load_futures_params`; `resolve_backend`; `draw_population`; `contact_index`; `lagged_contact_stats`; `step_m06_kernel`; `run_serial_reference`; `run_batch`; `build_future_specs`; `world_intervention`; `future_completed_events`; `FuturesRunner.step(t)`.  
**Tests:** a single future matches the main engine on the same seed, within tolerance. The same seed on the CPU path repeats. Zero attention does not fail. A guarantee or an insured community preset does not run. GPU copies are not used when the backend is CPU.  
**Acceptance: 16 futures × 250k** depositors × one day finish in under 5 seconds on the GPU (§9). CPU CI runs the same kernel on a small shape.

### M27 Calibration and validation harness

**Files:** /calibration/fit.py, /calibration/run_fit.py, /calibration/persist_fit.py, /calibration/nonbank.py, /calibration/run_nonbank_fit.py, /calibration/diagnostics/, /calibration/invariance.py, /engine/gpu_sim.py, /engine/gpu_sim_kernels.py, /engine/gpu_full.py, /engine/gpu_full_io.py  
**Implementation:** the SVB objective is the weighted squared log-error against the E1 targets on **requested** withdrawals by day-one close (0.249) and by next open (0.81) (caps and closure off), averaged over eight shared seeds, with `fixed_softness()` (ADR-083). Three identifiable parameters are fit in log space (worry bias, mobile share, decision-to-request rate; ADR-084, ADR-085); price impact stays at its registry value and the E1 stock ratio is reported, not fitted. Contact weight, feedback gain, and the media awareness rate stay at their registry priors. The other vector slots stay fixed. A batched sweep brackets the tipping point for the CMA-ES start only, then `cma.CMAEvolutionStrategy` runs with `_theta_bounds`, `es.pkl` checkpoints, and IPOP on stagnation (ADR-054, ADR-060). A holdout test then runs on E5. `validate_fitted_parameters` replays SVB with caps on. With `--gpu` and `--gpu-sim`, a generation evaluates candidates as one in-process batch: depositor state is array math, and the rest of the tick stays on the shared module steps (ADR-051). `--gpu-full` also runs attention, the market quote, and payments as array kernels, and a generation is one candidate×seed batch (ADR-056, ADR-058). Packed static state is cached per seed; later generations update only the parameter rows (ADR-061). Stage 1 scores `simulate_requested_fractions` (no per-lane engines or events; ADR-062). `simulate_logs` and CPU `MarketFake` quotes use the same `calibration_uses_scripted_drawdown()` helper whenever the calibration engine is on (ADR-086, ADR-087). Stage 2 calls `simulate_logs` once after stage 1 (caps on, 16 log-spaced multipliers, monotone log interpolation to settled day-one 0.24; ADR-066). Per-depositor arrays are float32 and each layer is one batched float32 SpMM (ADR-063). M05 is sampled once per seed on the host (ADR-059). A separate harness fits the equity-company price path (E13) and holds out crypto and rates (E15, E16) without refitting and without editing the SVB scenarios. Holdout horizons are recorded separately; `--crypto-holdout-minutes` lengthens E15 so `reachable_targets` keeps multi-day days-to-bottom terms (ADR-089). With `--gpu` and `--gpu-sim` that harness batches the same kernel: the equity generation is one batch, and the holdouts are crypto and rates (ADR-052). The flag-off path stays the fake 3-vector scale fit. Each run writes `/calibration/diagnostics`. Unrelated text is a no-op on the shared rumor classifier. Non-bank deposit requests stay inside the ASSUMED 0.1%/day bound or the loss takes a penalty (ADR-053).  
**Tests:** the SVB fit is reproducible from its seed; the three-parameter requested objective is not flat and is monotone-ish through the bracket; a candidate that hits next-open and settled but saturates day-one requests scores worse (`tests/test_svb34_day1_timing.py`); validation with caps on runs; parallel `workers` match sequential eval on a fake objective; `--gpu` falls back on CPU CI. A reduced population matches the CPU engine on NumPy within the ASSUMED relative tolerance (`tests/test_m27_gpu_sim.py`), including the non-bank equity, crypto, and rates price paths. `--gpu-full` matches that NumPy path in `tests/test_m27_gpu_full.py`, including batched versus single-lane seeds. Three generations pack static state once per seed and keep the same-seed scores (`tests/calibration/test_svb09_once.py`). The lean stage-1 path matches event-scored day-one, next-open, and stock-ratio columns within 1e-6 (`tests/engine/test_svb07_lean_stage1.py`). Caps-off `simulate_logs` requested fractions match the lean path within 1e-4 at seed 3060 (`tests/engine/test_svb37_board_fit_scripted.py`). Float32 SpMM spread stays within 1e-5 of the float64 product and requests within 1e-4 (`tests/engine/test_svb08_f32_spmm.py`). Lean, event, and float32 agree at finite `h` (prior 1/600 and 1/60); fused `svb28_once_amount` matches the NumPy fallback (`tests/engine/test_svb35b_paths.py`). Default `run_fit` flags run one seed pass. Stage 2 hits settled 0.24 ± 0.002 in at most two batched calls (`tests/calibration/test_svb21_stage2.py`). The mobile-share cap stays at or under the FIT fraction of initial uninsured and matches the previous amounts at 1.0 (`tests/engine/test_svb24_mobile_share.py`). The non-bank harness is reproducible on a tiny fake-mode run, the loss skips UNSET targets, `shape_loss_terms` sums to `shape_loss`, and the SVB, Signature, and placebo paths stay bit-identical for a fixed seed. A checkpointed non-bank run writes `converged.json` and does not write FITTED rows. Diagnostics files, exact-zero invariance, and the non-bank bound are in `tests/test_m27_diagnostics_constraints.py`. `write_fitted_registry` round-trips a registry that has extra CSV cells. `persist_fit --stage2` sets the same engine env as `run_fit._run --gpu-full` and the solve honors `calibration_uses_scripted_drawdown()` (`tests/calibration/test_persist_fit.py`).  
**Acceptance:** the §13 board fills in automatically. Non-bank targets live in §8.4 and are not part of that board. Unrelated probes leave the event log unchanged. A GameStop-like shock stays inside the daily request bound.
### M28 Scenario builder, presets, and replay

**Files:** /scenarios/  
**Implementation:** a pydantic scenario schema, state snapshots for branching, and branch comparison.  
**Tests:** the same scenario and seed always produce the same event-log hash.  
**Acceptance:** the preset menu and branch compare work.

### M33 Observability, determinism, and demo mode

**Files:** /ops/observability.py, /ops/rng.py, /ops/event_log.py, /ops/llm_cache.py  
**Implementation:**

* Philox4x32-10 draws for seeded runs (CuPy batch on the GPU when available; CPU fallback for CI).  
* Canonical JSON event-log hash (same as the Wave A fake) plus optional Parquet flush.  
* SQLite LLM cache seeded for the two-minute demo script; offline flag when cache is warm.  
* Per-tick metrics overlay: tick time, GPU memory, LLM cache lookup latency.

**Tests:** Philox determinism; Parquet round-trip; offline smoke (`scripts/offline_smoke.py`); real module matches fake log hash on a short run.  
**Acceptance:** the offline smoke test passes.

## 18.7 Interface

### UI specs for M29, M30, M33, and M34

* **M29 crowd screen:**  
  * **Files:** `/ui/crowd/` (`orbs.py`, `layout.py`, `frames.py`, `feed.py`, React shell).  
  * Binary frames: positions are Float32 and computed once; state is a Uint8 per orb per frame. Orb count is the registry render budget, not the depositor count.  
  * The layout is computed once at startup, grouped by network community.  
  * **Tests:** layout determinism; msgpack round-trip; `CrowdFeed` sees `PaymentSettled`; snapshot tracks fake panic scenario; orb block size, majority color, and radius cap.  
  * **Acceptance (shell):** panels render from `CrowdFrame` in fake mode without CuPy. A 1M crowd is 2,048 orbs (488 people each), not 1M drawn points (ADR-044). Playback interpolates orb state on that fixed layout (ADR-042).  
* **M30 market screen:** uPlot price chart and futures fan, plus depth bars. **Shell:** `ui/market` (React), `ui/market/snapshot.py`, `ui/market/server.py`; frames built from fake `M30` + crisis verdicts on `M24`. Acceptance (shell): fake `?fake=1` and pytest `test_m30_market_ui` pass; full UI: 16 futures render smoothly.  
* **M33 observability:** Philox random-number generation on the GPU (seeded), Parquet logs, demo mode, an offline flag, and a metrics overlay. Acceptance: the offline smoke test passes.  
* **M34 guardrails:** /agents/guardrails.py. A blocklist of real bank names and tickers with fuzzy matching, rumor moderation, and live mode off by default. Acceptance: real bank names are rejected.

When a module changes, update its row in §4, its spec in §5, the contracts in §6 if needed, the parameters in §8, and the change log in §16. Bump the version.

# 19\. Function reference

Each row lists the main component a module exposes: what it does, what it takes in, what it produces or changes, the events it publishes and consumes on the event bus (§6.1), and the modules or services it calls. Names describe responsibilities; run(until) and step(t) are the scheduler interface named in §18.1. Every module consumes Tick from M00.

| Module | Component | Purpose | Inputs | Outputs and side effects | Events published / consumed | Calls |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| M00 | Clock and scheduler (run(until), with each module's step(t)) | Advances sim time and fires scheduled and triggered events in a fixed phase order | Calendar table (business hours, wire windows, facility hours, weekends); heap event queue ordered by time, phase, sequence number | Calls each module's step(t) in phase order: macro, information, depositors, payments, banks, markets, system and regulators, UI snapshot | Publishes: Tick and every scheduled event (§6). Consumes: —. | Every module's step(t) |
| M01 | Yield-curve and property-path generator | Sets the external economy | Scenario paths (piecewise-linear schedules) and optional shock events | Nelson–Siegel yield curve mapped onto maturity buckets; policy rate; three property-price series; unemployment | Publishes: RatesChanged. Consumes: Tick. | — |
| M02 | Network builder | Builds the three graph layers that decide who sees what | Preset depositor population | Public feed (preferential attachment), group chats (stochastic block model), professional ties (bipartite); each layer stored in CSR on CuPy when CUDA is available and on NumPy otherwise, with node credibility and channel weight (ADR-027) | Publishes: —. Consumes: Tick. | M05 (population) |
| M03 | Attention propagator and rumor spreader | Moves panic by attention, not tone | Layer matrices (M02), rumors, context notes | Per-node attention updated each tick with decay and CuPy sparse matrix-vector multiplication; reposts drawn on the GPU weighted by credibility; about 1% of reposts sent to the mutator | Publishes: AttentionTick, RumorMutated. Consumes: Tick, RumorPosted, HeadlinePublished, AnalystNote, LiquidityDrawn, TradingHalt, StatementIssued, BankClosed, VIPDecision, ThesisPublished. | M02, Grok rumor mutator (§7) |
| M04 | Media and analyst agent runner | Turns events into coverage | Attention and price moves against agent thresholds | Queues a headline job when a threshold is crossed; headlines and analyst notes cite event IDs | Publishes: HeadlinePublished, AnalystNote. Consumes: Tick, AttentionTick, PriceTick, LossRealized. | Grok headline writer and analyst agent (§7) |
| M05 | Population sampler and insurance calculator | Generates each bank's customer base and insured amounts | Preset segment shares, insured share, top-10 concentration | Segments sampled; lognormal balances with a Pareto tail solved by bisection; insured amount \= min(balance, \$250k × ownership categories); second-bank flag | Publishes: —. Consumes: Tick. | — |
| M06 | Depositor decision kernel (CuPy or NumPy arrays) | Decides whether and how much each depositor withdraws | Attention, share of contacts who withdrew, insolvency belief, stock return, directive, payroll urgency, uninsured gap, persona-panel shifts (M25) | Worry passed through a sigmoid of weighted inputs; insolvency belief updated from public signals; withdrawal when worry exceeds threshold, amount set by segment and uninsured gap | Publishes: WithdrawalRequested. Consumes: Tick, AttentionTick, PriceTick, StatementIssued, PolicyEnacted, DepositRateChanged, DirectiveIssued, VIPDecision. | M25 output |
| M07 | Payment channel processor | Makes withdrawals take real time | WithdrawalRequested events; per-channel capacity, cutoff, daily limit | App transfers instant up to a daily limit; FIFO wire queue with per-tick capacity; ACH in batches; slow branch withdrawals; queued wires roll to the next business day | Publishes: PaymentSettled. Consumes: Tick, WithdrawalRequested, VIPDecision. | M08 (via events) |
| M08 | Double-entry ledger | Keeps each bank's balance sheet and capital | Every withdrawal, asset sale, loss, and liquidity draw | Debit and credit postings; CET1 with the AOCI toggle; invariant check (assets \= liabilities \+ equity) every tick | Publishes: AssetSold, CapitalChanged. Consumes: Tick, PaymentSettled, LiquidityDrawn, LossRealized, DepositRateChanged. | — |
| M09 | Securities valuer | Values bond holdings against the yield curve | Bond buckets (book value, coupon, duration, convexity); yield changes (M01) | Price change ≈ −D·Δy \+ ½·C·Δy²; a sale realizes the market-minus-book gap; HTM sale can reclassify the rest of the portfolio (toggle) | Publishes: LossRealized. Consumes: Tick, RatesChanged, AssetSold. | M08 (losses hit capital) |
| M12 | Liquidity-ladder allocator | Fills cash needs from funding sources | Cash need; each source's capacity, haircut, time to cash, operating hours, collateral pool, stigma; policy events | Fills from the fastest feasible source first; tracks collateral so nothing is pledged twice | Publishes: LiquidityDrawn. Consumes: Tick, PolicyEnacted. | — |
| M14 | Order-book matcher and trader agents | Prices each bank's stock | Trader signals: attention, headlines, capital, rates | Price-time-priority matching compiled with Numba; trader types submit orders each tick; trading halts on a large move within a window (X% within Y minutes, values not specified) | Publishes: PriceTick, TradingHalt. Consumes: Tick, RatesChanged, AttentionTick, HeadlinePublished, AnalystNote, CapitalChanged, ThesisPublished. | — |
| M20 | Supervision and closure rule engine | Decides when a bank closes | Cash after every feasible source; capital against its threshold; policy events | Business-day or weekend closure per timing policy; supervisory status per bank | Publishes: BankClosed, SupervisoryAction. Consumes: Tick, CapitalChanged, PolicyEnacted. | M12 |
| M22 | Policy lever generators | Turns policy choices into scheduled events with costs | Lever, schedule, scope | Each lever is an event generator with an effect hook and a cost model; moral-hazard meter tracks breadth and frequency of guarantees | Publishes: PolicyEnacted. Consumes: Tick. | — |
| M23 | Bank management rule core | Lets the bank act, with consequences | Cash, capital, stock price, attention; CEO actions (M42) | Borrow, raise capital, cut dividend, sell assets, issue statements, call top clients; capital-raise success model; a failed raise becomes a headline | Publishes: StatementIssued. Consumes: Tick, PriceTick, CEOAction. | M12, M24 |
| M24 | Crisis-room pipeline (asyncio) | Finds the true message that saves the bank | Claims ledger, current state | Writer drafts 32 statements; claims extracted as JSON; numeric claims checked in code; Grok checks vague wording; false claims rejected with reason; survivors scored and run forward; results ranked by deposits kept, stock drawdown, survival | Publishes: StatementIssued. Consumes: Tick, FutureCompleted. | M25, M26, Grok crisis writer and truth filter (§7) |
| M25 | Persona-panel scorer | Estimates each depositor type's reaction to a statement | Statement; about 20 archetype cards per bank type | One batched Grok call per statement returning change in worry, change in threshold, share acting now; validated and cached by statement hash and archetype version | Publishes: —. Consumes: Tick. | Grok (§7) |
| M26 | Batched futures runner (`FuturesRunner.step`, `run_batch`, `run_serial_reference`, `step_m06_kernel`) | Turns single runs into ranges | Scenario, seeds, perturbed parameters or interventions | Tensors shaped \[futures × agents\] running batched M06 kernels at 5-minute ticks; per-future metrics; failure probability, outflow and price bands, time-to-failure distributions | Publishes: FutureCompleted. Consumes: Tick. Reads AttentionTick, PriceTick, StatementIssued, PolicyEnacted, DirectiveIssued, and VIPDecision already on the bus when the batch launches. | M06 kernel formulas in `/engine/futures/batch.py` (M06 RawKernel file is still a stub; ADR-032) |
| M27 | Calibration fitter and holdout tester (`simulate_requested_fractions` for stage 1; `simulate_logs` for validation/holdouts and stage 2; `--gpu-full` also arrays attention, market, and payments) | Fits behavioral parameters on SVB and tests elsewhere. A separate harness fits an equity price path and holds out crypto and rates | E1 targets; three identifiable log-space parameters (slots 7,15,16; ADR-085); seed. Non-bank: E13–E16 price-path statistics | Weighted squared log-error on requested outflow by day-one close and by next open (caps off), eight shared seeds, `fixed_softness()`; bracket start then `cma.CMAEvolutionStrategy` (ADR-054, ADR-060, ADR-069, ADR-083, ADR-084). Holdout on E5 with the crypto-heavy preset and the SVB-fitted capacity held fixed; fills the §13 board (including `E1-requested-day1` and the E1 stock ratio). Caps-on validation is a separate entry. Stage 1 keeps attention, decisions, and price feedback on the device (ADR-062). Per-depositor arrays are float32 with one batched SpMM per layer (ADR-063). Opt-in batch keeps depositor state on the array device (ADR-051). Static packs are cached per seed (ADR-061). `--gpu-full` also arrays attention, market, and payments (ADR-056). Non-bank loss reads `PriceTick` only and skips UNSET targets. With both GPU flags the equity generation and the crypto and rates holdouts use that batch (ADR-052). Default engine path unchanged. No FITTED rows from the non-bank CLI. Diagnostics under `/calibration/diagnostics`. Exact-zero invariance and the ASSUMED 0.1%/day request bound (ADR-053). | Publishes: —. Consumes: Tick. | Engine runs; opt-in `/engine/gpu_sim.py` |
| M28 | Scenario schema, snapshots, and branch compare | Builds, replays, and branches scenarios | Scenario controls (bank, rumor and poster, rate path, property shock, time, policies) | Pydantic scenario schema; state snapshots for branching; side-by-side branch comparison; same scenario and seed give the same event-log hash | Publishes: RumorPosted. Consumes: Tick. | M33 (event log) |
| M29 | Crowd-screen renderer | Shows depositors as a point cloud | Binary frames over WebSocket | Registry orb budget (default 2,048 blocks; color is the block majority; uninsured radius capped). Positions are float32, computed once; state is one byte per orb. 2D canvas by default, deck.gl opt-in. Layout grouped by network community; panels for cash gauge, time to failure, tipping point, rumor box | Publishes: RumorPosted. Consumes: Tick, PaymentSettled. | Server frames |
| M30 | Market and crisis-room screen | Shows prices, depth, halts, statements, and the futures fan | PriceTick, TradingHalt, FutureCompleted, crisis-room results | uPlot price chart with event markers and futures fan; depth bars; halt banner; accepted or rejected statements with reasons | Publishes: —. Consumes: Tick, PriceTick, TradingHalt, FutureCompleted. | Server frames |
| M33 | Observability and demo-mode services | Makes runs seeded, logged, and replayable | Seed; run events | Philox RNG on the GPU; Parquet event logs; cached LLM outputs for the demo path; offline flag; metrics overlay (tick time, GPU memory, LLM latency) | Publishes: —. Consumes: Tick. | — |
| M34 | Guardrail filters | Keeps names fictional and content safe | Rumor text, bank names, tickers, statements | Fuzzy-matched blocklist of real bank names and tickers; rumor moderation; live mode off by default | Publishes: —. Consumes: Tick. | — |
| M41 | VIP agent batcher and rule reducer | Runs the top 15 depositors as Grok agents | Persona cards, 5-event memory, four tools (check exposure, move funds, post, message group) | Batched 10 per call with a JSON schema every 30 simulated minutes or on trigger; each decision reduced to a threshold and sensitivities for M26 | Publishes: DirectiveIssued, VIPDecision. Consumes: Tick. | Grok, M26 |
| M42 | CEO strategist | Chooses bank actions through Grok tool calls | Bank state, events hitting the bank | Tool calls mapped onto M23's action menu; unaffordable actions rejected by the ledger; per-bank CEO log with rationale | Publishes: DepositRateChanged, CEOAction. Consumes: Tick. | M23, M08, Grok |
| M44 | Short-seller and activist agents | Publishes theses against vulnerable banks | Public metrics | Vulnerability score; above a threshold, a Grok-written thesis and short orders in M14 | Publishes: ThesisPublished. Consumes: Tick. | M14, Grok |

# 20\. Run and setup

## 20.1 Prerequisites

* Python 3.11 with CuPy for GPU arrays and custom CUDA kernels (RawKernel). PyTorch is the fallback if the CuPy install fails.  
* An NVIDIA CUDA toolchain compatible with CuPy, on a desktop RTX 3060 with 12 GB VRAM (§9).  
* Python libraries from the stack (§18.0): Numba, pydantic, FastAPI with WebSockets and msgpack, xai-sdk, SQLite (LLM cache and event index), Parquet support for event logs, cma or Optuna, pytest, and Hypothesis.  
* An xAI API key for xai-sdk, configured through an environment variable.  
* Node for the React frontend (React, deck.gl, uPlot).

## 20.2 Install

1. Clone the repository. The layout is shown in §21.  
2. Install the CUDA toolchain and CuPy, then confirm the GPU is visible to CuPy.  
3. Install the Python dependencies listed in §18.0 into a Python 3.11 environment.  
4. Set the xAI API key environment variable for xai-sdk.  
5. Install the Node dependencies for /ui.  
6. Check that /params/registry.csv and the presets in /scenarios are present. Every parameter in the engine is read from the registry.

## 20.3 Running

1. Backend: start the FastAPI server. It runs the engine and streams binary msgpack frames over WebSockets at 10 Hz.  
2. Frontend: start the React app in /ui. The dashboard's crowd view (M29) and its market and crisis-room view (M30) connect to the server's WebSocket frames. The dashboard shows the run timeline and compares the run with and without regulator action (M22, M28).  
3. Simulation: pick a preset and seed through the scenario builder (M28). The clock (M00) runs run(until) and calls each module's step(t) in phase order. Demo mode and the offline flag (M33) use cached LLM outputs and work with Wi-Fi off.  
4. Calibration: run the calibration harness (M27, /calibration/fit.py). It fits three identifiable log-space parameters against requested E1 outflow by day-one close and by next open (caps off), then runs a holdout test on E5, a caps-on validation, and fills the §13 board.  
5. Non-bank price paths, only after that SVB fit finishes:

```
python -m calibration.run_nonbank_fit --seed 3060 --generations 40 --population 12 --eval-minutes 390
```

Run it from the repo root. The default engine is fake (`STAMPEDE_CALIBRATION_ENGINE` unset or `fake`). The command reads price paths for the equity, crypto, and rates presets. It does not rewrite the SVB, Signature, or placebo scenarios, it does not write FITTED rows, and it does not write under `calibration/runs/`.

The full-scale run on an RTX 3060 uses the batched kernel. Patience 8 stops a seed after eight generations without a relative improvement:

```
python -m calibration.run_nonbank_fit --gpu-full --seed 3060 --population 12 --eval-minutes 390 --generations 40 --patience 8
```

The SVB E1 fit on the same card:

```
python -m calibration.run_fit --gpu-full --seed 3060 --extra-seeds 3061,3062,3063 --population 12 --eval-minutes 390 --holdout-minutes 780 --workers 1
```

`--gpu-full` sets `STAMPEDE_BACKEND=gpu`, `STAMPEDE_CALIB_GPU_SIM=1`, and `STAMPEDE_CALIB_GPU_FULL=1`, and selects the real engine. Attention, contagion, the market quote, and payments run on CuPy when a device is visible (NumPy otherwise). Each generation scores the equity preset as one batch. Crypto and rates are holdouts on the same kernel and are not refit. The run writes `checkpoint.json` and `converged.json` under `calibration/runs/<run-id>/` and does not write FITTED rows. `--workers` parallelizes the legacy per-candidate engine when `--gpu-sim` is off; it does not split a GPU batch. Offline tests call the same module with a 1-generation, 4-member, 4-minute horizon and with a NumPy batch (ADR-052, ADR-056).

6. Tests: run pytest. The gates in §10.3 (contract tests, accounting invariant, golden scenarios, performance, demo smoke test) run in CI.

## 20.4 QLoRA adapter training (offline)

`training/qlora_train.py` trains a 4-bit QLoRA adapter for Llama 3.1 8B Instruct on the RTX 3060 (12 GB), with no network access. Everything lives under `$STAMPEDE_TRAIN_ROOT` (default `./training_runs`). The base model (HF-format safetensors) is found with no flags: `$STAMPEDE_QLORA_MODEL`, then `hf/Meta-Llama-3.1-8B-Instruct`, then `models/Meta-Llama-3.1-8B-Instruct`. `hf` is also `HF_HOME`; adapters go to `models/adapters/`; the venv is `venvs/qlora` (`training/requirements-qlora.txt`). The script sets `HF_HUB_OFFLINE=1` and loads with `local_files_only=True`. Data is local JSONL of chat `messages` (ending with an assistant turn) or `prompt`/`completion` pairs; loss is on the final assistant turn only.

Run it only when the GPU is free: it refuses to start if another process holds more than 500 MiB on the card, unless `--force`. Smoke run (5 steps on a built-in set; saves an adapter and `stampede_adapter.json`):

```
$STAMPEDE_TRAIN_ROOT/venvs/qlora/bin/python training/qlora_train.py --smoke
```

**Data.** `training/export_jsonl.py` turns a local SQLite table of prompt/response text into JSONL, with a deterministic 5% eval split. It never guesses the schema (`--list` first) and refuses a prompt column that holds only hashes. Use local-model outputs that passed validation, or the repo's own rule-path outputs; check a provider's terms before training on its outputs (ADR-078).

```
python training/export_jsonl.py --db <cache.sqlite> --list
python training/export_jsonl.py --db <cache.sqlite> --table <t> --prompt-col <p> --response-col <r> --out $STAMPEDE_TRAIN_ROOT/data/train.jsonl
```

**Full run** (about 1 epoch; defaults r=16, alpha 32, 1,024 tokens, batch 1 × 16 accumulation, paged 8-bit AdamW, bf16, gradient checkpointing):

```
$STAMPEDE_TRAIN_ROOT/venvs/qlora/bin/python training/qlora_train.py --data $STAMPEDE_TRAIN_ROOT/data/train.jsonl --epochs 1
```

The local server serves a GGUF through llama.cpp (ADR-057), so using an adapter there needs a conversion (llama.cpp `convert_lora_to_gguf.py`, then the server's `--lora` flag); that step isn't automated (ADR-075).

# 21\. Repository layout

The tree is built from the file paths given in the module specs (§18) and the code-area folders (§10.1). Folders for M29, M30, and M33 come from the code-area folders in §10.1. M34 is /agents/guardrails.py. M06 is /engine/agents/decide.py. M23 is /engine/bank/management.py and is owned by L5 (ADR-016).  
`/`  
`├── README.md                      Overview, results, and how to run`  
`├── docs/`  
`│   └── ARCHITECTURE.md            This document, the single source of truth`  
`├── contracts/                     Event and schema contracts (§6)`  
`├── params/`  
`│   └── registry.csv               Parameter registry with sources (§8.1)`  
`├── training/                      Offline QLoRA adapter training for the local model (§20.4)`  
`├── engine/                        Python 3.11 + CuPy simulation engine`  
`│   ├── gpu_sim.py                 M27 batched candidate eval (ADR-051, ADR-058)`  
`│   ├── gpu_sim_kernels.py         M27 fused depositor kernels (ADR-058)`  
`│   ├── gpu_full.py                M27 --gpu-full attention kernels (ADR-056, ADR-063)`  
`│   ├── gpu_full_io.py             M27 --gpu-full market and payments (ADR-056)`  
`│   ├── core/`  
`│   │   └── clock.py               M00 clock, calendar, scheduler`  
`│   ├── macro/                     M01 yield curve, policy rate, property prices`  
`│   ├── agents/`  
`│   │   ├── network.py             M02 multilayer information network`  
`│   │   ├── attention.py           M03 attention and rumor engine`  
`│   │   ├── population.py          M05 depositor population and insurance`  
`│   │   └── decide.py              M06 depositor decision model (CuPy or NumPy; ADR-029)`  
`│   ├── bank/`  
`│   │   ├── payments.py            M07 payments and withdrawal channels`  
`│   │   ├── ledger.py              M08 balance sheet and accounting`  
`│   │   ├── securities.py          M09 securities and interest-rate risk`  
`│   │   └── management.py          M23 bank management rule core (owned by L5)`  
`│   ├── liquidity/`  
`│   │   └── ladder.py              M12 liquidity and funding facilities`  
`│   ├── markets/`  
`│   │   └── book.py                M14 per-bank order book and traders`  
`│   ├── system/`  
`│   │   └── regulators.py          M20 regulators and closure triggers`  
`│   ├── policy/`  
`│   │   └── levers.py              M22 policy lab`  
`│   └── futures/`  
`│       └── batch.py               M26 ensemble and futures engine`  
`├── agents/                        Grok-driven agents (xai-sdk)`  
`│   ├── media.py                   M04 media and analyst agents`  
`│   ├── guardrails.py              M34 safety and content guardrails`  
`│   ├── rumor_input.py             M34 rumor classifier and live credibility (ADR-055)`  
`│   ├── rumor_inbox.py             M34 live rumor caps, merge, and cooldown (ADR-055)`  
`│   ├── vip.py                     M41 named VIP depositor agents`  
`│   ├── ceo.py                     M42 bank CEO agents`  
`│   ├── local_llm.py               Local/Grok backend settings, fake, GPU gate (§7.1)`  
`│   ├── llm_cache.py               LLM cache and STAMPEDE_OFFLINE fallback (§7.1)`  
`│   ├── shorts.py                  M44 short-seller and activist agents`  
`│   ├── crisis_room.py             M24 crisis comms room`  
`│   └── personas.py                M25 persona panel`  
`├── calibration/`  
`│   ├── fit.py                     M27 SVB fit, holdout, and §13 board`  
`│   ├── run_fit.py                 M27 long SVB fit CLI (--gpu, --gpu-sim, --workers)`  
`│   ├── nonbank.py                 M27 non-bank price-path fit and holdout`  
`│   ├── run_nonbank_fit.py         M27 non-bank fit CLI (--gpu, --gpu-sim, --workers)`  
`│   ├── diagnostics/               M27 run diagnostics, summary CLI, and identifiability sweep`  
`│   ├── invariance.py              M27 exact-zero unrelated-input check`  
`│   └── rumor_labels.py            shared classifier labels for fit claims`  
`├── scenarios/                     M28 scenario builder, presets, replay`  
`├── ui/                            React + deck.gl + uPlot frontend`  
`│   ├── crowd/                     M29 crowd screen (layout.py, frames.py, feed.py, React shell)`  
`│   └── market/                    M30 market screen and crisis room`  
`└── ops/                           Reliability and observability (M33; folder per work stream L8)`  
`    ├── observability.py           M33 module, config, metrics`  
`    ├── rng.py                     M33 Philox RNG (GPU path when CuPy is available)`  
`    ├── event_log.py               M33 Parquet logs and replay hash`  
`    └── llm_cache.py               M33 SQLite LLM cache for offline demo`
