# Rumor role reach in M03 (flags off)

## What changed

**How the role reaches M03.** No contract change. The live UI already sends the role id as `RumorPosted.source_node`: `ui/live/session.py` passes `source_node=poster`. Until now, M03 posted every role id from the hub. SVB-41 adds two independent flags. Both are **off**.

| Flag | When on | Map row |
|---|---|---|
| `M03 role exposure multiplier enabled` | Post impulse = credibility × `M03 rumor exposure` × multiplier[role]. Reposts are unchanged. | `M03 role exposure multipliers` |
| `M03 role source node enabled` | A role posts from multiplier map's node: `hub-public-1`, `node-2` or `small-account`. | `M03 role source nodes` |

A source that is not a role id is never touched. That covers `hub-public-1`, M28 `poster:` notes and scenario shocks.

**Code changes.**

- New `RumorAmplifierParams` and `load_rumor_amplifiers()`, kept separate from `AttentionParams` so the GPU mirror's inputs don't change.
- New `_source_index()` and `_post_scale()` in `AttentionEngine`.
- With a flag off, the multiply is skipped entirely, not done with 1.0, so logs are byte-identical.

## Option (a), not (b)

Role nodes are named nodes of the existing 16-node fallback graph. Wiring the M02 network into M03 (option b) would change every real-mode log and the SVB seeded run, and needs its own change.

What the fallback graph offers, as one-tick spread mass (public 1.0, group 0.6, professional 0.4):

| Node | Mass | Who posts here |
|---|---|---|
| `hub-public-1` | 14.6 | Public broadcast: journalist, VIP, regulator, short-seller, bank CEO, top shareholder. |
| `node-2` | 3.4 | The node that even-numbered nodes hear on the professional layer. M02 builds that layer between treasurers and advisors (media plus a sample of VC and founder nodes), so this is where VC, founder circle and treasury desk post. |
| `small-account` | 1.6 | One public follower: regular user, anonymous group chat. |

**This deviates from the original proposal**, which suggested journalist → a media node and VC → a VC-block node. In the fallback graph, a media outlet's reach is the public hub. A VC's channel is the advisor tie to treasurers, which is `node-2`. A ring node would give a VC 0.6 mass, less than a regular user.

**Consequence when this flag is on:** the default role `vc-partner` posts from `node-2`, which has about a quarter of the hub's one-tick mass. Its 0.9 credibility does the rest.

## Multiplier values (all ASSUMED, ordinal)

The multiplier stands for reach *outside* the simulated graph: followers and media pickup. The ordering follows the SVB-39 cases.

| Multiplier | Roles | Case |
|---|---|---|
| 3.0 | VIP | Ante 2023: Musk tweets, tens of millions of followers. |
| 2.0 | Journalist, regulator | The hacked AP account had about 1.9M followers; regulator is wide pickup by assumption. |
| 1.5 | Short-seller, bank CEO, top shareholder | Their statements travel through media coverage: Adani 2023, the SNB chair on Bloomberg TV. |
| 1.0 | VC partner, founder circle, treasury desk, anonymous group chat | Direct and group channels. Their pull in the cases is credibility and action (Founders Fund; Cookson et al.: networked depositors), not extra reach. |
| 0.5 | Regular user | Few followers. |

**Don't turn both flags on.** With both on, reach counts twice: once through graph position, once through the multiplier.

## GPU path

`engine/gpu_sim.py` and `engine/gpu_full.py` mirror M03 for batched calibration and do not read these flags. `AttentionEngine.__init__` raises `RegistryError` if any flag is on while `STAMPEDE_CALIB_GPU_SIM=1` or `STAMPEDE_CALIB_GPU_FULL=1`. This turns a silent CPU/GPU mismatch into an error.

The guard does not cover the batched kernels themselves, because they don't construct `AttentionEngine`. Those files were not bundled. If the flags ever go on for calibration, the kernels need the same terms.

## Verified

- A 96-run fingerprint taken at b674fa7 (`tests/fixtures/m03_flags_off_b674fa7.json`) is reproduced bit-for-bit with the flags off.
- The 10 of 11 M03 tests that don't need `run_scenario` pass unchanged.
- The 11th M03 test and the seeded-hash tests need the engine registry; the executor runs them.

## Not changed

- Contracts, `ui/`, `mocks/`, calibration, the GPU kernels, and any FITTED row.
