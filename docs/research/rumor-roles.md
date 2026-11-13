# Rumor poster roles and the credibility breakdown

## What changed

**The role table.** The poster row `M29 fake poster credibility (ASSUMED)` grows from 5 to 11 roles. The old five keep their exact values. Every value is **ASSUMED**: none is calibrated to SVB, and none is SOURCED.

**The breakdown.** `agents/rumor_input.py` gains:

- `rumor_roles()`, which parses the role table;
- `credibility_breakdown()`, which returns role base, specificity bonus, model shift, raw, floor, cap and final value;
- a `breakdown` field on `RumorMarketClassification`.

The arithmetic is unchanged (`base + hits·w/4 + shift`, clamped to 0.05–0.9).

**Where the breakdown appears.**

- `POST /api/rumor/interpret` returns `role` and `credibility_breakdown`.
- `POST /api/rumor` returns `credibility` and `credibility_breakdown`.
- New endpoint: `GET /api/rumor/roles`.

**The UI.** The picker lists the 11 roles, weakest to strongest, and shows one line under the rumor: `Credibility: The bank's CEO / CFO 0.85 + specificity + 0.05 = 0.90 (capped)`.

**Fake-mode playback.** Playback now scales effect size by credibility divided by the cap. At the cap this gives 1, so the default VC role plays exactly as before, and weaker roles play weaker. Before this change, the role had no visible effect in the default fake UI.

**What stayed the same.**

- The engine is unchanged. The role id is still `source_node`, and unmapped ids post as the hub.
- `mocks/demo_arc.py` is unchanged. It already reads the same row, and a test checks it agrees for every role.

## How a documented price move maps to a credibility value

It doesn't map directly. In this engine, credibility multiplies the exposure at the posting node and the repost probability (`attention.py` lines 651 and 362–378). It stands for how likely a reader is to believe the claim and pass it on.

A price move after a post mixes four things:

- who posted it (the role);
- how many people saw it (reach);
- how fragile the target already was;
- what else happened that day.

Cookson et al. find that Twitter attention, not negative tone, predicted run-period losses, and that the effect depended on the bank's existing run risk. So no single case gives "credibility = x".

What the cases do support is an **ordering**, built on three questions:

1. Does the source have inside information?
2. Is the source acting with its own money or capital, rather than only talking?
3. Does the channel have verification norms, and does the claim reverse when debunked?

The values below place roles on the existing 0.05–0.9 scale so that this ordering holds. The five old values anchor the scale.

## Evidence used (checked Nov 2–3, 2026)

| Role (id) | Base | Case | What happened | Why it sits here |
|---|---|---|---|---|
| Regular user (`ui-user`) | 0.28 | none (unchanged) | n/a | No inside information, no following. |
| Anonymous group chat (`group-chat-anon`) | 0.40 | Metro Bank, May 2019 | A WhatsApp message told customers to withdraw cash and empty safe-deposit boxes. Branches in west London had queues, and shares fell to an all-time low (about 11% on the day per City A.M.). | Trusted community channel with no named source. It can move local depositors, but the bank's rebuttal held. |
| VIP / celebrity (`vip`) | 0.55 | Ante (2023), *Technological Forecasting and Social Change* 186, event study of 47 Musk crypto tweets | Single tweets moved Bitcoin up about 16.9% or down about 11.8%. | Very high reach but no inside information about a bank. Pull comes mainly from reach, which belongs in a separate reach parameter (see PA-C). |
| Treasury desk (`treasury-desk`) | 0.55 | none (unchanged) | n/a | Professional counterparty. |
| Short-seller research firm (`short-seller`) | 0.60 | Hindenburg report on Adani, Jan 24, 2023 | Adani group lost more than $100B of market value within about a week, and the $2.5B share sale was cancelled (Reuters, AP). | Documented, specific claims; discounted because the firm profits from the fall. |
| Financial journalist / wire (`journalist`) | 0.70 | Hacked @AP account, Apr 23, 2013 | A false tweet briefly erased about $136B of S&P 500 value (Dow about −143 points), recovered within minutes once AP said it was hacked. | A trusted channel moves markets instantly; a debunk reverses it fast. That reversal is what the existing context-note ×0.5 row models. |
| Founder circle (`founder-circle`) | 0.72 | Cookson et al., NBER conference draft | Within the run, high-frequency effects were stronger for tweets by tech-startup users likely to be SVB depositors. | Depositors talking to depositors. |
| Bank CEO / CFO (`bank-ceo`) | 0.85 | Credit Suisse, Oct 2022; SVB, Mar 9, 2023 | Körner's "strong capital" memo was followed by heavy outflows: about CHF 84B by mid-November and CHF 111B in Q4 per the FT. SVB's CEO asked clients to stay calm and the run continued. | Best-informed source. A fear claim from the CEO is close to an admission. The cases also show that **reassurance can backfire**; that is a tone question, see Decision 6. |
| Top shareholder (`major-shareholder`) | 0.85 | Saudi National Bank chair, Mar 15, 2023 (Bloomberg TV) | Said "absolutely not" to more Credit Suisse capital. Shares fell 20–31% intraday by various reports; UBS takeover four days later. | Holder of capital refusing to act. |
| VC partner (`vc-partner`) | 0.90 | Founders Fund, Coatue, USV and others, Mar 9, 2023 | Advised portfolio companies to move money out. SVB's attempted withdrawals reached about $42B that day (California DFPI order), leaving a negative cash balance of about $958M. | Directs other people's money; advice turns into coordinated action. At the cap. |
| Regulator / central bank (`regulator`) | 0.90 | **none verified in this pass** | n/a | Highest verification norm and legal authority. ASSUMED at the cap; needs a case before anyone calls it sourced. |

### Sources

- Cookson, Fox, Gil-Bazo, Imbet, Schiller, "Social Media as a Bank Run Catalyst". SSRN 4422754, published in *JFE* 176 (2026), doi 10.1016/j.jfineco.2025.104218. Note that the reported effect size differs between versions: 4.3 pp in the SSRN abstract, 6.6 pp in the NBER draft.
- Axios, 2023-03-14, the story behind Founders Fund and the SVB run.
- Bloomberg / InvestmentNews and The National, 2023-03-15, Credit Suisse and the Saudi National Bank.
- Fortune and Fox Business, Oct–Nov 2022, Credit Suisse outflows.
- Reuters and AP, 2023-02-02, Adani.
- Christian Science Monitor and Bloomberg, 2013-04-23, the AP hack.
- City A.M. and ITV, 2019-05-13, Metro Bank.
- Ante, L. (2023), RePEc tefoso v186.

### Not used

From Grok's pointer list, these were not used:

- **Do Kwon / LUNA.** It is the founder of the asset, not a rumor about a bank.
- **Powell, March 2023.** A macro event, not a rumor.
- **Fake @SECGov post, Jan 2024.** Not verified in this pass.

No sell-side analyst or rating-agency case was verified, so no analyst role was added. Moody's March 8, 2023 action on SVB is a candidate for the next round.

## Not changed

- Engine: `attention.py` and `network.py` are untouched.
- SVB FITTED rows, the calibration scenarios, and the non-bank preset credibilities.
- `scenarios/builder.py`.
- Offline credibility, strength, effect size and read-as text for every text and the old five roles. A golden table of 196 cases from `77db0e2` is in `tests/fixtures/rumor_classification_77db0e2.json`.
