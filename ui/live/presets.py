"""P-A: the live demo's preset table (diag-09/11 validated these posts on the 3060 desktop).

A preset is a list of posts, each at a simulated minute. ``guarantee`` marks the regulator's
"deposits are fully insured" chip: with ``M29 live insured chip guarantee`` on, it enacts the
deposit guarantee (``PolicyEnacted(guarantee_deposits)``; M06 drops the uninsured gap, SVB's
Mar 12 2023 backstop) instead of spreading as a rumor. The lexicon reads that sentence as
neutral, so as a rumor it would add attention (diag-11). Only this chip enacts the guarantee;
calming text typed by a viewer keeps SVB-59's weak cure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresetPost:
    minute: int
    claim: str
    poster: str
    credibility: float
    guarantee: bool = False
    lever: str = ""  # PA-A: a response other than the guarantee (PolicyEnacted lever); "" for a rumor
    response: str = ""  # PA-A: the response's name, for labels and timing


@dataclass(frozen=True, slots=True)
class LivePreset:
    name: str
    label: str
    posts: tuple[PresetPost, ...]
    note: str = ""
    expect: str = ""  # UI-2: what the viewer should expect (diag-17, 10 seeds)


LOST_BONDS = PresetPost(0, "Copperleaf lost $1.8B on bonds and is raising capital", "vc-partner", 0.90)
# Sebastian (Nov 4): stays at 0.65, about half of lost-bonds at minute 390 with the fitted feedback (SVB-65).
FROZEN_BRANCHES = PresetPost(0, "Copperleaf has frozen withdrawals at two branches", "ui-user", 0.65)
INSURED_CHIP = PresetPost(240, "Regulators say all Copperleaf deposits are fully insured", "regulator", 0.90,
                          guarantee=True)
STADIUM_TEXT = "The new stadium in Quayside opens Friday"

LIVE_PRESETS: dict[str, LivePreset] = {
    "lost-bonds": LivePreset("lost-bonds", "Lost $1.8B on bonds", (LOST_BONDS,), expect=(
        "The 2023 replay (fitted). A VC partner posts it at 9:30 AM. Expect about 25-27% of deposits asked for by "
        "the 4 PM close, trading halted around 3:45-4:00 PM, the bank out of cash around 1:30-2:00 AM, and closed at "
        "the next morning's open.")),
    "frozen-branches": LivePreset("frozen-branches", "Frozen withdrawals at two branches", (FROZEN_BRANCHES,), expect=(
        "A regular customer posts it. Expect about half the bond rumor's pace on day one (12-14% by the close), "
        "but about 80% of deposits asked for by morning.")),
    "lost-bonds+insured": LivePreset(
        "lost-bonds+insured", "Lost $1.8B on bonds, then deposits guaranteed", (LOST_BONDS, INSURED_CHIP),
        note="The guarantee lands at minute 240 (1:30 PM on day one). In 2023 it came two days later, after the bank had closed.",
        expect="The bond rumor, then at 1:30 PM regulators guarantee every deposit. Expect withdrawals to stop at "
               "3-5% of deposits, and the bank to stay open.",
    ),
    "stadium": LivePreset(
        "stadium", "Stadium opens Friday", (),
        note=f"Irrelevant ({STADIUM_TEXT!r}): nothing is posted, so this is the no-rumor baseline.",
        expect="Nothing about the bank. Expect under 0.05% of deposits asked for: the engine's background noise.",
    ),
}


@dataclass(frozen=True, slots=True)
class Response:
    """PA-A: a countermeasure, enacted at ``minute`` (240 = 1:30 PM, as the insured preset)."""

    name: str
    lever: str
    label: str
    note: str
    minute: int = 240


# PA-A (Sebastian, Nov 4): this order; the limit uses the engine's existing lever as it is (assumed size,
# labelled scenario). CM-2 (Sebastian, Nov 5): the credit line is on the 2023 BTFP terms (M12 rows, live opt-in).
RESPONSES: dict[str, Response] = {
    "guarantee": Response("guarantee", "guarantee_deposits", "Deposit guarantee",
                          "All deposits count as insured from this minute, so no one has an uninsured reason to run. "
                          "Depositors who already asked for their money stay withdrawn."),
    "credit-line": Response("credit-line", "emergency_lending", "Emergency credit line",
                            "The 2023 emergency terms: once its own cash is gone, the bank borrows whatever it lacks "
                            "against its bonds at face value, up to all of them (about $119B), at most 1% of deposits "
                            "a minute (assumed). Depositors do not see it."),
    "withdrawal-limit": Response("withdrawal-limit", "cap_withdrawals", "Withdrawal limit",
                                 "Each depositor can get 30% of what they ask for before the next open; the rest waits "
                                 "(assumed size)."),
}


# AU-1: more steps a plan can use (the search's plan generator and live runs from the wall). Kept out of RESPONSES,
# which the cards, the batch table and the UI-3 search modes read.
TOOLS: dict[str, Response] = {
    "prepared": Response("prepared", "pledge_collateral", "Bonds pledged in advance",
                         "Before the day, the bank pledged its bonds at the Fed's discount window: it can borrow up to "
                         "their market value less the Fed's margin (assumed 95%).", minute=0),
    "big-bank-deposits": Response("big-bank-deposits", "big_bank_deposits", "Deposits from big banks",
                                  "Other banks deposit $30B (as 11 banks did for First Republic on Mar 16 2023)."),
    "statement": Response("statement", "statement", "Reassuring statement",
                          "The bank tells all depositors their money is safe. Lowers worry and raises the bar to "
                          "withdraw, but also draws attention (assumed effects)."),
    "rate-rise": Response("rate-rise", "deposit_rate", "Deposit rate raised to 8%",
                          "The bank pays 8% on deposits to hold them (assumed effect: worry falls by 0.8 x the rate)."),
}
STEPS: dict[str, Response] = {**RESPONSES, **TOOLS}


def response_post(name: str, minute: int) -> PresetPost:
    response = STEPS[name]
    if response.lever == "guarantee_deposits":
        return PresetPost(minute, INSURED_CHIP.claim, INSURED_CHIP.poster, INSURED_CHIP.credibility, guarantee=True,
                          response=name)
    return PresetPost(minute, "", "regulator", 0.0, lever=response.lever, response=name)


def response_table() -> list[dict[str, object]]:
    return [{"name": r.name, "label": r.label, "note": r.note, "minute": r.minute, "lever": r.lever}
            for r in RESPONSES.values()]


def preset_table() -> list[dict[str, object]]:
    """JSON-ready table for the UI (chips) and /api/presets."""
    return [
        {"name": p.name, "label": p.label, "note": p.note, "expect": p.expect,
         "posts": [{"minute": q.minute, "claim": q.claim, "poster": q.poster, "credibility": q.credibility,
                    "guarantee": q.guarantee} for q in p.posts]}
        for p in LIVE_PRESETS.values()
    ]
