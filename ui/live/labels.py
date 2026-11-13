"""PA-C: honesty labels and the limits screen (Sebastian approved the copy, Nov 4).

Only the 2023 replay is fitted: the lost-bonds preset with no response and no typed rumor, whose run
reaches 26% of deposits requested by the close and 82% by the next morning (diag-11 v2 / diag-14,
seed 42) against about 25% and 81% in 2023. Every other run is a scenario. No real bank names and no
forecasting words appear on screen.
"""

from __future__ import annotations

FITTED_PRESET = "lost-bonds"
FITTED_NAME = "2023 replay"
FITTED_BADGE = "Calibrated to a real 2023 bank run"
SCENARIO_BADGE = "What-if on the same calibrated model"
FREE_TEXT_OFF = "Free-text rumors are off. Pick a preset rumor."

LIMITS: tuple[tuple[str, str], ...] = (
    ("What this is",
     "A stress test. Pick a rumor; the engine tries the responses over many simulated days and shows which plan "
     "kept the bank open, so responses can be compared."),
    ("What it is not",
     "Not a forecaster. It does not say what any real bank will do."),
    ("What is fitted",
     "The 2023 replay. Its depositor model was tuned to a real 2023 run: the replay reaches 26% of deposits "
     "requested by the close and 82% by the next morning, against about 25% and 81% in 2023."),
    ("What is a scenario",
     "Everything else: other rumors, responses and their sizes. These use the same depositor model with settings "
     "we chose, and they have not been checked against real events."),
    ("Known gaps",
     "The share price follows withdrawals; it is not a market model. Depositors who withdraw do not come back. "
     "Only depositors with uninsured money run, about half of them: insured depositors never run, which held in "
     "March 2023 but not in every run. The bank's closure waits for the next morning, as in 2023. The guarantee "
     "and the credit line follow 2023's terms; the withdrawal limit's size and moral hazard are assumed."),
)


def run_label(preset: str | None, preset_label: str | None, responses: tuple[str, ...], free_text: bool) -> dict:
    """``fitted`` only for the 2023 replay; everything else, including no run yet, is a scenario."""
    fitted = preset == FITTED_PRESET and not responses and not free_text
    return {
        "kind": "fitted" if fitted else "scenario",
        "badge": FITTED_BADGE if fitted else SCENARIO_BADGE,
        "name": FITTED_NAME if fitted else (preset_label or "Live run"),
        "preset": preset,
        "responses": list(responses),
    }


def limits_screen() -> dict:
    return {"title": "What this is, and what it isn't",
            "sections": [{"heading": h, "body": b} for h, b in LIMITS],
            "badges": {"fitted": FITTED_BADGE, "scenario": SCENARIO_BADGE}}
