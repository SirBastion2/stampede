"""Calibration rumor claims and classifier labels (read-only; does not change fit vectors)."""

from __future__ import annotations

from agents.rumor_input import RumorMarketClassification, classify_rumor_market

SVB_FIT_CLAIM = "Uninsured deposits are leaving Fictional Valley Bank"
SIGNATURE_FIT_CLAIM = "Depositors question Fictional Harbor Bank reserves"
PLACEBO_FIT_CLAIM = "Fictional Community Bank is well capitalized"

CALIBRATION_RUMOR_CLAIMS: tuple[str, ...] = (
    SVB_FIT_CLAIM,
    SIGNATURE_FIT_CLAIM,
    PLACEBO_FIT_CLAIM,
)


def calibration_rumor_classification(claim: str) -> RumorMarketClassification:
    """Label a calibration scenario claim using the same lexicon as the UI rumor box."""
    return classify_rumor_market(claim)
