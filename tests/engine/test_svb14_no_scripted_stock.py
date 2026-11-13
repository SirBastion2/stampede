"""SVB-14: calibration stock path without the demo's scripted drawdown."""

from __future__ import annotations

import numpy as np

from engine.gpu_full_io import price_quote_batch
from mocks.demo_arc import price_quote
from mocks.support import OPENING_PRICE

_TIMES = (0.0, 12.0, 60.0, 390.0, 780.0)
_STRENGTH = np.array([0.0, 0.4, 0.9, 1.5])
_ZERO_ATTN = np.zeros(_STRENGTH.shape)


def test_unscripted_zero_attention_is_opening_price() -> None:
    for t in _TIMES:
        quotes = price_quote_batch(np, t, _STRENGTH, _ZERO_ATTN, scripted=False)
        assert np.array_equal(quotes, np.full(_STRENGTH.shape, float(OPENING_PRICE)))
        for index in range(len(_STRENGTH)):
            assert price_quote(
                t, float(_STRENGTH[index]), float(_ZERO_ATTN[index]), scripted=False
            ) == float(OPENING_PRICE)


def test_scripted_true_matches_old_price_quote() -> None:
    attn = np.array([0.0, 0.1, 0.5, 0.8])
    for t in _TIMES:
        quotes = price_quote_batch(np, t, _STRENGTH, attn, scripted=True)
        defaulted = price_quote_batch(np, t, _STRENGTH, attn)
        for index in range(len(_STRENGTH)):
            expected = price_quote(t, float(_STRENGTH[index]), float(attn[index]))
            assert quotes[index] == expected
            assert defaulted[index] == expected
