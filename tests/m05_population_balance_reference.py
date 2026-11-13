"""Pure-Python reference for M05 balance refinement (pre-vectorization baseline)."""

from __future__ import annotations

import math


def caps_for_segments(segments: list[str], insurance_limit: float) -> list[float]:
    ownership = {
        "startup": 1,
        "vc": 1,
        "founder": 1,
        "treasurer": 1,
        "retail": 1,
        "municipal": 2,
        "crypto": 1,
    }
    return [insurance_limit * ownership.get(segment, 1) for segment in segments]


def scale_to_total(balances: list[float], total: float) -> list[float]:
    current = sum(balances)
    if current <= 0.0:
        return balances
    factor = total / current
    return [value * factor for value in balances]


def insured_total(balances: list[float], caps: list[float]) -> float:
    return sum(min(balance, cap) for balance, cap in zip(balances, caps, strict=True))


def transfer_mass(
    balances: list[float],
    caps: list[float],
    donors: list[int],
    receivers: list[int],
    amount: float,
) -> None:
    if amount <= 0.0 or not donors or not receivers:
        return
    remaining = amount
    for index in donors:
        if remaining <= 0.0:
            break
        room = balances[index] - 1.0
        take = min(room, remaining / max(len(donors), 1))
        if take <= 0.0:
            continue
        balances[index] -= take
        remaining -= take
    if remaining <= 0.0:
        return
    share = remaining / len(receivers)
    for index in receivers:
        balances[index] += share


def refine_balances(
    balances: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
) -> list[float]:
    caps = caps_for_segments(segments, insurance_limit)
    current = [max(1.0, value) for value in balances]
    count = len(current)
    if count < 10:
        return scale_to_total(current, total_deposits)

    for _ in range(96):
        current = scale_to_total(current, total_deposits)
        ranked = sorted(range(count), key=lambda index: current[index])
        top_idx = ranked[-10:]
        top_sum = sum(current[index] for index in top_idx)
        if top_sum > 0.0:
            factor = target_top10 / top_sum
            for index in top_idx:
                current[index] *= factor
        current = scale_to_total(current, total_deposits)

        insured_total_value = insured_total(current, caps)
        insured_share = insured_total_value / total_deposits
        gap = (target_insured_share - insured_share) * total_deposits
        if abs(gap) < total_deposits * 0.0005:
            break
        shift = min(abs(gap) * 0.5, total_deposits * 0.03)
        if gap > 0.0:
            receivers = [index for index in ranked if current[index] < caps[index]]
            donors = ranked[count // 3 :]
            transfer_mass(current, caps, donors, receivers, shift)
        else:
            donors = [index for index in ranked if current[index] <= caps[index] * 0.98]
            receivers = ranked[count // 2 :]
            transfer_mass(current, caps, donors, receivers, shift)

    return [max(1.0, value) for value in scale_to_total(current, total_deposits)]


def solve_balances(
    z_scores: list[float],
    segments: list[str],
    total_deposits: float,
    insurance_limit: float,
    target_insured_share: float,
    target_top10: float,
    *,
    lognormal_mu: float = 11.0,
    lognormal_sigma: float = 1.35,
) -> list[float]:
    raw = [(max(value, 1.0) ** 1.15) for value in [math.exp(lognormal_mu + lognormal_sigma * z) for z in z_scores]]
    indexed = sorted(enumerate(raw), key=lambda item: item[1], reverse=True)
    boosted = list(raw)
    multiplier = math.exp(math.log(12.0))
    for rank, (index, _value) in enumerate(indexed[:10]):
        weight = 1.0 + (10 - rank) / 10.0
        boosted[index] *= 1.0 + (multiplier - 1.0) * weight
    seed = scale_to_total(boosted, total_deposits)
    return refine_balances(
        seed,
        segments,
        total_deposits,
        insurance_limit,
        target_insured_share,
        target_top10,
    )
