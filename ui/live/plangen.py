"""AU-2: the search writes its own plans (Sebastian, Nov 5: "a lot of automatic things, so let's stick with that").

Every plan is built from the tools and triggers below, alone or as a ladder (a smaller step, then a bigger one, at a
later trigger), optionally with the bonds pledged before the day. Tools that change what the bank can pay or what is
asked in a fixed way (guarantee, withdrawal limit, the Fed's line, big-bank deposits, pledged bonds) are worked out
from the no-response runs only where the required path is known (``ui/live/useful.py``). Cash conditions
after an earlier action require their own simulated cash path and are withheld from ranking until checked.
The generator retains those policies for coverage reporting. Tools that change how depositors behave
(a statement, a deposit-rate rise) need real simulations:
``messaging_plans`` lists the few the pipeline simulates on top of the best ladders, within a budget.
"""

from __future__ import annotations

from itertools import product

REQ = (0.005, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2)
CASH = (0.75, 0.5)
TRIGGERS = tuple(("requested", x) for x in REQ) + tuple(("cash", x) for x in CASH)
SINGLE = ("guarantee", "credit-line", "withdrawal-limit", "big-bank-deposits")
# a ladder is a smaller step first, then a bigger one: (first, then)
LADDERS = (("credit-line", "guarantee"), ("big-bank-deposits", "guarantee"), ("credit-line", "big-bank-deposits"),
           ("withdrawal-limit", "guarantee"))
LABEL = {"guarantee": "Deposit guarantee", "credit-line": "Fed lending program", "withdrawal-limit": "Withdrawal limit",
         "big-bank-deposits": "Deposits from big banks", "prepared": "Bonds pledged in advance",
         "statement": "Reassuring statement", "rate-rise": "Deposit rate raised to 8%"}


def _wid(w) -> str:
    if w == 0:
        return "0"
    return f"{'req' if w[0] == 'requested' else 'cash'}{w[1]:g}"


def _when(w) -> str:
    if w == 0:
        return ""
    return f"once withdrawals pass {w[1] * 100:g}% of deposits" if w[0] == "requested" else f"once cash falls below {w[1] * 100:g}%"


def _order(w) -> float:
    """When a trigger tends to fire: cash below 75% comes near 2-5% asked, below 50% near 5-8% (the UI-4 runs)."""
    if w == 0:
        return -1.0
    return w[1] if w[0] == "requested" else {0.75: 0.03, 0.5: 0.06}.get(w[1], 0.1)


def plan_id(steps) -> str:
    return "+".join(f"{n}@{_wid(w)}" for n, w in steps) or "none"


def plan_label(steps) -> str:
    if not steps:
        return "Nothing done"
    return " + ".join(LABEL[n] if w == 0 else f"{LABEL[n]} {_when(w)}" for n, w in steps)


def generate() -> list[dict]:
    """Every single tool at every trigger, every ladder with its second step later than its first, each with and
    without the bonds pledged in advance; plus the pledge alone."""
    plans: list[list] = [[("prepared", 0)]]
    for tool, w in product(SINGLE, TRIGGERS):
        plans.append([(tool, w)])
    for (a, b), w1, w2 in product(LADDERS, TRIGGERS, TRIGGERS):
        if _order(w2) > _order(w1):
            plans.append([(a, w1), (b, w2)])
    with_pledge = [[("prepared", 0)] + p for p in plans[1:] if not any(n == "withdrawal-limit" for n, _ in p)]
    out, seen = [], set()
    for p in plans + with_pledge:
        pid = plan_id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"plan": pid, "label": plan_label(p), "steps": [[n, list(w) if w != 0 else 0] for n, w in p],
                    "generated": True})
    return out


def messaging_plans(best: list[list], budget_plans: int = 4) -> list[list]:
    """AU-2: the behaviour-changing variants worth simulating: a statement, a rate rise, or both, once the run is a
    threat (5% asked), added in front of the best ladders found by the shortcut."""
    out = []
    for steps in best[:max(0, budget_plans)]:
        for extra in ([("statement", ("requested", 0.05))], [("rate-rise", ("requested", 0.05))],
                      [("statement", ("requested", 0.05)), ("rate-rise", ("requested", 0.05))]):
            names = {n for n, _ in steps}
            if not names & {"statement", "rate-rise"}:
                out.append(list(steps) + extra)
    return out
