"""Fingerprint of real M03 output for the flags-off guard (ADR-093).

Uses only APIs that exist at b674fa7, so the same function produced
``tests/fixtures/m03_flags_off_b674fa7.json`` on the base tree.
"""
from __future__ import annotations
import dataclasses
from contracts.events import AnalystNote, AttentionTick, RumorMutated, RumorPosted
from contracts.schemas import Scenario
from engine.agents.attention import AttentionEngine
from mocks.bus import EventBus

FEAR = "Kestrel Bank is out of cash and may delay uninsured wires, pull your money now"
CALM = "Kestrel Bank is well capitalized and stable"
SOURCES = ("hub-public-1", "small-account", "node-2", "vc-partner", "journalist", "ui-user", "bank-ceo", "not-a-node")
SEEDS = (1, 5, 11)
STEPS = 24

def _post(seed, rid, source, claim, cred, t):
    return RumorPosted(t=t, seed=seed, bank_id="bank-fvb", cause_id="fp", rumor_id=rid, claim=claim, source_node=source, credibility=cred)

def run_case(seed, source, claim, mutate):
    engine = AttentionEngine("M03")
    engine.attach(EventBus(), Scenario(preset="tech-concentrated", seed=seed))
    if mutate:
        engine.params = dataclasses.replace(engine.params, sample_fraction=1.0)
    sums, levels, mutations = [], [], []
    engine.inbox.append(_post(seed, "r1", source, claim, 0.9, 0.0))
    for tick in range(STEPS):
        if tick == 3:
            engine.inbox.append(_post(seed, "r2", source, claim, 0.5, float(tick)))
        if tick == 6:
            engine.inbox.append(AnalystNote(t=float(tick), seed=seed, bank_id="bank-fvb", cause_id="n", text="context", cited_event_ids=["fp"]))
        for event in engine.step(float(tick)):
            if isinstance(event, AttentionTick):
                levels.append([event.level, event.delta_4h])
            elif isinstance(event, RumorMutated):
                mutations.append([tick, event.rumor_id, event.claim, event.exaggeration_score])
        sums.append(float(engine.attention.sum()))
    return {"seed": seed, "source": source, "claim": claim, "mutate": mutate,
            "sums": sums, "levels": levels, "mutations": mutations,
            "final": [float(x) for x in engine.attention]}

def fingerprint():
    return [run_case(s, src, c, m) for s in SEEDS for src in SOURCES for c in (FEAR, CALM) for m in (False, True)]
