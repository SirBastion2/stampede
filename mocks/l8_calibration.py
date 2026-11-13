"""L8 fakes: calibration summary, scenario rumors, replay hash."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel

from contracts.events import PaymentSettled, RumorPosted
from mocks.base import FakeModule
from mocks.params_access import get_param
from mocks.support import claim_allowed, clamp, deposits_base, is_community


class CalibrationFake(FakeModule):
    """M27. Records a reproducible gap versus the first-day outflow. Does not run CMA-ES."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.objective = 0.0
        self.placebo_survives = True
        self.outflow_fraction = 0.0
        self.target_outflow = 0.0

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del t, new_events
        if self.scenario is None:
            return []
        paid = 0.0
        for event in self.log():
            if isinstance(event, PaymentSettled):
                paid += event.amount
        deposits = deposits_base()
        target = get_param("Signature-like: first-day outflow")
        fraction = paid / deposits if deposits else 0.0
        # Log-error is undefined at zero outflow; use a squared gap instead.
        self.objective = (fraction - target) ** 2
        self.outflow_fraction = fraction
        self.target_outflow = target
        self.placebo_survives = is_community(self.scenario.preset) or fraction < target
        return []


class ScenarioFake(FakeModule):
    """M28. Posts each scenario rumor once, dropping claims the blocklist rejects."""

    publishes: ClassVar[tuple[str, ...]] = ("RumorPosted",)

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self._posted: list[int] = []
        self.dropped: list[str] = []

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        del new_events
        if self.scenario is None:
            return []
        out: list[BaseModel] = []
        for index, rumor in enumerate(self.scenario.rumors):
            if index in self._posted:
                continue
            self._posted.append(index)
            if not claim_allowed(rumor.claim):
                self.dropped.append(rumor.id)
                continue
            out.append(
                RumorPosted(
                    t=t,
                    seed=self.seed,
                    bank_id=self.bank_id,
                    cause_id=self.cause(rumor.id),
                    rumor_id=rumor.id,
                    claim=rumor.claim,
                    source_node="hub-public-1",
                    credibility=clamp(rumor.credibility, 0.0, 1.0),
                )
            )
        return out


class ObservabilityFake(FakeModule):
    """M33. Replay hash of the log. No clock and no GPU sample."""

    def __init__(self, module_id: str) -> None:
        super().__init__(module_id)
        self.event_count = 0
        self.log_hash = ""
        self.offline = True
        self._hasher = hashlib.sha256(b"[")
        self._hashed_log: list[BaseModel] | None = None
        self._last: BaseModel | None = None

    def on_tick(self, t: float, new_events: list[BaseModel]) -> list[BaseModel]:
        """sha256 of ``json.dumps([event rows], sort_keys=True, separators=(",", ":"))``.

        SVB-69: hashes only the events since the last tick. That JSON is "[" + rows joined by "," + "]",
        so feeding the same row bytes to a running sha256 gives the same digest as re-dumping the whole
        log, which cost most of the step's growth over a run (diag-16 v2).
        """
        del t, new_events
        log = self.log()
        n = self.event_count
        if log is not self._hashed_log or n > len(log) or (n and log[n - 1] is not self._last):
            self._hasher, self._hashed_log, n = hashlib.sha256(b"["), log, 0
        for index in range(n, len(log)):
            payload = log[index].model_dump(mode="json", by_alias=True)
            if index:
                self._hasher.update(b",")
            self._hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        self.event_count = len(log)
        self._last = log[-1] if log else None
        final = self._hasher.copy()
        final.update(b"]")
        self.log_hash = final.hexdigest()
        return []


def l8_modules() -> dict[str, FakeModule]:
    return {
        "M27": CalibrationFake("M27"),
        "M28": ScenarioFake("M28"),
        "M33": ObservabilityFake("M33"),
    }
