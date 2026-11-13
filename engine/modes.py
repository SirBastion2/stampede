"""Per-module implementation mode.

off: the module is left out of the engine and its events never appear.
fake: deterministic stand-in on the in-memory bus (the default).
real: production module code (M00, M01, M02, M03, M04, M05, M06, M07, M08,
M09, M12, M14, M20, M22, M23, M24, M25, M26, M27, M28, M33, M34, M41, M42,
M44);
others still raise NotImplementedError.
"""

from enum import StrEnum
from typing import Self


class ModuleMode(StrEnum):
    OFF = "off"
    FAKE = "fake"
    REAL = "real"

    @classmethod
    def parse(cls, value: str | None) -> Self:
        if value is None:
            return cls.FAKE
        return cls(value.lower())
