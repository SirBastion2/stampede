"""VF-2: the --kernel-only diagnostic (only the depositor kernel on CuPy) and the record of the --unfused result."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _parity():
    spec = importlib.util.spec_from_file_location("gpu_parity_vf2", "scripts/gpu_parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kernel_only_keeps_the_other_modules_on_numpy() -> None:
    gp = _parity()
    assert gp.batched_backend("cupy", False) == "gpu"     # every array path on CuPy, as the search runs
    assert gp.batched_backend("auto", False) == "gpu"
    assert gp.batched_backend("cupy", True) == "cpu"      # M02, M03, M05 on NumPy; the kernel follows --xp cupy
    assert gp.batched_backend("numpy", False) == "cpu"


def test_kernel_only_needs_the_cupy_kernel(tmp_path: Path) -> None:
    gp = _parity()
    with pytest.raises(SystemExit, match="needs --xp cupy"):
        gp.main(["--seeds", "1", "--minutes", "1", "--workers", "1", "--xp", "numpy", "--kernel-only",
                 "--out", str(tmp_path / "out.json")])


def test_the_unfused_result_is_recorded_not_the_old_guess() -> None:
    src = Path("scripts/gpu_parity.py").read_text(encoding="utf-8")
    assert "the fused kernels are not" in src and '"kernel_only": bool(args.kernel_only)' in src
    assert "The likely cause is CuPy's fused float32 kernels" not in src
    assert "--unfused" in src and "--kernel-only" in src
