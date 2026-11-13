"""Tests for /params/registry.csv and loader (§8.1)."""

from __future__ import annotations

from params.loader import (
    RegistryError,
    get_parameter,
    list_by_module,
    list_by_status,
    load_registry,
)


def test_every_row_has_name_and_status() -> None:
    for row in load_registry():
        assert row.parameter
        assert row.status


def test_no_duplicate_parameter_names() -> None:
    names = [row.parameter for row in load_registry()]
    assert len(names) == len(set(names))


def test_lookup_known_parameters() -> None:
    insurance = get_parameter("Insurance limit")
    assert insurance.status == "VERIFIED (statutory)"
    assert "250,000" in insurance.value

    deposits = get_parameter("SVB-like preset: deposits")
    assert deposits.status == "VERIFIED"
    assert deposits.source == "E1"

    window = get_parameter("Attention memory window")
    assert window.value == "4 hours"
    assert window.module == "M03"


def test_missing_parameter_raises() -> None:
    try:
        get_parameter("not-a-real-parameter")
    except RegistryError as exc:
        assert "unknown parameter" in str(exc)
    else:
        raise AssertionError("expected RegistryError")


def test_list_by_status_and_module() -> None:
    verified = list_by_status("VERIFIED")
    assert get_parameter("SVB-like: uninsured share") in verified
    assert all(row.status == "VERIFIED" for row in verified)

    m03 = list_by_module("M03")
    assert any(row.parameter == "Attention memory window" for row in m03)
    assert all(row.module == "M03" for row in m03)
