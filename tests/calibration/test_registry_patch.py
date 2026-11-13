"""Row-preserving registry patches: only the named rows change, every other byte stays."""
from __future__ import annotations

import pytest

from calibration.registry_patch import RegistryPatchError, patch_registry_rows

_HEADER = "Parameter,Value,Status,Source,Module,Unit"


def _registry(tmp_path, ending="\r\n"):
    lines = [
        _HEADER,
        'Keep quoted,"a, b",ASSUMED,"src ""x""",M00,u',
        "M14 price impact (ASSUMED),1.13,ASSUMED,SVB-25 prior,M14,1/share",
        'Multi line,"line one' + ending + 'line two",ASSUMED,s,M01,',
        "Depositor mobile share of uninsured deposits (FIT),0.88,ASSUMED,SVB-24,M06,fraction",
        "Tail row,1,ASSUMED,s,M02,u",
    ]
    path = tmp_path / "registry.csv"
    path.write_bytes((ending.join(lines) + ending).encode("utf-8"))
    return path, lines


def test_only_named_rows_change_and_line_endings_survive(tmp_path):
    path, lines = _registry(tmp_path)
    before = path.read_bytes()
    status = patch_registry_rows(
        path,
        {"M14 price impact (ASSUMED)": {"Value": "1.18464", "Status": "FITTED", "Source": "run seed=3060"}},
    )
    assert status == {"M14 price impact (ASSUMED)": "updated"}
    after = path.read_bytes()
    old_lines = before.split(b"\r\n")
    new_lines = after.split(b"\r\n")
    assert len(old_lines) == len(new_lines)
    changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
    assert len(changed) == 1
    assert new_lines[changed[0]] == b"M14 price impact (ASSUMED),1.18464,FITTED,run seed=3060,M14,1/share"
    assert b"\r\n" in after and b"\n\n" not in after


def test_quoted_and_multiline_records_are_untouched(tmp_path):
    path, _ = _registry(tmp_path)
    before = path.read_bytes()
    patch_registry_rows(path, {"Tail row": {"Value": "2"}})
    after = path.read_bytes()
    assert after.replace(b"Tail row,2,", b"Tail row,1,") == before


def test_append_missing_row_uses_file_line_ending(tmp_path):
    path, _ = _registry(tmp_path, ending="\n")
    status = patch_registry_rows(
        path,
        {"New row": {"Value": "0.5"}},
        append_missing={"New row": {"Status": "FITTED", "Source": "x", "Module": "M27", "Unit": "fraction"}},
    )
    assert status == {"New row": "appended"}
    assert path.read_text(encoding="utf-8").endswith("New row,0.5,FITTED,x,M27,fraction\n")


def test_unknown_row_or_column_raises(tmp_path):
    path, _ = _registry(tmp_path)
    with pytest.raises(RegistryPatchError):
        patch_registry_rows(path, {"No such row": {"Value": "1"}})
    with pytest.raises(RegistryPatchError):
        patch_registry_rows(path, {"Tail row": {"Nope": "1"}})


def test_patching_twice_is_idempotent(tmp_path):
    path, _ = _registry(tmp_path)
    updates = {"Tail row": {"Value": "7", "Status": "FITTED"}}
    patch_registry_rows(path, updates)
    once = path.read_bytes()
    assert patch_registry_rows(path, updates) == {"Tail row": "unchanged"}
    assert path.read_bytes() == once


def test_skip_missing_leaves_absent_rows_out(tmp_path):
    path, _ = _registry(tmp_path)
    before = path.read_bytes()
    status = patch_registry_rows(path, {"Not here": {"Value": "1"}}, skip_missing=True)
    assert status == {"Not here": "absent"}
    assert path.read_bytes() == before


def test_apply_update_file_patches_only_spec_rows(tmp_path):
    import json

    from calibration.registry_patch import apply_update_file

    reg = tmp_path / "registry.csv"
    reg.write_text(
        "Parameter,Value,Status,Source,Module,Unit\n"
        "Non-bank E15 LUNA peak-to-trough fraction,UNSET,UNSET,old,M27,fraction\n"
        "Other,1,ASSUMED,s,M00,u\n",
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"updates": {"Non-bank E15 LUNA peak-to-trough fraction": {
        "Value": "0.9999996", "Status": "SOURCED", "Source": "derived, with a comma"}}}), encoding="utf-8")
    assert apply_update_file(spec, reg) == {"Non-bank E15 LUNA peak-to-trough fraction": "updated"}
    lines = reg.read_text(encoding="utf-8").splitlines()
    assert lines[1] == 'Non-bank E15 LUNA peak-to-trough fraction,0.9999996,SOURCED,"derived, with a comma",M27,fraction'
    assert lines[2] == "Other,1,ASSUMED,s,M00,u"
