from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_rada_trees_quality_windows_authoritative.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_privacy_provenance", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_privacy_reference_is_explicit_feature_branch_sync_not_main_merge() -> None:
    assert (
        MODULE.PRIVACY_REPAIR_FEATURE_BRANCH_SYNC_MERGE_SHA
        == "c5b9d0922cd63ade1a90c7f0163325dcf35d0b41"
    )
    assert not hasattr(MODULE, "PRIVACY_REPAIR_REFERENCE_MERGE_SHA")

    source = TOOL.read_text(encoding="utf-8")
    assert '"privacy_repair_reference_merge_sha"' not in source
    assert (
        '"privacy_repair_feature_branch_sync_merge_sha": '
        "PRIVACY_REPAIR_FEATURE_BRANCH_SYNC_MERGE_SHA"
    ) in source
