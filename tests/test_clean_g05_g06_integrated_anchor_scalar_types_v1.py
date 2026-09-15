from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.clean_g05_g06_successor_authority_v1 import (
    CleanG05G06SuccessorError,
    build_prepared_clean_g05_g06_successor_authority,
    verify_clean_g05_g06_successor_authority,
)


def _cjson(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _seal(document: dict[str, object], field: str) -> dict[str, object]:
    result = copy.deepcopy(document)
    result.pop(field, None)
    result[field] = hashlib.sha256(_cjson(result)).hexdigest()
    return result


@pytest.mark.parametrize(
    "field",
    ["physical_run_id", "physical_job_id", "physical_artifact_id"],
)
def test_rejects_float_aliases_in_fixed_physical_anchor(field: str) -> None:
    authority = build_prepared_clean_g05_g06_successor_authority()
    anchor = authority["source_anchor"]
    assert isinstance(anchor, dict)
    value = anchor[field]
    assert isinstance(value, int) and not isinstance(value, bool)
    anchor[field] = float(value)
    authority["source_anchor"] = _seal(anchor, "source_anchor_identity_sha256")
    authority = _seal(authority, "successor_authority_identity_sha256")

    with pytest.raises(
        CleanG05G06SuccessorError,
        match=f"integrated source anchor drift: {field}",
    ):
        verify_clean_g05_g06_successor_authority(authority)
