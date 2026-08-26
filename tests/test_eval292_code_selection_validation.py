from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.eval292_code_selection_validation import (
    Eval292Error,
    build_authority,
    validate_authority,
    verify_eval233_boundary,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/eval292/code-selection-validation-v1.json"


def _rehash(authority: dict[str, object]) -> None:
    unsigned = dict(authority)
    unsigned.pop("authority_identity_sha256", None)
    raw = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    authority["authority_identity_sha256"] = hashlib.sha256(raw).hexdigest()


def test_terminal_eval233_boundary_is_exact() -> None:
    verify_eval233_boundary(ROOT)


def test_committed_authority_is_deterministic_build() -> None:
    expected = build_authority()
    actual = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert actual == expected
    validate_authority(actual)


def test_selection_is_empty_and_physically_separate() -> None:
    authority = build_authority()
    selection = authority["selection_set"]
    separation = authority["separation"]

    assert selection["documents"] == 0
    assert selection["records"] == []
    assert selection["unique_content_bytes"] == 0
    assert separation["selected_source_ids"] == []
    assert separation["future_training_source_id_overlap_count"] == 0
    assert separation["selected_content_hash_overlap_count"] == 0
    assert separation["final_test_records_copied"] == 0
    assert separation["final_test_outcomes_inspected"] is False
    assert separation["final_test_bytes_consumed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidates", 0, "selection_admitted"), True),
        (("candidates", 0, "evaluation_use_explicitly_authorized"), True),
        (("candidates", 0, "reserved_from_all_training"), True),
        (("separation", "final_test_outcomes_inspected"), True),
        (("separation", "final_test_records_copied"), 1),
        (("verdict", "release_allowed"), True),
    ],
)
def test_fail_closed_mutations_are_rejected(
    path: tuple[object, ...],
    value: object,
) -> None:
    authority = copy.deepcopy(build_authority())
    cursor: object = authority
    for key in path[:-1]:
        if isinstance(key, int):
            assert isinstance(cursor, list)
            cursor = cursor[key]
        else:
            assert isinstance(cursor, dict)
            cursor = cursor[key]
    final_key = path[-1]
    if isinstance(final_key, int):
        assert isinstance(cursor, list)
        cursor[final_key] = value
    else:
        assert isinstance(cursor, dict)
        cursor[final_key] = value
    _rehash(authority)

    with pytest.raises(Eval292Error):
        validate_authority(authority)
