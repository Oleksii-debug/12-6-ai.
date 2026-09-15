from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "run_d03_clean_g05_g06_replay_v1.py"
SPEC = importlib.util.spec_from_file_location("clean_g05_g06_retained_replay_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def test_retained_materialization_authority_is_exact() -> None:
    replay._validate_retained_materialization_authority(
        {
            "execution_head_sha": replay.RETAINED_EXECUTION_HEAD,
            "evidence_identity_sha256": replay.RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
        }
    )


@pytest.mark.parametrize(
    "mutated",
    [
        {
            "execution_head_sha": "0" * 40,
            "evidence_identity_sha256": replay.RETAINED_DATA526_EVIDENCE_IDENTITY_SHA256,
        },
        {
            "execution_head_sha": replay.RETAINED_EXECUTION_HEAD,
            "evidence_identity_sha256": "0" * 64,
        },
    ],
)
def test_current_or_self_authored_materialization_identity_is_rejected(
    mutated: dict[str, str],
) -> None:
    with pytest.raises(replay.CleanG05G06ReplayError):
        replay._validate_retained_materialization_authority(mutated)


def test_product_and_corpus_authority_heads_are_not_conflated() -> None:
    source = TOOL.read_text(encoding="utf-8")
    assert "corpus_materialization_authority_head_sha" in source
    assert "product_execution_head_sha" in source
    assert "fresh clean DATA526 evidence was not produced on this execution head" not in source
