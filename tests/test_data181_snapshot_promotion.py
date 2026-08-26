from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.data.external_sources import EligibilityResolver
from twelve_six.data.snapshot_promotion import (
    SnapshotPromotionError,
    _chunk_text,
    _verify_download,
    load_and_verify_registry,
    load_promotion_plan,
)
from twelve_six.data.source_intake import DownloadedBytes


PLAN = Path("configs/data/data181_real_snapshot_promotion_v1.json")
REGISTRY = Path("data/external/external_sources.json")


def test_committed_data181_plan_registry_and_rights_evidence_are_exact() -> None:
    plan = load_promotion_plan(PLAN)
    registry, sources = load_and_verify_registry(Path("."), plan, REGISTRY)
    assert registry["registry_identity_sha256"] == (
        "82abd7dca04947d72a6d07d8228025c58373d17018fa8dc3a7bca30f7a2714c2"
    )
    assert len(sources) == 3
    resolver = EligibilityResolver(registry)
    for source in sources:
        decision = resolver.assert_model_training_eligible(
            source.source_id, source.source_version, source.source_manifest_sha256
        )
        assert decision.model_training_eligible is True
        assert decision.acquisition == "ALLOWED"
        assert decision.storage == "ALLOWED"
        assert decision.analysis == "ALLOWED"
        assert decision.model_training == "ALLOWED"
        assert decision.redistribution == "ALLOWED"
        assert len(decision.evidence_ids) == 2


def test_promotion_plan_retains_exact_data21_22_object_identities() -> None:
    plan = load_promotion_plan(PLAN)
    observed = {
        item["record_id"]: (
            item["raw_sha256"],
            item["raw_bytes"],
            item["normalized_sha256"],
            item["normalized_utf8_bytes"],
            item["parent_source_identity_sha256"],
        )
        for item in plan["objects"]
    }
    assert observed == {
        "ext-ba861bf058ce23e02cc569d1a63de897": (
            "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4",
            332400,
            "72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50",
            88565,
            "b8f1d2f99a3db71d894a3233e9417d6283d11768c41b1634bc8b096ab77aba4e",
        ),
        "ext-34108f6cb4826107ff3b53be7b172eb0": (
            "21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860",
            68812,
            "154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83",
            48002,
            "ba622171b752c4d411bd0b93a94dad14b7ff0e5ac88064678d1b91a551c01be3",
        ),
        "ext-b47c3572ee24c4641b86e91804ed04fe": (
            "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509",
            37299,
            "94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a",
            36791,
            "ba622171b752c4d411bd0b93a94dad14b7ff0e5ac88064678d1b91a551c01be3",
        ),
    }


def test_raw_drift_fails_closed_before_promotion() -> None:
    plan = load_promotion_plan(PLAN)
    item = plan["objects"][0]
    with pytest.raises(SnapshotPromotionError, match="raw size drift"):
        _verify_download(DownloadedBytes(b"wrong"), item, "test")


def test_generic_chunker_is_deterministic_bounded_and_not_source_specific() -> None:
    text = "\n".join(
        [
            "The deterministic chunker groups ordinary natural text without a source exception. " * 8,
            "Український текст також проходить той самий загальний алгоритм сегментації. " * 8,
        ]
    )
    first = _chunk_text(text, max_chars=300, min_chars=80)
    second = _chunk_text(text, max_chars=300, min_chars=80)
    assert first == second
    assert len(first) >= 2
    assert all(80 <= len(item) <= 300 for item in first)
