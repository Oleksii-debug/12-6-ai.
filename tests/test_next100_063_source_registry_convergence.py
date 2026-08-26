from __future__ import annotations

from collections import Counter

import pytest

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import source_registry_convergence_v1 as convergence
from twelve_six.data.pipeline import _quality_reason, normalize_text


def _synthetic_addition(raw: bytes) -> dict[str, object]:
    normalized = normalize_text(raw.decode("utf-8"))
    chunks = convergence._chunk_text(normalized, max_chars=1200, min_chars=80)
    policy = {
        "min_chars": 60,
        "max_chars": 1600,
        "min_alpha_ratio": 0.35,
        "reject_control_characters": True,
        "reject_email": True,
        "reject_phone": True,
    }
    stub = {
        "quality_privacy": {
            "quality_policy": policy,
        }
    }
    quality_config = convergence._quality_config(stub)
    accepted_hashes: list[str] = []
    rejection_reasons: Counter[str] = Counter()
    for chunk in chunks:
        reason = _quality_reason(chunk, quality_config)
        if reason is not None:
            rejection_reasons[reason] += 1
            continue
        accepted_hashes.append(v1._sha256(normalize_text(chunk).encode("utf-8")))

    return {
        "source": {
            "raw_bytes": len(raw),
            "raw_sha256": v1._sha256(raw),
            "source_git_blob_sha1": v1._git_blob_sha1(raw),
            "normalization": {
                "truncate_chars": 50000,
                "normalized_utf8_bytes": len(normalized.encode("utf-8")),
                "normalized_sha256": v1._sha256(normalized.encode("utf-8")),
            },
        },
        "quality_privacy": {
            "materialization_policy": convergence.MATERIALIZATION_POLICY,
            "chunk_count": len(chunks),
            "chunking": {"max_chars": 1200, "min_chars": 80},
            "quality_policy": policy,
            "accepted_chunk_count": len(accepted_hashes),
            "accepted_normalized_sha256": accepted_hashes,
            "rejected_chunk_count": sum(rejection_reasons.values()),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
        },
    }


def test_materialization_excludes_privacy_rejected_chunk() -> None:
    clean = ("alpha language model training evidence " * 24).strip()
    rejected = (("beta technical documentation " * 24) + " call +380 50 123 4567").strip()
    raw = f"{clean}\n\n{rejected}\n".encode("utf-8")
    addition = _synthetic_addition(raw)

    materialized = convergence.materialize_authorized_text(raw, addition)

    assert materialized["accepted_chunk_count"] == 1
    assert materialized["rejected_chunk_count"] == 1
    assert materialized["rejection_reasons"] == {"pii_phone": 1}
    assert b"380 50 123 4567" not in materialized["payload"]
    assert materialized["declared_capacity_bytes"] < len(normalize_text(raw.decode()).encode())


def test_materialization_rejects_authority_hash_drift() -> None:
    clean = ("alpha deterministic accepted document " * 30).strip()
    raw = f"{clean}\n".encode("utf-8")
    addition = _synthetic_addition(raw)
    addition["quality_privacy"]["accepted_normalized_sha256"][0] = "0" * 64

    with pytest.raises(convergence.SourceRegistryConvergenceError, match="identity/order drift"):
        convergence.materialize_authorized_text(raw, addition)


def test_family_gate_resolves_en_two_family_minimum() -> None:
    report = {
        "terminal_candidates": {
            "by_modality": {
                "code": {"declared_source_family_count": 4},
                "en": {"declared_source_family_count": 2},
                "uk": {"declared_source_family_count": 2},
            }
        },
        "matches": [],
    }
    config = {
        "family_gate": {
            "minimum_independent_families_per_stratum": 2,
            "required_family_counts_after": {"code": 4, "en": 2, "uk": 2},
            "forbid_cross_family_capacity_collapse_for_added_source": True,
        }
    }

    assert convergence._assert_family_gate(report, config, "cpython") == {
        "code": 4,
        "en": 2,
        "uk": 2,
    }


def test_family_gate_rejects_cross_family_copy_for_added_source() -> None:
    report = {
        "terminal_candidates": {
            "by_modality": {
                "code": {"declared_source_family_count": 4},
                "en": {"declared_source_family_count": 2},
                "uk": {"declared_source_family_count": 2},
            }
        },
        "matches": [
            {
                "left_source_id": "cpython",
                "right_source_id": "standardebooks",
                "cross_source_family": True,
                "capacity_collapsing": True,
            }
        ],
    }
    config = {
        "family_gate": {
            "minimum_independent_families_per_stratum": 2,
            "required_family_counts_after": {"code": 4, "en": 2, "uk": 2},
            "forbid_cross_family_capacity_collapse_for_added_source": True,
        }
    }

    with pytest.raises(convergence.SourceRegistryConvergenceError, match="cross-family"):
        convergence._assert_family_gate(report, config, "cpython")
