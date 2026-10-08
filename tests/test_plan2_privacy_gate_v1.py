from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy

ROOT = Path(__file__).resolve().parents[1]


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _fixture(*lines: str):
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    index = []
    start = 0
    for number, row in enumerate(payload.splitlines(keepends=True)):
        index.append({"index": number, "start_byte": start,
                      "end_byte": start + len(row), "sha256": _sha(row)})
        start += len(row)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": "fixture.test",
        "record_count": len(index),
        "records": index,
        "normalized_bytes": len(payload),
        "normalized_sha256": _sha(payload),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    return {**core, "manifest_sha256": _sha(privacy._canonical(core))}, payload


def _inspect(*lines: str, removed=()):
    receipt, payload = _fixture(*lines)
    return privacy.inspect_normalized(
        receipt, payload, policy_sha256="f" * 64, tombstones=removed)


def test_allow_only_and_exact_restart_are_deterministic():
    lines = ("Це відкритий публічний навчальний документ без приватних даних.",
             "Опис відкритої документації та технічних параметрів.")
    one = _inspect(*lines)
    two = _inspect(*lines)
    assert one == two
    assert one["candidate_record_count"] == 2
    assert one["excluded_records"] == []
    assert one["counts"]["allow"] == 2
    assert one["g06_execution_identity_sha256"]
    assert one["training_corpus_authorized"] is False
    assert one["tokenizer_fit_authorized"] is False
    assert "документ" not in json.dumps(one, ensure_ascii=False)
    assert privacy._sha(privacy._canonical({
        k: v for k, v in one.items() if k != "manifest_sha256"
    })) == one["manifest_sha256"]


def test_contact_and_credentials_are_entirely_excluded():
    lines = ("Звичайний відкритий україномовний текст.",
             "Напишіть на contact@real-domain.dev для розгляду.",
             "sk-proj-" + "Q" * 36)
    result = _inspect(*lines)
    assert result["candidate_record_count"] == 1
    assert len(result["excluded_records"]) == 2
    assert {v["reason"] for v in result["excluded_records"]} >= {"REDACT", "EXCLUDE"}
    rendered = json.dumps(result, ensure_ascii=False)
    assert "contact@" not in rendered
    assert "sk-proj-" not in rendered
    assert "Q" * 20 not in rendered
    assert result["detector_counts"]["email"] >= 1
    assert result["detector_counts"]["api_token"] >= 1


def test_manual_removal_creates_new_immutable_identity():
    lines = ("Рядок один для відкритого корпусу.", "Рядок два для корпусу.")
    before = _inspect(*lines)
    after = _inspect(*lines, removed=("fixture.test:r00000001",))
    assert before["candidate_record_count"] == 2
    assert after["candidate_record_count"] == 1
    assert after["excluded_records"] == [
        {"record_id": "fixture.test:r00000001", "reason": "REMOVAL"}
    ]
    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert before["g06_execution_identity_sha256"] == after["g06_execution_identity_sha256"]


@pytest.mark.parametrize("tombstones", [
    ("missing",), ("fixture.test:r00000000", "fixture.test:r00000000"),
    (123,),
])
def test_unknown_duplicate_or_wrong_type_removals_fail_closed(tombstones):
    manifest, payload = _fixture("Рядок.")
    with pytest.raises(privacy.Plan2PrivacyError, match="tombstone"):
        privacy.inspect_normalized(
            manifest, payload, policy_sha256="f" * 64,
            tombstones=tombstones)


def test_stale_bytes_record_hash_and_manifest_are_all_rejected():
    manifest, payload = _fixture("Рядок один.", "Рядок два.")
    original = copy.deepcopy(manifest)
    with pytest.raises(privacy.Plan2PrivacyError, match="payload"):
        privacy.inspect_normalized(manifest, payload + b"x", policy_sha256="f" * 64)
    mutated = copy.deepcopy(original)
    mutated["records"][0]["sha256"] = "0" * 64
    mutated["manifest_sha256"] = _sha(privacy._canonical({
        k: v for k, v in mutated.items() if k != "manifest_sha256"
    }))
    with pytest.raises(privacy.Plan2PrivacyError, match="digest"):
        privacy.inspect_normalized(mutated, payload, policy_sha256="f" * 64)
    mutated = copy.deepcopy(original)
    mutated["manifest_sha256"] = "0" * 64
    with pytest.raises(privacy.Plan2PrivacyError, match="self-hash"):
        privacy.inspect_normalized(mutated, payload, policy_sha256="f" * 64)


def test_record_boundary_gap_and_unindexed_suffix_rejected():
    manifest, payload = _fixture("Рядок один.", "Рядок два.")
    for broken in ("start_byte", "end_byte"):
        changed = copy.deepcopy(manifest)
        changed["records"][1][broken] += 1
        changed["manifest_sha256"] = _sha(privacy._canonical({
            k: v for k, v in changed.items() if k != "manifest_sha256"
        }))
        with pytest.raises(privacy.Plan2PrivacyError):
            privacy.inspect_normalized(
                changed, payload, policy_sha256="f" * 64)


def test_policy_is_exact_and_immutable(tmp_path):
    import shutil
    directory = tmp_path / "configs" / "data"
    directory.mkdir(parents=True)
    policy_file = directory / "plan2_privacy_policy_v1.json"
    shutil.copyfile(ROOT / privacy.POLICY_PATH, policy_file)
    assert privacy._policy(tmp_path) == _sha(privacy._canonical(privacy.POLICY))
    policy_file.write_text(json.dumps({**privacy.POLICY, "decision": "ALLOW_ALL"}))
    with pytest.raises(privacy.Plan2PrivacyError, match="drift"):
        privacy._policy(tmp_path)


def test_real_s3_s4_g06_physical_staging_replay_and_no_clobber(tmp_path):
    out = tmp_path / "p2"
    first = privacy.stage_privacy(ROOT, out)
    second = privacy.stage_privacy(ROOT, out)
    assert first == second
    assert first["input_record_count"] == 50
    assert first["candidate_record_count"] + len(first["excluded_records"]) == 50
    assert not first["training_corpus_authorized"]
    assert json.loads((out / "privacy-manifest.json").read_text()) == first
    assert privacy.stage_privacy(ROOT, tmp_path / "clean-restart") == first
    (out / "privacy-manifest.json").write_text('{"tampered":true}')
    with pytest.raises(privacy.Plan2PrivacyError, match="immutable"):
        privacy.stage_privacy(ROOT, out)


def test_removal_rebuild_is_new_destination_and_old_identity_stays_pinned(tmp_path):
    first = privacy.stage_privacy(ROOT, tmp_path / "old")
    record_id = f"{first['source_id']}:r00000000"
    revised = privacy.stage_privacy(
        ROOT, tmp_path / "rebuild", tombstones=(record_id,))
    assert revised["manifest_sha256"] != first["manifest_sha256"]
    assert any(row["record_id"] == record_id
               for row in revised["excluded_records"])
    with pytest.raises(privacy.Plan2PrivacyError, match="immutable"):
        privacy.stage_privacy(
            ROOT, tmp_path / "old", tombstones=(record_id,))


def test_destination_symlink_is_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(
        tmp_path / "real", target_is_directory=True)
    with pytest.raises(privacy.Plan2PrivacyError, match="symlink"):
        privacy.stage_privacy(ROOT, tmp_path / "link" / "run")
