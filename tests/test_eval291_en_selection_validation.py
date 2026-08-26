from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from twelve_six.eval291_en_selection_validation import (
    AUTHORITY_REL,
    CONFIG_REL,
    SELECTION_REL,
    AuthorityError,
    materialize,
    verify,
)


def _copy_authority_tree(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for rel in (
        CONFIG_REL,
        Path("data/evaluation/eval291/source-snapshots/httpx-timeouts.md"),
        Path("data/evaluation/eval291/source-snapshots/requests-authentication.rst"),
        Path("data/evaluation/eval291/rights-evidence/httpx-LICENSE.md"),
        Path("data/evaluation/eval291/rights-evidence/requests-APACHE-2.0.txt"),
        Path("data/evaluation/eval291/rights-evidence/requests-NOTICE"),
        SELECTION_REL,
        AUTHORITY_REL,
    ):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_root / rel, target)
    return tmp_path


def test_committed_authority_rebuilds_byte_for_byte() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    verify(repo_root)
    rebuilt_selection, rebuilt_authority = materialize(repo_root)
    assert rebuilt_selection == (repo_root / SELECTION_REL).read_bytes()
    assert rebuilt_authority == (repo_root / AUTHORITY_REL).read_bytes()


def test_authority_is_external_real_en_selection_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    authority = json.loads((repo_root / AUTHORITY_REL).read_text(encoding="utf-8"))
    assert authority["status"] == "IMMUTABLE_EXTERNAL_REAL_EN_SELECTION_VALIDATION"
    assert authority["language"] == "en"
    assert authority["purpose"] == "selection_validation"
    assert authority["documents"] == 2
    assert authority["source_families"] == ["github:encode/httpx", "github:psf/requests"]
    assert authority["firewalls"] == {
        "data227_training_objects_reused": False,
        "final_test_outcomes_influence_selection_construction": False,
        "final_test_outcomes_inspected": False,
        "final_test_payload_used_as_builder_input": False,
        "offline_rebuild_requires_network": False,
        "selection_bytes_are_final_test_eligible": False,
        "selection_bytes_are_tokenizer_fit_eligible": False,
        "selection_bytes_are_training_eligible": False,
    }


def test_exact_selection_objects_are_not_data227_training_objects() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads((repo_root / CONFIG_REL).read_text(encoding="utf-8"))
    for source in config["sources"]:
        training = source["project_reservation"]["training_object_from_terminal_admission"]
        assert source["path"] != training["path"]
        assert source["git_blob_sha1"] != training["git_blob_sha1"]
        assert source["project_reservation"]["selection_validation"] is True
        assert source["project_reservation"]["training"] is False
        assert source["project_reservation"]["tokenizer_fit"] is False
        assert source["project_reservation"]["final_test"] is False


def test_final_test_is_metadata_only_and_never_a_builder_input() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads((repo_root / CONFIG_REL).read_text(encoding="utf-8"))
    boundary = config["final_test_boundary"]
    assert boundary["payload_read_for_construction"] is False
    assert boundary["outcomes_read_for_construction"] is False
    assert boundary["outcomes_allowed_in_config"] is False
    assert set(boundary["admitted_source_ids"]).isdisjoint(
        {source["source_family"] for source in config["sources"]}
    )
    assert all("recover174_real_holdout_seed" not in source["snapshot_path"] for source in config["sources"])


def test_mutated_source_bytes_fail_closed(tmp_path: Path) -> None:
    root = _copy_authority_tree(tmp_path)
    target = root / "data/evaluation/eval291/source-snapshots/httpx-timeouts.md"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(AuthorityError, match="byte count mismatch"):
        materialize(root)


def test_attempted_training_object_reuse_fails_closed(tmp_path: Path) -> None:
    root = _copy_authority_tree(tmp_path)
    config_path = root / CONFIG_REL
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["sources"][0]
    source["git_blob_sha1"] = source["project_reservation"]["training_object_from_terminal_admission"]["git_blob_sha1"]
    config_path.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AuthorityError, match="reuses DATA-227 training object"):
        materialize(root)


def test_any_final_test_payload_reference_fails_closed(tmp_path: Path) -> None:
    root = _copy_authority_tree(tmp_path)
    config_path = root / CONFIG_REL
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sources"][0]["snapshot_path"] = "data/evaluation/recover174_real_holdout_seed.jsonl.gz"
    config_path.write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(AuthorityError, match="final-test payload paths"):
        materialize(root)


def test_rights_evidence_is_hash_bound(tmp_path: Path) -> None:
    root = _copy_authority_tree(tmp_path)
    notice = root / "data/evaluation/eval291/rights-evidence/requests-NOTICE"
    notice.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(AuthorityError, match="NOTICE evidence mismatch"):
        materialize(root)
