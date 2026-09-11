from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "configs/data/next100_037_python_docs_source_authority_v1.json"
EVIDENCE = ROOT / "evidence/d03/cpython_docs_current_main_convergence_v1.json"
VALIDATOR = ROOT / "tools/validate_next100_037_python_docs_source_authority.py"

EXPECTED_BLOBS = {
    "configs/data/next100_037_python_docs_source_authority_v1.json": (
        "b15abac8744ccda9fe58d1351f7925b6ab328034"
    ),
    "docs/NEXT100_037_PYTHON_DOCS_SOURCE_AUTHORITY.md": (
        "701790c6826aca9cda1a57100762ab3dbd33627c"
    ),
    "tools/materialize_next100_037_python_docs_accepted_chunks.py": (
        "0635bb18cec8b9db2d339b044485aa8ab5c28847"
    ),
    "tools/validate_next100_037_python_docs_source_authority.py": (
        "9b2f7e7895fc620e5f152c00d893866c28afdcc1"
    ),
}
EXPECTED_EVIDENCE_ID = "23ad2d93d3e4f1d830feeed0ef446e52bffa201ae517590accf08c679ba3c14c"


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def test_scientific_files_are_exact_historical_pr467_blobs() -> None:
    for relative_path, expected_blob in EXPECTED_BLOBS.items():
        assert _git_blob_sha1(ROOT / relative_path) == expected_blob


def test_historical_authority_validator_passes_on_exact_bytes() -> None:
    namespace = runpy.run_path(str(VALIDATOR), run_name="cpython_docs_validator")
    result = namespace["validate"](AUTHORITY)
    assert result["status"] == "PASS"
    assert result["accepted_chunk_count"] == 14
    assert result["rejected_chunk_count"] == 2
    assert result["evaluation_authorized"] is False


def test_historical_authority_validator_rejects_truth_boundary_drift(tmp_path: Path) -> None:
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    authority["claim_boundary"]["training_executed"] = True
    mutated = tmp_path / "mutated-authority.json"
    mutated.write_text(json.dumps(authority), encoding="utf-8")

    namespace = runpy.run_path(str(VALIDATOR), run_name="cpython_docs_validator_mutation")
    with pytest.raises(SystemExit):
        namespace["validate"](mutated)


def test_convergence_evidence_is_self_bound_and_zero_credit() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    supplied = evidence.pop("evidence_identity_sha256")
    assert supplied == EXPECTED_EVIDENCE_ID
    assert hashlib.sha256(_canonical(evidence)).hexdigest() == supplied

    execution = evidence["historical_real_execution"]
    assert execution["workflow_run"] == 33006886741
    assert execution["artifact_id"] == 9624065504
    assert execution["accepted_chunk_count"] == 14
    assert execution["rejected_chunk_count"] == 2
    assert execution["accepted_normalized_utf8_bytes"] == 15540
    assert execution["ledger_identity_sha256"] == (
        "fa16cb7e4015480500ba74fb28ecfa6af60e5f0685c90a5c822db04d480b44c2"
    )

    boundary = evidence["claim_boundary"]
    for key in (
        "training_authorized_bytes",
        "unique_loss_positions_authorized",
        "authorized_optimized_target_exposure",
        "optimizer_updates",
    ):
        assert type(boundary[key]) is int
        assert boundary[key] == 0
    for key in (
        "corpus_frozen",
        "global_cross_source_dedup_complete",
        "evaluation_decontamination_complete",
        "tokenizer_fit_executed",
        "model_training_executed",
        "learned_weights_created",
        "final_test_accessed",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        assert boundary[key] is False
