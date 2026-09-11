from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

BRANCH = "swarm/1274-loc-real-execution-v1"
EXPECTED_SOURCE_BYTES = 358_594_502
EXPECTED_SOURCE_SHA256 = (
    "a6a6023d9cae067b5531ba84537877c180ab3ff1579f99aeac446d65d465ae0f"
)
CONFIG_PATH = Path("configs/data/d03_common_pile_loc_intake_v1.json")
RIGHTS_POLICY_PATH = Path("configs/data/d03_common_pile_loc_source_rights_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-swarm-1274-loc-real-execution-v1"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def _run_materializer(repo_root: Path, source: Path, output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "tools/materialize_d03_common_pile_loc.py",
            "--repo-root",
            str(repo_root),
            "--output-dir",
            str(output_dir),
            "--shard",
            str(source),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_swarm1274_real_loc_two_clean_execution(tmp_path: Path) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        pytest.skip("real payload execution is shared-CI-only")
    if os.environ.get("GITHUB_HEAD_REF") != BRANCH:
        pytest.skip("real payload execution is bound to the SWARM-1274 branch")

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_json(repo_root / CONFIG_PATH)
    upstream = config["upstream"]
    assert upstream["shard_compressed_bytes"] == EXPECTED_SOURCE_BYTES
    assert upstream["shard_lfs_sha256"] == EXPECTED_SOURCE_SHA256

    subprocess.run(
        [
            sys.executable,
            "tools/validate_d03_common_pile_loc_source_rights_v1.py",
            "--repo-root",
            str(repo_root),
            "--policy",
            str(repo_root / RIGHTS_POLICY_PATH),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    source_a = tmp_path / "source-a.jsonl.gz"
    source_b = tmp_path / "source-b.jsonl.gz"
    _download(upstream["resolve_url"], source_a)
    assert source_a.stat().st_size == EXPECTED_SOURCE_BYTES
    assert _sha256(source_a) == EXPECTED_SOURCE_SHA256

    out_a = tmp_path / "out-a"
    _run_materializer(repo_root, source_a, out_a)

    _download(upstream["resolve_url"], source_b)
    assert source_b.stat().st_size == EXPECTED_SOURCE_BYTES
    assert _sha256(source_b) == EXPECTED_SOURCE_SHA256

    out_b = tmp_path / "out-b"
    _run_materializer(repo_root, source_b, out_b)

    candidate_a = out_a / "loc-candidate.jsonl"
    candidate_b = out_b / "loc-candidate.jsonl"
    report_a_path = out_a / "loc-report.json"
    report_b_path = out_b / "loc-report.json"

    assert candidate_a.read_bytes() == candidate_b.read_bytes()
    assert report_a_path.read_bytes() == report_b_path.read_bytes()

    report = _load_json(report_a_path)
    assert report["full_shard_hash_verified"] is True
    assert report["status"] == "CANDIDATE_MATERIALIZED_ZERO_CREDIT"
    assert type(report["training_authorized_bytes"]) is int
    assert report["training_authorized_bytes"] == 0
    assert type(report["authorized_unique_loss_positions"]) is int
    assert report["authorized_unique_loss_positions"] == 0
    assert type(report["canonical_capacity_credit_bytes"]) is int
    assert report["canonical_capacity_credit_bytes"] == 0
    assert report["corpus_admitted"] is False
    assert report["evaluation_eligible"] is False
    assert report["model_training_executed"] is False
    assert report["final_test_payload_accessed"] is False
    assert report["paid_compute_used"] is False

    evidence = {
        "schema_version": "12-6.d03-loc-real-execution-evidence.v1",
        "execution_class": "LOCAL_FREE_GITHUB_ACTIONS_PUBLIC_REPO",
        "source_acquisitions": 2,
        "source_bytes_a": source_a.stat().st_size,
        "source_bytes_b": source_b.stat().st_size,
        "source_sha256_a": _sha256(source_a),
        "source_sha256_b": _sha256(source_b),
        "candidate_bytes": candidate_a.stat().st_size,
        "candidate_sha256": _sha256(candidate_a),
        "report_bytes": report_a_path.stat().st_size,
        "report_sha256": _sha256(report_a_path),
        "accepted_documents": report["accepted_documents"],
        "accepted_normalized_utf8_bytes": report[
            "accepted_normalized_utf8_bytes"
        ],
        "examined_documents": report["examined_documents"],
        "rejection_counts": report["rejection_counts"],
        "candidate_inventory_identity_sha256": report[
            "candidate_inventory_identity_sha256"
        ],
        "rights_provenance_evidence_identity_sha256": report[
            "rights_provenance_evidence_identity_sha256"
        ],
        "contract_identity_sha256": report["contract_identity_sha256"],
        "full_shard_hash_verified": report["full_shard_hash_verified"],
        "corpus_admitted": report["corpus_admitted"],
        "training_authorized_bytes": report["training_authorized_bytes"],
        "authorized_unique_loss_positions": report[
            "authorized_unique_loss_positions"
        ],
        "canonical_capacity_credit_bytes": report[
            "canonical_capacity_credit_bytes"
        ],
        "model_training_executed": report["model_training_executed"],
        "final_test_payload_accessed": report["final_test_payload_accessed"],
        "paid_compute_used": report["paid_compute_used"],
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    print("SWARM1274_LOC_REAL_EXECUTION=" + json.dumps(evidence, sort_keys=True))
