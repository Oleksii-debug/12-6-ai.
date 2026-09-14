from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from twelve_six.data._data232_decontamination_matching import DecontaminationError

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/execute_eval647_current_corpus_overlap_v1.py"
spec = importlib.util.spec_from_file_location("eval647_overlap_executor", TOOL)
assert spec is not None and spec.loader is not None
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], dict[str, bytes | str]]:
    reserved_a = b"def reserved_alpha(x):\n    return x * 101\n"
    reserved_b = b"def reserved_beta(y):\n    return y - 303\n"
    training_record = {
        "record_id": "train:1",
        "source_id": "local:training",
        "source_family": "local:training",
        "modality": "code",
        "text": "def train_only(value):\n    return value + 17\n",
    }
    training_raw = (json.dumps(training_record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    training_sha = _sha(training_raw)
    manifest = {
        "objects": [
            {
                "source_family": "github:example/one",
                "repository": "example/one",
                "revision": "1" * 40,
                "path": "one.py",
                "expected_raw_bytes": len(reserved_a),
                "raw_sha256": _sha(reserved_a),
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            },
            {
                "source_family": "github:example/two",
                "repository": "example/two",
                "revision": "2" * 40,
                "path": "two.py",
                "expected_raw_bytes": len(reserved_b),
                "raw_sha256": _sha(reserved_b),
                "training_allowed": False,
                "tokenizer_fit_allowed": False,
                "permanent_future_training_exclusion": True,
            },
        ],
        "materialization_evidence": {"identity_sha256": "3" * 64},
    }
    evidence = {
        "schema_version": "12-6.d03-post-g05-g06-materialization.v1",
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "materialization_identity_sha256": "4" * 64,
        "result": {"record_payload_jsonl_sha256": training_sha, "record_count": 1},
        "truth_boundary": {
            "final_test_outcomes_read": False,
            "training_executed": False,
            "paid_compute_used": False,
        },
    }
    paths = {
        "training": tmp_path / "records.jsonl",
        "evidence": tmp_path / "evidence.json",
        "manifest": tmp_path / "manifest.json",
        "output": tmp_path / "overlap.json",
    }
    paths["training"].write_bytes(training_raw)
    paths["evidence"].write_text(json.dumps(evidence), encoding="utf-8")
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths, {
        "reserved_a": reserved_a,
        "reserved_b": reserved_b,
        "training_sha": training_sha,
    }


def test_execute_writes_hash_only_clean_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, values = _fixture(tmp_path)

    def fake_fetch(url: str, _timeout: int) -> bytes:
        return values["reserved_a"] if "example/one" in url else values["reserved_b"]  # type: ignore[return-value]

    monkeypatch.setattr(executor.source_materializer, "_fetch", fake_fetch)
    report = executor.execute(
        training_records_jsonl=paths["training"],
        materialization_evidence_json=paths["evidence"],
        manifest_json=paths["manifest"],
        output_json=paths["output"],
        expected_materialization_identity_sha256="4" * 64,
        expected_training_jsonl_sha256=str(values["training_sha"]),
    )
    assert report["status"] == "PASS_CURRENT_CORPUS_OVERLAP_ONLY"
    assert report["selection_validation_records_authorized"] == 0
    durable = paths["output"].read_text(encoding="utf-8")
    assert "reserved_alpha" not in durable
    assert "reserved_beta" not in durable
    assert "train_only" not in durable


def test_execute_rejects_self_selected_training_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, values = _fixture(tmp_path)

    def fake_fetch(url: str, _timeout: int) -> bytes:
        return values["reserved_a"] if "example/one" in url else values["reserved_b"]  # type: ignore[return-value]

    monkeypatch.setattr(executor.source_materializer, "_fetch", fake_fetch)
    with pytest.raises(DecontaminationError, match="independent current-corpus authority"):
        executor.execute(
            training_records_jsonl=paths["training"],
            materialization_evidence_json=paths["evidence"],
            manifest_json=paths["manifest"],
            output_json=paths["output"],
            expected_materialization_identity_sha256="4" * 64,
            expected_training_jsonl_sha256="6" * 64,
        )
