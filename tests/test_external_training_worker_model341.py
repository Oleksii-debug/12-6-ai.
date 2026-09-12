from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TRAINER_SHA = "a" * 64
BASE_SHA = "b" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request() -> tuple[dict[str, object], str]:
    job = {
        "base_artifact": {
            "artifact_ref": "model341-random-init",
            "sha256": BASE_SHA,
        },
        "candidate_artifact_ref": "candidate/model341",
        "job_id": "model341-external-worker-mechanics",
        "max_steps": 1,
        "owner_id": "swarm-1592",
        "project_id": "nika-model-training",
        "resource_scope": "local-free",
        "task_id": "model341-mechanics-step-1",
    }
    job_fingerprint = hashlib.sha256(
        b"nika-training-job-v1\x00" + _canonical(job)
    ).hexdigest()
    step_id = hashlib.sha256(
        (
            f"nika-training-step-v1\x00{job_fingerprint}\x00{TRAINER_SHA}\x000"
        ).encode()
    ).hexdigest()
    return (
        {
            "job": job,
            "job_fingerprint": job_fingerprint,
            "previous_step_id": None,
            "protocol_version": 1,
            "resume_state": {},
            "step_id": step_id,
            "step_index": 0,
            "trainer_sha256": TRAINER_SHA,
        },
        job_fingerprint,
    )


def test_model341_subprocess_commits_one_real_optimizer_step_and_replays(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    tool = repository_root / "tools" / "run_external_training_worker.py"
    state_root = tmp_path / "external-training-state"
    request, job_fingerprint = _request()
    payload = _canonical(request) + b"\n"
    environment = dict(os.environ)
    environment["TWELVE_SIX_EXTERNAL_TRAINING_ROOT"] = str(state_root)
    environment["TWELVE_SIX_TRAINER_AUTHORITY_SHA256"] = TRAINER_SHA

    first = subprocess.run(
        [sys.executable, str(tool)],
        input=payload,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        check=False,
        timeout=300,
    )

    assert first.returncode == 0, first.stderr.decode("utf-8", errors="replace")
    response = json.loads(first.stdout.decode("utf-8"))
    assert set(response) == {
        "candidate_sha256",
        "completed",
        "protocol_version",
        "resume_state",
        "step_id",
    }
    assert response["completed"] is True
    assert response["protocol_version"] == 1
    assert response["step_id"] == request["step_id"]
    assert response["resume_state"]["mode"] == "synthetic-mechanics"
    assert response["resume_state"]["optimizer_step"] == 1
    assert response["resume_state"]["model_spec_sha256"] == (
        "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    )

    job_root = state_root / job_fingerprint
    checkpoint = job_root / "checkpoints" / "step-00000001"
    weights = checkpoint / "weights.safetensors"
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    confirmed = json.loads(
        (job_root / "results" / f"{request['step_id']}.json").read_text(encoding="utf-8")
    )

    assert response["candidate_sha256"] == _sha256_file(weights)
    assert response["candidate_sha256"] == manifest["files"]["weights.safetensors"]["sha256"]
    assert response["candidate_sha256"] == response["resume_state"]["checkpoint_weights_sha256"]
    assert confirmed["model_state_before_sha256"] != confirmed["model_state_after_sha256"]
    assert confirmed["checkpoint_id"] == manifest["checkpoint_id"]
    assert manifest["identity"]["step"] == 1
    assert manifest["identity"]["training_config"]["mode"] == "synthetic-mechanics"

    checkpoint_inventory_before = sorted(path.name for path in (job_root / "checkpoints").iterdir())
    second = subprocess.run(
        [sys.executable, str(tool)],
        input=payload,
        capture_output=True,
        cwd=repository_root,
        env=environment,
        check=False,
        timeout=60,
    )

    assert second.returncode == 0, second.stderr.decode("utf-8", errors="replace")
    assert second.stdout == first.stdout
    assert sorted(path.name for path in (job_root / "checkpoints").iterdir()) == (
        checkpoint_inventory_before
    )
