from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.training import external_worker
from twelve_six.training.external_worker import (
    ExternalTrainingWorkerError,
    handle_request,
)

TRAINER_SHA = "a" * 64
BASE_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64
WEIGHTS_SHA = "d" * 64
BEFORE_SHA = "e" * 64
AFTER_SHA = "f" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _job(*, max_steps: int = 1, job_id: str = "job-1") -> dict[str, object]:
    return {
        "base_artifact": {
            "artifact_ref": "model341-random-init",
            "sha256": BASE_SHA,
        },
        "candidate_artifact_ref": "candidate/model341",
        "job_id": job_id,
        "max_steps": max_steps,
        "owner_id": "owner-1",
        "project_id": "nika-model-training",
        "resource_scope": "local-free",
        "task_id": "task-1",
    }


def _fingerprint(job: dict[str, object]) -> str:
    return hashlib.sha256(b"nika-training-job-v1\x00" + _canonical(job)).hexdigest()


def _step_id(fingerprint: str, step_index: int) -> str:
    material = (
        f"nika-training-step-v1\x00{fingerprint}\x00{TRAINER_SHA}\x00{step_index}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _request(
    *,
    max_steps: int = 1,
    step_index: int = 0,
    resume_state: dict[str, object] | None = None,
    previous_step_id: str | None = None,
    job_id: str = "job-1",
) -> dict[str, object]:
    job = _job(max_steps=max_steps, job_id=job_id)
    fingerprint = _fingerprint(job)
    return {
        "job": job,
        "job_fingerprint": fingerprint,
        "previous_step_id": previous_step_id,
        "protocol_version": 1,
        "resume_state": {} if resume_state is None else resume_state,
        "step_id": _step_id(fingerprint, step_index),
        "step_index": step_index,
        "trainer_sha256": TRAINER_SHA,
    }


def _configure(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("TWELVE_SIX_EXTERNAL_TRAINING_ROOT", str(root))
    monkeypatch.setenv("TWELVE_SIX_TRAINER_AUTHORITY_SHA256", TRAINER_SHA)


def test_request_identity_matches_nika_subprocess_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure(monkeypatch, tmp_path / "state")
    request = _request()

    validated = external_worker._validated_request(_canonical(request))

    assert validated["job_fingerprint"] == _fingerprint(request["job"])
    assert validated["step_id"] == _step_id(validated["job_fingerprint"], 0)


def test_duplicate_json_key_is_rejected_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    request = _canonical(_request())
    duplicated = request.replace(b'{"job":', b'{"job":null,"job":', 1)

    with pytest.raises(ExternalTrainingWorkerError, match="strict JSON"):
        handle_request(duplicated)

    assert not root.exists()


def test_wrong_job_fingerprint_is_rejected_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    request = _request()
    request["job_fingerprint"] = "0" * 64

    with pytest.raises(ExternalTrainingWorkerError, match="fingerprint"):
        handle_request(_canonical(request))

    assert not root.exists()


def test_wrong_trainer_authority_is_rejected_before_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    monkeypatch.setenv("TWELVE_SIX_TRAINER_AUTHORITY_SHA256", "9" * 64)

    with pytest.raises(ExternalTrainingWorkerError, match="trusted environment"):
        handle_request(_canonical(_request()))

    assert not root.exists()


def test_unresolved_dispatch_refuses_blind_optimizer_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    calls = 0

    def ambiguous_effect(
        request: dict[str, object], *, job_root: Path
    ) -> dict[str, dict[str, object]]:
        nonlocal calls
        del request, job_root
        calls += 1
        raise ExternalTrainingWorkerError("synthetic crash after possible optimizer effect")

    monkeypatch.setattr(external_worker, "_execute_training_step", ambiguous_effect)
    payload = _canonical(_request())

    with pytest.raises(ExternalTrainingWorkerError, match="synthetic crash"):
        handle_request(payload)
    with pytest.raises(ExternalTrainingWorkerError, match="DISPATCHING"):
        handle_request(payload)

    assert calls == 1


def test_confirmed_step_replays_without_second_optimizer_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    calls = 0
    replay_verifications = 0

    def confirmed_effect(
        request: dict[str, object], *, job_root: Path
    ) -> dict[str, dict[str, object]]:
        nonlocal calls
        del job_root
        calls += 1
        response = {
            "candidate_sha256": WEIGHTS_SHA,
            "completed": True,
            "protocol_version": 1,
            "resume_state": {},
            "step_id": request["step_id"],
        }
        record = {
            "checkpoint_id": CHECKPOINT_SHA,
            "checkpoint_weights_sha256": WEIGHTS_SHA,
            "job_fingerprint": request["job_fingerprint"],
            "model_state_after_sha256": AFTER_SHA,
            "model_state_before_sha256": BEFORE_SHA,
            "response": response,
            "schema": "twelve-six-external-worker-confirmed-v1",
            "step_id": request["step_id"],
            "trainer_sha256": request["trainer_sha256"],
        }
        return {"response": response, "record": record}

    def verified_replay(**kwargs: object) -> None:
        nonlocal replay_verifications
        assert kwargs["job_root"] == root / _fingerprint(_job())
        replay_verifications += 1

    monkeypatch.setattr(external_worker, "_execute_training_step", confirmed_effect)
    monkeypatch.setattr(
        external_worker,
        "_verify_confirmed_replay_checkpoint",
        verified_replay,
    )
    payload = _canonical(_request())

    first = handle_request(payload)
    second = handle_request(payload)

    assert first == second
    assert calls == 1
    assert replay_verifications == 1
    parsed = json.loads(first)
    assert parsed["candidate_sha256"] == WEIGHTS_SHA
    assert parsed["completed"] is True


def test_resume_state_cannot_cross_job_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    _configure(monkeypatch, root)
    first = _request(max_steps=2)
    first_fingerprint = first["job_fingerprint"]
    first_step_id = first["step_id"]
    assert type(first_fingerprint) is str
    assert type(first_step_id) is str
    resume_state = {
        "checkpoint_id": CHECKPOINT_SHA,
        "checkpoint_weights_sha256": WEIGHTS_SHA,
        "init_spec_sha256": external_worker.INIT_SPEC_SHA256,
        "job_fingerprint": first_fingerprint,
        "last_step_id": first_step_id,
        "mode": external_worker.MODE,
        "model_spec_sha256": external_worker.MODEL_SPEC_SHA256,
        "model_state_sha256": AFTER_SHA,
        "optimizer_step": 1,
        "schema": external_worker.STATE_SCHEMA,
        "tokens_seen": 31,
        "trainer_sha256": TRAINER_SHA,
    }
    second = _request(
        max_steps=2,
        step_index=1,
        resume_state=resume_state,
        previous_step_id=first_step_id,
        job_id="job-2",
    )

    with pytest.raises(ExternalTrainingWorkerError, match="different job"):
        handle_request(_canonical(second))

    assert not root.exists()
