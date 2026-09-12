"""Strict JSON subprocess worker for one bounded MODEL-341/D02 optimizer step.

This module adapts the existing MODEL-341 model and D02 Trainer to an external
orchestrator. It does not own dataset authorization or long-running training.
Until a positive optimized-target authority is integrated, the only executable
mode is deterministic LOCAL_FREE synthetic mechanics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, NoReturn

import torch

from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.checkpoint import (
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer

PROTOCOL_VERSION = 1
STATE_SCHEMA = "twelve-six-external-worker-state-v1"
MODE = "synthetic-mechanics"
MODEL_CARRIER_GIT_SHA = "61aa37b340565dd1ba791adc16ce430f9b17fbaf"
MODEL_SPEC_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_PARAMETERS = 20_613_440
MODEL_SEED = 341
SEQUENCE_LENGTH = 32
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_JSON_DEPTH = 12
_MAX_JSON_NODES = 4096
_MAX_TEXT_BYTES = 4096
_MAX_STEPS = 10_000
_HEX = frozenset("0123456789abcdef")
_ROOT_ENV = "TWELVE_SIX_EXTERNAL_TRAINING_ROOT"
_TRAINER_AUTHORITY_ENV = "TWELVE_SIX_TRAINER_AUTHORITY_SHA256"

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_MODEL_CONFIG = _PACKAGE_ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"


class ExternalTrainingWorkerError(RuntimeError):
    """Safe public failure for the subprocess protocol boundary."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant: {value}")


def _validate_tree(value: object, *, label: str) -> None:
    nodes = 0

    def visit(item: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ExternalTrainingWorkerError(f"{label} contains too many JSON values")
        if depth > _MAX_JSON_DEPTH:
            raise ExternalTrainingWorkerError(f"{label} exceeds the JSON nesting limit")
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ExternalTrainingWorkerError(f"{label} contains a non-finite number")
            return
        if type(item) is list:
            for child in item:
                visit(child, depth + 1)
            return
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ExternalTrainingWorkerError(f"{label} contains a non-string key")
                visit(child, depth + 1)
            return
        raise ExternalTrainingWorkerError(f"{label} contains a non-JSON value")

    visit(value, 0)


def _canonical_json_bytes(value: object, *, label: str) -> bytes:
    _validate_tree(value, label=label)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExternalTrainingWorkerError(f"{label} is not canonical JSON") from exc


def _parse_json(raw: bytes) -> dict[str, object]:
    if not raw or len(raw) > _MAX_REQUEST_BYTES:
        raise ExternalTrainingWorkerError("request size is invalid")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExternalTrainingWorkerError("request is not valid strict JSON") from exc
    if type(value) is not dict:
        raise ExternalTrainingWorkerError("request must be a JSON object")
    _validate_tree(value, label="request")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ExternalTrainingWorkerError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ExternalTrainingWorkerError(f"{name} must be canonical non-empty text")
    encoded = value.encode("utf-8")
    if len(encoded) > _MAX_TEXT_BYTES or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ExternalTrainingWorkerError(f"{name} contains unsupported text")
    return value


def _require_exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ExternalTrainingWorkerError(f"{label} has unexpected fields")


def _job_identity(request: dict[str, object]) -> dict[str, object]:
    job = request["job"]
    if type(job) is not dict:
        raise ExternalTrainingWorkerError("job must be a JSON object")
    _require_exact_keys(
        job,
        {
            "base_artifact",
            "candidate_artifact_ref",
            "job_id",
            "max_steps",
            "owner_id",
            "project_id",
            "resource_scope",
            "task_id",
        },
        label="job",
    )
    base_artifact = job["base_artifact"]
    if type(base_artifact) is not dict:
        raise ExternalTrainingWorkerError("base_artifact must be a JSON object")
    _require_exact_keys(base_artifact, {"artifact_ref", "sha256"}, label="base_artifact")
    base_ref = _require_text(base_artifact["artifact_ref"], name="base_artifact.artifact_ref")
    base_sha = _require_sha256(base_artifact["sha256"], name="base_artifact.sha256")
    max_steps = job["max_steps"]
    if type(max_steps) is not int or not 1 <= max_steps <= _MAX_STEPS:
        raise ExternalTrainingWorkerError("job.max_steps is outside the supported bound")
    return {
        "base_artifact": {"artifact_ref": base_ref, "sha256": base_sha},
        "candidate_artifact_ref": _require_text(
            job["candidate_artifact_ref"], name="job.candidate_artifact_ref"
        ),
        "job_id": _require_text(job["job_id"], name="job.job_id"),
        "max_steps": max_steps,
        "owner_id": _require_text(job["owner_id"], name="job.owner_id"),
        "project_id": _require_text(job["project_id"], name="job.project_id"),
        "resource_scope": _require_text(job["resource_scope"], name="job.resource_scope"),
        "task_id": _require_text(job["task_id"], name="job.task_id"),
    }


def _nika_job_fingerprint(job: dict[str, object]) -> str:
    encoded = _canonical_json_bytes(job, label="job identity")
    return hashlib.sha256(b"nika-training-job-v1\x00" + encoded).hexdigest()


def _nika_step_id(job_fingerprint: str, trainer_sha256: str, step_index: int) -> str:
    material = (
        f"nika-training-step-v1\x00{job_fingerprint}\x00{trainer_sha256}\x00{step_index}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def _validated_request(raw: bytes) -> dict[str, object]:
    request = _parse_json(raw)
    _require_exact_keys(
        request,
        {
            "job",
            "job_fingerprint",
            "previous_step_id",
            "protocol_version",
            "resume_state",
            "step_id",
            "step_index",
            "trainer_sha256",
        },
        label="request",
    )
    if type(request["protocol_version"]) is not int or request["protocol_version"] != PROTOCOL_VERSION:
        raise ExternalTrainingWorkerError("unsupported protocol version")

    job = _job_identity(request)
    expected_job_fingerprint = _nika_job_fingerprint(job)
    supplied_job_fingerprint = _require_sha256(
        request["job_fingerprint"], name="job_fingerprint"
    )
    if supplied_job_fingerprint != expected_job_fingerprint:
        raise ExternalTrainingWorkerError("job fingerprint does not match job identity")

    trainer_sha256 = _require_sha256(request["trainer_sha256"], name="trainer_sha256")
    expected_trainer_sha256 = _require_sha256(
        os.environ.get(_TRAINER_AUTHORITY_ENV), name=_TRAINER_AUTHORITY_ENV
    )
    if trainer_sha256 != expected_trainer_sha256:
        raise ExternalTrainingWorkerError("trainer authority does not match trusted environment")

    step_index = request["step_index"]
    max_steps = job["max_steps"]
    assert type(max_steps) is int
    if type(step_index) is not int or not 0 <= step_index < max_steps:
        raise ExternalTrainingWorkerError("step_index is outside job bounds")

    expected_step_id = _nika_step_id(expected_job_fingerprint, trainer_sha256, step_index)
    supplied_step_id = _require_sha256(request["step_id"], name="step_id")
    if supplied_step_id != expected_step_id:
        raise ExternalTrainingWorkerError("step identity does not match request authority")

    resume_state = request["resume_state"]
    if type(resume_state) is not dict:
        raise ExternalTrainingWorkerError("resume_state must be a JSON object")
    previous_step_id = request["previous_step_id"]
    if step_index == 0:
        if previous_step_id is not None or resume_state:
            raise ExternalTrainingWorkerError("initial step must not carry prior state")
    else:
        previous_step_id = _require_sha256(previous_step_id, name="previous_step_id")
        _validate_resume_state(
            resume_state,
            job_fingerprint=expected_job_fingerprint,
            trainer_sha256=trainer_sha256,
            previous_step_id=previous_step_id,
            step_index=step_index,
        )

    return {
        **request,
        "job": job,
        "job_fingerprint": expected_job_fingerprint,
        "step_id": expected_step_id,
        "trainer_sha256": trainer_sha256,
    }


def _validate_resume_state(
    state: dict[str, object],
    *,
    job_fingerprint: str,
    trainer_sha256: str,
    previous_step_id: str,
    step_index: int,
) -> None:
    _require_exact_keys(
        state,
        {
            "checkpoint_id",
            "checkpoint_weights_sha256",
            "init_spec_sha256",
            "job_fingerprint",
            "last_step_id",
            "mode",
            "model_spec_sha256",
            "model_state_sha256",
            "optimizer_step",
            "schema",
            "tokens_seen",
            "trainer_sha256",
        },
        label="resume_state",
    )
    if state["schema"] != STATE_SCHEMA or type(state["schema"]) is not str:
        raise ExternalTrainingWorkerError("resume_state schema is invalid")
    if state["mode"] != MODE or type(state["mode"]) is not str:
        raise ExternalTrainingWorkerError("resume_state mode is invalid")
    if state["job_fingerprint"] != job_fingerprint:
        raise ExternalTrainingWorkerError("resume_state belongs to a different job")
    if state["trainer_sha256"] != trainer_sha256:
        raise ExternalTrainingWorkerError("resume_state belongs to a different trainer")
    if state["model_spec_sha256"] != MODEL_SPEC_SHA256:
        raise ExternalTrainingWorkerError("resume_state model authority is invalid")
    if state["init_spec_sha256"] != INIT_SPEC_SHA256:
        raise ExternalTrainingWorkerError("resume_state init authority is invalid")
    if state["last_step_id"] != previous_step_id:
        raise ExternalTrainingWorkerError("resume_state does not match previous step")
    if type(state["optimizer_step"]) is not int or state["optimizer_step"] != step_index:
        raise ExternalTrainingWorkerError("resume_state optimizer progress is invalid")
    if type(state["tokens_seen"]) is not int or state["tokens_seen"] < 0:
        raise ExternalTrainingWorkerError("resume_state token progress is invalid")
    _require_sha256(state["checkpoint_id"], name="resume_state.checkpoint_id")
    _require_sha256(
        state["checkpoint_weights_sha256"], name="resume_state.checkpoint_weights_sha256"
    )
    _require_sha256(state["model_state_sha256"], name="resume_state.model_state_sha256")


def _trusted_root() -> Path:
    raw = os.environ.get(_ROOT_ENV)
    if type(raw) is not str or not raw:
        raise ExternalTrainingWorkerError(f"{_ROOT_ENV} is required")
    root = Path(raw)
    if not root.is_absolute():
        raise ExternalTrainingWorkerError(f"{_ROOT_ENV} must be absolute")
    if root.exists() and root.is_symlink():
        raise ExternalTrainingWorkerError("training root must not be a symbolic link")
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ExternalTrainingWorkerError("training root is not accessible") from exc
    if not stat.S_ISDIR(mode):
        raise ExternalTrainingWorkerError("training root must be a directory")
    return resolved


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(",".join(str(size) for size in value.shape).encode("ascii"))
        digest.update(b"\x00")
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _mechanics_dataset_hash(job_fingerprint: str) -> str:
    return hashlib.sha256(
        b"twelve-six-synthetic-mechanics-dataset-v1\x00" + job_fingerprint.encode("ascii")
    ).hexdigest()


def _run_manifest_hash(job_fingerprint: str, trainer_sha256: str, max_steps: int) -> str:
    payload = {
        "job_fingerprint": job_fingerprint,
        "max_steps": max_steps,
        "mode": MODE,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "trainer_sha256": trainer_sha256,
    }
    return hashlib.sha256(_canonical_json_bytes(payload, label="run manifest")).hexdigest()


def _fixed_tokenizer_hash(label: bytes) -> str:
    return hashlib.sha256(label).hexdigest()


def _trainer_config(max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=MODEL_SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _build_model_and_trainer(max_steps: int) -> tuple[Any, Trainer, Any]:
    if not _MODEL_CONFIG.is_file():
        raise ExternalTrainingWorkerError("MODEL-341 configuration is unavailable")
    torch.manual_seed(MODEL_SEED)
    stage = load_stage_config(_MODEL_CONFIG)
    if stage.model.identity_sha256() != MODEL_SPEC_SHA256:
        raise ExternalTrainingWorkerError("MODEL-341 model identity changed")
    if stage.init.identity_sha256() != INIT_SPEC_SHA256:
        raise ExternalTrainingWorkerError("MODEL-341 init identity changed")
    model = TwelveSixDecoder(stage.model, stage.init)
    if count_trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ExternalTrainingWorkerError("MODEL-341 parameter count changed")
    trainer = Trainer(model, _trainer_config(max_steps), device="cpu")
    return model, trainer, stage


def _synthetic_batch(
    *, job_fingerprint: str, step_id: str, vocab_size: int
) -> dict[str, torch.Tensor]:
    seed_material = hashlib.sha256(
        b"twelve-six-external-worker-batch-v1\x00"
        + job_fingerprint.encode("ascii")
        + b"\x00"
        + step_id.encode("ascii")
    ).digest()
    seed = int.from_bytes(seed_material[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (1, SEQUENCE_LENGTH),
        dtype=torch.long,
        generator=generator,
    )
    return {"input_ids": input_ids}


def _job_root(root: Path, job_fingerprint: str) -> Path:
    path = root / job_fingerprint
    path.mkdir(parents=False, exist_ok=True)
    if path.is_symlink():
        raise ExternalTrainingWorkerError("job state directory must not be a symbolic link")
    for child in ("dispatch", "results", "checkpoints"):
        target = path / child
        target.mkdir(exist_ok=True)
        if target.is_symlink():
            raise ExternalTrainingWorkerError("job state subdirectory must not be a symbolic link")
    return path


def _checkpoint_path(job_root: Path, optimizer_step: int) -> Path:
    return job_root / "checkpoints" / f"step-{optimizer_step:08d}"


def _atomic_create_json(path: Path, value: object) -> None:
    encoded = _canonical_json_bytes(value, label="durable worker state") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ExternalTrainingWorkerError("durable worker state could not be created") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _atomic_replace_json(path: Path, value: object) -> None:
    encoded = _canonical_json_bytes(value, label="durable worker result") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ExternalTrainingWorkerError("durable worker result could not be published") from exc


def _read_durable_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExternalTrainingWorkerError(f"{label} could not be read") from exc
    return _parse_json(raw)


def _checkpoint_identity(
    *, stage: Any, trainer: Trainer, job_fingerprint: str, trainer_sha256: str
) -> CheckpointIdentity:
    config = asdict(trainer.config)
    return CheckpointIdentity(
        git_sha=MODEL_CARRIER_GIT_SHA,
        model_spec=stage.model.to_dict(),
        parameter_count=EXPECTED_PARAMETERS,
        tokenizer_hash=_fixed_tokenizer_hash(b"12-6-byte-tokenizer-mechanics-v1"),
        tokenizer_vocab_hash=_fixed_tokenizer_hash(b"12-6-byte-vocab-0-255-v1"),
        dataset_manifest_hash=_mechanics_dataset_hash(job_fingerprint),
        run_manifest_hash=_run_manifest_hash(
            job_fingerprint, trainer_sha256, trainer.config.max_steps
        ),
        training_config={
            "mode": MODE,
            "model_spec_sha256": MODEL_SPEC_SHA256,
            "init_spec_sha256": INIT_SPEC_SHA256,
            "nika_job_fingerprint": job_fingerprint,
            "trainer_authority_sha256": trainer_sha256,
            "trainer_config": config,
        },
        seed=MODEL_SEED,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": type(trainer.optimizer).__name__,
            "config": {
                "learning_rate": trainer.config.learning_rate,
                "betas": list(trainer.config.betas),
                "eps": trainer.config.eps,
                "weight_decay": trainer.config.weight_decay,
            },
        },
        scheduler=(
            None if trainer.scheduler is None else {"name": type(trainer.scheduler).__name__}
        ),
    )


def _resume_checkpoint(
    *, job_root: Path, request: dict[str, object], model: Any, trainer: Trainer
) -> dict[str, object] | None:
    step_index = request["step_index"]
    assert type(step_index) is int
    if step_index == 0:
        return None
    previous = _checkpoint_path(job_root, step_index)
    if not previous.is_dir():
        raise ExternalTrainingWorkerError("required previous checkpoint is unavailable")
    job_fingerprint = request["job_fingerprint"]
    trainer_sha256 = request["trainer_sha256"]
    assert type(job_fingerprint) is str
    assert type(trainer_sha256) is str
    try:
        result = load_trainer_checkpoint(
            previous,
            model=model,
            trainer=trainer,
            restore_rng=True,
            expected_git_sha=MODEL_CARRIER_GIT_SHA,
            expected_model_spec_hash=MODEL_SPEC_SHA256,
            expected_init_spec_hash=INIT_SPEC_SHA256,
            expected_dataset_manifest_hash=_mechanics_dataset_hash(job_fingerprint),
            expected_run_manifest_hash=_run_manifest_hash(
                job_fingerprint, trainer_sha256, trainer.config.max_steps
            ),
            expected_seed=MODEL_SEED,
            expected_step=step_index,
        )
    except Exception as exc:
        raise ExternalTrainingWorkerError("previous checkpoint failed trusted restore") from exc

    state = request["resume_state"]
    assert type(state) is dict
    if result.manifest.get("checkpoint_id") != state["checkpoint_id"]:
        raise ExternalTrainingWorkerError("resume checkpoint identity mismatch")
    files = result.manifest.get("files")
    if type(files) is not dict:
        raise ExternalTrainingWorkerError("resume checkpoint file evidence is missing")
    weights = files.get("weights.safetensors")
    if type(weights) is not dict or weights.get("sha256") != state["checkpoint_weights_sha256"]:
        raise ExternalTrainingWorkerError("resume checkpoint weights identity mismatch")
    current_state_digest = _state_digest(model)
    if current_state_digest != state["model_state_sha256"]:
        raise ExternalTrainingWorkerError("resume model state identity mismatch")
    if trainer.optimizer_step != step_index:
        raise ExternalTrainingWorkerError("restored trainer progress mismatch")
    return result.manifest


def _confirmed_record(
    request: dict[str, object],
    response: dict[str, object],
    *,
    model_state_before: str,
    model_state_after: str,
    checkpoint_id: str,
    checkpoint_weights_sha256: str,
) -> dict[str, object]:
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_weights_sha256": checkpoint_weights_sha256,
        "job_fingerprint": request["job_fingerprint"],
        "model_state_after_sha256": model_state_after,
        "model_state_before_sha256": model_state_before,
        "response": response,
        "schema": "twelve-six-external-worker-confirmed-v1",
        "step_id": request["step_id"],
        "trainer_sha256": request["trainer_sha256"],
    }


def _validate_response(response: dict[str, object], *, request: dict[str, object]) -> None:
    _require_exact_keys(
        response,
        {"candidate_sha256", "completed", "protocol_version", "resume_state", "step_id"},
        label="response",
    )
    if response["protocol_version"] != PROTOCOL_VERSION or type(response["protocol_version"]) is not int:
        raise ExternalTrainingWorkerError("response protocol is invalid")
    if response["step_id"] != request["step_id"]:
        raise ExternalTrainingWorkerError("response step identity is invalid")
    if type(response["completed"]) is not bool:
        raise ExternalTrainingWorkerError("response completion flag is invalid")
    if type(response["resume_state"]) is not dict:
        raise ExternalTrainingWorkerError("response resume_state is invalid")
    candidate = response["candidate_sha256"]
    if response["completed"]:
        _require_sha256(candidate, name="candidate_sha256")
    elif candidate is not None:
        raise ExternalTrainingWorkerError("incomplete response published candidate digest")


def _load_replay(result_path: Path, request: dict[str, object]) -> dict[str, object]:
    record = _read_durable_json(result_path, label="confirmed step result")
    _require_exact_keys(
        record,
        {
            "checkpoint_id",
            "checkpoint_weights_sha256",
            "job_fingerprint",
            "model_state_after_sha256",
            "model_state_before_sha256",
            "response",
            "schema",
            "step_id",
            "trainer_sha256",
        },
        label="confirmed step result",
    )
    if record["schema"] != "twelve-six-external-worker-confirmed-v1":
        raise ExternalTrainingWorkerError("confirmed step result schema is invalid")
    for key in ("job_fingerprint", "step_id", "trainer_sha256"):
        if record[key] != request[key]:
            raise ExternalTrainingWorkerError("confirmed step result authority mismatch")
    _require_sha256(record["checkpoint_id"], name="confirmed.checkpoint_id")
    _require_sha256(
        record["checkpoint_weights_sha256"], name="confirmed.checkpoint_weights_sha256"
    )
    before = _require_sha256(
        record["model_state_before_sha256"], name="confirmed.model_state_before_sha256"
    )
    after = _require_sha256(
        record["model_state_after_sha256"], name="confirmed.model_state_after_sha256"
    )
    if before == after:
        raise ExternalTrainingWorkerError("confirmed step does not prove changed weights")
    response = record["response"]
    if type(response) is not dict:
        raise ExternalTrainingWorkerError("confirmed response is invalid")
    _validate_response(response, request=request)
    return response


def _execute_training_step(
    request: dict[str, object], *, job_root: Path
) -> dict[str, dict[str, object]]:
    job = request["job"]
    assert type(job) is dict
    max_steps = job["max_steps"]
    step_index = request["step_index"]
    job_fingerprint = request["job_fingerprint"]
    trainer_sha256 = request["trainer_sha256"]
    step_id = request["step_id"]
    assert type(max_steps) is int
    assert type(step_index) is int
    assert type(job_fingerprint) is str
    assert type(trainer_sha256) is str
    assert type(step_id) is str

    model, trainer, stage = _build_model_and_trainer(max_steps)
    _resume_checkpoint(job_root=job_root, request=request, model=model, trainer=trainer)

    model_state_before = _state_digest(model)
    batch = _synthetic_batch(
        job_fingerprint=job_fingerprint,
        step_id=step_id,
        vocab_size=stage.model.vocab_size,
    )
    try:
        metrics = trainer.train_microbatch(batch)
        trainer.assert_checkpoint_safe()
    except Exception as exc:
        raise ExternalTrainingWorkerError("MODEL-341/D02 optimizer step failed") from exc
    if not metrics.optimizer_stepped or trainer.optimizer_step != step_index + 1:
        raise ExternalTrainingWorkerError("trainer did not commit exactly one optimizer transition")

    model_state_after = _state_digest(model)
    if model_state_after == model_state_before:
        raise ExternalTrainingWorkerError("optimizer transition did not change model weights")

    checkpoint_path = _checkpoint_path(job_root, trainer.optimizer_step)
    identity = _checkpoint_identity(
        stage=stage,
        trainer=trainer,
        job_fingerprint=job_fingerprint,
        trainer_sha256=trainer_sha256,
    )
    try:
        manifest = save_trainer_checkpoint(
            checkpoint_path,
            model=model,
            trainer=trainer,
            identity=identity,
        )
    except Exception as exc:
        raise ExternalTrainingWorkerError(
            "optimizer step committed but checkpoint publication failed; reconciliation required"
        ) from exc

    checkpoint_id = _require_sha256(manifest.get("checkpoint_id"), name="checkpoint_id")
    files = manifest.get("files")
    if type(files) is not dict:
        raise ExternalTrainingWorkerError("checkpoint file evidence is missing")
    weights = files.get("weights.safetensors")
    if type(weights) is not dict:
        raise ExternalTrainingWorkerError("checkpoint weights evidence is missing")
    weights_sha256 = _require_sha256(
        weights.get("sha256"), name="checkpoint weights sha256"
    )

    resume_state = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_weights_sha256": weights_sha256,
        "init_spec_sha256": INIT_SPEC_SHA256,
        "job_fingerprint": job_fingerprint,
        "last_step_id": step_id,
        "mode": MODE,
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "model_state_sha256": model_state_after,
        "optimizer_step": trainer.optimizer_step,
        "schema": STATE_SCHEMA,
        "tokens_seen": trainer.tokens_seen,
        "trainer_sha256": trainer_sha256,
    }
    completed = trainer.optimizer_step >= max_steps
    response = {
        "candidate_sha256": weights_sha256 if completed else None,
        "completed": completed,
        "protocol_version": PROTOCOL_VERSION,
        "resume_state": resume_state,
        "step_id": step_id,
    }
    _validate_response(response, request=request)
    record = _confirmed_record(
        request,
        response,
        model_state_before=model_state_before,
        model_state_after=model_state_after,
        checkpoint_id=checkpoint_id,
        checkpoint_weights_sha256=weights_sha256,
    )
    return {"response": response, "record": record}


def handle_request(raw: bytes) -> bytes:
    """Validate one Nika-compatible request and execute or replay one exact step."""

    request = _validated_request(raw)
    root = _trusted_root()
    job_fingerprint = request["job_fingerprint"]
    step_id = request["step_id"]
    assert type(job_fingerprint) is str
    assert type(step_id) is str
    job_root = _job_root(root, job_fingerprint)
    dispatch_path = job_root / "dispatch" / f"{step_id}.json"
    result_path = job_root / "results" / f"{step_id}.json"

    if result_path.exists():
        response = _load_replay(result_path, request)
        return _canonical_json_bytes(response, label="response") + b"\n"
    if dispatch_path.exists():
        raise ExternalTrainingWorkerError(
            "step has unresolved DISPATCHING state; refusing blind optimizer replay"
        )

    dispatch_record = {
        "job_fingerprint": job_fingerprint,
        "schema": "twelve-six-external-worker-dispatch-v1",
        "step_id": step_id,
        "trainer_sha256": request["trainer_sha256"],
    }
    try:
        _atomic_create_json(dispatch_path, dispatch_record)
    except FileExistsError:
        if result_path.exists():
            response = _load_replay(result_path, request)
            return _canonical_json_bytes(response, label="response") + b"\n"
        raise ExternalTrainingWorkerError(
            "concurrent or unresolved step dispatch; refusing optimizer effect"
        ) from None

    executed = _execute_training_step(request, job_root=job_root)
    response = executed["response"]
    record = executed["record"]
    _atomic_replace_json(result_path, record)
    return _canonical_json_bytes(response, label="response") + b"\n"


def main() -> None:
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    try:
        response = handle_request(raw)
    except ExternalTrainingWorkerError as exc:
        print(f"external training worker rejected request: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()


__all__ = ["ExternalTrainingWorkerError", "handle_request", "main"]
