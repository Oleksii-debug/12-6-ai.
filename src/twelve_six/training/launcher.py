"""One native training lifecycle over replaceable local, runner and server transports."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from multiprocessing.connection import Connection
from typing import Any, Literal

from ..checkpoint import hash_json
from ..checkpoint.core import (
    FORMAT_NAME,
    FORMAT_VERSION,
    STATE_TENSORS_NAME,
    STATE_TREE_NAME,
    WEIGHTS_NAME,
    _require_exact_hex,
    _strict_json_object,
    canonical_json_bytes,
)
from ..checkpoint.transaction import checkpoint_transaction
from .portable_packet import PortableTrainingRun
from .run_control import RunBoundary, RunPhase, RunReceipt

_REQUEST_SCHEMA = "12-6.training-launch-request.v1"
_RECEIPT_SCHEMA = "12-6.training-launch-receipt.v1"
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

LaunchAction = Literal["prepare", "run", "status", "artifacts"]
LaunchAdapter = Literal["local", "runner", "server"]
TrainingTransport = Callable[[bytes], bytes]


class TrainingLaunchError(RuntimeError):
    """The selected transport did not complete a verified launcher operation."""


def _decode(payload: bytes, artifact: str) -> dict[str, Any]:
    if type(payload) is not bytes or len(payload) > _MAX_MESSAGE_BYTES:
        raise TrainingLaunchError(f"{artifact} exceeds the wire contract")
    return _strict_json_object(payload, artifact=artifact, invalid_message=f"invalid {artifact}")


def _require_digest(value: Any, field: str) -> None:
    try:
        _require_exact_hex(value, field=field, lengths={64})
    except ValueError as exc:
        raise TrainingLaunchError(str(exc)) from exc


def _read_run_receipt(state: Any, expected_run: str) -> RunReceipt:
    if not isinstance(state, dict) or set(state) != {field.name for field in fields(RunReceipt)}:
        raise TrainingLaunchError("launcher response does not contain a complete run receipt")
    for field in ("run_sha256", "identity_sha256", "checkpoint_id", "receipt_sha256"):
        _require_digest(state[field], field)
    for field in ("parent_sha256", "pending_cursor_sha256", "control_policy_sha256"):
        if state[field] is not None:
            _require_digest(state[field], field)
    for field in (
        "generation",
        "control_epoch",
        "committed_step",
        "tokens_seen",
        "checkpoint_epoch",
    ):
        if type(state[field]) is not int or state[field] < 0:
            raise TrainingLaunchError(f"invalid launcher run receipt count {field}")
    for field in ("pending_step", "pending_tokens"):
        if state[field] is not None and (type(state[field]) is not int or state[field] < 0):
            raise TrainingLaunchError(f"invalid launcher run receipt count {field}")
    for field in ("control_enabled", "control_pending"):
        if type(state[field]) is not bool:
            raise TrainingLaunchError(f"invalid launcher run receipt flag {field}")
    schema = state["exposure_cursor_schema"]
    if schema is not None and (type(schema) is not str or not schema or schema.strip() != schema):
        raise TrainingLaunchError("invalid launcher exposure cursor schema")
    try:
        phase = RunPhase(state["phase"])
    except (TypeError, ValueError) as exc:
        raise TrainingLaunchError("unknown launcher run receipt phase") from exc
    receipt_fields = {key: item for key, item in state.items() if key != "receipt_sha256"}
    digest = hash_json(receipt_fields)
    if digest != state["receipt_sha256"] and state["control_policy_sha256"] is None:
        receipt_fields.pop("control_policy_sha256")
        digest = hash_json(receipt_fields)
    if digest != state["receipt_sha256"] or state["run_sha256"] != expected_run:
        raise TrainingLaunchError("launcher run receipt failed identity verification")
    return RunReceipt(**{**state, "phase": phase})


def _validate_artifacts(artifacts: Any, request: LaunchRequest, receipt: RunReceipt) -> None:
    if request.action != "artifacts":
        if artifacts is not None:
            raise TrainingLaunchError("unexpected launcher artifacts")
        return
    if (
        not isinstance(artifacts, dict)
        or set(artifacts)
        != {
            "checkpoint_id",
            "checkpoint_manifest_identity_sha256",
            "exposure_cursor_sha256",
            "manifest",
        }
        or artifacts["checkpoint_id"] != receipt.checkpoint_id
        or receipt.receipt_sha256 != request.expected_receipt_sha256
    ):
        raise TrainingLaunchError("launcher artifacts differ from the committed receipt")
    _require_digest(artifacts["checkpoint_manifest_identity_sha256"], "checkpoint manifest")
    if artifacts["exposure_cursor_sha256"] is not None:
        _require_digest(artifacts["exposure_cursor_sha256"], "exposure cursor")
    manifest = artifacts["manifest"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT_NAME
        or type(manifest.get("format_version")) is not int
        or manifest["format_version"] != FORMAT_VERSION
        or manifest.get("checkpoint_id") != receipt.checkpoint_id
        or hash_json(manifest) != artifacts["checkpoint_manifest_identity_sha256"]
    ):
        raise TrainingLaunchError("launcher artifact manifest failed identity verification")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        WEIGHTS_NAME,
        STATE_TREE_NAME,
        STATE_TENSORS_NAME,
    }:
        raise TrainingLaunchError("launcher artifact inventory differs from checkpoint-v1")
    for name, record in files.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
        ):
            raise TrainingLaunchError(f"invalid launcher artifact record {name}")
        _require_digest(record["sha256"], name)
    identity = manifest.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("run_manifest_hash") != receipt.run_sha256
        or type(identity.get("step")) is not int
        or identity["step"] != receipt.committed_step
        or type(identity.get("tokens_seen")) is not int
        or identity["tokens_seen"] != receipt.tokens_seen
        or hash_json({"identity": identity, "files": files}) != receipt.checkpoint_id
    ):
        raise TrainingLaunchError("launcher artifact files do not bind the checkpoint identity")
    transaction = checkpoint_transaction(manifest)
    cursor = None if transaction is None else transaction.exposure_cursor_sha256
    if cursor != artifacts["exposure_cursor_sha256"]:
        raise TrainingLaunchError("launcher artifact exposure cursor failed identity verification")


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    packet_sha256: str
    run_manifest_sha256: str
    action: LaunchAction
    expected_receipt_sha256: str | None = None
    recover: bool = False
    boundary_step: int | None = None
    boundary_phase: RunPhase | None = None

    def __post_init__(self) -> None:
        _require_exact_hex(self.packet_sha256, field="packet_sha256", lengths={64})
        _require_exact_hex(self.run_manifest_sha256, field="run_manifest_sha256", lengths={64})
        if self.action not in ("prepare", "run", "status", "artifacts"):
            raise ValueError("unsupported training launcher operation")
        if type(self.recover) is not bool:
            raise ValueError("launcher recovery flag must be boolean")
        if self.action in ("run", "artifacts"):
            _require_exact_hex(
                self.expected_receipt_sha256,
                field="expected_receipt_sha256",
                lengths={64},
            )
        elif self.expected_receipt_sha256 is not None:
            raise ValueError("launcher operation does not accept a prior receipt")
        if self.action != "run" and (
            self.recover or self.boundary_step is not None or self.boundary_phase is not None
        ):
            raise ValueError("only run accepts recovery and boundary control")
        if self.boundary_step is None and self.boundary_phase is None:
            return
        if (
            type(self.boundary_step) is not int
            or self.boundary_step < 1
            or (
                self.boundary_phase is not RunPhase.PAUSED
                and self.boundary_phase is not RunPhase.STOPPED
            )
        ):
            raise ValueError("launcher boundary requires a positive step and PAUSED or STOPPED")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": _REQUEST_SCHEMA, **asdict(self)}

    @classmethod
    def from_bytes(cls, payload: bytes) -> LaunchRequest:
        value = _decode(payload, "training launch request")
        if set(value) != {"schema", *(field.name for field in fields(cls))}:
            raise ValueError("launcher request fields differ from the wire schema")
        if value.pop("schema") != _REQUEST_SCHEMA:
            raise ValueError("unsupported training launcher request schema")
        if value["boundary_phase"] is not None:
            value["boundary_phase"] = RunPhase(value["boundary_phase"])
        return cls(**value)

    def control(self, boundary: RunBoundary) -> RunPhase | None:
        return self.boundary_phase if boundary.committed_step == self.boundary_step else None

    def control_policy_sha256(self) -> str | None:
        if self.boundary_step is None:
            return None
        return hash_json(
            {"boundary_step": self.boundary_step, "boundary_phase": self.boundary_phase}
        )


@dataclass(frozen=True, slots=True)
class LaunchReceipt:
    request_sha256: str
    run: RunReceipt
    artifacts: dict[str, Any] | None


class NativeTrainingEndpoint:
    """Dispatch the same bounded wire operations to the same portable runtime."""

    def __init__(self, run: PortableTrainingRun) -> None:
        self._run = run

    def __call__(self, payload: bytes) -> bytes:
        request = LaunchRequest.from_bytes(payload)
        run = self._run
        if (
            request.packet_sha256 != run.packet_sha256
            or request.run_manifest_sha256 != run.run_manifest_sha256
        ):
            raise TrainingLaunchError("launcher request does not bind the configured run")
        artifacts = None
        if request.action == "prepare":
            receipt = run.prepare()
        elif request.action == "run":
            assert request.expected_receipt_sha256 is not None
            receipt = run.run(
                expected_receipt_sha256=request.expected_receipt_sha256,
                recover=request.recover,
                control=request.control if request.boundary_step is not None else None,
                control_policy_sha256=request.control_policy_sha256(),
            )
        else:
            receipt = run.status()
            if request.action == "artifacts":
                if request.expected_receipt_sha256 != receipt.receipt_sha256:
                    raise TrainingLaunchError("stale run receipt for artifact readback")
                artifacts = run.checkpoint_artifacts(receipt)
        body = {
            "schema": _RECEIPT_SCHEMA,
            "request_sha256": hash_json(request.to_dict()),
            "run": asdict(receipt),
            "artifacts": artifacts,
        }
        return canonical_json_bytes({**body, "receipt_sha256": hash_json(body)})


class TrainingLauncher:
    """Use a configured transport without adding provider settings to science."""

    def __init__(
        self,
        adapter: LaunchAdapter,
        transport: TrainingTransport,
        *,
        authorize_server: Callable[[LaunchRequest], bool] | None = None,
    ) -> None:
        if adapter not in ("local", "runner", "server") or not callable(transport):
            raise ValueError("invalid training launcher adapter or transport")
        self._adapter = adapter
        self._transport = transport
        self._authorize_server = authorize_server

    def invoke(self, request: LaunchRequest) -> LaunchReceipt:
        if self._adapter == "server" and (
            self._authorize_server is None or self._authorize_server(request) is not True
        ):
            raise TrainingLaunchError("server compute dispatch has no explicit authorization")
        expected_request = hash_json(request.to_dict())
        raw = self._transport(canonical_json_bytes(request.to_dict()))
        value = _decode(raw, "training launcher receipt")
        if set(value) != {"schema", "request_sha256", "run", "artifacts", "receipt_sha256"}:
            raise TrainingLaunchError("launcher receipt fields differ from the wire schema")
        claimed = value.pop("receipt_sha256")
        if (
            value["schema"] != _RECEIPT_SCHEMA
            or claimed != hash_json(value)
            or value["request_sha256"] != expected_request
        ):
            raise TrainingLaunchError("launcher receipt identity differs from the request")
        state = _read_run_receipt(value["run"], request.run_manifest_sha256)
        artifacts = value["artifacts"]
        _validate_artifacts(artifacts, request, state)
        return LaunchReceipt(expected_request, state, artifacts)


def _process_dispatch(
    factory: Callable[[], PortableTrainingRun],
    payload: bytes,
    sender: Connection,
) -> None:
    with sender:
        try:
            reply = b"\x00" + NativeTrainingEndpoint(factory())(payload)
        except Exception as exc:  # noqa: BLE001 - preserve errors from configured worker code
            reply = b"\x01" + f"{type(exc).__name__}: {exc}".encode()[:4096]
        sender.send_bytes(reply)


class ProcessTrainingTransport:
    """Open the same packet in a fresh configured runner process for each operation."""

    def __init__(
        self,
        factory: Callable[[], PortableTrainingRun],
        *,
        timeout_seconds: int = 60,
    ) -> None:
        if not callable(factory) or type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("runner requires a factory and a positive timeout")
        self._factory = factory
        self._timeout = timeout_seconds

    def __call__(self, payload: bytes) -> bytes:
        context = multiprocessing.get_context("spawn")
        reader, sender = context.Pipe(duplex=False)
        with reader, sender:
            process = context.Process(
                target=_process_dispatch, args=(self._factory, payload, sender)
            )
            process.start()
            sender.close()
            try:
                if not reader.poll(self._timeout):
                    raise TrainingLaunchError(
                        "runner timed out; query the authoritative run status"
                    )
                try:
                    raw = reader.recv_bytes(maxlength=_MAX_MESSAGE_BYTES + 1)
                except (EOFError, OSError) as exc:
                    raise TrainingLaunchError(
                        "runner disconnected; query the authoritative run status"
                    ) from exc
            finally:
                process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                if process.is_alive():
                    raise TrainingLaunchError("runner did not exit after cancellation")
                process.close()
        if raw[:1] != b"\x00":
            raise TrainingLaunchError(raw[1:].decode("utf-8", errors="replace"))
        return raw[1:]
