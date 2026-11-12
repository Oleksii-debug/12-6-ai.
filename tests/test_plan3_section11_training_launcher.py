from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.request import ProxyHandler, Request, build_opener

import pytest
import torch
from test_plan3_section9_portable_packet import _batches, _settings
from test_plan3_section9_portable_packet import training_inputs as _portable_training_inputs

from twelve_six.checkpoint import CheckpointIntegrityError, hash_json
from twelve_six.training.launcher import (
    LaunchRequest,
    NativeTrainingEndpoint,
    ProcessTrainingTransport,
    TrainingLauncher,
    TrainingLaunchError,
)
from twelve_six.training.portable_packet import PortableTrainingPacket, PortableTrainingRun
from twelve_six.training.run_control import RunControlError, RunController, RunPhase

training_inputs = _portable_training_inputs


def _fixture_batches(_settings, index):
    return _batches(index)


def _disconnected(_settings, _index):
    raise OSError("configured fixture runner disconnected before opening source")


@dataclass(frozen=True)
class _Factory:
    packet: PortableTrainingPacket
    root: Path
    refs: dict
    adapter: str
    process_log: Path | None = None
    fail: bool = False

    def __call__(self):
        if self.process_log is not None:
            torch.set_num_threads(1)
            self.process_log.write_text(str(os.getpid()), encoding="ascii")
        source = _disconnected if self.fail else _fixture_batches
        return PortableTrainingRun(
            self.packet,
            _settings(self.packet, self.root, self.adapter, self.adapter),
            expected_packet_sha256=self.packet.identity_sha256(),
            sources={self.adapter: source},
            **self.refs,
        )


def _request(packet, action, receipt=None, *, recover=False, pause_at=None):
    return LaunchRequest(
        packet.identity_sha256(),
        packet.checkpoint_identity().run_manifest_hash,
        action,
        expected_receipt_sha256=None if receipt is None else receipt.receipt_sha256,
        recover=recover,
        boundary_step=pause_at,
        boundary_phase=None if pause_at is None else RunPhase.PAUSED,
    )


def _http_transport(endpoint):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            reply = endpoint(self.rfile.read(size))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint_url = f"http://127.0.0.1:{server.server_port}/train"
    opener = build_opener(ProxyHandler({}))

    def invoke(payload):
        request = Request(
            endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=15) as response:
            return response.read()

    return server, worker, invoke


def test_local_runner_failure_and_loopback_server_resume_one_canonical_run(
    training_inputs,
    tmp_path: Path,
):
    packet, refs = training_inputs
    baseline = TrainingLauncher(
        "local",
        NativeTrainingEndpoint(_Factory(packet, tmp_path / "baseline", refs, "local")()),
    )
    root = tmp_path / "authoritative"
    local = TrainingLauncher(
        "local", NativeTrainingEndpoint(_Factory(packet, root, refs, "local")())
    )
    base_start = baseline.invoke(_request(packet, "prepare")).run
    start = local.invoke(_request(packet, "prepare")).run
    assert start.checkpoint_id == base_start.checkpoint_id
    full = baseline.invoke(_request(packet, "run", base_start)).run
    paused = local.invoke(_request(packet, "run", start, pause_at=1)).run
    assert (paused.phase, paused.committed_step) == (RunPhase.PAUSED, 1)

    process_log = tmp_path / "runner-pid.txt"
    failing = TrainingLauncher(
        "runner",
        ProcessTrainingTransport(_Factory(packet, root, refs, "free", process_log, fail=True)),
    )
    with pytest.raises(TrainingLaunchError, match="runner disconnected"):
        failing.invoke(_request(packet, "run", paused))
    assert int(process_log.read_text(encoding="ascii")) != os.getpid()
    failed = local.invoke(_request(packet, "status")).run
    assert (failed.phase, failed.committed_step, failed.checkpoint_id) == (
        RunPhase.FAILED,
        1,
        paused.checkpoint_id,
    )
    runner = TrainingLauncher(
        "runner",
        ProcessTrainingTransport(_Factory(packet, root, refs, "free", process_log)),
    )
    halfway = runner.invoke(_request(packet, "run", failed, recover=True, pause_at=2)).run
    assert (halfway.phase, halfway.committed_step) == (RunPhase.PAUSED, 2)
    assert int(process_log.read_text(encoding="ascii")) != os.getpid()

    server, worker, transport = _http_transport(
        NativeTrainingEndpoint(_Factory(packet, root, refs, "server")()),
    )
    try:
        remote = TrainingLauncher(
            "server",
            transport,
            authorize_server=lambda request: request.packet_sha256 == packet.identity_sha256(),
        )
        done = remote.invoke(_request(packet, "run", halfway)).run
        artifacts = remote.invoke(_request(packet, "artifacts", done)).artifacts
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert (done.phase, done.committed_step, done.tokens_seen) == (RunPhase.COMPLETED, 3, 4)
    assert done.run_sha256 == start.run_sha256 == full.run_sha256
    assert done.checkpoint_id == full.checkpoint_id
    uninterrupted = baseline.invoke(_request(packet, "artifacts", full)).artifacts
    assert artifacts is not None and uninterrupted is not None
    assert artifacts["manifest"]["files"] == uninterrupted["manifest"]["files"]
    assert artifacts["exposure_cursor_sha256"] == uninterrupted["exposure_cursor_sha256"]


def test_server_without_explicit_authorization_does_not_contact_transport(
    training_inputs,
):
    packet, _ = training_inputs

    def unexpected(_payload):
        pytest.fail("unauthorized server transport was contacted")

    for check in (None, lambda _: False, lambda _: 1):
        launcher = TrainingLauncher("server", unexpected, authorize_server=check)
        with pytest.raises(TrainingLaunchError, match="no explicit authorization"):
            launcher.invoke(_request(packet, "prepare"))


@pytest.mark.parametrize("mutation", ["negative", "boolean", "phase", "artifact", "file_hash"])
def test_malformed_self_hashed_wire_receipt_is_rejected(
    training_inputs,
    tmp_path: Path,
    mutation: str,
):
    packet, refs = training_inputs
    endpoint = NativeTrainingEndpoint(_Factory(packet, tmp_path, refs, "local")())
    local = TrainingLauncher("local", endpoint)
    prepared = local.invoke(_request(packet, "prepare")).run

    def forged(payload):
        reply = json.loads(endpoint(payload))
        state = reply["run"]
        if mutation == "negative":
            state["tokens_seen"] = -1
        elif mutation == "boolean":
            state["committed_step"] = True
        elif mutation == "phase":
            state["phase"] = "UNKNOWN"
        else:
            artifact = reply["artifacts"]
            manifest = artifact["manifest"]
            if mutation == "artifact":
                manifest["files"]["extra.bin"] = {"bytes": 1, "sha256": "a" * 64}
            else:
                manifest["files"]["weights.safetensors"]["sha256"] = "e" * 64
            artifact["checkpoint_manifest_identity_sha256"] = hash_json(manifest)
        state["receipt_sha256"] = hash_json(
            {key: value for key, value in state.items() if key != "receipt_sha256"}
        )
        reply["receipt_sha256"] = hash_json(
            {key: value for key, value in reply.items() if key != "receipt_sha256"}
        )
        return json.dumps(reply).encode()

    launcher = TrainingLauncher("local", forged)
    request = (
        _request(packet, "artifacts", prepared)
        if mutation in {"artifact", "file_hash"}
        else _request(packet, "status")
    )
    with pytest.raises(TrainingLaunchError, match="invalid|unknown|inventory|bind the checkpoint"):
        launcher.invoke(request)
    assert local.invoke(_request(packet, "status")).run == prepared


def test_artifact_requests_require_current_receipt_and_verified_payload(
    training_inputs,
    tmp_path: Path,
):
    packet, refs = training_inputs
    local = TrainingLauncher(
        "local", NativeTrainingEndpoint(_Factory(packet, tmp_path, refs, "local")())
    )
    prepared = local.invoke(_request(packet, "prepare")).run
    paused = local.invoke(_request(packet, "run", prepared, pause_at=1)).run
    with pytest.raises(TrainingLaunchError, match="stale"):
        local.invoke(_request(packet, "artifacts", prepared))
    path = (
        tmp_path
        / paused.run_sha256
        / "checkpoints"
        / f"{paused.checkpoint_epoch:08d}-{paused.committed_step:08d}"
        / "weights.safetensors"
    )
    path.write_bytes(b"corrupt")
    with pytest.raises(CheckpointIntegrityError):
        local.invoke(_request(packet, "artifacts", paused))


def test_replacement_adapter_replays_exact_durable_stop_policy(
    training_inputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    packet, refs = training_inputs
    local = TrainingLauncher(
        "local",
        NativeTrainingEndpoint(_Factory(packet, tmp_path, refs, "local")()),
    )
    prepared = local.invoke(_request(packet, "prepare")).run
    original = RunController._save_checkpoint

    def fail_after_publication(path, trainer, identity):
        original(path, trainer, identity)
        raise OSError("interrupted after publication")

    stop = LaunchRequest(
        packet.identity_sha256(),
        prepared.run_sha256,
        "run",
        expected_receipt_sha256=prepared.receipt_sha256,
        boundary_step=1,
        boundary_phase=RunPhase.STOPPED,
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(RunController, "_save_checkpoint", staticmethod(fail_after_publication))
        with pytest.raises(OSError, match="interrupted after publication"):
            local.invoke(stop)
    failed = local.invoke(_request(packet, "status")).run
    assert (failed.phase, failed.committed_step, failed.pending_step) == (RunPhase.FAILED, 0, 1)
    assert failed.control_policy_sha256 == stop.control_policy_sha256()

    replacement = TrainingLauncher(
        "server",
        NativeTrainingEndpoint(_Factory(packet, tmp_path, refs, "server")()),
        authorize_server=lambda _: True,
    )
    with pytest.raises(RunControlError, match="control policy differs"):
        replacement.invoke(_request(packet, "run", failed, recover=True, pause_at=2))
    assert replacement.invoke(_request(packet, "status")).run == failed
    replay = LaunchRequest(
        packet.identity_sha256(),
        failed.run_sha256,
        "run",
        expected_receipt_sha256=failed.receipt_sha256,
        recover=True,
        boundary_step=1,
        boundary_phase=RunPhase.STOPPED,
    )
    stopped = replacement.invoke(replay).run
    assert (stopped.phase, stopped.committed_step, stopped.tokens_seen) == (RunPhase.STOPPED, 1, 1)
