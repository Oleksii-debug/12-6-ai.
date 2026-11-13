"""Prepare or explicitly run the four-update SCALE-500M cybersecurity pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools import benchmark_plan3_native as native
from tools import cyber_pilot_stream
from tools import plan9_scale500m_launch_preparation_v1 as preparation
from twelve_six.checkpoint import CheckpointIdentity, bind_checkpoint_identity, hash_json
from twelve_six.model import build_canonical_base
from twelve_six.training import Trainer, runtime_release
from twelve_six.training.config import TrainingRecipe
from twelve_six.training.resource_envelope import (
    DISK_FLOOR_DENIAL,
    assess_resource_envelope,
    observe_host_resources,
)
from twelve_six.training.run_control import RunController, RunPhase

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "12-6.plan9-scale500m-cyber-pilot.v1"
PROFILE = "SCALE-500M"
RECIPE = "configs/plan9/scale500m_cyber_pilot_recipe_v1.json"
ROLE = "evidence/plan9_cyber_candidate/pilot_data_scope_v1.json"
RIGHTS_DECISION = "evidence/plan9_cyber_candidate/pilot_source_decision_v1.json"
REQUIRED_INPUTS = ("receipt.json", "packing-manifest.json", "shards/shard-000000.json", "shards/shard-000001.json")
CHECKPOINT_HEADROOM = 1024**3
HALF = 2


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(root: Path, commit: str, relative: str) -> bytes:
    return native._committed_file(root, commit, relative)


def _read(path: Path, limit: int = 40_000_000) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked pilot input: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"pilot input exceeds its size limit: {path.name}")
    return data


def build_packet(root: Path, bundle: Path, profile_id: str) -> tuple[dict, object]:
    if profile_id not in preparation.EXTERNAL_GPU_PROFILES:
        raise ValueError("pilot requires a committed Linux lock profile")
    commit = native._git(root, "rev-parse", "HEAD").decode().strip()
    for module in (native, cyber_pilot_stream, preparation, runtime_release):
        _exact(root, commit, Path(module.__file__).resolve().relative_to(root).as_posix())
    _exact(root, commit, Path(__file__).resolve().relative_to(root).as_posix())
    runtime_release.verify_runtime_release(root, release_version=2)
    recipe = TrainingRecipe.from_dict(json.loads(_exact(root, commit, RECIPE)))
    cfg, initialization, profile_sha = native._profile(root, commit, PROFILE)
    stream = cyber_pilot_stream.open_pilot_stream(
        root, bundle, recipe_sha256=recipe.identity_sha256(), optimizer_updates=recipe.trainer.max_steps,
        microbatch_size=recipe.microbatch_size, gradient_accumulation_steps=recipe.trainer.gradient_accumulation_steps,
    )
    if (
        recipe.trainer.max_steps != 4 or recipe.sequence_length != 32
        or recipe.maximum_total_exposures != stream.prefix_total_targets
        or recipe.target_unique_loss_positions != stream.prefix_total_targets
        or recipe.max_exposures_per_unique_position != 1
        or recipe.trainer.seed != initialization.seed
    ):
        raise ValueError("pilot recipe differs from its exact physical stream or profile")
    lock = preparation._lock_binding(root, commit, profile_id)
    scope_raw = _exact(root, commit, ROLE)
    scope = json.loads(scope_raw)
    if sha(_exact(root, commit, RIGHTS_DECISION)) != scope["source_decision_sha256"]:
        raise ValueError("pilot source decision differs from the pinned data scope")
    corpus_raw = _read(bundle / "receipt.json")
    pack_raw = _read(bundle / "packing-manifest.json")
    rights_raw = _read(bundle / "rights/receipt.json", 250_000)
    corpus, pack, rights = map(json.loads, (corpus_raw, pack_raw, rights_raw))
    if (
        sha(rights_raw) != scope["rights_receipt_sha256"]
        or rights["bundle_receipt_sha256"] != scope["corpus_receipt_sha256"]
        or scope["corpus_receipt_sha256"] != sha(corpus_raw)
        or corpus["native_packing_manifest_sha256"] != sha(pack_raw)
        or scope["input_sha256"]["packing-manifest.json"] != sha(pack_raw)
        or scope["input_sha256"]["receipt.json"] != sha(corpus_raw)
    ):
        raise ValueError("pilot rights do not match the exact corpus")
    for name, digest in rights["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("invalid source notice path")
        if sha(_read(bundle / "rights" / relative, 250_000)) != digest:
            raise ValueError(f"pilot source notice changed: {name}")
    if scope["input_sha256"] != {
        "receipt.json": sha(corpus_raw), "packing-manifest.json": sha(pack_raw),
        **{entry["path"]: entry["sha256"] for entry in pack["shards"][:2]},
    }:
        raise ValueError("transport scope differs from the exact pilot shards")
    cursors = [stream.cursor(index * recipe.trainer.gradient_accumulation_steps)
               for index in range(recipe.trainer.max_steps + 1)]
    training = recipe.run_training_fields()
    tokenizer = corpus["tokenizer"]["tokenizer_identity"]
    packing = {"config_sha256": corpus["native_packing_manifest_sha256"],
               "version": pack["schema_version"], "sequence_length": recipe.sequence_length}
    run = {
        "run_id": "scale500m-cyber-prefix-pilot-v1",
        "stage": PROFILE,
        "run_kind": "bounded_cybersecurity_pilot",
        "candidate": {
            "git_sha": commit, "modelspec_sha256": cfg.model.identity_sha256(),
            "initspec_sha256": cfg.init.identity_sha256(), "parameter_count": cfg.model.parameter_count(),
            "initialization": initialization.to_dict(),
        },
        "data": {
            "dataset_manifest_sha256": scope["corpus_receipt_sha256"],
            "split_identity": stream.stream_identity_sha256,
            "tokenizer_sha256": tokenizer["config_sha256"],
            "tokenizer_vocab_sha256": tokenizer["vocab_sha256"],
            "tokenizer_version": tokenizer["version"],
            "packing_sha256": packing["config_sha256"], "packing_version": packing["version"],
            "prefix_sha256": stream.prefix_sha256,
        },
        "training": training,
        "environment": {"lock_sha256": lock["runtime_lock_sha256"], "profile_id": profile_id},
        "data_scope_sha256": sha(scope_raw),
        "runtime_release_sha256": runtime_release.RELEASE_V2_SHA256,
        "control": {"pause_at_update": HALF, "max_updates": 4, "checkpoint_each_committed_update": True},
    }
    identity = bind_checkpoint_identity(
        run_manifest=run, model_spec=cfg.model.to_dict(), init_spec=cfg.init.to_dict(),
        tokenizer_identity=tokenizer, packing_identity=packing, step=0, tokens_seen=0,
    )
    core = {
        "schema": SCHEMA, "scope": "FOUR_UPDATE_CYBERSECURITY_PILOT",
        "run": run, "run_manifest_sha256": identity.run_manifest_hash, "identity": asdict(identity),
        "profile_sha256": profile_sha, "checkpoints": cursors, "data_scope": scope,
        "resource_floors": preparation._gpu_resource_floors(cfg.model, remaining_updates=4),
        "execution_authorized_by_packet": False, "production_campaign_authorized": False,
    }
    return {**core, "packet_sha256": hash_json(core)}, stream


def stage_input(bundle: Path, output: Path, packet: dict) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    scope = packet["data_scope"]
    rights = json.loads(_read(bundle / "rights/receipt.json", 250_000))
    inputs = {**scope["input_sha256"], "rights/receipt.json": scope["rights_receipt_sha256"],
              **{f"rights/{name}": digest for name, digest in rights["files"].items()}}
    if set(scope["input_sha256"]) != set(REQUIRED_INPUTS):
        raise ValueError("unexpected pilot input transport inventory")
    manifest = {
        "schema": "12-6.plan9-cyber-pilot-local-input.v1",
        "packet_sha256": packet["packet_sha256"],
        "files": inputs,
        "staged_blocks": scope["staged_blocks"],
        "evaluation_payloads_included": False,
        "external_transfer_authorized": False,
    }
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / "pilot-data"
        for name, digest in inputs.items():
            raw = _read(bundle / name)
            if sha(raw) != digest:
                raise ValueError(f"pilot transport member changed: {name}")
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            if sha(_read(path)) != digest:
                raise ValueError(f"pilot transport readback differs: {name}")
        (staged / "pilot-input-manifest.json").write_bytes(canonical(manifest))
        staged.rename(output)
    return {**manifest, "manifest_file_sha256": sha(canonical(manifest))}


class PilotCursor:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.position = 0

    def restore(self, value: Mapping) -> None:
        cursor = dict(value)
        self.stream.verify_cursor(cursor, cursor["cursor_sha256"])
        self.position = cursor["next_microbatch_index"]

    def batches(self, receipt):
        cursor = self.stream.cursor(self.position)
        if (
            cursor["optimizer_step"] != receipt.committed_step
            or cursor["consumed_loss_positions"] != receipt.tokens_seen
        ):
            raise ValueError("pilot receipt and physical cursor disagree")
        for batch in self.stream.iter_batches(cursor, cursor["cursor_sha256"]):
            self.position += 1
            yield {field: torch.tensor(value, dtype=torch.long) for field, value in batch.items()}

    def read(self, trainer) -> dict:
        cursor = self.stream.cursor(self.position)
        if trainer.micro_step != self.position or trainer.tokens_seen != cursor["consumed_loss_positions"]:
            raise ValueError("trainer counters differ from the verified pilot cursor")
        return cursor


def _admit(cfg, scratch: Path, *, remaining: int, prepare: bool, defer_disk: bool = False) -> None:
    observed = observe_host_resources(scratch, "cuda:0")
    report = assess_resource_envelope(cfg.model, observed, remaining_updates=remaining,
                                      initial_checkpoint_needed=prepare)
    denials = [reason for reason in report["denials"] if not (defer_disk and reason == DISK_FLOOR_DENIAL)]
    if not defer_disk and observed.checkpoint_volume_free_bytes < (
        report["unpruned_checkpoint_volume_floor_bytes"] + CHECKPOINT_HEADROOM
    ):
        denials.append("PILOT_DISK_HEADROOM_UNAVAILABLE")
    if denials:
        raise RuntimeError("pilot resource admission denied: " + ", ".join(denials))


def execute(root: Path, packet: dict, stream, args) -> dict:
    controller = RunController(args.scratch, packet["run_manifest_sha256"])
    if args.action == "status":
        return asdict(controller.status())
    if sys.platform != "linux":
        raise ValueError("pilot execution requires Linux on an external CUDA host")
    if not args.execute or args.expected_packet_sha256 != packet["packet_sha256"]:
        raise ValueError("execution requires --execute and the exact reviewed packet digest")
    if args.install_report is None or not native.SHA256.fullmatch(args.install_report_sha256 or ""):
        raise ValueError("execution requires the exact Linux wheel-install report")
    run = packet["run"]
    commit = run["candidate"]["git_sha"]
    native._verify_measurement_inputs(root, commit, run["environment"]["lock_sha256"],
                                      args.install_report, args.install_report_sha256, 2)
    if native.dependency_lock.current_profile_id() != run["environment"]["profile_id"]:
        raise ValueError("pilot packet Linux lock does not match this host")
    native._cuda_environment(":4096:8")
    cfg, init, _ = native._profile(root, commit, PROFILE)
    recipe = TrainingRecipe.from_dict(run["training"]["recipe"])
    identity = CheckpointIdentity(**packet["identity"])
    current = None if args.action == "prepare" else controller.status()
    if current is not None and current.receipt_sha256 != args.expected_receipt_sha256:
        raise ValueError("pilot execution requires the current durable receipt digest")
    if args.recover and current is None:
        raise ValueError("recovery requires an existing pilot receipt")
    if current is not None:
        allowed = (RunPhase.FAILED, RunPhase.RUNNING) if args.recover else (RunPhase.PREPARED, RunPhase.PAUSED)
        if current.phase not in allowed:
            raise ValueError("pilot phase does not allow this action")
    remaining = recipe.trainer.max_steps if current is None else recipe.trainer.max_steps - current.committed_step
    pending = args.recover and current is not None and current.pending_step is not None and current.pending_tokens is not None
    _admit(cfg, args.scratch, remaining=remaining, prepare=current is None, defer_disk=pending)
    if torch.get_default_dtype() != torch.float32 or torch.get_default_device().type != "cpu":
        raise ValueError("pilot requires default FP32 model allocation on CPU before placement")
    np.random.seed(recipe.trainer.seed)
    trainer = Trainer(build_canonical_base(cfg.model, cfg.init, init), recipe.trainer, device="cuda:0")
    if current is None:
        return asdict(controller.prepare(trainer, identity, exposure_cursor=stream.cursor(0)))
    hooks = PilotCursor(stream)

    def control(boundary):
        return RunPhase.PAUSED if boundary.committed_step == HALF else None

    def admit_recovery(receipt):
        _admit(cfg, args.scratch, remaining=recipe.trainer.max_steps - receipt.committed_step, prepare=False)

    return asdict(controller.run(
        trainer, identity, hooks.batches, expected_receipt_sha256=args.expected_receipt_sha256,
        recover=args.recover, control=control, read_exposure_cursor=hooks.read, restore_exposure_cursor=hooks.restore,
        control_policy_sha256=hash_json(run["control"]), admit_progress=admit_recovery if pending else None,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "stage-input", "status", "prepare", "run"), nargs="?", default="plan")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--lock-profile", default="linux-x86_64", choices=preparation.EXTERNAL_GPU_PROFILES)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-packet-sha256")
    parser.add_argument("--expected-receipt-sha256")
    parser.add_argument("--install-report", type=Path)
    parser.add_argument("--install-report-sha256")
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args(argv)
    try:
        packet, stream = build_packet(ROOT, args.bundle, args.lock_profile)
        if args.action == "plan":
            result = packet
        elif args.action == "stage-input":
            if args.stage_dir is None:
                raise ValueError("staging requires --stage-dir")
            result = stage_input(args.bundle, args.stage_dir, packet)
        else:
            if args.scratch is None or not args.scratch.is_dir():
                raise ValueError("pilot requires an existing --scratch directory")
            result = execute(ROOT, packet, stream, args)
        sys.stdout.buffer.write(canonical(result))
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"pilot refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
