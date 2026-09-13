"""Fail-closed execution gate for the learned-20M bounded LOCAL_FREE pilot.

The gate consumes canonical readiness/run-packet, D10 launch-input, and D04
ordered/content exposure authorities. It does not authorize training itself.
Every learned-target optimizer transition is bound to the exact live model,
optimizer, Trainer state, packet roots, and loss-bearing content immediately
before ``optimizer.step()``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from twelve_six.data.deterministic_exposure_order import ordered_next_exposure_identity
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.loss_bearing_content_binding_v1 import verify_live_loss_bearing_batch
from twelve_six.model import TwelveSixDecoder
from twelve_six.portable_run_binding import PortableRunBinding, canonical_sha256

from .single_gpu import SingleDeviceStepMetrics, SingleDeviceStepRunner
from .trainer import Trainer, build_optimizer

Batch = Mapping[str, Tensor]


class BoundedPilotAuthorizationError(RuntimeError):
    """Raised when a learned-target optimizer update is not exactly authorized."""


class BoundedPilotRecoveryRequiredError(BoundedPilotAuthorizationError):
    """Raised once execution may have crossed a learned-state commit boundary."""


@dataclass(frozen=True, slots=True)
class BoundedPilotStepReceipt:
    """Text-free evidence for one exact authorized optimizer transition."""

    optimizer_step_before: int
    optimizer_step_after: int
    actual_nonignored_targets: int
    exposure_identity_sha256: str
    consumed_loss_positions_after: int
    packet_sha256: str
    modelspec_sha256: str
    initspec_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_step_before": self.optimizer_step_before,
            "optimizer_step_after": self.optimizer_step_after,
            "actual_nonignored_targets": self.actual_nonignored_targets,
            "exposure_identity_sha256": self.exposure_identity_sha256,
            "consumed_loss_positions_after": self.consumed_loss_positions_after,
            "packet_sha256": self.packet_sha256,
            "modelspec_sha256": self.modelspec_sha256,
            "initspec_sha256": self.initspec_sha256,
        }


@dataclass(slots=True)
class _PendingAuthorization:
    batch: Batch
    batch_index: int
    expected_identity: str
    actual_targets: int


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: {label} is not canonical sha256"
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _authority_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_LAUNCH_INPUT_SCHEMA = "12-6.learned20m-launch-input-authority.v2"
_LAUNCH_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "binding_status",
        "data_spine",
        "carrier",
        "claim_boundary",
        "authority_identity_sha256",
    }
)
_LAUNCH_DATA_SPINE_KEYS = frozenset(
    {
        "terminal_corpus_authority_identity_sha256",
        "stage_bindings",
        "deterministic_double_pack_proof_identity_sha256",
        "terminal_record_inventory_digest_sha256",
        "terminal_payload_inventory_digest_sha256",
        "terminal_split_application_identity_sha256",
        "terminal_split_spec_identity_sha256",
        "terminal_split_train_record_membership_sha256",
        "canonical_build_sha256",
        "two_clean_proof_identity_sha256",
        "two_clean_input_packet_identity_sha256",
        "two_clean_runtime_identity_sha256",
        "materialization_identity_sha256",
        "unique_loss_ledger_identity_sha256",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
        "one_pass_unique_nonignored_causal_loss_positions",
        "requested_unique_loss_positions",
    }
)
_LAUNCH_CARRIER_KEYS = frozenset(
    {
        "repository",
        "git_sha",
        "modelspec_sha256",
        "initialization_identity_sha256",
        "canonical_base",
        "foreign_pretrained_weights_used",
        "terminal",
        "workflow_run_id",
        "workflow_status",
        "workflow_conclusion",
        "workflow_head_sha",
        "evidence_sha256",
    }
)
_LAUNCH_CLAIM_BOUNDARY = {
    "contains_source_text": False,
    "final_test_payload_consumed": False,
    "authorizes_training": False,
    "authorizes_compute": False,
    "authorized_optimized_target_exposure": 0,
    "replay_padding_or_replacement_can_increase_unique_capacity": False,
}


def _verify_launch_input_v2_authority(
    authority: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_modelspec_sha256: str,
    expected_initspec_sha256: str,
    expected_ledger_sha256: str,
    expected_requested_unique_loss_positions: int,
) -> None:
    expected = _require_sha256(expected_identity_sha256, label="external D10 launch-input root")
    if not isinstance(authority, Mapping) or set(authority) != set(_LAUNCH_ROOT_KEYS):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 launch-input authority root is not closed-world"
        )
    value = copy.deepcopy(dict(authority))
    if value.get("schema_version") != _LAUNCH_INPUT_SCHEMA:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 launch-input v2 authority required"
        )
    if value.get("binding_status") != "READY_FOR_READINESS_BINDING":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 launch-input authority is not terminal for readiness"
        )
    observed = _require_sha256(value.get("authority_identity_sha256"), label="D10 authority root")
    body = copy.deepcopy(value)
    body.pop("authority_identity_sha256", None)
    if _authority_sha256(body) != observed or observed != expected:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 launch-input authority identity mismatch"
        )
    data_spine = value.get("data_spine")
    carrier = value.get("carrier")
    claim_boundary = value.get("claim_boundary")
    if not isinstance(data_spine, Mapping) or set(data_spine) != set(_LAUNCH_DATA_SPINE_KEYS):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 data spine is not closed-world"
        )
    if not isinstance(carrier, Mapping) or set(carrier) != set(_LAUNCH_CARRIER_KEYS):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier is not closed-world"
        )
    if not isinstance(claim_boundary, Mapping) or dict(claim_boundary) != _LAUNCH_CLAIM_BOUNDARY:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 claim boundary widened"
        )
    if carrier.get("modelspec_sha256") != expected_modelspec_sha256:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier ModelSpec differs from runtime packet"
        )
    if carrier.get("initialization_identity_sha256") != expected_initspec_sha256:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier InitSpec differs from runtime packet"
        )
    if carrier.get("canonical_base") != "random_init":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier is not random_init"
        )
    if carrier.get("foreign_pretrained_weights_used") is not False:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier permits foreign pretrained weights"
        )
    if carrier.get("terminal") is not True or carrier.get("workflow_conclusion") != "success":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 carrier is not terminal-success"
        )
    if data_spine.get("unique_loss_ledger_identity_sha256") != expected_ledger_sha256:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 unique-loss ledger differs from runtime D04 ledger"
        )
    capacity = data_spine.get("one_pass_unique_nonignored_causal_loss_positions")
    requested = data_spine.get("requested_unique_loss_positions")
    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
        or requested > capacity
    ):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 unique-loss capacity/request is invalid"
        )
    if requested != expected_requested_unique_loss_positions:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 requested unique-loss budget differs from runtime packet"
        )


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_typed_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_typed_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256(b"12-6.model-state.v1\x00")
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        digest.update(_canonical_json_bytes(metadata))
        digest.update(b"\x00")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _require_fresh_start_model_state(trainer: Trainer) -> str:
    model = trainer.model
    if type(model) is not TwelveSixDecoder:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: canonical TwelveSixDecoder required"
        )
    if trainer.optimizer_step != 0 or trainer.tokens_seen != 0 or trainer.micro_step != 0:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: FRESH_START Trainer counters must be zero"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(trainer.config.seed)
        reference = TwelveSixDecoder(model.spec, model.init_spec)
    expected = _model_state_sha256(reference)
    observed = _model_state_sha256(model)
    if observed != expected:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: live model tensors differ from canonical random initialization"
        )
    return observed


def _batch_projection(batch: Batch) -> tuple[Tensor, Tensor, Tensor | None, bool]:
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise BoundedPilotAuthorizationError("batch input_ids must be a rank-2 tensor")
    if "labels" in batch and "target_ids" in batch:
        raise BoundedPilotAuthorizationError("batch must not contain both labels and target_ids")
    if "target_ids" in batch:
        targets = batch["target_ids"]
        if not isinstance(targets, Tensor) or targets.ndim != 2:
            raise BoundedPilotAuthorizationError("target_ids must be a rank-2 tensor")
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None and (
            not isinstance(loss_mask, Tensor) or loss_mask.shape != targets.shape
        ):
            raise BoundedPilotAuthorizationError("loss_mask must match target_ids")
        return input_ids, targets, loss_mask, False
    if "loss_mask" in batch:
        raise BoundedPilotAuthorizationError(
            "shifted causal batches do not accept a separate loss_mask"
        )
    labels = batch.get("labels", input_ids)
    if not isinstance(labels, Tensor) or labels.ndim != 2:
        raise BoundedPilotAuthorizationError("shifted labels must be a rank-2 tensor")
    return input_ids, labels, None, True


def _actual_nonignored_targets(batch: Batch) -> int:
    _, targets, loss_mask, shifted = _batch_projection(batch)
    if shifted:
        return 0 if targets.shape[1] < 2 else int(targets[:, 1:].ne(-100).sum().item())
    valid = targets.ne(-100)
    if loss_mask is not None:
        valid = valid & loss_mask.bool()
    return int(valid.sum().item())


def _require_packet_roots(
    packet: Mapping[str, Any],
    *,
    expected_portable_execution_sha256: str,
    expected_launch_input_authority_identity_sha256: str,
) -> None:
    binding = packet.get("binding")
    if not isinstance(binding, Mapping):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet binding roots are missing"
        )
    expected_execution = _require_sha256(
        expected_portable_execution_sha256,
        label="external portable execution root",
    )
    expected_launch = _require_sha256(
        expected_launch_input_authority_identity_sha256,
        label="external D10 launch-input root",
    )
    if binding.get("portable_execution_sha256") != expected_execution:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: portable execution root differs from external authority"
        )
    if binding.get("launch_input_authority_identity_sha256") != expected_launch:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: D10 launch-input root differs from packet binding"
        )


def _require_local_free_packet(
    binding: PortableRunBinding,
    *,
    expected_packet_sha256: str,
    expected_portable_execution_sha256: str,
    expected_launch_input_authority_identity_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected_root = _require_sha256(expected_packet_sha256, label="external packet root")
    if (
        not binding.binding_ready
        or not binding.readiness_ready
        or not binding.overlay_contract_valid
        or not binding.packet_contract_valid
        or binding.packet is None
        or binding.packet_sha256 is None
    ):
        blockers = ",".join(binding.blockers) if binding.blockers else "binding_not_ready"
        raise BoundedPilotAuthorizationError(
            f"BLOCKED_PRE_STEP_1: portable run binding is not runnable: {blockers}"
        )
    if binding.mode != "FRESH_START":
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: current bounded runtime supports authenticated FRESH_START only"
        )

    binding_root = _require_sha256(binding.packet_sha256, label="binding packet root")
    packet = copy.deepcopy(binding.packet)
    observed_root = canonical_sha256(packet)
    if observed_root != binding_root:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet content differs from binding packet root"
        )
    if observed_root != expected_root:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet root differs from external authority"
        )
    _require_packet_roots(
        packet,
        expected_portable_execution_sha256=expected_portable_execution_sha256,
        expected_launch_input_authority_identity_sha256=(
            expected_launch_input_authority_identity_sha256
        ),
    )

    resource = packet.get("resource")
    truth = packet.get("truth_boundary")
    identities = packet.get("identities")
    recipe = packet.get("recipe")
    if not all(isinstance(value, dict) for value in (resource, truth, identities, recipe)):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: packet sections malformed")
    assert isinstance(resource, dict) and isinstance(truth, dict)
    assert isinstance(identities, dict) and isinstance(recipe, dict)
    if resource.get("resource_class") != "LOCAL_FREE":
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: LOCAL_FREE required")
    if resource.get("maximum_cost_usd") != 0 or resource.get("materially_paid") is not False:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: paid compute forbidden")
    if truth.get("final_test_payload_accessed") is not False:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: final test must stay sealed")
    if identities.get("canonical_base") != "random_init":
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: Base must be random_init")
    target = recipe.get("target_unique_loss_positions")
    maximum = recipe.get("maximum_total_exposures")
    available = recipe.get("available_unique_loss_positions")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (target, maximum, available)
    ):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimized-target authority must be positive"
        )
    if maximum != target or recipe.get("max_exposures_per_unique_position") != 1:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: exposure budget not exact")
    return packet, observed_root


def _actual_model_identities(trainer: Trainer) -> tuple[str, str]:
    model = trainer.model
    spec = getattr(model, "spec", None)
    init_spec = getattr(model, "init_spec", None)
    spec_identity = getattr(spec, "identity_sha256", None)
    init_identity = getattr(init_spec, "identity_sha256", None)
    if not callable(spec_identity) or not callable(init_identity):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: actual model does not expose ModelSpec/InitSpec identities"
        )
    model_sha = spec_identity()
    init_sha = init_identity()
    if not isinstance(model_sha, str) or not isinstance(init_sha, str):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: model identities malformed")
    parameter_count = getattr(spec, "parameter_count", None)
    if not callable(parameter_count):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: ModelSpec lacks parameter_count")
    actual_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count() != actual_parameters:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual parameter count drift")
    return model_sha, init_sha


def _actual_overlay_projection(trainer: Trainer) -> dict[str, Any]:
    return {
        "optimizer": trainer.optimizer.__class__.__name__,
        "scheduler": trainer.config.scheduler,
        "precision": trainer.config.precision,
    }


def _require_optimizer_parameter_coverage(trainer: Trainer) -> None:
    model_parameters = [
        parameter for parameter in trainer.model.parameters() if parameter.requires_grad
    ]
    optimizer_parameters = [
        parameter
        for group in trainer.optimizer.param_groups
        for parameter in group.get("params", ())
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimizer contains duplicate trainable parameters"
        )
    if set(optimizer_ids) != {id(parameter) for parameter in model_parameters}:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: optimizer parameter coverage differs from model"
        )


def _require_live_optimizer_matches_canonical_builder(trainer: Trainer) -> None:
    _require_optimizer_parameter_coverage(trainer)
    reference = build_optimizer(trainer.model, trainer.config)
    if type(trainer.optimizer) is not type(reference):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: live optimizer type differs from canonical builder"
        )
    if not _typed_equal(trainer.optimizer.defaults, reference.defaults):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: live optimizer defaults differ from canonical builder"
        )
    if len(trainer.optimizer.param_groups) != len(reference.param_groups):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: live optimizer param-group count differs from canonical builder"
        )
    for index, (actual, expected) in enumerate(
        zip(trainer.optimizer.param_groups, reference.param_groups, strict=True)
    ):
        actual_options = {key: value for key, value in actual.items() if key != "params"}
        expected_options = {key: value for key, value in expected.items() if key != "params"}
        if not _typed_equal(actual_options, expected_options):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live optimizer group semantics differ "
                f"from canonical builder (group {index})"
            )
    if trainer.config.scheduler != "constant" or trainer.config.warmup_steps != 0:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: learned-20M bounded pilot requires constant scheduler/warmup0"
        )
    if trainer.scheduler is not None:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: canonical constant scheduler must not inject a scheduler object"
        )


def _require_actual_execution_matches_packet(
    trainer: Trainer,
    packet: Mapping[str, Any],
) -> tuple[str, str]:
    identities = packet["identities"]
    recipe = packet["recipe"]
    model_sha, init_sha = _actual_model_identities(trainer)
    if identities.get("modelspec_sha256") != model_sha:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual ModelSpec differs")
    if identities.get("initspec_sha256") != init_sha:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: actual InitSpec differs")
    expected_overlay = recipe.get("optimizer_scheduler_precision")
    actual_overlay = _actual_overlay_projection(trainer)
    if not isinstance(expected_overlay, dict) or set(expected_overlay) != set(actual_overlay):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: packet lacks canonical optimizer/scheduler/precision overlay"
        )
    if expected_overlay != actual_overlay:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: actual optimizer/scheduler/precision differs from packet"
        )
    if recipe.get("seed") != trainer.config.seed:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: Trainer seed differs")
    if trainer.config.gradient_accumulation_steps != 1:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: bounded pilot requires one microbatch per optimizer step"
        )
    _require_live_optimizer_matches_canonical_builder(trainer)
    return model_sha, init_sha


def _require_live_replay_state_consistency(
    trainer: Trainer,
    replay_guard: IdentitySafeExposureReplayGuard,
    *,
    expected_inflight_targets: int = 0,
) -> None:
    if (
        isinstance(expected_inflight_targets, bool)
        or not isinstance(expected_inflight_targets, int)
        or expected_inflight_targets < 0
    ):
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: expected in-flight target count is invalid"
        )
    if replay_guard.trainer_state_binding.get("optimizer_step") != trainer.optimizer_step:
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: replay optimizer state stale")
    if (
        replay_guard.trainer_state_binding.get("trainer_nonignored_target_count")
        != replay_guard.consumed_loss_positions
    ):
        raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: replay target counter stale")
    expected_tokens_seen = replay_guard.consumed_loss_positions + expected_inflight_targets
    if trainer.tokens_seen != expected_tokens_seen:
        raise BoundedPilotAuthorizationError(
            "BLOCKED_PRE_STEP_1: Trainer target counter differs from phase-aware D04 replay state"
        )


class BoundedPilotStepRunner:
    """Consume D04 authority at the final pre-optimizer-step boundary."""

    def __init__(
        self,
        runner: SingleDeviceStepRunner,
        *,
        binding: PortableRunBinding,
        expected_packet_sha256: str,
        expected_portable_execution_sha256: str,
        launch_input_authority: Mapping[str, Any],
        expected_launch_input_authority_identity_sha256: str,
        replay_guard: IdentitySafeExposureReplayGuard,
        loss_bearing_content_manifest: Mapping[str, Any],
        expected_loss_bearing_manifest_identity_sha256: str,
        exposure_plan: Mapping[str, Any],
        expected_plan_identity_sha256: str,
    ) -> None:
        if type(runner) is not SingleDeviceStepRunner:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: exact canonical SingleDeviceStepRunner required"
            )
        self.runner = runner
        self.trainer: Trainer = runner.trainer
        self.binding = binding
        launch_root = _require_sha256(
            expected_launch_input_authority_identity_sha256,
            label="external D10 launch-input root",
        )
        self.packet, self._packet_sha256 = _require_local_free_packet(
            binding,
            expected_packet_sha256=expected_packet_sha256,
            expected_portable_execution_sha256=expected_portable_execution_sha256,
            expected_launch_input_authority_identity_sha256=launch_root,
        )
        self.replay_guard = replay_guard
        manifest_root = _require_sha256(
            expected_loss_bearing_manifest_identity_sha256,
            label="external loss-bearing content manifest root",
        )
        if (
            not replay_guard.content_authority_configured
            or replay_guard.loss_bearing_manifest_identity_sha256 != manifest_root
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 replay guard lacks the externally rooted content manifest"
            )
        self.loss_bearing_content_manifest = copy.deepcopy(loss_bearing_content_manifest)
        self._manifest_root = manifest_root
        self.exposure_plan = copy.deepcopy(exposure_plan)
        self.expected_plan_identity_sha256 = _require_sha256(
            expected_plan_identity_sha256,
            label="external D04 plan root",
        )
        if self.exposure_plan.get("plan_identity_sha256") != self.expected_plan_identity_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 plan differs from external handoff"
            )
        self._pending: _PendingAuthorization | None = None
        self._authorized_identity: str | None = None
        self._poisoned_reason: str | None = None
        self._optimizer_object = self.trainer.optimizer
        self._modelspec_sha256, self._initspec_sha256 = _require_actual_execution_matches_packet(
            self.trainer,
            self.packet,
        )
        identities = self.packet["identities"]
        recipe = self.packet["recipe"]
        _verify_launch_input_v2_authority(
            launch_input_authority,
            expected_identity_sha256=launch_root,
            expected_modelspec_sha256=str(identities.get("modelspec_sha256")),
            expected_initspec_sha256=str(identities.get("initspec_sha256")),
            expected_ledger_sha256=replay_guard.ledger_identity_sha256,
            expected_requested_unique_loss_positions=recipe["target_unique_loss_positions"],
        )
        if identities.get("unique_loss_ledger_sha256") != replay_guard.ledger_identity_sha256:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: D04 ledger differs")
        if recipe.get("maximum_total_exposures") != replay_guard.authorized_budget:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: D04 budget differs")
        _require_live_replay_state_consistency(self.trainer, replay_guard)
        self._expected_model_state_sha256 = _require_fresh_start_model_state(self.trainer)

        register = getattr(self.trainer.optimizer, "register_step_pre_hook", None)
        if register is None:
            raise BoundedPilotAuthorizationError("BLOCKED_PRE_STEP_1: no optimizer pre-hook")
        self._hook_handle = register(self._authorize_immediately_before_optimizer_step)

    @property
    def poisoned(self) -> bool:
        return self._poisoned_reason is not None

    def close(self) -> None:
        handle = getattr(self, "_hook_handle", None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def _raise_recovery_required(self) -> None:
        reason = self._poisoned_reason or "learned-state commit status is ambiguous"
        raise BoundedPilotRecoveryRequiredError(
            "BLOCKED_RECOVERY_REQUIRED: fresh verified recovery required; " + reason
        )

    def _require_live_execution_chain(self, *, expected_inflight_targets: int = 0) -> None:
        if self.trainer.optimizer is not self._optimizer_object:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer object replaced after "
                "authorization hook installation"
            )
        _require_actual_execution_matches_packet(self.trainer, self.packet)
        _require_live_replay_state_consistency(
            self.trainer,
            self.replay_guard,
            expected_inflight_targets=expected_inflight_targets,
        )
        observed_state = _model_state_sha256(self.trainer.model)
        if observed_state != self._expected_model_state_sha256:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: live model state changed outside authorized optimizer chain"
            )

    def _verify_live_content(self, batch: Batch, *, batch_index: int) -> str:
        input_ids, targets, loss_mask, shifted = _batch_projection(batch)
        try:
            return verify_live_loss_bearing_batch(
                self.loss_bearing_content_manifest,
                expected_manifest_identity_sha256=self._manifest_root,
                batch_index=batch_index,
                input_ids=input_ids,
                target_ids=targets,
                loss_mask=loss_mask,
                shifted=shifted,
            )
        except (TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: live loss-bearing content rejected: {exc}"
            ) from exc

    def _preflight_handoff(
        self,
        *,
        batch: Batch,
        batch_index: int,
        expected_identity: str,
        actual_targets: int,
        expected_inflight_targets: int = 0,
    ) -> None:
        self._require_live_execution_chain(
            expected_inflight_targets=expected_inflight_targets,
        )
        self._verify_live_content(batch, batch_index=batch_index)
        observed = ordered_next_exposure_identity(
            self.replay_guard,
            self.exposure_plan,
            batch_index=batch_index,
            expected_plan_identity_sha256=self.expected_plan_identity_sha256,
        )
        if observed != expected_identity:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: next exposure identity differs from external handoff"
            )
        plan_batch = self.exposure_plan.get("batches", [])[batch_index]
        if plan_batch.get("actual_nonignored_targets") != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: D04 cardinality differs from Trainer batch"
            )

    def _authorize_immediately_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        if self.poisoned:
            self._raise_recovery_required()
        if self._pending is None:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer step lacks D04 handoff"
            )
        if (
            optimizer is not self._optimizer_object
            or self.trainer.optimizer is not self._optimizer_object
        ):
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: optimizer object differs from authorized hook target"
            )
        pending = self._pending
        self._preflight_handoff(
            batch=pending.batch,
            batch_index=pending.batch_index,
            expected_identity=pending.expected_identity,
            actual_targets=pending.actual_targets,
            expected_inflight_targets=pending.actual_targets,
        )
        plan_batch = self.exposure_plan["batches"][pending.batch_index]
        input_ids, targets, loss_mask, shifted = _batch_projection(pending.batch)
        try:
            self._authorized_identity = self.replay_guard.authorize_live_batch_with_identity(
                plan_batch["claims"],
                actual_nonignored_targets=pending.actual_targets,
                expected_next_exposure_identity_sha256=pending.expected_identity,
                batch_index=pending.batch_index,
                input_ids=input_ids,
                target_ids=targets,
                loss_mask=loss_mask,
                shifted=shifted,
            )
        except (TypeError, ValueError) as exc:
            raise BoundedPilotAuthorizationError(
                f"BLOCKED_PRE_STEP_1: D04 live authorization rejected: {exc}"
            ) from exc

    def _bind_live_post_step_state(self) -> None:
        if self.trainer.tokens_seen != self.replay_guard.consumed_loss_positions:
            self._poison("Trainer/D04 target counters diverged after optimizer transition")
            self._raise_recovery_required()
        state = dict(self.replay_guard.trainer_state_binding)
        state["optimizer_step"] = self.trainer.optimizer_step
        state["trainer_nonignored_target_count"] = self.trainer.tokens_seen
        try:
            self.replay_guard.bind_checkpoint_state(state)
        except Exception as exc:  # noqa: BLE001 - any post-commit failure must poison.
            self._poison(f"post-step D04 state binding failed: {exc}")
            self._raise_recovery_required()
        self._expected_model_state_sha256 = _model_state_sha256(self.trainer.model)

    def train_authorized_microbatch(
        self,
        batch: Batch,
        *,
        batch_index: int,
        expected_next_exposure_identity_sha256: str,
    ) -> tuple[SingleDeviceStepMetrics, BoundedPilotStepReceipt]:
        if self.poisoned:
            self._raise_recovery_required()
        if self._pending is not None:
            raise BoundedPilotAuthorizationError("prior authorization handoff still pending")
        if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
            raise BoundedPilotAuthorizationError("batch_index must be non-negative integer")
        batches = self.exposure_plan.get("batches")
        if not isinstance(batches, list) or batch_index >= len(batches):
            raise BoundedPilotAuthorizationError("batch_index outside exposure plan")
        frozen_batch = {
            key: tensor.detach().clone() for key, tensor in batch.items()
        }
        actual_targets = _actual_nonignored_targets(frozen_batch)
        planned_targets = batches[batch_index].get("actual_nonignored_targets")
        if actual_targets <= 0 or planned_targets != actual_targets:
            raise BoundedPilotAuthorizationError(
                "BLOCKED_PRE_STEP_1: loss-bearing cardinality is not exact"
            )
        self._preflight_handoff(
            batch=frozen_batch,
            batch_index=batch_index,
            expected_identity=expected_next_exposure_identity_sha256,
            actual_targets=actual_targets,
        )

        before_step = self.trainer.optimizer_step
        self._pending = _PendingAuthorization(
            batch=frozen_batch,
            batch_index=batch_index,
            expected_identity=expected_next_exposure_identity_sha256,
            actual_targets=actual_targets,
        )
        self._authorized_identity = None
        try:
            metrics = self.runner.train_microbatch(frozen_batch)
        except Exception as exc:  # noqa: BLE001 - ambiguous transition must fail closed.
            if self._authorized_identity is None:
                reason = "Trainer execution failed after entering the transition boundary"
            else:
                reason = "Trainer execution failed after D04 authorization was consumed"
            self._poison(f"{reason}: {exc}")
            self._raise_recovery_required()
        finally:
            self._pending = None

        if not metrics.trainer.optimizer_stepped or self._authorized_identity is None:
            self._poison("Trainer returned without one consumed optimizer authorization")
            self._raise_recovery_required()
        if self.trainer.optimizer_step != before_step + 1:
            self._poison("optimizer step did not advance exactly once")
            self._raise_recovery_required()

        self._bind_live_post_step_state()
        receipt = BoundedPilotStepReceipt(
            optimizer_step_before=before_step,
            optimizer_step_after=self.trainer.optimizer_step,
            actual_nonignored_targets=actual_targets,
            exposure_identity_sha256=self._authorized_identity,
            consumed_loss_positions_after=self.replay_guard.consumed_loss_positions,
            packet_sha256=self._packet_sha256,
            modelspec_sha256=self._modelspec_sha256,
            initspec_sha256=self._initspec_sha256,
        )
        return metrics, receipt
