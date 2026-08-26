"""Terminal convergence gate for POSTBASE-353 SFT mechanics.

This module is a control/evidence layer over :mod:`sft_runner`; it does not add a
trainer, model backend, dataset, authorization, or campaign path.  It exists to bind
POSTBASE-353 mechanics to the live POSTBASE-253/352 boundaries and to produce one
scoped, auditable terminal receipt from tiny fixture-only execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from twelve_six.post_base.contract import TokenizerCompatibility, snapshot_directory
from twelve_six.post_base.sft_runner import (
    MAX_MECHANICS_STEPS,
    SFT_CHECKPOINT_NAMESPACE,
    SFT_EVALUATION_NAMESPACE,
    SFT_MECHANICS_SCHEMA,
    SFTCheckpointStore,
    SFTMechanicsBackend,
    SFTMechanicsDataset,
    SFTMechanicsPlan,
    SFTMechanicsReceipt,
    run_sft_mechanics,
)
from twelve_six.posttraining.contracts import ComputeClass

POSTBASE253_AUTHORITY = "postbase253/communication-layer-contract-20260826"
POSTBASE352_AUTHORITY = "postbase352/communication-data-contract-v1-20260826"
POSTBASE352_TOKENIZER_ID = "s0-byte-v1"
POSTBASE352_VOCAB_SIZE = 256
TERMINAL_EVIDENCE_NAMESPACE = "evidence/post_base/sft/terminal"
TERMINAL_SCHEMA = "12-6.post-base.sft-mechanics-terminal.v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_payload(tokenizer: TokenizerCompatibility) -> dict[str, object]:
    return {
        "tokenizer_id": tokenizer.tokenizer_id,
        "config_sha256": tokenizer.config_sha256,
        "vocab_sha256": tokenizer.vocab_sha256,
        "vocab_size": tokenizer.vocab_size,
    }


def _dataset_payload(dataset: SFTMechanicsDataset) -> dict[str, object]:
    return {
        "dataset_id": dataset.dataset_id,
        "manifest_sha256": dataset.manifest_sha256,
        "source_registry_sha256": dataset.source_registry_sha256,
        "train_split_sha256": dataset.train_split_sha256,
        "evaluation_split_sha256": dataset.evaluation_split_sha256,
        "fixture_only": dataset.fixture_only,
    }


def require_terminal_inputs(plan: SFTMechanicsPlan, dataset: SFTMechanicsDataset) -> None:
    """Fail closed unless the mechanics input obeys POSTBASE-253/352 convergence."""
    if plan.compute_class is not ComputeClass.LOCAL_FREE:
        raise ValueError("terminal POSTBASE-353 remains LOCAL_FREE only")
    if not 1 <= plan.max_steps <= MAX_MECHANICS_STEPS:
        raise ValueError("terminal mechanics steps exceed the bounded POSTBASE-353 envelope")
    if plan.real_campaign_authorization_id is not None or plan.contract.execution_authorized:
        raise ValueError("terminal POSTBASE-353 carries no real communication campaign authority")
    if not dataset.fixture_only:
        raise ValueError("terminal POSTBASE-353 accepts fixture-only mechanics data")

    plan.contract.require_tokenizer(plan.tokenizer)
    dataset.require_contract_match(plan.contract)
    if plan.tokenizer.tokenizer_id != POSTBASE352_TOKENIZER_ID:
        raise ValueError("POSTBASE-352 requires the s0-byte-v1 logical tokenizer profile")
    if plan.tokenizer.vocab_size != POSTBASE352_VOCAB_SIZE:
        raise ValueError("POSTBASE-352 requires the 256-byte logical vocabulary")

    for record in (*dataset.train, *dataset.evaluation):
        if record.provenance.foreign_model_output:
            raise ValueError("foreign model output is forbidden in POSTBASE-353 fixtures")


def _expected_base_payload(plan: SFTMechanicsPlan) -> dict[str, object]:
    checkpoint = plan.contract.base_checkpoint
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "sha256": checkpoint.sha256,
        "git_sha": checkpoint.git_sha,
        "stage": checkpoint.stage,
        "lineage": checkpoint.lineage.value,
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _verify_active_pointer(
    store: SFTCheckpointStore,
    *,
    expected_generation: int,
) -> str:
    pointer = _read_json(store.active_pointer)
    if pointer.get("schema") != SFT_MECHANICS_SCHEMA:
        raise RuntimeError("active rollback pointer schema mismatch")
    if pointer.get("run_id") != store.run_id:
        raise RuntimeError("active rollback pointer run mismatch")
    if pointer.get("generation") != expected_generation:
        raise RuntimeError("active rollback pointer generation mismatch")

    manifest_path = store.generation_path(expected_generation) / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_sha256 = _sha256_bytes(_canonical_bytes(manifest))
    if pointer.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("active rollback pointer does not bind the target manifest")
    return manifest_sha256


def _verify_generation_chain(
    *,
    store: SFTCheckpointStore,
    backend: SFTMechanicsBackend,
    plan: SFTMechanicsPlan,
    dataset: SFTMechanicsDataset,
    receipt: SFTMechanicsReceipt,
) -> tuple[str, ...]:
    manifest_hashes: list[str] = []
    expected_base = _expected_base_payload(plan)
    expected_generations = tuple(range(receipt.training_steps + 1))
    actual_generations = tuple(
        sorted(
            int(path.name.removeprefix("generation_"))
            for path in store.root.glob("generation_*")
            if path.is_dir()
        )
    )
    if actual_generations != expected_generations:
        raise RuntimeError("checkpoint generations are not the exact immutable expected sequence")

    for generation in expected_generations:
        manifest_path = store.generation_path(generation) / "manifest.json"
        manifest = _read_json(manifest_path)
        expected_parent = None if generation == 0 else generation - 1
        if manifest.get("schema") != SFT_MECHANICS_SCHEMA:
            raise RuntimeError("checkpoint manifest schema drift")
        if manifest.get("run_id") != plan.run_id:
            raise RuntimeError("checkpoint manifest run binding drift")
        if manifest.get("generation") != generation:
            raise RuntimeError("checkpoint manifest generation drift")
        if manifest.get("parent_generation") != expected_parent:
            raise RuntimeError("checkpoint immutable generation chain is broken")
        if manifest.get("backend_id") != plan.backend_id:
            raise RuntimeError("checkpoint backend binding drift")
        if manifest.get("input_snapshot_sha256") != receipt.workspace.cloned_snapshot_sha256:
            raise RuntimeError("checkpoint input-clone binding drift")
        if manifest.get("dataset_manifest_sha256") != dataset.manifest_sha256:
            raise RuntimeError("checkpoint dataset binding drift")
        if manifest.get("base_checkpoint") != expected_base:
            raise RuntimeError("checkpoint Base binding drift")

        # Existing POSTBASE-353 load_state is the byte-tamper gate for backend state.
        store.load_state(backend, generation)
        manifest_hashes.append(_sha256_bytes(_canonical_bytes(manifest)))

    return tuple(manifest_hashes)


def _expected_evaluation_payload(
    *,
    run_id: str,
    generation: int,
    phase: str,
    metrics: Mapping[str, float],
    dataset: SFTMechanicsDataset,
) -> dict[str, object]:
    return {
        "schema": SFT_MECHANICS_SCHEMA,
        "run_id": run_id,
        "generation": generation,
        "phase": phase,
        "evaluation_split_sha256": dataset.evaluation_split_sha256,
        "metrics": dict(metrics),
    }


def _verify_evaluation_separation(
    receipt: SFTMechanicsReceipt,
    dataset: SFTMechanicsDataset,
    plan: SFTMechanicsPlan,
) -> None:
    if receipt.checkpoint_namespace == receipt.evaluation_namespace:
        raise RuntimeError("checkpoint artifacts and evaluation evidence share a namespace")
    if receipt.evaluation_namespace.is_relative_to(receipt.checkpoint_namespace):
        raise RuntimeError("evaluation evidence is nested inside checkpoint artifacts")
    if receipt.checkpoint_namespace.is_relative_to(receipt.evaluation_namespace):
        raise RuntimeError("checkpoint artifacts are nested inside evaluation evidence")

    expected = (
        (
            receipt.baseline_evaluation,
            _expected_evaluation_payload(
                run_id=plan.run_id,
                generation=0,
                phase="baseline",
                metrics=receipt.baseline_metrics,
                dataset=dataset,
            ),
        ),
        (
            receipt.final_evaluation,
            _expected_evaluation_payload(
                run_id=plan.run_id,
                generation=receipt.final_checkpoint.generation,
                phase="final",
                metrics=receipt.final_metrics,
                dataset=dataset,
            ),
        ),
    )
    expected_names = {path.name for path, _ in expected}
    actual_names = {
        path.name
        for path in receipt.evaluation_namespace.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_names != expected_names:
        raise RuntimeError("evaluation evidence set drift")

    for path, expected_payload in expected:
        if path.parent != receipt.evaluation_namespace or path.is_symlink():
            raise RuntimeError("evaluation evidence path escaped its immutable namespace")
        payload = _read_json(path)
        if payload != expected_payload:
            raise RuntimeError("evaluation evidence payload drift")


def _prove_rollback_pointer(
    *,
    store: SFTCheckpointStore,
    backend: SFTMechanicsBackend,
    final_generation: int,
) -> None:
    before = {
        generation: snapshot_directory(store.generation_path(generation))
        for generation in range(final_generation + 1)
    }
    store.rollback_to(0)
    _verify_active_pointer(store, expected_generation=0)
    store.load_state(backend, 0)
    after = {
        generation: snapshot_directory(store.generation_path(generation))
        for generation in range(final_generation + 1)
    }
    if before != after:
        raise RuntimeError("rollback pointer rewrote an immutable checkpoint generation")

    # Restore the final active pointer without changing any generation bytes.
    store._activate(final_generation, reason="terminal_rollback_proof_restore")
    _verify_active_pointer(store, expected_generation=final_generation)


@dataclass(frozen=True, slots=True)
class TerminalSFTReceipt:
    mechanics: SFTMechanicsReceipt
    authority_path: Path
    authority_sha256: str


def terminalize_sft_mechanics(
    *,
    plan: SFTMechanicsPlan,
    dataset: SFTMechanicsDataset,
    canonical_base_root: Path,
    experiment_root: Path,
    backend: SFTMechanicsBackend,
) -> TerminalSFTReceipt:
    """Run and independently terminal-gate bounded fixture-only SFT mechanics."""
    require_terminal_inputs(plan, dataset)
    receipt = run_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=canonical_base_root,
        experiment_root=experiment_root,
        backend=backend,
    )

    if receipt.training_steps != plan.max_steps:
        raise RuntimeError("mechanics step count does not match the bounded plan")
    if receipt.checkpoint_namespace.relative_to(experiment_root).as_posix() != (
        SFT_CHECKPOINT_NAMESPACE
    ):
        raise RuntimeError("checkpoint namespace escaped the POSTBASE-353 boundary")
    if receipt.evaluation_namespace.relative_to(experiment_root).as_posix() != (
        SFT_EVALUATION_NAMESPACE
    ):
        raise RuntimeError("evaluation namespace escaped the POSTBASE-353 boundary")

    canonical_after = snapshot_directory(canonical_base_root)
    clone_after = snapshot_directory(receipt.workspace.cloned_checkpoint_root)
    if receipt.canonical_snapshot_before != canonical_after.identity_sha256:
        raise RuntimeError("canonical Base changed during terminal mechanics")
    if receipt.workspace.cloned_snapshot_sha256 != clone_after.identity_sha256:
        raise RuntimeError("input Base checkpoint clone changed during terminal mechanics")

    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    manifest_hashes = _verify_generation_chain(
        store=store,
        backend=backend,
        plan=plan,
        dataset=dataset,
        receipt=receipt,
    )
    _verify_active_pointer(store, expected_generation=receipt.final_checkpoint.generation)
    _verify_evaluation_separation(receipt, dataset, plan)
    _prove_rollback_pointer(
        store=store,
        backend=backend,
        final_generation=receipt.final_checkpoint.generation,
    )

    authority_root = experiment_root / TERMINAL_EVIDENCE_NAMESPACE
    authority_root.mkdir(parents=True, exist_ok=False)
    authority_path = authority_root / "authority.json"
    authority = {
        "schema": TERMINAL_SCHEMA,
        "run_id": plan.run_id,
        "status": "PASS",
        "scope": "fixture_only_sft_mechanics",
        "execution_profile": "LOCAL_FREE",
        "postbase253_authority": POSTBASE253_AUTHORITY,
        "postbase352_authority": POSTBASE352_AUTHORITY,
        "real_communication_campaign_authorized": False,
        "max_mechanics_steps": MAX_MECHANICS_STEPS,
        "executed_steps": receipt.training_steps,
        "base_checkpoint": _expected_base_payload(plan),
        "canonical_base_snapshot_sha256": receipt.canonical_snapshot_after,
        "input_clone_snapshot_sha256": receipt.workspace.cloned_snapshot_sha256,
        "tokenizer": _tokenizer_payload(plan.tokenizer),
        "dataset": _dataset_payload(dataset),
        "checkpoint_namespace": SFT_CHECKPOINT_NAMESPACE,
        "evaluation_namespace": SFT_EVALUATION_NAMESPACE,
        "generation_manifest_sha256": list(manifest_hashes),
        "rollback_generation": receipt.rollback_generation,
    }
    authority_bytes = _canonical_bytes(authority)
    authority_path.write_bytes(authority_bytes)
    return TerminalSFTReceipt(
        mechanics=receipt,
        authority_path=authority_path,
        authority_sha256=_sha256_bytes(authority_bytes),
    )
