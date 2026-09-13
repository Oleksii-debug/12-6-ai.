from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from test_bounded_pilot import _authority, _batch, _materialization, _next
from test_bounded_pilot_durable_attempt import _constructor_inputs, _trainer

from twelve_six.data.unique_loss_ledger_v2 import build_ledger
from twelve_six.training.bounded_pilot import BoundedPilotAuthorizationError
from twelve_six.training.d04_authenticated_batch import (
    AuthenticatedD04BatchSource,
    AuthenticatedD04BoundedPilotStepRunner,
)


def _source(
    plan: dict,
    manifest: dict,
    *,
    materialization: dict | None = None,
    ledger: dict | None = None,
    expected_materialization_identity_sha256: str | None = None,
) -> AuthenticatedD04BatchSource:
    selected = copy.deepcopy(
        materialization if materialization is not None else _materialization()
    )
    selected_ledger = copy.deepcopy(ledger) if ledger is not None else build_ledger(selected)
    return AuthenticatedD04BatchSource(
        materialization=selected,
        ledger=selected_ledger,
        exposure_plan=plan,
        loss_bearing_content_manifest=manifest,
        expected_materialization_identity_sha256=(
            expected_materialization_identity_sha256
            if expected_materialization_identity_sha256 is not None
            else selected["materialization_identity_sha256"]
        ),
        expected_ledger_identity_sha256=selected_ledger["ledger_identity_sha256"],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
        expected_loss_bearing_manifest_identity_sha256=manifest[
            "manifest_identity_sha256"
        ],
    )


def _gate(
    tmp_path: Path,
    *,
    batch_count: int = 1,
) -> tuple[
    AuthenticatedD04BoundedPilotStepRunner,
    object,
    dict,
    dict,
]:
    guard, plan, manifest = _authority(batch_count=batch_count)
    trainer = _trainer(max_steps=batch_count)
    runner, kwargs, store = _constructor_inputs(
        tmp_path / "authenticated-d04",
        trainer,
        guard,
        plan,
        manifest,
        run_id="authenticated-d04-runtime",
    )
    gate = AuthenticatedD04BoundedPilotStepRunner(
        runner,
        batch_source=_source(plan, manifest),
        **kwargs,
    )
    return gate, store, guard, plan


def test_source_derives_exact_predictor_and_target_rows_from_postpack_tokens() -> None:
    _guard, plan, manifest = _authority(batch_count=2)
    source = _source(plan, manifest)

    first = source.build_batch(batch_index=0, device="cpu")
    second = source.build_batch(batch_index=1, device="cpu")

    assert set(first) == {"input_ids", "target_ids"}
    assert first["input_ids"].tolist() == [[1, 2]]
    assert first["target_ids"].tolist() == [[2, 3]]
    assert second["input_ids"].tolist() == [[3, 4]]
    assert second["target_ids"].tolist() == [[4, 5]]
    assert first["input_ids"].dtype == torch.int64
    assert first["target_ids"].dtype == torch.int64


def test_authenticated_runner_executes_derived_batch_and_returns_ordered_identity(
    tmp_path: Path,
) -> None:
    gate, store, guard, plan = _gate(tmp_path)
    expected = _next(guard, plan, 0)

    metrics, receipt = gate.train_authenticated_batch(
        batch_index=0,
        expected_next_exposure_identity_sha256=expected,
    )
    gate.close()

    assert metrics.trainer.optimizer_step == 1
    assert receipt.optimizer_step_after == 1
    assert receipt.exposure_identity_sha256 == expected
    assert receipt.actual_nonignored_targets == 2
    assert guard.consumed_loss_positions == 2
    assert guard.claim_sequence == 1
    assert store.open()["phase"] == "COMPLETED"


def test_authenticated_runner_rejects_caller_authored_batch_before_mutation(
    tmp_path: Path,
) -> None:
    gate, store, guard, plan = _gate(tmp_path)
    before = [parameter.detach().clone() for parameter in gate.trainer.model.parameters()]

    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="caller-authored Batch is forbidden",
    ):
        gate.train_authorized_microbatch(
            _batch(0),
            batch_index=0,
            expected_next_exposure_identity_sha256=_next(guard, plan, 0),
        )
    gate.close()

    assert gate.trainer.optimizer_step == 0
    assert gate.trainer.tokens_seen == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0
    assert all(
        left.equal(right)
        for left, right in zip(before, gate.trainer.model.parameters(), strict=True)
    )


def test_wrong_or_stale_batch_order_fails_before_durable_attempt_or_model_mutation(
    tmp_path: Path,
) -> None:
    gate, store, guard, plan = _gate(tmp_path, batch_count=2)
    before = [parameter.detach().clone() for parameter in gate.trainer.model.parameters()]

    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="batch_index is not the next exposure",
    ):
        gate.train_authenticated_batch(
            batch_index=1,
            expected_next_exposure_identity_sha256=_next(guard, plan, 1),
        )
    gate.close()

    assert gate.trainer.optimizer_step == 0
    assert gate.trainer.tokens_seen == 0
    assert guard.consumed_loss_positions == 0
    assert store.open()["attempt"] == 0
    assert all(
        left.equal(right)
        for left, right in zip(before, gate.trainer.model.parameters(), strict=True)
    )


def test_postpack_predictor_or_target_token_mutation_cannot_rebuild_authorized_manifest() -> None:
    _guard, plan, manifest = _authority(batch_count=1)
    original = _materialization()
    original_root = original["materialization_identity_sha256"]
    original_ledger = build_ledger(original)

    for token_index in (0, 1):
        mutated = copy.deepcopy(original)
        mutated["packing"]["packs"][0]["token_ids"][token_index] += 7
        with pytest.raises(
            BoundedPilotAuthorizationError,
            match="authenticated D04 batch source rejected",
        ):
            _source(
                plan,
                manifest,
                materialization=mutated,
                ledger=original_ledger,
                expected_materialization_identity_sha256=original_root,
            )


def test_stale_external_materialization_root_fails_closed() -> None:
    _guard, plan, manifest = _authority(batch_count=1)

    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="authenticated D04 batch source rejected|materialization root differs",
    ):
        _source(
            plan,
            manifest,
            expected_materialization_identity_sha256="0" * 64,
        )


def test_stale_content_manifest_fails_before_batch_source_is_usable() -> None:
    _guard, plan, manifest = _authority(batch_count=1)
    stale = copy.deepcopy(manifest)
    stale["batches"][0]["claims"][0]["target_count"] += 1
    materialization = _materialization()
    ledger = build_ledger(materialization)

    with pytest.raises(
        BoundedPilotAuthorizationError,
        match="supplied D04 content manifest differs|content manifest",
    ):
        AuthenticatedD04BatchSource(
            materialization=materialization,
            ledger=ledger,
            exposure_plan=plan,
            loss_bearing_content_manifest=stale,
            expected_materialization_identity_sha256=materialization[
                "materialization_identity_sha256"
            ],
            expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
            expected_plan_identity_sha256=plan["plan_identity_sha256"],
            expected_loss_bearing_manifest_identity_sha256=manifest[
                "manifest_identity_sha256"
            ],
        )


def test_derived_batch_has_no_caller_controlled_mask_or_ignore_geometry() -> None:
    _guard, plan, manifest = _authority(batch_count=1)
    source = _source(plan, manifest)
    batch = source.build_batch(batch_index=0, device="cpu")

    assert "loss_mask" not in batch
    assert "labels" not in batch
    assert batch["input_ids"].shape == batch["target_ids"].shape == (1, 2)
    assert bool(batch["target_ids"].ne(-100).all())
