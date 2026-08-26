from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.post_base.contract import (
    CanonicalBasePolicy,
    PostBaseConsumptionContract,
    PostBaseStage,
    TokenizerCompatibility,
    snapshot_directory,
)
from twelve_six.post_base.sft_runner import (
    SFT_CHECKPOINT_NAMESPACE,
    SFT_EVALUATION_NAMESPACE,
    FixtureProvenance,
    FixtureSourceKind,
    MechanicsSplit,
    SFTCheckpointStore,
    SFTEvaluationStore,
    SFTMechanicsDataset,
    SFTMechanicsExample,
    SFTMechanicsPlan,
    run_sft_mechanics,
)
from twelve_six.posttraining.contracts import CheckpointRef, ComputeClass, LineageKind


class ToySFTBackend:
    backend_id = "local-toy-sft-v1"

    @staticmethod
    def _state(state: object) -> dict[str, float]:
        if not isinstance(state, dict) or "weight" not in state:
            raise TypeError("invalid toy state")
        return state

    def load_input_checkpoint(self, checkpoint_root: Path) -> object:
        payload = json.loads((checkpoint_root / "weights.json").read_text(encoding="utf-8"))
        return {"weight": float(payload["weight"])}

    def train_step(
        self,
        state: object,
        example: SFTMechanicsExample,
        *,
        step: int,
        seed: int,
    ) -> dict[str, float]:
        del step, seed
        toy = self._state(state)
        target = float(example.assistant_text)
        error = target - toy["weight"]
        loss = error * error
        toy["weight"] += 0.5 * error
        return {"loss": loss}

    def evaluate(
        self,
        state: object,
        examples: tuple[SFTMechanicsExample, ...],
    ) -> dict[str, float]:
        toy = self._state(state)
        losses = [
            (float(example.assistant_text) - toy["weight"]) ** 2
            for example in examples
        ]
        return {"mse": sum(losses) / len(losses)}

    def save_checkpoint(self, state: object, checkpoint_root: Path) -> None:
        toy = self._state(state)
        (checkpoint_root / "weights.json").write_text(
            json.dumps({"weight": toy["weight"]}, sort_keys=True),
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint_root: Path) -> object:
        return self.load_input_checkpoint(checkpoint_root)


def _provenance(source_id: str) -> FixtureProvenance:
    return FixtureProvenance(
        source_id=source_id,
        source_kind=FixtureSourceKind.SYNTHETIC_LOCAL,
        source_revision="fixture-v1",
        generator_id="local:deterministic-scalar-fixture",
    )


def _dataset(*, eval_target: str = "1.0") -> SFTMechanicsDataset:
    return SFTMechanicsDataset(
        dataset_id="postbase353-fixture-v1",
        train=(
            SFTMechanicsExample(
                record_id="train-1",
                split=MechanicsSplit.TRAIN,
                user_text="Return the numeric target.",
                assistant_text="1.0",
                provenance=_provenance("fixture/train-1"),
            ),
            SFTMechanicsExample(
                record_id="train-2",
                split=MechanicsSplit.TRAIN,
                user_text="Return the same numeric target.",
                assistant_text="1.0",
                provenance=_provenance("fixture/train-2"),
            ),
        ),
        evaluation=(
            SFTMechanicsExample(
                record_id="eval-1",
                split=MechanicsSplit.EVALUATION,
                user_text="Evaluate the numeric target.",
                assistant_text=eval_target,
                provenance=FixtureProvenance(
                    source_id="project/eval-1",
                    source_kind=FixtureSourceKind.PROJECT_OWNED,
                    source_revision="fixture-v1",
                ),
            ),
        ),
    )


def _tokenizer() -> TokenizerCompatibility:
    return TokenizerCompatibility(
        tokenizer_id="fixture-byte-v1",
        config_sha256="1" * 64,
        vocab_sha256="2" * 64,
        vocab_size=256,
    )


def _base_checkpoint(base_root: Path) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="fixture/base/postbase353",
        sha256=snapshot_directory(base_root).identity_sha256,
        git_sha="abcdef0",
        stage="learned_fixture",
        lineage=LineageKind.BASE,
    )


def _contract(
    dataset: SFTMechanicsDataset,
    base_checkpoint: CheckpointRef,
    tokenizer: TokenizerCompatibility,
    *,
    stage: PostBaseStage = PostBaseStage.COMMUNICATION_SUPERVISION,
) -> PostBaseConsumptionContract:
    return PostBaseConsumptionContract(
        contract_id="postbase353-fixture-contract",
        base_checkpoint=base_checkpoint,
        base_policy=CanonicalBasePolicy(),
        tokenizer=tokenizer,
        dataset=dataset.to_contract_provenance(),
        stage=stage,
    )


def _base_root(tmp_path: Path) -> Path:
    root = tmp_path / "canonical_base"
    root.mkdir()
    (root / "weights.json").write_text('{"weight": 0.0}', encoding="utf-8")
    return root


def _plan(
    dataset: SFTMechanicsDataset,
    base_root: Path,
    *,
    max_steps: int = 2,
) -> tuple[SFTMechanicsPlan, CheckpointRef]:
    tokenizer = _tokenizer()
    base_checkpoint = _base_checkpoint(base_root)
    contract = _contract(dataset, base_checkpoint, tokenizer)
    return (
        SFTMechanicsPlan(
            run_id="postbase353-mechanics-test",
            backend_id=ToySFTBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=max_steps,
            seed=20260826,
        ),
        base_checkpoint,
    )


def test_runner_isolates_base_namespaces_evaluation_and_rollback(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    base_before = snapshot_directory(base_root)
    dataset = _dataset()
    plan, base_checkpoint = _plan(dataset, base_root)
    experiment_root = tmp_path / "post_base_run"
    backend = ToySFTBackend()

    receipt = run_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=backend,
    )

    assert receipt.training_steps == 2
    assert receipt.baseline_checkpoint.generation == 0
    assert receipt.final_checkpoint.generation == 2
    assert receipt.rollback_generation == 0
    assert receipt.rollback_base_checkpoint == base_checkpoint
    assert receipt.canonical_snapshot_before == receipt.canonical_snapshot_after
    assert snapshot_directory(base_root) == base_before
    assert (base_root / "weights.json").read_text(encoding="utf-8") == '{"weight": 0.0}'
    assert (
        receipt.workspace.cloned_checkpoint_root / "weights.json"
    ).read_text(encoding="utf-8") == '{"weight": 0.0}'

    assert (
        receipt.checkpoint_namespace.relative_to(experiment_root).as_posix()
        == SFT_CHECKPOINT_NAMESPACE
    )
    assert (
        receipt.evaluation_namespace.relative_to(experiment_root).as_posix()
        == SFT_EVALUATION_NAMESPACE
    )
    assert not receipt.evaluation_namespace.is_relative_to(receipt.checkpoint_namespace)
    assert receipt.final_metrics["mse"] < receipt.baseline_metrics["mse"]

    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    assert store.active_generation() == 2
    final_before = snapshot_directory(store.generation_path(2))
    rollback_path = store.rollback_to(0)
    assert rollback_path == store.generation_path(0)
    assert store.active_generation() == 0
    rollback_state = store.load_state(backend, 0)
    assert ToySFTBackend._state(rollback_state)["weight"] == 0.0
    assert snapshot_directory(store.generation_path(2)) == final_before


def test_runner_refuses_existing_experiment_root_instead_of_overwriting(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan, _ = _plan(dataset, base_root, max_steps=1)
    experiment_root = tmp_path / "post_base_run"

    run_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=ToySFTBackend(),
    )

    with pytest.raises(FileExistsError, match="must not already exist"):
        run_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ToySFTBackend(),
        )


def test_runner_binds_exact_dataset_identity_before_workspace_creation(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    contracted = _dataset(eval_target="1.0")
    supplied = _dataset(eval_target="0.0")
    plan, _ = _plan(contracted, base_root)
    experiment_root = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="do not match"):
        run_sft_mechanics(
            plan=plan,
            dataset=supplied,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ToySFTBackend(),
        )

    assert not experiment_root.exists()


def test_foreign_model_output_is_not_admitted_as_fixture() -> None:
    with pytest.raises(ValueError, match="foreign-model"):
        FixtureProvenance(
            source_id="foreign/output",
            source_kind=FixtureSourceKind.SYNTHETIC_LOCAL,
            source_revision="v1",
            generator_id="local:fixture",
            foreign_model_output=True,
        )


def test_real_campaign_and_paid_compute_are_not_authorized(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    tokenizer = _tokenizer()
    contract = _contract(dataset, _base_checkpoint(base_root), tokenizer)

    with pytest.raises(ValueError, match="does not authorize a real communication campaign"):
        SFTMechanicsPlan(
            run_id="forbidden-real-run",
            backend_id=ToySFTBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=1,
            seed=0,
            real_campaign_authorization_id="not-valid-in-postbase353",
        )

    with pytest.raises(ValueError, match="LOCAL_FREE"):
        SFTMechanicsPlan(
            run_id="forbidden-paid-run",
            backend_id=ToySFTBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=1,
            seed=0,
            compute_class=ComputeClass.MATERIAL_PAID,
        )


def test_sft_plan_rejects_non_supervision_stage(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    tokenizer = _tokenizer()
    contract = _contract(
        dataset,
        _base_checkpoint(base_root),
        tokenizer,
        stage=PostBaseStage.PREFERENCE_OPTIMIZATION,
    )

    with pytest.raises(ValueError, match="communication supervision"):
        SFTMechanicsPlan(
            run_id="wrong-stage",
            backend_id=ToySFTBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=1,
            seed=0,
        )


def test_checkpoint_manifest_detects_generation_byte_mutation(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan, _ = _plan(dataset, base_root, max_steps=1)
    experiment_root = tmp_path / "post_base_run"
    backend = ToySFTBackend()

    run_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=backend,
    )

    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    mutated = store.generation_path(1) / "backend" / "weights.json"
    mutated.write_text('{"weight": 999.0}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable manifest"):
        store.load_state(backend, 1)


def test_evaluation_evidence_is_immutable(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan, _ = _plan(dataset, base_root, max_steps=1)
    experiment_root = tmp_path / "post_base_run"

    receipt = run_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=ToySFTBackend(),
    )

    store = SFTEvaluationStore(experiment_root, plan.run_id, create=False)
    with pytest.raises(FileExistsError, match="immutable"):
        store.publish(
            generation=0,
            phase="baseline",
            metrics=receipt.baseline_metrics,
            dataset=dataset,
        )
