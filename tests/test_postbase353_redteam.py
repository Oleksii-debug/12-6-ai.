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
from twelve_six.post_base.sft_recovery import (
    POSTBASE352_TOKENIZER_ID,
    TERMINAL_EVIDENCE_NAMESPACE,
    terminalize_sft_mechanics,
)
from twelve_six.post_base.sft_runner import (
    MAX_MECHANICS_STEPS,
    SFT_CHECKPOINT_NAMESPACE,
    SFT_EVALUATION_NAMESPACE,
    FixtureProvenance,
    FixtureSourceKind,
    MechanicsSplit,
    SFTCheckpointStore,
    SFTMechanicsDataset,
    SFTMechanicsExample,
    SFTMechanicsPlan,
)
from twelve_six.posttraining.contracts import CheckpointRef, LineageKind


class ProjectOwnedToyBackend:
    backend_id = "local-project-owned-redteam-sft-v1"

    def __init__(self) -> None:
        self.train_calls = 0
        self.evaluate_calls = 0

    @staticmethod
    def _state(state: object) -> dict[str, float]:
        if not isinstance(state, dict) or "weight" not in state:
            raise TypeError("invalid red-team toy state")
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
        self.train_calls += 1
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
        self.evaluate_calls += 1
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


def _project_provenance(source_id: str) -> FixtureProvenance:
    return FixtureProvenance(
        source_id=source_id,
        source_kind=FixtureSourceKind.PROJECT_OWNED,
        source_revision="next100-088-project-fixture-v1",
    )


def _dataset(*, target: str = "1.0", forged_foreign: bool = False) -> SFTMechanicsDataset:
    train_one = _project_provenance("project:redteam/train-1")
    if forged_foreign:
        object.__setattr__(train_one, "foreign_model_output", True)
    return SFTMechanicsDataset(
        dataset_id="next100-088-project-owned-mechanics-v1",
        train=(
            SFTMechanicsExample(
                record_id="redteam-train-1",
                split=MechanicsSplit.TRAIN,
                user_text="Return the project-owned scalar target.",
                assistant_text=target,
                provenance=train_one,
            ),
            SFTMechanicsExample(
                record_id="redteam-train-2",
                split=MechanicsSplit.TRAIN,
                user_text="Return the same project-owned scalar target.",
                assistant_text=target,
                provenance=_project_provenance("project:redteam/train-2"),
            ),
        ),
        evaluation=(
            SFTMechanicsExample(
                record_id="redteam-eval-1",
                split=MechanicsSplit.EVALUATION,
                user_text="Evaluate the project-owned scalar target.",
                assistant_text=target,
                provenance=_project_provenance("project:redteam/eval-1"),
            ),
        ),
    )


def _tokenizer(*, tokenizer_id: str = POSTBASE352_TOKENIZER_ID) -> TokenizerCompatibility:
    return TokenizerCompatibility(
        tokenizer_id=tokenizer_id,
        config_sha256="6" * 64,
        vocab_sha256="7" * 64,
        vocab_size=256,
    )


def _base_root(tmp_path: Path) -> Path:
    root = tmp_path / "canonical_base"
    root.mkdir()
    (root / "weights.json").write_text('{"weight": 0.0}', encoding="utf-8")
    return root


def _base_checkpoint(base_root: Path) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="fixture/base/next100-088",
        sha256=snapshot_directory(base_root).identity_sha256,
        git_sha="abcdef0",
        stage="learned_fixture",
        lineage=LineageKind.BASE,
    )


def _contract(
    *,
    dataset: SFTMechanicsDataset,
    base_root: Path,
    tokenizer: TokenizerCompatibility,
) -> PostBaseConsumptionContract:
    return PostBaseConsumptionContract(
        contract_id="next100-088-redteam-contract",
        base_checkpoint=_base_checkpoint(base_root),
        base_policy=CanonicalBasePolicy(),
        tokenizer=tokenizer,
        dataset=dataset.to_contract_provenance(),
        stage=PostBaseStage.COMMUNICATION_SUPERVISION,
    )


def _plan(
    *,
    dataset: SFTMechanicsDataset,
    base_root: Path,
    tokenizer: TokenizerCompatibility | None = None,
    max_steps: int = 2,
) -> SFTMechanicsPlan:
    bound = tokenizer or _tokenizer()
    return SFTMechanicsPlan(
        run_id="next100-088-redteam",
        backend_id=ProjectOwnedToyBackend.backend_id,
        contract=_contract(dataset=dataset, base_root=base_root, tokenizer=bound),
        tokenizer=bound,
        max_steps=max_steps,
        seed=20260826,
    )


def _authority_path(experiment_root: Path) -> Path:
    return experiment_root / TERMINAL_EVIDENCE_NAMESPACE / "authority.json"


def test_project_owned_fixture_performs_deterministic_update_and_pointer_rollback(
    tmp_path: Path,
) -> None:
    base_root = _base_root(tmp_path)
    base_before = snapshot_directory(base_root)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=2)
    backend = ProjectOwnedToyBackend()
    experiment_root = tmp_path / "valid"

    terminal = terminalize_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=backend,
    )

    assert terminal.mechanics.training_steps == 2
    assert backend.train_calls == 2
    assert snapshot_directory(base_root) == base_before
    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    final_state = ProjectOwnedToyBackend._state(store.load_state(backend, 2))
    assert final_state["weight"] == pytest.approx(0.75)

    generations_before = {
        generation: snapshot_directory(store.generation_path(generation))
        for generation in range(3)
    }
    store.rollback_to(0)
    rollback_state = ProjectOwnedToyBackend._state(store.load_state(backend, 0))
    assert rollback_state["weight"] == 0.0
    assert {
        generation: snapshot_directory(store.generation_path(generation))
        for generation in range(3)
    } == generations_before


def test_canonical_base_overwrite_attempt_fails_closed(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "canonical-overwrite"

    class BaseOverwriter(ProjectOwnedToyBackend):
        def train_step(
            self,
            state: object,
            example: SFTMechanicsExample,
            *,
            step: int,
            seed: int,
        ) -> dict[str, float]:
            (base_root / "weights.json").write_text('{"weight": 999.0}', encoding="utf-8")
            return super().train_step(state, example, step=step, seed=seed)

    with pytest.raises(RuntimeError, match="canonical Base mutated"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=BaseOverwriter(),
        )
    assert not _authority_path(experiment_root).exists()


def test_checkpoint_namespace_escape_is_rejected_before_workspace(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    tokenizer = _tokenizer()
    contract = _contract(dataset=dataset, base_root=base_root, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="dedicated post-Base namespace"):
        SFTMechanicsPlan(
            run_id="namespace-escape",
            backend_id=ProjectOwnedToyBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=1,
            seed=0,
            checkpoint_namespace="../canonical_base",
        )


def test_rollback_cannot_overwrite_immutable_generation(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    backend = ProjectOwnedToyBackend()
    experiment_root = tmp_path / "rollback-overwrite"
    terminal = terminalize_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=backend,
    )
    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    generation_zero_before = snapshot_directory(store.generation_path(0))
    generation_one_before = snapshot_directory(store.generation_path(1))

    store.rollback_to(0)
    assert snapshot_directory(store.generation_path(0)) == generation_zero_before
    assert snapshot_directory(store.generation_path(1)) == generation_one_before
    with pytest.raises(FileExistsError, match="already exists"):
        store.publish(
            backend,
            {"weight": 123.0},
            generation=0,
            parent_generation=None,
            input_snapshot_sha256=terminal.mechanics.workspace.cloned_snapshot_sha256,
            base_checkpoint=plan.contract.base_checkpoint,
            dataset_manifest_sha256=dataset.manifest_sha256,
        )


def test_dataset_hash_substitution_is_rejected_before_workspace(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    contracted = _dataset(target="1.0")
    substituted = _dataset(target="2.0")
    plan = _plan(dataset=contracted, base_root=base_root)
    experiment_root = tmp_path / "dataset-substitution"

    with pytest.raises(ValueError, match="do not match"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=substituted,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ProjectOwnedToyBackend(),
        )
    assert not experiment_root.exists()


def test_tokenizer_mismatch_is_rejected_before_workspace(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    incompatible = _tokenizer(tokenizer_id="foreign-tokenizer-v1")
    plan = _plan(dataset=dataset, base_root=base_root, tokenizer=incompatible)
    experiment_root = tmp_path / "tokenizer-mismatch"

    with pytest.raises(ValueError, match="s0-byte-v1"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ProjectOwnedToyBackend(),
        )
    assert not experiment_root.exists()


def test_forged_foreign_model_example_is_rejected_before_workspace(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset(forged_foreign=True)
    plan = _plan(dataset=dataset, base_root=base_root)
    experiment_root = tmp_path / "foreign-example"

    with pytest.raises(ValueError, match="foreign model output"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ProjectOwnedToyBackend(),
        )
    assert not experiment_root.exists()


def test_evaluation_metric_mutation_is_detected_before_authority(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "evaluation-mutation"

    class EvaluationMutator(ProjectOwnedToyBackend):
        def train_step(
            self,
            state: object,
            example: SFTMechanicsExample,
            *,
            step: int,
            seed: int,
        ) -> dict[str, float]:
            path = (
                experiment_root
                / SFT_EVALUATION_NAMESPACE
                / "generation_000000_baseline.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metrics"] = {"mse": 999.0}
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            return super().train_step(state, example, step=step, seed=seed)

    with pytest.raises(RuntimeError, match="evaluation evidence payload drift"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=EvaluationMutator(),
        )
    assert not _authority_path(experiment_root).exists()


def test_checkpoint_generation_corruption_is_detected_before_authority(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "generation-corruption"

    class GenerationMutator(ProjectOwnedToyBackend):
        def train_step(
            self,
            state: object,
            example: SFTMechanicsExample,
            *,
            step: int,
            seed: int,
        ) -> dict[str, float]:
            path = (
                experiment_root
                / SFT_CHECKPOINT_NAMESPACE
                / "generation_000000"
                / "backend"
                / "weights.json"
            )
            path.write_text('{"weight": 777.0}', encoding="utf-8")
            return super().train_step(state, example, step=step, seed=seed)

    with pytest.raises(RuntimeError, match="immutable manifest"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=GenerationMutator(),
        )
    assert not _authority_path(experiment_root).exists()


def test_parent_chain_forgery_is_detected_before_authority(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "parent-forgery"

    class ParentChainForger(ProjectOwnedToyBackend):
        def evaluate(
            self,
            state: object,
            examples: tuple[SFTMechanicsExample, ...],
        ) -> dict[str, float]:
            metrics = super().evaluate(state, examples)
            if self.evaluate_calls == 2:
                path = (
                    experiment_root
                    / SFT_CHECKPOINT_NAMESPACE
                    / "generation_000001"
                    / "manifest.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["parent_generation"] = None
                path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            return metrics

    with pytest.raises(RuntimeError, match="generation chain is broken"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=ParentChainForger(),
        )
    assert not _authority_path(experiment_root).exists()


def test_step_budget_bypass_is_rejected_at_plan_construction(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    tokenizer = _tokenizer()
    contract = _contract(dataset=dataset, base_root=base_root, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="max_steps"):
        SFTMechanicsPlan(
            run_id="step-budget-bypass",
            backend_id=ProjectOwnedToyBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=MAX_MECHANICS_STEPS + 1,
            seed=0,
        )
