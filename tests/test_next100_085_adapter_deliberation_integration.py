from __future__ import annotations

import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import torch

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.inference.contracts import GenerationResult
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.postbase import (
    BaseCheckpointEvidence,
    ControllerGenerationRequest,
    ControllerGenerationResponse,
    PostBaseGenerationEvidence,
    PostBaseModelAdapter,
)
from twelve_six.postbase.controller_integration import (
    DeliberationBaseBridge,
    HypothesisBaseBridge,
)
from twelve_six.postbase_deliberation import Budget, Config, DeliberationController, Verification
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _identity(spec: ModelSpec, tokenizer: ByteTokenizer) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": spec.max_seq_len},
        },
        seed=17,
        precision="fp32",
        step=3,
        tokens_seen=96,
        optimizer={"name": "adamw"},
        scheduler=None,
        environment_lock_hash="d" * 64,
    )


def _checkpoint(tmp_path: Path) -> Path:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    spec = replace(stage.model, max_seq_len=1024)
    tokenizer = ByteTokenizer()
    checkpoint = tmp_path / "base-checkpoint"
    save_checkpoint(
        checkpoint,
        model=TwelveSixDecoder(spec, stage.init),
        identity=_identity(spec, tokenizer),
    )
    return checkpoint


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _weight_digest(adapter: PostBaseModelAdapter) -> str:
    # Audit-only access: the production adapter still exposes no mutable model handle.
    digest = hashlib.sha256()
    state = adapter._backend.model.state_dict()  # noqa: SLF001
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.flatten().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _AlwaysPassVerifier:
    def evaluate(
        self,
        task: str,
        text: str,
        branch_id: str,
        iteration: int,
    ) -> Verification:
        return Verification(score=1.0, confidence=1.0, summary="mechanics-only fixture")


class _RecordingPort:
    def __init__(self) -> None:
        self.requests: list[ControllerGenerationRequest] = []
        self.base_evidence = BaseCheckpointEvidence(
            checkpoint_id="fixture",
            git_sha="a" * 40,
            model_spec_sha256="b" * 64,
            parameter_count=10,
            vocab_size=256,
            max_context_tokens=1024,
            tokenizer_version="s0-byte-v1",
            tokenizer_config_sha256="c" * 64,
            tokenizer_vocab_sha256="d" * 64,
            dataset_manifest_sha256="e" * 64,
            run_manifest_sha256="f" * 64,
            step=1,
            tokens_seen=1,
            device="cpu",
        )

    def generate(
        self,
        request: ControllerGenerationRequest,
    ) -> ControllerGenerationResponse:
        self.requests.append(request)
        text = (
            "mechanics hypothesis"
            if request.controller == "hypothesis"
            else "mechanics candidate"
        )
        token_ids = tuple(text.encode("utf-8")[: request.config.max_new_tokens])
        generation = GenerationResult(
            prompt_token_ids=(1,),
            generated_token_ids=token_ids,
            text=text,
            stop_reason="max_new_tokens",
        )
        post = PostBaseGenerationEvidence(
            evidence_namespace="post_base",
            adapter_version="fixture",
            runtime_policy="LOCAL_FREE",
            controller=request.controller,
            generation_config_sha256="1" * 64,
            prompt_utf8_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
            prompt_token_count=1,
            generated_token_count=len(token_ids),
            generated_token_ids_sha256="2" * 64,
            stop_reason="max_new_tokens",
        )
        return ControllerGenerationResponse(generation, self.base_evidence, post)


def test_deliberation_bridge_translates_values_without_aliasing_controller_state() -> None:
    from twelve_six.postbase_deliberation import Request

    port = _RecordingPort()
    bridge = DeliberationBaseBridge(port)
    original = Request(
        task="fixture task",
        stage="propose",
        branch_id="branch-1",
        candidate_id="candidate-1",
        iteration=0,
        max_generated_tokens=4,
        max_tool_calls=7,
        deadline_monotonic=123.0,
    )

    bridge.generate(original)
    translated = port.requests[0]
    assert translated is not original
    assert translated.controller == "deliberation"
    assert translated.config.max_new_tokens == 4
    assert '"task":"fixture task"' in translated.prompt
    assert "max_tool_calls" not in translated.prompt
    assert "deadline_monotonic" not in translated.prompt
    assert bridge.post_base_evidence[0].evidence_namespace == "post_base"


def test_hypothesis_bridge_keeps_graph_and_generation_evidence_post_base_only() -> None:
    port = _RecordingPort()
    bridge = HypothesisBaseBridge(port)

    hypothesis = bridge.propose("mechanics task", max_new_tokens=4)
    bridge.critique(hypothesis.id, "mechanics task", max_new_tokens=4)
    bridge.search.test(
        hypothesis.id,
        name="deterministic equality",
        prediction=1,
        observed=1,
        source="next100-085 mechanics fixture",
    )

    assert all(request.controller == "hypothesis" for request in port.requests)
    assert all(
        item.evidence_namespace == "post_base"
        for item in bridge.post_base_evidence
    )
    serialized = str(bridge.search.export())
    assert "checkpoint_id" not in serialized
    assert "model_spec_sha256" not in serialized
    assert port.base_evidence.evidence_namespace == "base"


def test_repeated_deliberation_controller_calls_preserve_checkpoint_and_weights(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    files_before = _snapshot_files(checkpoint)
    adapter = PostBaseModelAdapter.from_checkpoint(checkpoint)
    weights_before = _weight_digest(adapter)
    base_before = asdict(adapter.base_evidence)
    bridge = DeliberationBaseBridge(adapter)
    controller = DeliberationController(
        bridge,
        _AlwaysPassVerifier(),
        config=Config(initial_branches=1, target_score=1.0, min_confidence=1.0),
    )

    budget = Budget(model_calls=1, generated_tokens=2, candidate_branches=1)
    controller.run("mechanics run one", budget)
    controller.run("mechanics run two", budget)

    assert _snapshot_files(checkpoint) == files_before
    assert _weight_digest(adapter) == weights_before
    assert asdict(adapter.base_evidence) == base_before
    assert len(bridge.post_base_evidence) == 2
    assert all(
        item.evidence_namespace == "post_base"
        for item in bridge.post_base_evidence
    )


def test_post_base_integration_evidence_cannot_silently_absorb_base_provenance() -> None:
    port = _RecordingPort()
    bridge = HypothesisBaseBridge(port)
    bridge.propose("namespace firewall", max_new_tokens=2)

    post_text = str(asdict(bridge.post_base_evidence[0]))
    for forbidden in (
        "checkpoint_id",
        "git_sha",
        "model_spec_sha256",
        "parameter_count",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
        "tokens_seen",
    ):
        assert forbidden not in post_text
