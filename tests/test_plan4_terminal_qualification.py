"""Plan 4 S10: LOCAL_FREE terminal integration on a one-step trained CPU model.

These are component/contract tests, not champion or whole-product evidence.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from tools.evaluation_harness import FrozenProtocol, FrozenSuite, evaluate
from tools.inference_runtime import GenerationConfig, ReferenceInference, _model_weights_sha
from tools.plan4_backend_parity import DENIAL, qualify_cpu_dynamic_int8, verify_report
from tools.plan4_model_service import LocalClient, ModelService, SCHEMA as SERVICE_SCHEMA
from tools.plan4_multimodel_serving import MultiModelServing, SCHEMA as SERVING_SCHEMA
from tools.plan4_safe_export import export_safe_bundle, verify_safe_export
from tools.plan4_serving_observability import ServingObserver


def _call(client, op, **args):
    return client.call({"schema": SERVICE_SCHEMA, "op": op, "args": args})


def _request(op, **args):
    return {"schema": SERVING_SCHEMA, "op": op, "args": args}


@pytest.fixture
def trained():
    """One real optimizer step; no network, corpus, CUDA or paid infrastructure."""
    spec = ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=32, d_model=16,
        n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32,
        rope_rotary_dim=8, tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(421)
        model = TwelveSixDecoder(spec)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tokens = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    loss = F.cross_entropy(
        model(tokens).logits[:, :-1, :].reshape(-1, spec.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    return model


def test_trained_checkpoint_eval_export_quantization_and_serving(tmp_path, trained):
    """One learned fixture identity traverses all accepted Plan-4 boundaries."""
    runtime = ReferenceInference(trained, ByteTokenizer())
    weights = _model_weights_sha(trained)
    config = GenerationConfig(max_new_tokens=2)
    direct = runtime.generate("A", config)

    # S2 frozen public evaluation: score may be poor, but cannot be invented.
    suite = FrozenSuite("plan4-s10-fixture", "v1", (("a", "A", "Z"),))
    protocol = FrozenProtocol(suite.identity, (17,), "exact_match", "s10-v1")
    report = evaluate(
        suite, protocol, candidate_sha256=weights,
        predict=lambda prompt, seed: runtime.generate(prompt, config).text,
    )
    assert report["candidate_sha256"] == weights
    assert report["observation_count"] == 1
    assert 0 <= report["score"] <= 1

    identity = CheckpointIdentity(
        git_sha="a" * 40, model_spec=asdict(trained.spec),
        parameter_count=trained.spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash="3" * 64, run_manifest_hash="4" * 64,
        training_config={"steps": 1}, seed=421, precision="float32",
        step=1, tokens_seen=4, optimizer={"name": "sgd"}, scheduler=None,
    )
    source = save_checkpoint(tmp_path / "source", model=trained, identity=identity)
    export_safe_bundle(tmp_path / "source", tmp_path / "bundle")
    exported = verify_safe_export(
        tmp_path / "bundle", expected_checkpoint_id=source["checkpoint_id"],
    )
    assert exported["tokenizer_hash"] == BYTE_TOKENIZER_HASH
    assert exported["interoperability"] == "HF_STYLE_ONLY_NO_TRANSFORMERS_PARITY_CLAIM"

    parity = qualify_cpu_dynamic_int8(trained)
    verify_report(parity)
    assert parity["reference_weights_sha256"] == weights
    assert parity["production_promotion"] == DENIAL

    service = ModelService(max_sessions=4)
    service.register("trained", runtime)
    client = LocalClient(service)
    assert _call(client, "load", model="trained")["ok"]
    service_identity = _call(client, "identity")["result"]
    assert service_identity["model_weights_sha256"] == weights

    plane = MultiModelServing(budget_bytes=16, max_sessions=4)
    plane.publish(
        slot_id="trained-v1", role="answer", provider="fixture",
        model="trained", transport="local", client=client,
        priority=1, reserved_bytes=4,
    )
    observer = ServingObserver(plane, max_inflight=1)
    request = _request(
        "generate", request_id="s10-generate", role="answer",
        prompt="A", config={"max_new_tokens": 2}, failover=False,
    )
    observed = observer.dispatch(request)
    assert observed["ok"], observed
    assert observed["selected"]["identity"] == service_identity
    generated = observed["result"]["data"]["generation"]
    assert tuple(generated["output_token_ids"]) == direct.output_token_ids
    assert generated["result_sha256"] == direct.result_sha256
    assert observer.metrics()["successful_requests"] == 1

    # Restart destroys readiness. The old pinned route cannot silently recover.
    assert _call(client, "restart")["ok"]
    old = observer.dispatch(request)
    assert not old["ok"]
    assert _call(client, "load", model="trained")["ok"]
    plane.replace(
        "trained-v1", slot_id="trained-v2", role="answer",
        provider="fixture", model="trained", transport="local",
        client=client, priority=1, reserved_bytes=4,
    )
    restored = ServingObserver(plane).dispatch(request)
    assert restored["ok"]
    assert restored["selected"]["slot_id"] == "trained-v2"
    assert restored["selected"]["identity"]["model_weights_sha256"] == weights


def test_stream_cancel_canary_and_external_denial_on_trained_fixture(trained):
    """Cancellation and permission fences survive the full S6->S7->S8->S9 path."""
    service = ModelService(max_sessions=4)
    service.register("trained", ReferenceInference(trained, ByteTokenizer()))
    client = LocalClient(service)
    assert _call(client, "load", model="trained")["ok"]
    plane = MultiModelServing(budget_bytes=16, max_sessions=4)
    plane.publish(
        slot_id="local", role="answer", provider="fixture",
        model="trained", transport="local", client=client,
        priority=1, reserved_bytes=4,
    )
    observer = ServingObserver(plane)
    first = observer.dispatch(_request(
        "stream.start", request_id="s10-stream", role="answer",
        prompt="A", config={"max_new_tokens": 3}, failover=False,
    ))
    assert first["ok"], first
    session = first["result"]["data"]["session_id"]
    cancelled = observer.dispatch(_request("stream.cancel", session_id=session))
    assert cancelled["ok"], cancelled
    event = observer.dispatch(_request(
        "stream.next", session_id=session, cursor=0,
    ))
    assert event["ok"] and event["result"]["data"]["event"]["kind"] == "CANCELLED"
    assert observer.metrics()["cancellations"] == 1
    assert observer.canary(role="answer")["evaluation"] is False
    assert observer.canary(role="answer")["ready"] is True

    external = MultiModelServing(budget_bytes=16)
    external.publish(
        slot_id="remote", role="answer", provider="remote",
        model="trained", transport="external", client=client,
        priority=1, reserved_bytes=4, authorize_external=True,
    )
    denied = ServingObserver(external).canary(role="answer")
    assert denied["ready"] is False
    assert denied["reason"] == "EXTERNAL_DENIED"
    assert denied["evaluation"] is False
