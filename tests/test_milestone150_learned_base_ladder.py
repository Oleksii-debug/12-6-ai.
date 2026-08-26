from __future__ import annotations

from twelve_six.milestone150_entrypoint import json_normalize
from twelve_six.milestone150_learned_base_ladder import (
    EXPECTED_CORPUS_ID,
    SCALE_ORDER,
    SCALE_SPECS,
    _read_json,
    _run_manifest,
    _write_json,
    evaluation_identity,
    init_spec,
    model_spec,
    trainer_config,
)
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.tokenization import ByteTokenizer


def test_milestone150_scale_family_is_exact_and_byte_native() -> None:
    expected = {
        "100k": (95_568, "4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470"),
        "500k": (467_808, "208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a"),
        "1m": (1_037_696, "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07"),
    }
    assert SCALE_ORDER == ("100k", "500k", "1m")
    assert set(SCALE_SPECS) == set(expected)
    assert "10m" not in SCALE_SPECS
    for scale, (parameters, identity) in expected.items():
        spec = model_spec(scale)
        assert spec.vocab_size == 256
        assert spec.max_seq_len == 256
        assert spec.parameter_count() == parameters
        assert spec.identity_sha256() == identity


def test_milestone150_rungs_are_exact_members_of_inherited_controlled_family() -> None:
    family = controlled_specs()
    by_parameters = {spec.parameter_count(): spec for spec in family}
    for scale in SCALE_ORDER:
        retained = model_spec(scale)
        inherited = by_parameters[retained.parameter_count()]
        assert retained.to_dict() == inherited.to_dict()
        assert retained.identity_sha256() == inherited.identity_sha256()


def test_milestone150_init_and_common_evaluation_identity_are_frozen() -> None:
    init = init_spec()
    assert init.identity_sha256() == "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"

    tok = ByteTokenizer()
    manifest = {"corpus_identity_sha256": EXPECTED_CORPUS_ID}
    first = evaluation_identity(tok, manifest)
    second = evaluation_identity(tok, manifest)
    assert first == second
    assert first["corpus_identity_sha256"] == EXPECTED_CORPUS_ID
    assert first["split"] == "validation"
    assert first["strata_order"] == ["uk", "en", "code"]
    assert first["tokenizer"]["version"] == "s0-byte-v1"
    assert first["tokenizer"]["vocab_size"] == 256
    assert first["tokenizer"]["special_tokens"] == {}
    assert first["packing"]["sequence_length"] == 128
    assert first["packing"]["cross_document"] is False


def test_milestone150_fresh_process_manifest_representation_is_json_stable() -> None:
    in_process = {
        "trainer_config": {"betas": (0.9, 0.95), "precision": "fp32"},
        "checkpoint_steps": (0, 250, 500, 750, 1000),
        "identity_sha256": "representation-only-sentinel",
    }
    normalized = json_normalize(in_process)
    assert normalized["trainer_config"]["betas"] == [0.9, 0.95]
    assert normalized["checkpoint_steps"] == [0, 250, 500, 750, 1000]
    assert normalized["identity_sha256"] == in_process["identity_sha256"]
    assert json_normalize(normalized) == normalized


def test_milestone150_core_run_manifest_roundtrips_without_shim(tmp_path) -> None:
    tok = ByteTokenizer()
    manifest = {"corpus_identity_sha256": EXPECTED_CORPUS_ID}
    eval_id = evaluation_identity(tok, manifest)
    run = _run_manifest(
        "0" * 40,
        "100k",
        model_spec("100k"),
        init_spec(),
        tok,
        manifest,
        eval_id,
        trainer_config(),
        {"combined_sha256": "1" * 64},
    )
    path = tmp_path / "run-manifest.json"
    _write_json(path, run)
    persisted = _read_json(path)
    assert persisted == run
    assert run["trainer_config"]["betas"] == [0.9, 0.95]


def test_milestone150_trainer_manifest_fields_are_json_native_before_execution() -> None:
    tok = ByteTokenizer()
    manifest = {"corpus_identity_sha256": EXPECTED_CORPUS_ID}
    eval_id = evaluation_identity(tok, manifest)
    for scale in SCALE_ORDER:
        run = _run_manifest(
            "0" * 40,
            scale,
            model_spec(scale),
            init_spec(),
            tok,
            manifest,
            eval_id,
            trainer_config(),
            {"combined_sha256": "1" * 64},
        )
        assert json_normalize(run) == run
        assert run["trainer_config"]["betas"] == [0.9, 0.95]
        assert run["checkpoint_steps"] == [0, 250, 500, 750, 1000]
