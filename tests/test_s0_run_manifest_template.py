import json
from pathlib import Path

MANIFEST_PATH = Path("configs/runs/s0_10k.local_cpu.example.json")


def _get_path(document: dict, dotted_path: str):
    value = document
    for part in dotted_path.split("."):
        value = value[part]
    return value


def test_s0_local_cpu_manifest_is_parseable_and_fail_closed():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["stage"] == "S0"
    assert manifest["state"] == "PREPARED_NOT_LAUNCHED"
    assert manifest["authorization"]["class"] == "LOCAL_FREE"
    assert manifest["authorization"]["metered_cost_expected"] is False
    assert manifest["authorization"]["estimated_cost_max"] == 0
    assert manifest["authorization"]["spending_ceiling"] == 0
    assert manifest["launch_gate"]["fail_closed"] is True


def test_s0_local_cpu_example_cannot_be_mistaken_for_launch_ready():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    unresolved = []
    for dotted_path in manifest["launch_gate"]["required_non_null"]:
        value = _get_path(manifest, dotted_path)
        if value in (None, "", "UNRESOLVED"):
            unresolved.append(dotted_path)

    assert unresolved, "example manifest must retain unresolved launch identities"
    for required_identity in (
        "candidate.git_sha",
        "candidate.modelspec_sha256",
        "candidate.initspec_sha256",
        "data.dataset_manifest_sha256",
        "data.tokenizer_sha256",
        "data.tokenizer_vocab_sha256",
        "data.split_identity",
        "data.packing_sha256",
        "data.packing_version",
        "environment.lock_sha256",
    ):
        assert required_identity in unresolved


def test_checkpoint_lineage_dimensions_are_independent_launch_gates():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required = manifest["launch_gate"]["required_non_null"]

    expected = (
        "candidate.modelspec_sha256",
        "candidate.initspec_sha256",
        "data.tokenizer_sha256",
        "data.tokenizer_vocab_sha256",
        "data.dataset_manifest_sha256",
        "data.split_identity",
        "data.packing_sha256",
        "data.packing_version",
        "environment.lock_sha256",
    )
    for dotted_path in expected:
        assert required.count(dotted_path) == 1
