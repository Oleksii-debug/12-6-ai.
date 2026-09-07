from __future__ import annotations

import json
import subprocess
from pathlib import Path

from twelve_six.inference.server import make_server
from twelve_six.integration import S0TorchInferenceBackend
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.s0_candidate_evaluation import build_reports, collect_s0_candidate_evidence
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
COMPOSITION = (
    ROOT
    / "configs/releases/s0_candidate_successor_convergence_20260825.experimental.json"
)


def _git_head() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert len(head) == 40
    assert head == head.lower()
    assert all(ch in "0123456789abcdef" for ch in head)
    return head


def test_successor_manifest_is_fail_closed_and_ancestry_complete() -> None:
    payload = json.loads(COMPOSITION.read_text(encoding="utf-8"))
    head = _git_head()

    assert payload["status"] == "experimental"
    assert payload["canonical_base"] == "random_init_pretraining_only"
    assert payload["composition_complete"] is True
    assert payload["promotion_eligible"] is False
    assert payload["authority_snapshot"]["main_protected"] is False
    assert payload["audits"]["AUDIT-A"]["verdict"] == "CHANGES_REQUIRED"
    assert payload["audits"]["AUDIT-B"]["verdict"] == "CHANGES_REQUIRED"
    assert payload["integration_method"]["changed_path_overlap"] == 0
    assert payload["intake_disposition"]["history_destroyed"] is False

    accepted = {row["pr_number"]: row["source_sha"] for row in payload["late_wave_intake"]}
    assert accepted == {
        82: "e5a3b551fa509fd6d36f51915cd887f5cc352f69",
        84: "c23b14c7fc23f089309926e2870d6c32d0cd7f02",
        85: "e9ecbccafaf2e9191e946b819319caf191f31353",
        86: "11755855fd136709599ff13e514c9cc8256df011",
        89: "c631c024e641dac102036fafee6d78ba31c067cd",
    }
    assert set(payload["required_git_ancestry"]) == set(accepted.values())

    for source_sha in payload["required_git_ancestry"]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_sha, head],
            cwd=ROOT,
            check=True,
        )

    repeatability = payload["repeatability_evidence"]
    assert repeatability["source_sha"] == accepted[89]
    assert repeatability["same_seed_exact_equivalence"] is True
    assert repeatability["different_seed_initialization_diverges"] is True
    assert repeatability["different_seed_training_diverges"] is True
    assert repeatability["validation_optimized_tokens"] == 0
    assert repeatability["cross_hardware_bitwise_reproducibility_claimed"] is False
    assert repeatability["distributed_reproducibility_claimed"] is False
    assert repeatability["gpu_reproducibility_claimed"] is False
    assert repeatability["promotion_claimed"] is False

    required_surfaces = [
        "src/twelve_six/model.py",
        "src/twelve_six/tokenization/byte.py",
        "data/s0/packaged/manifest.json",
        "src/twelve_six/training/trainer.py",
        "src/twelve_six/training/s0_evidence_contract.py",
        "src/twelve_six/training/s0_repeatability.py",
        "src/twelve_six/checkpoint/core.py",
        "src/twelve_six/s0_candidate_evaluation.py",
        "src/twelve_six/inference/first_party.py",
        "src/twelve_six/inference/server.py",
    ]
    assert all((ROOT / path).is_file() for path in required_surfaces)


def test_exact_head_successor_train_checkpoint_eval_inference_server_contract() -> None:
    head = _git_head()

    # D04's exact-candidate collector is itself a compact cross-lane oracle: it
    # consumes committed data through the canonical tokenizer, trains the random
    # S0 model, performs transactional checkpoint save/load + deterministic
    # resume, evaluates stage gates, and verifies first-party checkpoint
    # inference parity. Four steps keep this LOCAL_FREE test focused and cheap.
    evidence = collect_s0_candidate_evidence(ROOT, head, train_steps=4)
    gate_report, promotion_report = build_reports(evidence)

    assert evidence["candidate"]["sha"] == head
    assert evidence["candidate"]["random_init"] is True
    assert evidence["candidate"]["model_vocab_size"] == 256
    assert evidence["checkpoint"]["save_load_verified"] is True
    assert evidence["checkpoint"]["resume_verified"] is True
    assert evidence["checkpoint"]["serialization_pickle"] is False
    assert gate_report["summary"]["evaluation_complete"] is True
    assert gate_report["summary"]["counts"]["FAIL"] == 0
    assert gate_report["summary"]["counts"]["NOT_TESTED"] == 0
    assert promotion_report["promotion_eligible"] is False
    assert promotion_report["promotion_authority_status"] == "NOT_TESTED"

    # Compose the accepted D07 server with the same canonical model/tokenizer
    # contract. Binding to port 0 opens no external service and remains
    # loopback-only; closing immediately keeps the test deterministic.
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    assert stage.canonical_base == "random_init"
    model = TwelveSixDecoder(stage.model, stage.init)
    backend = S0TorchInferenceBackend(model, tokenizer)
    server = make_server(backend, host="127.0.0.1", port=0, model_name="12-6-s0-test")
    try:
        host, port = server.server_address[:2]
        assert str(host) == "127.0.0.1"
        assert int(port) > 0
        assert server.model_name == "12-6-s0-test"
    finally:
        server.server_close()
