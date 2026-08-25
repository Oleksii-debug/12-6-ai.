from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/tok112/MACHINE_REPORT_LOCAL_FREE_20260826.json"
CONFIG_PATH = ROOT / "configs/experiments/tok112_milestone100_v1.json"
DIAG_PATH = ROOT / "data/diagnostics/ua_tokenization_v1.jsonl"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_reserved_ua_diagnostic_identity_and_coverage() -> None:
    cfg = load_json(CONFIG_PATH)
    expected = cfg["reserved_ua_diagnostic"]["sha256"]
    assert hashlib.sha256(DIAG_PATH.read_bytes()).hexdigest() == expected
    rows = [json.loads(line) for line in DIAG_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert cfg["reserved_ua_diagnostic"]["may_train_tokenizer"] is False
    assert cfg["reserved_ua_diagnostic"]["may_train_language_model"] is False
    categories = {row["category"] for row in rows}
    required = {
        "inflection", "case_endings", "verb_forms", "prefix_suffix", "apostrophe",
        "soft_sign", "hyphenated", "proper_names", "numbers", "abbreviations",
        "mixed_technical", "long_morphology",
    }
    assert required <= categories


def test_tok37_artifact_reproduced_and_no_unknowns() -> None:
    report = load_json(REPORT_PATH)
    assert all(report["tok37_artifact"]["fixed_probe_reproduction_exact"].values())
    for diag in report["tokenizer_diagnostics"]:
        if diag["tokenizer"].startswith("bpe"):
            assert diag["unknown_tokens"] == 0


def test_parameter_matched_probes_select_bpe437() -> None:
    report = load_json(REPORT_PATH)
    probes = report["matched_ua_bpb_probes"]
    for probe in probes:
        target = 100_000 if probe["scale"] == "100K" else 500_000
        assert abs(probe["parameters"] - target) / target < 0.01
    for scale in ("100K", "500K"):
        group = [p for p in probes if p["scale"] == scale]
        assert min(group, key=lambda p: p["final_bpb"])["tokenizer"] == "bpe437"
    assert report["tokenizer_recommendation"]["ukrainian_specific_tokenizer_needed"] is False


def test_learned_base_end_to_end_proof() -> None:
    report = load_json(REPORT_PATH)
    final = report["final_learned_base"]
    assert final["random_initialization"] is True
    assert final["parameters"] == 467_808
    assert final["optimized_tokens"] == 201_600
    assert final["optimizer_steps"] == 800
    assert final["train_loss_last32"] < final["train_loss_first32"]
    assert final["fresh_process_resume"]["passed"] is True
    assert final["fresh_process_resume"]["phase1_pid"] != final["fresh_process_resume"]["phase2_pid"]
    assert final["evaluation_non_mutation"] is True
    assert final["generation_before"] != final["generation_after"]
    for domain, initial in final["heldout_bpb_initial"].items():
        assert final["heldout_bpb_final"][domain] < initial
    assert len(final["checkpoint_sha256"]) == 5


def test_truth_boundary_remains_fail_closed() -> None:
    report = load_json(REPORT_PATH)
    assert report["tok111"]["bpe_candidates_trained"] == 0
    assert report["verdict"]["genuinely_learned_local_base"] == "PASS"
    assert report["verdict"]["strict_real_representative_corpus_gate"] == "FAIL_UNMET"
    cfg = load_json(CONFIG_PATH)
    assert cfg["decision_gates"]["external_real_world_corpus_representative"] is False
    assert cfg["decision_gates"]["bpe_allowed_for_complete_first_party_inference"] is False
