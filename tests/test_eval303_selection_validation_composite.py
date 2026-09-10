import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_eval303_selection_validation_composite.py"
MANIFEST = ROOT / "configs/evaluation/eval303_selection_validation_composite_v1.json"
PROOF = ROOT / "evidence/eval303/data300-exact-exclusion-proof-v1.json"
EXPECTED_SELECTION_IDENTITY = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify() -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--repo-root", str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_eval303_composite_verifies():
    result = _verify()
    assert result["status"] == "PASS"
    assert result["selection_identity_sha256"] == EXPECTED_SELECTION_IDENTITY
    assert result["documents"] == 10


def test_selection_only_and_code_fail_closed():
    manifest = _load_json(MANIFEST)
    usage = manifest["usage_contract"]
    assert usage["selection_only"] is True
    assert usage["may_train"] is False
    assert usage["may_fit_tokenizer"] is False
    assert usage["may_update_model"] is False
    assert usage["may_report_final_test"] is False
    assert manifest["strata"]["code"]["documents"] == 0
    assert manifest["strata"]["code"]["selection_eligible"] is False


def test_data300_exact_exclusion_is_hash_bound_not_family_washed():
    proof = _load_json(PROOF)
    comparisons = proof["comparisons"]
    assert comparisons["selected_content_vs_training_raw_or_normalized_sha256_overlap"] == []
    assert comparisons["selected_git_blob_vs_training_git_blob_overlap"] == []
    assert comparisons["selected_source_family_vs_training_source_family_overlap"] == [
        "github:encode/httpx",
        "github:psf/requests",
    ]
    assert len(comparisons["same_family_distinct_object_evidence"]) == 2
    assert proof["verdict"]["near_copy_or_dedup_cluster_scan_claimed"] is False


def test_final_test_outcomes_and_payload_not_consumed_by_eval303():
    firewall = _load_json(MANIFEST)["final_test_firewall"]
    assert firewall["outcomes_read_by_eval303"] is False
    assert firewall["final_test_payload_read_by_eval303"] is False
    assert firewall["final_test_bytes_copied_into_composite"] is False
