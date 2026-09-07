from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"


def d03_job_block() -> str:
    text = CI.read_text(encoding="utf-8")
    marker = "  d03-rada-trees-secondary-role:\n"
    assert marker in text
    return text.split(marker, 1)[1]


def test_secondary_role_job_uses_shared_ci_and_pinned_runtime_actions() -> None:
    block = d03_job_block()
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in block
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in block
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in block
    assert 'python-version: "3.11.16"' in block
    assert "actions/checkout@v4" not in block
    assert "actions/setup-python@v5" not in block
    assert "actions/upload-artifact@v4" not in block


def test_secondary_role_job_binds_consumed_extractor_and_runtime_identity() -> None:
    block = d03_job_block()
    assert "inventory.find_extractor(None, accepted)" in block
    assert "inventory.extractor_version(extractor)" in block
    assert "rada-secondary-runtime-evidence.json" in block
    assert "runtime_identity_sha256" in block
    assert 'assert platform.python_version() == "3.11.16"' in block
    assert 'current_extractor == runtime["extractor_command"]' in block
    assert 'inventory.extractor_version(current_extractor) == runtime["extractor_version"]' in block


def test_secondary_role_evidence_remains_zero_credit() -> None:
    block = d03_job_block()
    assert 'boundary["training_authorized_bytes"] == 0' in block
    assert 'boundary["unique_causal_loss_positions_authorized"] == 0' in block
    assert 'boundary["tokenizer_fit_authorized"] is False' in block
    assert 'report["decision"]["training_admission_claimed"] is False' in block
