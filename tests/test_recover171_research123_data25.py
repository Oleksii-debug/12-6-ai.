"""Static preflight for RECOVER-171 before the locked torch runtime is installed."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "tools/research123_data25_adapter.py"
HARNESS = ROOT / "tools/research123_real_tn_scaling.py"
RUNNER = ROOT / "tools/run_research123_real_tn_scaling.py"
WORKFLOW = ROOT / ".github/workflows/research123-real-tn-scaling.yml"


def _literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return result


def test_frozen_research123_family_and_grid_are_retained() -> None:
    source = HARNESS.read_text(encoding="utf-8")
    assert "expected = (95_568, 267_912, 467_808, 1_037_696)" in source
    assert "TARGET_TN_RATIOS = (1.0 / 32.0, 1.0 / 8.0, 1.0 / 2.0, 2.0)" in source
    assert "evaluation_optimized_tokens\": 0" in source


def test_data25_adapter_preregisters_common_bounded_trace() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    constants = _literal_constants(ADAPTER)
    assert constants["SCHEMA_VERSION"] == "12-6.research123-data25-tn-scaling.v2"
    assert constants["TRAIN_BATCHES_BY_STRATUM"] == {"uk": 180, "en": 140, "code": 80}
    assert constants["VALIDATION_BATCHES_BY_STRATUM"] == {"uk": 64, "en": 33, "code": 32}
    assert constants["BOOTSTRAP_SAMPLES"] == 400
    assert constants["CURVE_BOOTSTRAP_SAMPLES"] == 300
    assert "EXPECTED_CORPUS_ID = m150.EXPECTED_CORPUS_ID" in source
    assert "BATCH_SIZE = m150.BATCH" in source
    assert "SEQUENCE_LENGTH = m150.SEQ" in source
    assert "cross_document=False" in source


def test_universal_bootstrap_is_paired_and_nonparametric() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    assert "paired_common_nonparametric_packed_batch_resampling" in source
    assert "shared_resample_indices_across_checkpoints_and_scales\": True" in source
    assert "rng.randrange(len(observations))" in source
    assert "parametric_eval_bootstrap_95ci" not in source


def test_runner_installs_adapter_before_frozen_harness_import() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    install = source.index("adapter.install_batch_noise_probe_stub()")
    frozen_import = source.index("import research123_real_tn_scaling as experiment")
    configure = source.index("adapter.configure_experiment(experiment)")
    assert install < frozen_import < configure
    assert "experiment._checkpoint_identity = checkpoint_identity" in source


def test_workflow_preflights_before_any_heavy_locked_runtime_step() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    preflight = source.index("Scope-check only the new orchestration harness")
    locked = source.index("Verify exact locked runtime identity")
    heavy = source.index("Create exact locked experiment environment")
    assert preflight < locked < heavy


def test_recovery_keeps_unsupported_10m_transfer_absent() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    assert '"ten_million_transfer": "ABSENT_NOT_SUPPORTED_BY_RECOVER171"' in source
    assert '"ten_million_status": "ABSENT_NO_RECOVER171_10M_RUN"' in source
