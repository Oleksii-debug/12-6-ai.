from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "run_d03_caselaw_source_admission_replay_v1.py"
spec = importlib.util.spec_from_file_location("caselaw_source_admission_real_probe", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true",
    reason="real immutable payload replay is intentionally limited to GitHub Actions",
)

EXPECTED_ADMITTED_RECORDS = 5_658
EXPECTED_ADMITTED_NORMALIZED_BYTES = 5_962_147
EXPECTED_ADMITTED_SHA256 = "45043f528ee83cb87a94f42ca9164c16463502308bf849a6736498d3b8caa273"


def _recover_product(root: Path) -> tuple[Path, Path]:
    tool = root / "materialize.py"
    config = root / "config.json"
    root.mkdir(parents=True, exist_ok=True)
    tool.write_bytes(
        mod.read_commit_file(
            mod.REPO_ROOT,
            mod.PRODUCT_SEMANTIC_COMMIT,
            mod.PRODUCT_MATERIALIZER_PATH,
        )
    )
    config.write_bytes(
        mod.read_commit_file(
            mod.REPO_ROOT,
            mod.PRODUCT_SEMANTIC_COMMIT,
            mod.PRODUCT_CONFIG_PATH,
        )
    )
    return tool, config


def _run_once(root: Path, product_tool: Path, product_config: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return mod.execute(
        product_tool_path=product_tool,
        product_config_path=product_config,
        admission_tool_path=mod.REPO_ROOT / mod.ADMISSION_TOOL,
        admission_policy_path=mod.REPO_ROOT / mod.ADMISSION_POLICY,
        source_gzips=[root / "cap_00044.jsonl.gz", root / "cap_00043.jsonl.gz"],
        candidate_path=root / "candidate.jsonl",
        admitted_candidate_path=root / "admitted.jsonl",
        materializer_report_path=root / "materializer-report.json",
        report_path=root / "source-admission-report.json",
        download=True,
        repo_root=mod.REPO_ROOT,
    )


def test_repaired_real_caselaw_replay_is_two_run_deterministic(tmp_path: Path) -> None:
    product_tool, product_config = _recover_product(tmp_path / "product")
    report_a = _run_once(tmp_path / "a", product_tool, product_config)
    report_b = _run_once(tmp_path / "b", product_tool, product_config)

    for name in (
        "cap_00044.jsonl.gz",
        "cap_00043.jsonl.gz",
        "candidate.jsonl",
        "admitted.jsonl",
        "materializer-report.json",
        "source-admission-report.json",
    ):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    assert report_a == report_b
    assert report_a["historical_product_replay_reproduced"] is True
    assert report_a["product_commit_path_membership_verified"] is True
    assert report_a["product_binding_observed_from_authenticated_config"] is True
    assert report_a["source_admission_commit_path_membership_verified"] is True
    assert report_a["input_records"] == mod.HISTORICAL_RETAINED_RECORDS
    assert report_a["input_normalized_utf8_bytes"] == mod.HISTORICAL_RETAINED_NORMALIZED_BYTES
    assert report_a["source_admitted_records"] == EXPECTED_ADMITTED_RECORDS
    assert (
        report_a["source_admitted_normalized_utf8_bytes"]
        == EXPECTED_ADMITTED_NORMALIZED_BYTES
    )
    assert report_a["source_admitted_candidate_sha256"] == EXPECTED_ADMITTED_SHA256
    assert report_a["candidate_denied_reason_counts"] == {}
    assert report_a["canonical_capacity_credited"] == 0
    assert report_a["training_authorized_bytes"] == 0
    assert report_a["unique_causal_loss_positions_authorized"] == 0
    assert report_a["tokenizer_fit_authorized"] is False
    assert report_a["model_training_executed"] is False
    assert report_a["learned_weights_created"] is False
    assert report_a["final_test_accessed"] is False
    assert report_a["paid_compute_used"] is False

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        summary = {
            "source_admitted_records": report_a["source_admitted_records"],
            "source_admitted_normalized_utf8_bytes": report_a[
                "source_admitted_normalized_utf8_bytes"
            ],
            "source_admitted_candidate_sha256": report_a[
                "source_admitted_candidate_sha256"
            ],
            "report_identity_sha256": report_a["report_identity_sha256"],
        }
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n### Repaired Caselaw real replay\n\n")
            handle.write(f"```json\n{json.dumps(summary, sort_keys=True)}\n```\n")
