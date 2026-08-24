from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from twelve_six.inference.acceptance import (
    SCHEMA_VERSION,
    run_s0_inference_acceptance,
    validate_s0_inference_acceptance,
)

ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def test_exact_candidate_trained_reload_cli_http_and_fail_closed(tmp_path: Path) -> None:
    output_dir = tmp_path / "acceptance"
    evidence = run_s0_inference_acceptance(
        ROOT,
        output_dir,
        candidate_sha=_head(),
        train_steps=8,
        seed=20260825,
    )

    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["candidate_sha"] == _head()
    assert evidence["checkpoint"]["step"] == 8
    assert evidence["identity"]["parameter_count"] == 10_140
    assert evidence["identity"]["max_context_tokens"] == 128
    assert evidence["greedy"]["direct_reload_equal"] is True
    assert evidence["seeded_sampling"]["same_seed_repeat_equal"] is True
    assert evidence["seeded_sampling"]["direct_reload_equal"] is True
    assert evidence["parity"]["passed"] is True
    assert evidence["parity"]["max_abs_error"] == 0.0
    assert evidence["parity"]["max_rel_error"] == 0.0
    assert all(evidence["stop_and_context"].values())
    assert evidence["cli"]["returncode"] == 0
    assert evidence["cli"]["ansi_escape_sequences"] is False
    assert evidence["cli"]["json_diagnostics"] is True
    assert evidence["server"]["health_status"] == "ok"
    assert evidence["server"]["matches_direct_generation"] is True
    assert evidence["server"]["chat_semantics"] is False
    assert all(evidence["fail_closed"].values())
    assert evidence["raw_base_semantics"] == {
        "chat_template": False,
        "hidden_system_prompt": False,
        "instruction_alignment": False,
        "refusal_layer": False,
    }
    assert evidence["paid_compute"] is False
    assert evidence["foreign_pretrained_weights"] is False
    assert evidence["audits_pass"] is False
    assert evidence["promotion_eligible"] is False
    assert evidence["windows_nvda"] == "NOT_TESTED_BLOCKED_BY_REPOSITORY_IDENTITY"

    written = json.loads(
        (output_dir / "inference-acceptance.json").read_text(encoding="utf-8")
    )
    assert written == evidence
    validate_s0_inference_acceptance(written)


def test_acceptance_manifest_rejects_tamper(tmp_path: Path) -> None:
    evidence = run_s0_inference_acceptance(
        ROOT,
        tmp_path / "acceptance",
        candidate_sha=_head(),
        train_steps=4,
        seed=17,
    )
    tampered = copy.deepcopy(evidence)
    tampered["server"]["matches_direct_generation"] = False
    with pytest.raises(ValueError, match="loopback server"):
        validate_s0_inference_acceptance(tampered)


def test_acceptance_rejects_stale_candidate_before_execution(tmp_path: Path) -> None:
    stale = "a" * 40
    if stale == _head():
        stale = "b" * 40
    with pytest.raises(ValueError, match="exact checkout HEAD"):
        run_s0_inference_acceptance(
            ROOT,
            tmp_path / "stale",
            candidate_sha=stale,
            train_steps=4,
        )
