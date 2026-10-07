from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_d03_rada_trees_quality_windows_isolated.py"
EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evidence"
    / "d03-rada-trees"
    / "quality-window-authoritative-real-replay-v4.json"
)
SPEC = importlib.util.spec_from_file_location("rada_isolated_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

EXPECTED_LAUNCHER_BLOB = "5f6688cc45a0fea5fe2c2690252a2a2db756af57"
EXPECTED_EXECUTION_HEAD = "d3f619fa5b7e4d0805ee9a76f2f3aa6a6ddd4c49"
EXPECTED_EXECUTION_RUN = 34733315129
EXPECTED_SUMMARY_SHA256 = "c22c18d354cd8e7a55acd57919148d4a3e058e916c230c4229379294d85a7d6a"


def _blob(payload: bytes) -> str:
    return runner._git_blob_sha_bytes(payload)


def test_terminal_execution_evidence_is_self_hashed_and_zero_credit() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    claimed = evidence.pop("summary_sha256")
    canonical = (
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()

    assert hashlib.sha256(canonical).hexdigest() == claimed == EXPECTED_SUMMARY_SHA256
    assert evidence["workflow_run_id"] == EXPECTED_EXECUTION_RUN
    assert evidence["pr_head_sha"] == EXPECTED_EXECUTION_HEAD
    assert evidence["classification"] == "CURRENT_AUTHORITY_NO_SITE_ISOLATED_REAL_REPLAY_ZERO_CREDIT"
    assert evidence["two_fresh_isolated_outputs_byte_identical"] is True
    assert evidence["two_fresh_isolated_reports_byte_identical"] is True
    assert evidence["two_fresh_isolated_closure_receipts_byte_identical"] is True
    assert evidence["training_authorized_bytes"] == 0
    assert evidence["unique_causal_loss_positions_authorized"] == 0
    assert evidence["authorized_optimized_target_exposure"] == 0
    assert evidence["tokenizer_fit_authorized"] is False
    assert evidence["optimizer_updates"] == 0
    assert evidence["model_training_executed"] is False
    assert evidence["learned_weights_created"] is False
    assert evidence["final_test_outcomes_read"] is False
    assert evidence["paid_compute_used"] is False
    assert evidence["foreign_pretrained_weights"] is False
    assert evidence["external_llm_or_api_data_or_intelligence"] is False


def test_live_behavior_closure_and_launcher_are_exact() -> None:
    repo = RUNNER_PATH.parents[1]
    assert _blob(RUNNER_PATH.read_bytes()) == EXPECTED_LAUNCHER_BLOB
    payloads = runner._collect_authenticated_sources(
        repo, runner.AUTHENTICATED_SOURCE_CLOSURE
    )
    assert set(payloads) == set(runner.AUTHENTICATED_SOURCE_CLOSURE)
    for relative, payload in payloads.items():
        assert _blob(payload) == runner.AUTHENTICATED_SOURCE_CLOSURE[relative]


def test_staged_tree_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"VALUE = 'bound'\n"
    path = source / "src" / "victim.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    expected = {"src/victim.py": _blob(payload)}
    collected = runner._collect_authenticated_sources(source, expected)

    staged = tmp_path / "staged"
    runner._stage_authenticated_tree(staged, collected)
    runner._verify_staged_tree(staged, expected)

    (staged / "src" / "victim.py").write_bytes(payload + b"# tamper\n")
    with pytest.raises(runner.IsolatedAuthorityError, match="Git blob drift"):
        runner._verify_staged_tree(staged, expected)


def test_fresh_no_site_child_ignores_parent_poison_hostile_pythonpath_and_checkout_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    victim = source / "src" / "victim.py"
    probe = source / "tools" / "probe.py"
    victim.parent.mkdir(parents=True)
    probe.parent.mkdir(parents=True)
    victim_payload = b"VALUE = 'authenticated'\n"
    probe_payload = (
        b"import sys\n"
        b"from pathlib import Path\n"
        b"import victim\n"
        b"Path(sys.argv[1]).write_text("
        b"victim.VALUE + '|' + str('site' in sys.modules) + '|' + str('sitecustomize' in sys.modules), "
        b"encoding='utf-8')\n"
    )
    victim.write_bytes(victim_payload)
    probe.write_bytes(probe_payload)
    expected = {
        "src/victim.py": _blob(victim_payload),
        "tools/probe.py": _blob(probe_payload),
    }
    collected = runner._collect_authenticated_sources(source, expected)

    staged = tmp_path / "staged"
    runner._stage_authenticated_tree(staged, collected)
    runner._verify_staged_tree(staged, expected)

    victim.write_text("VALUE = 'mutated-live-checkout'\n", encoding="utf-8")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    (hostile / "victim.py").write_text("VALUE = 'hostile-pythonpath'\n", encoding="utf-8")
    (hostile / "sitecustomize.py").write_text(
        "raise RuntimeError('hostile sitecustomize executed')\n", encoding="utf-8"
    )

    poisoned = types.ModuleType("victim")
    poisoned.VALUE = "parent-sys-modules-poison"
    previous = sys.modules.get("victim")
    sys.modules["victim"] = poisoned
    try:
        output = tmp_path / "child-result.txt"
        result = runner._run_isolated_entrypoint(
            staged,
            "tools/probe.py",
            [str(output)],
            cwd=tmp_path / "child-cwd",
            extra_env={"PYTHONPATH": str(hostile), "PYTHONHOME": str(hostile)},
        )
    finally:
        if previous is None:
            sys.modules.pop("victim", None)
        else:
            sys.modules["victim"] = previous

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert output.read_text(encoding="utf-8") == "authenticated|False|False"


def test_isolated_environment_strips_python_import_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/hostile")
    monkeypatch.setenv("PYTHONHOME", "/hostile")
    monkeypatch.setenv("PYTHONSTARTUP", "/hostile/start.py")
    monkeypatch.setenv("PYTHONUSERBASE", "/hostile/user")
    env = runner._isolated_environment()
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "PYTHONSTARTUP" not in env
    assert "PYTHONUSERBASE" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
