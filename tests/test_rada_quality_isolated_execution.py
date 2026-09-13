from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_d03_rada_trees_quality_windows_isolated.py"
SPEC = importlib.util.spec_from_file_location("rada_isolated_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

EXPECTED_LAUNCHER_BLOB = "2e225e44746c60ea7c383cef323b1f17c332d774"


def _blob(payload: bytes) -> str:
    return runner._git_blob_sha_bytes(payload)


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

    (staged / "src" / "victim.py").write_bytes(payload + b"# one byte family tamper\n")
    with pytest.raises(runner.IsolatedAuthorityError, match="Git blob drift"):
        runner._verify_staged_tree(staged, expected)


def test_fresh_isolated_child_ignores_parent_poison_hostile_pythonpath_and_checkout_mutation(
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
        b"victim.VALUE + '|' + str('sitecustomize' in sys.modules), encoding='utf-8')\n"
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

    # Mutate the live source after authentication/staging. The child must still
    # execute the already-authenticated bytes.
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
    assert output.read_text(encoding="utf-8") == "authenticated|False"


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
