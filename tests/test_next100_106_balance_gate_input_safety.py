"""Strict JSON evidence parser and CLI errors for the NEXT100-106 balance gate."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/next100_106_balance_gate.py"
POLICY = ROOT / "configs/data/next100_106_balance_gate_policy_v1.json"


def _gate():
    spec = importlib.util.spec_from_file_location("strict_balance_gate", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        pytest.param(b'{"terminal":false,"terminal":true}', "duplicate", id="root-duplicate"),
        pytest.param(b'{"obj":{"x":1,"x":2}}', "duplicate", id="nested-duplicate"),
        pytest.param(b'{"obj":NaN}', "non-finite", id="nan"),
        pytest.param(b'{"obj":Infinity}', "non-finite", id="infinity"),
        pytest.param(b'{"obj":1e9999}', "not finite", id="overflow"),
        pytest.param(b'{"obj":1e-9999}', "underflow", id="positive-underflow"),
        pytest.param(b'{"obj":-1e-9999}', "underflow", id="negative-underflow"),
        pytest.param(b'{"obj":' + b"9" * 65 + b"}", "digit limit", id="huge-int"),
        pytest.param(
            b'{"a":' + b"[" * 65 + b"0" + b"]" * 65 + b"}",
            "structure limit",
            id="overdepth",
        ),
        pytest.param(
            b'{"a":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}",
            "nesting limit",
            id="parser-recursion",
        ),
        pytest.param(
            b'{"a":[' + b",".join([b"0"] * 10010) + b"]}",
            "structure limit",
            id="too-many-nodes",
        ),
        pytest.param(br'{"\ud800":"invalid"}', "surrogates not allowed", id="surrogate-key"),
        pytest.param(br'{"a":"\ud800"}', "surrogates not allowed", id="surrogate-value"),
        pytest.param(b'{"a":"\xff"}', "decode", id="invalid-utf8"),
        pytest.param(b" " * (1_048_576 + 1), "byte limit", id="oversize"),
        pytest.param(b"[]", "JSON object", id="nonobject-root"),
    ],
)
def test_rejects_untrustworthy_json(
    tmp_path: Path, raw: bytes, message: str,
) -> None:
    source = tmp_path / "вхідні дані з пробілами.json"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match=message):
        _gate().load_json(source)


@pytest.mark.parametrize("token", ["0e-9999", "-0.000e-9999", "1.25", "0.0"])
def test_preserves_valid_finite_numbers(tmp_path: Path, token: str) -> None:
    source = tmp_path / "vector.json"
    source.write_text('{"value":' + token + "}", encoding="utf-8")
    value = _gate().load_json(source)
    assert value["value"] == float(token)


def test_accepts_exact_byte_limit(tmp_path: Path) -> None:
    gate = _gate()
    source = tmp_path / "vector.json"
    raw = b'{"pad":"' + b"x" * (gate.MAX_BALANCE_JSON_BYTES - 10) + b'"}'
    assert len(raw) == gate.MAX_BALANCE_JSON_BYTES
    source.write_bytes(raw)
    assert len(gate.load_json(source)["pad"]) == gate.MAX_BALANCE_JSON_BYTES - 10


def test_preserves_canonical_policy_identity() -> None:
    gate = _gate()
    policy = gate.load_json(POLICY)
    gate.validate_policy(policy)
    assert policy["policy_identity_sha256"] == gate.canonical_sha(
        policy, "policy_identity_sha256"
    )


def test_cli_rejects_ambiguous_terminal_input_without_output(tmp_path: Path) -> None:
    source = tmp_path / "неоднозначний вхід.json"
    source.write_bytes(b'{"terminal":false,"terminal":true}')
    output = tmp_path / "balance-result.json"
    run = subprocess.run(
        [
            sys.executable, str(TOOL), "evaluate", str(source), "--output", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert run.stderr == ""
    assert json.loads(run.stdout)["status"] == "BLOCKED_INVALID_INPUT"
    assert not output.exists()


def test_cli_keeps_valid_policy_and_zero_training_credit() -> None:
    run = subprocess.run(
        [sys.executable, str(TOOL), "validate-policy"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0
    assert run.stdout.startswith("NEXT100-106 policy PASS")
    assert run.stderr == ""

    families = [
        {"family_id": f"{stratum}.{i}", "stratum": stratum, "unique_bytes": cap}
        for stratum, cap in (("ua", 4_500_000), ("en", 3_500_000), ("code", 2_000_000))
        for i in range(2)
    ]
    counts = {"ua": 2, "en": 2, "code": 2}
    raw = {
        "schema_version": _gate().INPUT_SCHEMA,
        "terminal": True,
        "dedup_authority": {
            "worker_id": "synthetic-test-only",
            "head_sha": "a" * 40,
            "evidence_identity_sha256": "b" * 64,
            "terminal_verdict": "PASS",
        },
        "families": families,
        "totals": {
            "total_unique_bytes": 20_000_000,
            "by_stratum": {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000},
            "family_count": counts,
        },
    }
    assert sum(f["unique_bytes"] for f in families) == 20_000_000
    result = _gate().evaluate(_gate().load_json(POLICY), raw)
    assert result["maximum_feasible_total_source_bytes"] == 20_000_000
    assert result["claim_boundary"]["model_training_authorized"] is False
    assert result["claim_boundary"]["tokenizer_fit_authorized"] is False
    assert result["claim_boundary"]["authorized_training_exposure_loss_positions"] == 0


def test_accepts_exact_structure_limits(tmp_path: Path) -> None:
    gate = _gate()
    deep = tmp_path / "depth.json"
    deep.write_bytes(b'{"a":' + b"[" * 64 + b"0" + b"]" * 64 + b"}")
    # Root and the nested leaf count toward the structure budget.
    with pytest.raises(ValueError, match="structure limit"):
        gate.load_json(deep)

    wide = tmp_path / "nodes.json"
    wide.write_bytes(b'{"a":[' + b",".join([b"0"] * 9998) + b"]}")
    assert len(gate.load_json(wide)["a"]) == 9998


def test_real_current_clean_vector_reproduces_committed_balance(
    tmp_path: Path,
) -> None:
    """Use committed physical-evidence metadata, not a synthetic 20M vector."""
    gate = _gate()
    report = gate.load_json(
        ROOT / "reports/d03/current_clean_balance_local_candidate_execution_v1.json"
    )
    vector = report["outputs"]["next100-input.json"]["document"]
    vector_path = tmp_path / "current clean vector.json"
    vector_path.write_text(
        json.dumps(vector, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = gate.evaluate(gate.load_json(POLICY), gate.load_json(vector_path))
    assert result == report["outputs"]["balance-result.json"]["document"]
    assert result["maximum_feasible_total_source_bytes"] == 7300
    assert result["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    assert result["claim_boundary"]["model_training_authorized"] is False
    assert result["claim_boundary"]["tokenizer_fit_authorized"] is False
    assert report["canonical_capacity_granted"] == 0

    output = tmp_path / "real physical metadata result.json"
    run = subprocess.run(
        [
            sys.executable, str(TOOL), "evaluate", str(vector_path),
            "--output", str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0
    assert run.stderr == ""
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert output.read_bytes().endswith(b"\n")
    assert b"\r\n" not in output.read_bytes()


def _isolated_balance_cli(tmp_path: Path) -> tuple[Path, Path]:
    """Use a disposable policy copy so alias regressions cannot damage the repo."""
    root = tmp_path / "isolated"
    tool = root / "tools" / TOOL.name
    policy = root / "configs" / "data" / POLICY.name
    tool.parent.mkdir(parents=True)
    policy.parent.mkdir(parents=True)
    shutil.copyfile(TOOL, tool)
    shutil.copyfile(POLICY, policy)
    return tool, policy


@pytest.mark.parametrize(
    "target", ["policy", "input", "existing", "symlink"],
)
def test_cli_refuses_destructive_output_aliases(
    tmp_path: Path, target: str,
) -> None:
    gate = _gate()
    tool, policy = _isolated_balance_cli(tmp_path)
    report = gate.load_json(
        ROOT / "reports/d03/current_clean_balance_local_candidate_execution_v1.json"
    )
    vector = tmp_path / "input.json"
    vector.write_text(
        json.dumps(
            report["outputs"]["next100-input.json"]["document"],
            ensure_ascii=False, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    existing = tmp_path / "prior-result.json"
    existing.write_bytes(b"previous verified result")
    symlink = tmp_path / "linked-source.json"
    if target == "symlink":
        try:
            symlink.symlink_to(vector)
        except OSError:
            pytest.skip("file symlinks unavailable on this platform")
    destinations = {
        "policy": policy, "input": vector,
        "existing": existing, "symlink": symlink,
    }
    before = {
        "policy": policy.read_bytes(),
        "input": vector.read_bytes(),
        "existing": existing.read_bytes(),
    }
    run = subprocess.run(
        [
            sys.executable, str(tool), "evaluate", str(vector),
            "--output", str(destinations[target]),
        ],
        cwd=tool.parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert run.stderr == ""
    assert json.loads(run.stdout)["status"] == "BLOCKED_INVALID_INPUT"
    assert policy.read_bytes() == before["policy"]
    assert vector.read_bytes() == before["input"]
    assert existing.read_bytes() == before["existing"]
    if target == "symlink":
        assert symlink.is_symlink()


def test_cli_accepts_new_output_and_unchanged_stdout(tmp_path: Path) -> None:
    gate = _gate()
    tool, _ = _isolated_balance_cli(tmp_path)
    report = gate.load_json(
        ROOT / "reports/d03/current_clean_balance_local_candidate_execution_v1.json"
    )
    vector_value = report["outputs"]["next100-input.json"]["document"]
    vector = tmp_path / "current-clean-input.json"
    vector.write_text(
        json.dumps(vector_value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = gate.evaluate(gate.load_json(POLICY), vector_value)
    output = tmp_path / "new-output.json"
    saved = subprocess.run(
        [
            sys.executable, str(tool), "evaluate", str(vector),
            "--output", str(output),
        ],
        cwd=tool.parent.parent, capture_output=True, text=True, check=False,
    )
    assert saved.returncode == 0
    assert saved.stdout == ""
    assert saved.stderr == ""
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert output.read_bytes().endswith(b"\n")
    stdout = subprocess.run(
        [sys.executable, str(tool), "evaluate", str(vector)],
        cwd=tool.parent.parent, capture_output=True, text=True, check=False,
    )
    assert stdout.returncode == 0
    assert stdout.stderr == ""
    assert json.loads(stdout.stdout) == result
    assert result["claim_boundary"]["model_training_authorized"] is False


@pytest.mark.parametrize(
    ("field", "stratum"),
    [
        ("by_stratum", "ua"),
        ("by_stratum", "code"),
        ("family_count", "en"),
        ("family_count", "ua"),
    ],
)
def test_real_current_clean_declared_numeric_aliases_fail_closed(
    field: str, stratum: str,
) -> None:
    gate = _gate()
    report = gate.load_json(
        ROOT / "reports/d03/current_clean_balance_local_candidate_execution_v1.json"
    )
    vector = report["outputs"]["next100-input.json"]["document"]
    count = vector["totals"][field][stratum]
    assert isinstance(count, int) and not isinstance(count, bool)
    vector["totals"][field][stratum] = float(count)
    with pytest.raises(gate.GateError, match=f"declared {field}"):
        gate.evaluate(gate.load_json(POLICY), vector)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("policy", "target_total_source_bytes"), 20_000_000.0, "20M"),
        (("policy", "minimum_independent_families_per_stratum"), 2.0, "minimum family"),
        (("policy", "budget_quantum_bytes"), 100.0, "budget quantum"),
        (("policy", "strata", "ua", "target_numerator"), 9.0, "ua mixture numerator"),
        (("policy", "strata", "code", "target_denominator"), 5.0, "code mixture denominator"),
        (("policy", "max_family_fraction_total", "numerator"), 1.0, "global family cap"),
        (("policy", "max_family_fraction_total", "denominator"), 4.0, "global family cap"),
        (("policy", "max_family_fraction_own_stratum", "numerator"), 3.0, "within-stratum"),
        (("claim_boundary", "authorizes_model_training"), 0, "claim boundary"),
        (("claim_boundary", "authorizes_tokenizer_fit"), 0, "claim boundary"),
        (("claim_boundary", "computes_source_mixture_feasibility_only"), 1, "claim boundary"),
    ],
)
def test_self_resealed_policy_rejects_numeric_and_boolean_aliases(
    path: tuple[str, ...], value: object, message: str,
) -> None:
    gate = _gate()
    policy = copy.deepcopy(gate.load_json(POLICY))
    target = policy
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    # An attacker controlling the external policy bytes can re-sign its
    # self-hash; matching a recomputed digest is not a type guarantee.
    policy["policy_identity_sha256"] = gate.canonical_sha(
        policy, "policy_identity_sha256"
    )
    with pytest.raises(gate.GateError, match=message):
        gate.validate_policy(policy)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("policy", "strata", "ua", "target_denominator"), "45/35/20"),
        (("policy", "max_family_fraction_total", "numerator"), "global family cap"),
        (("claim_boundary", "authorizes_paid_compute"), "claim boundary"),
    ],
)
def test_self_resealed_policy_missing_required_fields_fail_closed(
    path: tuple[str, ...], message: str,
) -> None:
    gate = _gate()
    policy = copy.deepcopy(gate.load_json(POLICY))
    target = policy
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    policy["policy_identity_sha256"] = gate.canonical_sha(
        policy, "policy_identity_sha256"
    )
    with pytest.raises(gate.GateError, match=message):
        gate.validate_policy(policy)


def test_self_resealed_policy_rejects_extra_stratum() -> None:
    gate = _gate()
    policy = copy.deepcopy(gate.load_json(POLICY))
    policy["policy"]["strata"]["extra"] = {
        "target_numerator": 0, "target_denominator": 20,
    }
    policy["policy_identity_sha256"] = gate.canonical_sha(
        policy, "policy_identity_sha256"
    )
    with pytest.raises(gate.GateError, match="45/35/20"):
        gate.validate_policy(policy)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("issue",), 558),
        (("lineage", "base_head_sha"), "0" * 40),
        (("input_contract", "terminal_required"), False),
        (("input_contract", "family_capacity_semantics"), "RAW_SOURCE_BYTES"),
        (("new_training_authority",), True),
    ],
)
def test_resigned_policy_cannot_change_unreviewed_identity_fields(
    path: tuple[str, ...], value: object,
) -> None:
    gate = _gate()
    policy = copy.deepcopy(gate.load_json(POLICY))
    target = policy
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    policy["policy_identity_sha256"] = gate.canonical_sha(
        policy, "policy_identity_sha256"
    )
    with pytest.raises(gate.GateError, match="pinned authority"):
        gate.validate_policy(policy)


def test_canonical_policy_is_externally_pinned() -> None:
    gate = _gate()
    policy = gate.load_json(POLICY)
    assert policy["policy_identity_sha256"] == gate.EXPECTED_POLICY_IDENTITY_SHA256
    gate.validate_policy(policy)

def test_stage_short_write_is_detected() -> None:
    class ShortWriter:
        def write(self, payload: bytes) -> int:
            return len(payload) - 1

        def flush(self) -> None:
            raise AssertionError("must not flush a short write")

        def fileno(self) -> int:
            raise AssertionError("must not fsync a short write")

    with pytest.raises(OSError, match="incomplete staged balance output"):
        _gate()._write_staged_bytes(ShortWriter(), b"expected bytes")


@pytest.mark.parametrize("fault", ["partial", "fsync"])
def test_stage_failure_never_leaves_a_final_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    gate = _gate()
    source = tmp_path / "original vector.json"
    source.write_bytes(b"original vector")
    result = tmp_path / "balance-result.json"
    if fault == "partial":
        def partial(destination: object, payload: bytes) -> None:
            destination.write(payload[:3])
            raise OSError("injected disk full")
        monkeypatch.setattr(gate, "_write_staged_bytes", partial)
    else:
        def bad_fsync(_fd: int) -> None:
            raise OSError("injected fsync failure")
        monkeypatch.setattr(gate.os, "fsync", bad_fsync)
    with pytest.raises(OSError, match="injected"):
        gate._write_new_output(result, b"complete payload", input_path=source)
    assert not result.exists()
    assert not list(tmp_path.glob(f".{result.name}.*.tmp"))
    assert source.read_bytes() == b"original vector"


def test_atomic_create_preserves_late_competing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original")
    output = tmp_path / "result.json"
    real_link = gate.os.link

    def raced_link(stage: Path, final: Path) -> None:
        final.write_bytes(b"competitor verified result")
        real_link(stage, final)

    monkeypatch.setattr(gate.os, "link", raced_link)
    with pytest.raises(gate.GateError, match="refusing to overwrite"):
        gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"competitor verified result"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert source.read_bytes() == b"original"


def test_stage_mutation_after_sync_rolls_back_own_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original")
    output = tmp_path / "result.json"
    real_link = gate.os.link

    def tamper_link(stage: Path, final: Path) -> None:
        stage.write_bytes(b"wrong output")
        real_link(stage, final)

    monkeypatch.setattr(gate.os, "link", tamper_link)
    with pytest.raises(gate.GateError, match="failed byte/path verification"):
        gate._write_new_output(output, b"expected payload", input_path=source)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert source.read_bytes() == b"original"


def test_valid_output_is_one_shot_and_preserves_input(tmp_path: Path) -> None:
    gate = _gate()
    source = tmp_path / "дані з пробілами.json"
    source.write_bytes(b"original")
    output = tmp_path / "результат.json"
    payload = '{"text":"Україна"}\n'.encode("utf-8")
    gate._write_new_output(output, payload, input_path=source)
    assert output.read_bytes() == payload
    with pytest.raises(gate.GateError, match="refusing to overwrite"):
        gate._write_new_output(output, b"different bytes", input_path=source)
    assert output.read_bytes() == payload
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), object(), "\ud800"])
def test_invalid_adapter_result_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    bad: object,
) -> None:
    from types import SimpleNamespace

    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"source")
    output = tmp_path / "blocked.json"
    monkeypatch.setattr(
        gate, "parse_args",
        lambda: SimpleNamespace(command="evaluate", input=source, output=output),
    )
    monkeypatch.setattr(gate, "load_json", lambda _path: {})
    monkeypatch.setattr(gate, "validate_policy", lambda _policy: None)
    monkeypatch.setattr(gate, "evaluate", lambda _policy, _vector: {"bad": bad})
    assert gate.main() == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED_INVALID_INPUT"
    assert not output.exists()
    assert source.read_bytes() == b"source"


def test_valid_final_with_failed_temp_cleanup_reports_committed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gate = _gate()
    source = tmp_path / "original.json"
    output = tmp_path / "результат із пробілами.json"
    source.write_bytes(b"original authority")
    original_unlink = Path.unlink

    def stage_locked(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(f".{output.name}.") and self.suffix == ".tmp":
            raise PermissionError("injected stage sharing failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", stage_locked)
    payload = '{"text":"Україна"}\n'.encode("utf-8")
    gate._write_new_output(output, payload, input_path=source)
    emitted = capsys.readouterr()
    assert emitted.err.startswith("OUTPUT_COMMITTED_CLEANUP_PENDING: ")
    warning = json.loads(emitted.err.split(": ", 1)[1])
    assert warning["output"] == str(output)
    assert output.read_bytes() == payload
    assert len(list(tmp_path.glob(f".{output.name}.*.tmp"))) == 1
    assert source.read_bytes() == b"original authority"
    with pytest.raises(gate.GateError, match="refusing to overwrite"):
        gate._write_new_output(output, b"different", input_path=source)
    assert output.read_bytes() == payload


def test_failed_rollback_reports_invalid_published_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    output = tmp_path / "result.json"
    source.write_bytes(b"original authority")
    original_link = gate.os.link
    original_unlink = Path.unlink

    def tamper_then_link(stage: Path, final: Path) -> None:
        stage.write_bytes(b"invalid modified result")
        original_link(stage, final)

    def prevent_final_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == output:
            raise PermissionError("injected rollback sharing failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(gate.os, "link", tamper_then_link)
    monkeypatch.setattr(Path, "unlink", prevent_final_unlink)
    with pytest.raises(gate.GateError, match="ROLLBACK_INCOMPLETE"):
        gate._write_new_output(output, b"expected", input_path=source)
    assert output.read_bytes() == b"invalid modified result"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert source.read_bytes() == b"original authority"


def test_unpublished_stage_cleanup_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    output = tmp_path / "result.json"
    source.write_bytes(b"original authority")
    original_unlink = Path.unlink

    def stage_locked(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(f".{output.name}.") and self.suffix == ".tmp":
            raise PermissionError("injected stage sharing failure")
        original_unlink(self, *args, **kwargs)

    def unavailable_link(_stage: Path, _final: Path) -> None:
        raise OSError("injected unsupported link")

    monkeypatch.setattr(Path, "unlink", stage_locked)
    monkeypatch.setattr(gate.os, "link", unavailable_link)
    with pytest.raises(gate.GateError, match="STAGING_CLEANUP_INCOMPLETE"):
        gate._write_new_output(output, b"candidate", input_path=source)
    assert not output.exists()
    assert len(list(tmp_path.glob(f".{output.name}.*.tmp"))) == 1
    assert source.read_bytes() == b"original authority"


def test_failed_stage_fsync_and_cleanup_preserve_original_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "результат.json"
    real_unlink = Path.unlink

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected ENOSPC during staged fsync")

    def lock_stage(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(f".{output.name}.") and self.suffix == ".tmp":
            raise PermissionError("injected staging sharing violation")
        real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "fsync", fail_fsync)
        fault.setattr(Path, "unlink", lock_stage)
        with pytest.raises(gate.GateError, match="STAGING_CLEANUP_INCOMPLETE") as caught:
            gate._write_new_output(output, b"candidate", input_path=source)

    stages = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(stages) == 1
    assert str(stages[0]) in str(caught.value)
    assert "ENOSPC" in str(caught.value)
    assert "staging sharing violation" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert not output.exists()
    assert source.read_bytes() == b"original authority"
    stages[0].unlink()
    gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"candidate"


def test_failed_rollback_and_cleanup_report_both_orphans_and_primary_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "result.json"
    real_link = gate.os.link
    real_unlink = Path.unlink

    def tamper_then_link(stage: Path, final: Path) -> None:
        stage.write_bytes(b"tampered output")
        real_link(stage, final)

    def lock_final_and_stage(self: Path, *args: object, **kwargs: object) -> None:
        if self == output or (
            self.name.startswith(f".{output.name}.") and self.suffix == ".tmp"
        ):
            raise PermissionError("injected sharing violation")
        real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "link", tamper_then_link)
        fault.setattr(Path, "unlink", lock_final_and_stage)
        with pytest.raises(gate.GateError, match="ROLLBACK_INCOMPLETE") as caught:
            gate._write_new_output(output, b"expected", input_path=source)

    stages = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(stages) == 1
    assert str(output) in str(caught.value)
    assert str(stages[0]) in str(caught.value)
    assert "rollback failure" in str(caught.value)
    assert "staged cleanup also failed" in str(caught.value)
    assert "failed byte/path verification" in str(caught.value)
    assert isinstance(caught.value.__cause__, gate.GateError)
    assert stages[0].read_bytes() == output.read_bytes() == b"tampered output"
    assert source.read_bytes() == b"original authority"
    stages[0].unlink()
    output.unlink()
    gate._write_new_output(output, b"expected", input_path=source)
    assert output.read_bytes() == b"expected"


def test_link_created_then_error_rolls_back_only_owned_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error after link creation must not publish unverified output."""
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "результат із пробілами.json"
    real_link = gate.os.link
    published = False

    def link_then_raise(stage: Path, final: Path) -> None:
        nonlocal published
        real_link(stage, final)
        published = True
        raise OSError("injected postlink response loss")

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "link", link_then_raise)
        with pytest.raises(OSError, match="postlink response loss"):
            gate._write_new_output(output, b"candidate", input_path=source)

    assert published is True
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert source.read_bytes() == b"original authority"
    gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"candidate"


def test_link_created_then_error_and_rollback_failure_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery reports the original link fault and a stranded own-inode final."""
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "result.json"
    real_link = gate.os.link
    real_unlink = Path.unlink

    def link_then_raise(stage: Path, final: Path) -> None:
        real_link(stage, final)
        raise OSError("injected postlink transport failure")

    def lock_final(self: Path, *args: object, **kwargs: object) -> None:
        if self == output:
            raise PermissionError("injected locked final")
        real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "link", link_then_raise)
        fault.setattr(Path, "unlink", lock_final)
        with pytest.raises(gate.GateError, match="ROLLBACK_INCOMPLETE") as caught:
            gate._write_new_output(output, b"candidate", input_path=source)

    assert str(output) in str(caught.value)
    assert "postlink transport failure" in str(caught.value)
    assert "locked final" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert "postlink transport failure" in str(caught.value.__cause__)
    assert output.read_bytes() == b"candidate"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert source.read_bytes() == b"original authority"
    output.unlink()
    gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"candidate"


def test_ambiguous_postlink_stat_failure_preserves_stage_for_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently remove stage when final's identity is inaccessible."""
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "uncertain output.json"
    real_link = gate.os.link
    real_stat = Path.stat
    linked = False

    def link_then_error(stage: Path, final: Path) -> None:
        nonlocal linked
        real_link(stage, final)
        linked = True
        raise OSError("injected lost link response")

    def inaccessible_final(self: Path, *args: object, **kwargs: object):
        if self == output and linked:
            raise PermissionError("injected final stat denial")
        return real_stat(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "link", link_then_error)
        fault.setattr(Path, "stat", inaccessible_final)
        with pytest.raises(gate.GateError, match="PUBLICATION_INDETERMINATE") as caught:
            gate._write_new_output(output, b"candidate", input_path=source)

    stages = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert linked is True
    assert output.read_bytes() == b"candidate"
    assert len(stages) == 1
    assert stages[0].read_bytes() == b"candidate"
    assert str(output) in str(caught.value)
    assert str(stages[0]) in str(caught.value)
    assert "injected final stat denial" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert "lost link response" in str(caught.value.__cause__)
    assert source.read_bytes() == b"original authority"
    output.unlink()
    stages[0].unlink()
    gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"candidate"


def test_rollback_stat_failure_preserves_stage_and_original_validation_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A linked but uninspectable final is an explicit manual recovery case."""
    gate = _gate()
    source = tmp_path / "source.json"
    source.write_bytes(b"original authority")
    output = tmp_path / "result.json"
    real_link = gate.os.link
    real_stat = Path.stat
    linked = False

    def mark_link(stage: Path, final: Path) -> None:
        nonlocal linked
        real_link(stage, final)
        linked = True

    def inaccessible_final(self: Path, *args: object, **kwargs: object):
        if self == output and linked:
            raise PermissionError("injected rollback stat denial")
        return real_stat(self, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(gate.os, "link", mark_link)
        fault.setattr(Path, "stat", inaccessible_final)
        with pytest.raises(gate.GateError, match="ROLLBACK_INCOMPLETE") as caught:
            gate._write_new_output(output, b"candidate", input_path=source)

    stages = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert linked is True
    assert len(stages) == 1
    assert output.read_bytes() == stages[0].read_bytes() == b"candidate"
    assert str(output) in str(caught.value)
    assert str(stages[0]) in str(caught.value)
    assert "rollback stat denial" in str(caught.value)
    assert isinstance(caught.value.__cause__, gate.GateError)
    assert "byte/path verification" in str(caught.value.__cause__)
    assert source.read_bytes() == b"original authority"
    output.unlink()
    stages[0].unlink()
    gate._write_new_output(output, b"candidate", input_path=source)
    assert output.read_bytes() == b"candidate"
