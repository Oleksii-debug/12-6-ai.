"""Strict JSON evidence parser and CLI errors for the NEXT100-106 balance gate."""

from __future__ import annotations

import importlib.util
import json
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
