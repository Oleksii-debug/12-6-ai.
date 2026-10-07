from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from twelve_six.data.balanced_split_application_v1 import BalancedSplitApplicationError

MODULE = Path(__file__).parents[1] / "tools" / "run_d03_balanced_split_application_v1.py"
SPEC = importlib.util.spec_from_file_location("run_d03_balanced_split_application_v1", MODULE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

IDS = {
    "inventory": hashlib.sha256(b"inventory").hexdigest(),
    "decontam": hashlib.sha256(b"decontam").hexdigest(),
    "dedup": hashlib.sha256(b"dedup").hexdigest(),
    "policy": hashlib.sha256(b"policy").hexdigest(),
    "balance": hashlib.sha256(b"balance").hexdigest(),
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _rehash(document: dict[str, Any], field: str) -> None:
    core = dict(document)
    core.pop(field, None)
    encoded = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document[field] = hashlib.sha256(encoded.encode()).hexdigest()


def _raw_records() -> list[dict[str, Any]]:
    specs = [
        ("r1", "s1", "family-uk-a", "uk", "text", "c1", "Альфа один"),
        ("r2", "s2", "family-uk-b", "uk", "text", "c1", "Альфа два"),
        ("r3", "s3", "family-en-a", "en", "text", "c2", "English three"),
        ("r4", "s4", "family-en-b", "en", "text", "c3", "English four"),
        ("r5", "s5", "family-code-a", "code", "code", "c4", "def f(): return 5"),
        ("r6", "s6", "family-code-b", "code", "code", "c4", "def g(): return 6"),
        ("r7", "s7", "family-uk-a", "uk", "text", "c5", "Український сім"),
        ("r8", "s8", "family-en-a", "en", "text", "c6", "English eight"),
    ]
    return [
        {
            "record_id": record_id,
            "source_id": source_id,
            "family": family,
            "stratum": stratum,
            "modality": modality,
            "near_duplicate_cluster_id": cluster,
            "normalized_payload": payload,
            "purpose": "pretraining_eligible",
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        for record_id, source_id, family, stratum, modality, cluster, payload in specs
    ]


def _selection(raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    family_bytes: dict[str, int] = {}
    stratum_bytes: dict[str, int] = {}
    source_bytes = 0
    for raw in raw_records:
        payload = raw["normalized_payload"].encode()
        payload_bytes = len(payload)
        row = {
            "record_id": raw["record_id"],
            "source_id": raw["source_id"],
            "family": raw["family"],
            "stratum": raw["stratum"],
            "modality": raw["modality"],
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_bytes": payload_bytes,
            "near_duplicate_cluster_id": raw["near_duplicate_cluster_id"],
            "purpose": raw["purpose"],
            "training_eligible": False,
            "evaluation_eligible": False,
            "evaluation_reserved": False,
        }
        rows.append(row)
        source_bytes += payload_bytes
        family_bytes[row["family"]] = family_bytes.get(row["family"], 0) + payload_bytes
        stratum_bytes[row["stratum"]] = stratum_bytes.get(row["stratum"], 0) + payload_bytes

    document: dict[str, Any] = {
        "schema": "12-6.d03-balanced-selection-authority.v1",
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": IDS["inventory"],
        "decontamination_authority_sha256": IDS["decontam"],
        "dedup_authority_sha256": IDS["dedup"],
        "balance_policy_identity_sha256": IDS["policy"],
        "balance_result_identity_sha256": IDS["balance"],
        "records": rows,
        "totals": {
            "record_count": len(rows),
            "source_bytes": source_bytes,
            "family_source_bytes": dict(sorted(family_bytes.items())),
            "stratum_source_bytes": dict(sorted(stratum_bytes.items())),
        },
        "claim_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
        },
    }
    _rehash(document, "balanced_selection_identity_sha256")
    return document


def _expected(selection: dict[str, Any]) -> dict[str, str]:
    return {
        "expected_selection_identity_sha256": selection["balanced_selection_identity_sha256"],
        "expected_retained_inventory_identity_sha256": IDS["inventory"],
        "expected_decontamination_authority_sha256": IDS["decontam"],
        "expected_dedup_authority_sha256": IDS["dedup"],
        "expected_balance_policy_identity_sha256": IDS["policy"],
        "expected_balance_result_identity_sha256": IDS["balance"],
    }


def _write_inputs(
    tmp_path: Path,
    *,
    raw: list[dict[str, Any]] | None = None,
    selection: dict[str, Any] | None = None,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, Any]]:
    actual_raw = raw if raw is not None else _raw_records()
    actual_selection = selection if selection is not None else _selection(actual_raw)
    selection_path = tmp_path / "selection.json"
    raw_path = tmp_path / "raw.jsonl"
    selection_path.write_bytes(_canonical_bytes(actual_selection))
    raw_path.write_bytes(b"".join(_canonical_bytes(row) for row in actual_raw))
    return selection_path, raw_path, actual_raw, actual_selection


def _execute(
    tmp_path: Path,
    *,
    output_name: str = "split.json",
    raw: list[dict[str, Any]] | None = None,
    selection: dict[str, Any] | None = None,
    expected_override: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Path]:
    selection_path, raw_path, _raw, actual_selection = _write_inputs(
        tmp_path, raw=raw, selection=selection
    )
    expected = _expected(actual_selection)
    if expected_override:
        expected.update(expected_override)
    output = tmp_path / output_name
    application = RUNNER.execute_balanced_split_application(
        selection_path=selection_path,
        raw_records_path=raw_path,
        output_path=output,
        **expected,
    )
    return application, output


def test_executes_canonical_split_deterministically_and_text_free(tmp_path: Path) -> None:
    application, output = _execute(tmp_path, output_name="first.json")

    raw = list(reversed(_raw_records()))
    selection = _selection(_raw_records())
    repeated, second = _execute(
        tmp_path,
        output_name="second.json",
        raw=raw,
        selection=selection,
    )

    assert repeated == application
    assert second.read_bytes() == output.read_bytes() == _canonical_bytes(application)
    encoded = output.read_text(encoding="utf-8")
    assert "normalized_payload" not in encoded
    assert "Альфа один" not in encoded
    assert "English three" not in encoded
    assert application["status"] == "PASS_ZERO_CREDIT"
    assert application["claim_boundary"]["authorized_optimized_target_exposure"] == 0
    assert application["claim_boundary"]["model_training_authorized"] is False


def test_operator_does_not_expose_split_science_knobs() -> None:
    help_text = RUNNER._parser().format_help()
    assert "--variant-seed" not in help_text
    assert "--validation-fraction" not in help_text
    assert "--split-git-blob" not in help_text


@pytest.mark.parametrize(
    "bad_kind", ["selection_duplicate", "raw_duplicate", "blank_line", "nonfinite"]
)
def test_strict_input_parser_fails_before_publication(tmp_path: Path, bad_kind: str) -> None:
    selection_path, raw_path, _raw, selection = _write_inputs(tmp_path)
    output = tmp_path / "split.json"

    if bad_kind == "selection_duplicate":
        selection_path.write_text('{"x":1,"x":2}', encoding="utf-8")
    elif bad_kind == "raw_duplicate":
        raw_path.write_text('{"record_id":"a","record_id":"b"}\n', encoding="utf-8")
    elif bad_kind == "blank_line":
        lines = raw_path.read_text(encoding="utf-8").splitlines()
        raw_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    elif bad_kind == "nonfinite":
        selection_path.write_text('{"x":NaN}', encoding="utf-8")

    with pytest.raises(RUNNER.BalancedSplitRunnerError):
        RUNNER.execute_balanced_split_application(
            selection_path=selection_path,
            raw_records_path=raw_path,
            output_path=output,
            **_expected(selection),
        )
    assert not output.exists()


def test_wrong_independent_authority_or_payload_drift_leaves_no_output(tmp_path: Path) -> None:
    selection_path, raw_path, raw, selection = _write_inputs(tmp_path)
    output = tmp_path / "split.json"
    expected = _expected(selection)
    expected["expected_selection_identity_sha256"] = hashlib.sha256(b"wrong").hexdigest()

    with pytest.raises(BalancedSplitApplicationError, match="expected identity"):
        RUNNER.execute_balanced_split_application(
            selection_path=selection_path,
            raw_records_path=raw_path,
            output_path=output,
            **expected,
        )
    assert not output.exists()

    raw[0]["normalized_payload"] += " drift"
    raw_path.write_bytes(b"".join(_canonical_bytes(row) for row in raw))
    with pytest.raises(BalancedSplitApplicationError, match="payload bytes/hash drift"):
        RUNNER.execute_balanced_split_application(
            selection_path=selection_path,
            raw_records_path=raw_path,
            output_path=output,
            **_expected(selection),
        )
    assert not output.exists()


def test_verifier_failure_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection_path, raw_path, _raw, selection = _write_inputs(tmp_path)
    output = tmp_path / "split.json"

    def reject(*_args: Any, **_kwargs: Any) -> None:
        raise BalancedSplitApplicationError("synthetic verifier rejection")

    monkeypatch.setattr(RUNNER, "verify_balanced_split_application", reject)
    with pytest.raises(BalancedSplitApplicationError, match="synthetic verifier rejection"):
        RUNNER.execute_balanced_split_application(
            selection_path=selection_path,
            raw_records_path=raw_path,
            output_path=output,
            **_expected(selection),
        )
    assert not output.exists()


def test_existing_output_is_refused_without_modification(tmp_path: Path) -> None:
    selection_path, raw_path, _raw, selection = _write_inputs(tmp_path)
    output = tmp_path / "split.json"
    output.write_bytes(b"sentinel\n")

    with pytest.raises(RUNNER.BalancedSplitRunnerError, match="output already exists"):
        RUNNER.execute_balanced_split_application(
            selection_path=selection_path,
            raw_records_path=raw_path,
            output_path=output,
            **_expected(selection),
        )
    assert output.read_bytes() == b"sentinel\n"


def test_cli_failure_is_explicit_and_does_not_publish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    selection_path, raw_path, _raw, selection = _write_inputs(tmp_path)
    output = tmp_path / "split.json"
    expected = _expected(selection)

    rc = RUNNER.main(
        [
            "--selection",
            str(selection_path),
            "--raw-records",
            str(raw_path),
            "--output",
            str(output),
            "--expected-selection-identity-sha256",
            "f" * 64,
            "--expected-retained-inventory-identity-sha256",
            expected["expected_retained_inventory_identity_sha256"],
            "--expected-decontamination-authority-sha256",
            expected["expected_decontamination_authority_sha256"],
            "--expected-dedup-authority-sha256",
            expected["expected_dedup_authority_sha256"],
            "--expected-balance-policy-identity-sha256",
            expected["expected_balance_policy_identity_sha256"],
            "--expected-balance-result-identity-sha256",
            expected["expected_balance_result_identity_sha256"],
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert captured.err.startswith("BLOCKED: ")
    assert not output.exists()
