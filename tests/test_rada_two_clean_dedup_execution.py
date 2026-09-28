from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_d03_rada_two_clean_global_dedup as runner
from twelve_six.data import rada_two_clean_dedup_execution as carrier


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _authority() -> dict[str, object]:
    return {
        "schema_version": carrier.AUTHORITY_SCHEMA,
        "local_free_only": True,
        "current_main_consumable_rada_intake": True,
        "rada_intake_pr": 1787,
        "rada_intake_head_sha": "1" * 40,
        "rada_intake_audit_ref": "#2062 PASS",
        "rada_adapter_module": "twelve_six.data.rada_laws_qp_dedup_adapter",
        "rada_adapter_git_blob_sha1": "2" * 40,
        "matcher_pr": 1459,
        "matcher_head_sha": "3" * 40,
        "matcher_audit_ref": "#2018 PASS",
        "matcher_different_worker_pass": True,
        "indexed_module": "twelve_six.data.incumbent_dedup_indexed_execution",
        "indexed_module_git_blob_sha1": "4" * 40,
        "v3_module": "authority_runtime.cross_source_capacity_audit_v3",
        "incumbent_base_authority_ref": "clean-base-authority:test",
        "combined_inventory_sha256": "6" * 64,
        "incumbent_base_payload_map_sha256": "7" * 64,
        "max_candidate_pairs": 5_000_000,
        "max_index_postings": 100_000_000,
        "max_pair_expansions": 100_000_000,
        "worker_timeout_seconds": 3600,
        "canonical_capacity_credit": 0,
        "authorized_optimized_target_exposure": 0,
        "training_executed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }


def _source(source_id: str, capacity: int) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": "ua.rada.open-data.laws-texts",
        "modality": "uk",
        "declared_capacity_bytes": capacity,
        "verified_raw_sha256": "a" * 64,
        "normalized_sha256": "b" * 64,
        "stable_origin_id_sha256": "c" * 64,
        "stable_object_id_sha256": "d" * 64,
    }


def _report() -> dict[str, object]:
    return {
        "report_sha256": "e" * 64,
        "sources": [
            _source("a", 10),
            _source("b", 20),
            _source("c", 20),
        ],
        "terminal_candidates": {
            "duplicate_clusters": [["b", "a"]],
            "declared_capacity_bytes_before": 50,
            "conservative_unique_capacity_bytes_after": 40,
            "duplicate_discount_bytes": 10,
        },
    }


def _receipt(run_id: str, *, report_sha: str = "e" * 64) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": carrier.RECEIPT_SCHEMA,
        "run_id": run_id,
        "local_free_only": True,
        "completed": True,
        "dependency_authority_raw_sha256": "1" * 64,
        "inventory_raw_sha256": "2" * 64,
        "base_payload_map_raw_sha256": "5" * 64,
        "source_object_count": 3,
        "source_payload_utf8_bytes": 50,
        "work_limits": {
            "max_candidate_pairs": 5_000_000,
            "max_index_postings": 100_000_000,
            "max_pair_expansions": 100_000_000,
        },
        "v3_report_sha256": report_sha,
        "survivor_authority_sha256": "3" * 64,
        "survivor_source_object_count": 2,
        "survivor_declared_capacity_bytes": 40,
        "match_wall_clock_seconds": 1.0,
        "total_wall_clock_seconds": 2.0,
        "source_objects_per_match_second": 3.0,
        "payload_bytes_per_match_second": 50.0,
        "process_max_rss_kib": 100,
        "resource_measurement_complete": True,
        "raw_text_emitted": False,
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
    return {
        **core,
        "receipt_identity_sha256": hashlib.sha256(_canonical(core)).hexdigest(),
    }


def test_dependency_authority_requires_external_raw_hash_and_terminal_gates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    identity = _write_json(path, value)

    observed = carrier.validate_dependency_authority(
        path,
        expected_raw_sha256=identity,
    )
    assert observed == value

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="raw SHA-256 drift"):
        carrier.validate_dependency_authority(path, expected_raw_sha256="f" * 64)

    value["matcher_different_worker_pass"] = False
    changed = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="different-worker PASS"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=changed)


def test_dependency_authority_rejects_boolean_integer_and_extra_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    value["rada_intake_pr"] = True
    identity = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="positive exact int"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)

    value = _authority()
    value["unexpected"] = "not allowed"
    identity = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="schema drift"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_dependency_authority_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y"}',
        encoding="utf-8",
    )
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="duplicate JSON key"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)




def test_global_payload_composition_requires_incumbent_base_and_exact_rada_segment(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.bin"
    base_path.write_bytes(b"base-payload")
    base_row = {
        "source_id": "base-1",
        "source_family": "incumbent.family",
    }
    rada_row = {
        "source_id": "rada-1",
        "source_family": "ua.rada.open-data.laws-texts",
    }
    inventory = {"sources": [base_row, rada_row]}
    payloads = carrier._load_base_payloads(
        inventory,
        (rada_row,),
        {"rada-1": b"rada-payload"},
        {"base-1": str(base_path)},
        rada_source_family="ua.rada.open-data.laws-texts",
    )

    assert payloads == {
        "base-1": b"base-payload",
        "rada-1": b"rada-payload",
    }

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="lacks incumbent base"):
        carrier._load_base_payloads(
            {"sources": [rada_row]},
            (rada_row,),
            {"rada-1": b"rada-payload"},
            {},
            rada_source_family="ua.rada.open-data.laws-texts",
        )


def test_global_payload_composition_rejects_base_coverage_and_rada_projection_drift(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.bin"
    base_path.write_bytes(b"base-payload")
    base_row = {
        "source_id": "base-1",
        "source_family": "incumbent.family",
    }
    rada_row = {
        "source_id": "rada-1",
        "source_family": "ua.rada.open-data.laws-texts",
    }

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="base payload map coverage"):
        carrier._load_base_payloads(
            {"sources": [base_row, rada_row]},
            (rada_row,),
            {"rada-1": b"rada-payload"},
            {},
            rada_source_family="ua.rada.open-data.laws-texts",
        )

    substituted = dict(rada_row)
    substituted["source_id"] = "rada-substituted"
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="Rada segment"):
        carrier._load_base_payloads(
            {"sources": [base_row, substituted]},
            (rada_row,),
            {"rada-1": b"rada-payload"},
            {"base-1": str(base_path), "rada-substituted": str(base_path)},
            rada_source_family="ua.rada.open-data.laws-texts",
        )


def test_global_payload_composition_rejects_duplicate_combined_source_ids(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.bin"
    base_path.write_bytes(b"base-payload")
    base_row = {
        "source_id": "base-1",
        "source_family": "incumbent.family",
    }
    rada_row = {
        "source_id": "rada-1",
        "source_family": "ua.rada.open-data.laws-texts",
    }
    duplicate_base = dict(base_row)

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="duplicate source_id"):
        carrier._load_base_payloads(
            {"sources": [base_row, duplicate_base, rada_row]},
            (rada_row,),
            {"rada-1": b"rada-payload"},
            {"base-1": str(base_path)},
            rada_source_family="ua.rada.open-data.laws-texts",
        )


def test_survivor_authority_reuses_incumbent_capacity_representative_rule() -> None:
    authority = carrier.derive_survivor_authority(_report())

    assert authority["selection_rule"] == carrier.SELECTION_RULE
    assert authority["post_dedup_survivor_source_object_count"] == 2
    assert authority["post_dedup_declared_capacity_bytes"] == 40
    assert [row["source_id"] for row in authority["survivors"]] == ["b", "c"]
    assert authority["duplicate_clusters"] == [
        {
            "member_source_ids": ["a", "b"],
            "selected_source_id": "b",
            "selected_declared_capacity_bytes": 20,
        }
    ]
    assert authority["raw_text_emitted"] is False
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert "text" not in json.dumps(authority, sort_keys=True)


def test_survivor_authority_rejects_overlapping_clusters() -> None:
    report = _report()
    terminal = report["terminal_candidates"]
    assert isinstance(terminal, dict)
    terminal["duplicate_clusters"] = [["a", "b"], ["b", "c"]]

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="overlap"):
        carrier.derive_survivor_authority(report)


def test_survivor_authority_rejects_capacity_summary_drift() -> None:
    report = _report()
    terminal = report["terminal_candidates"]
    assert isinstance(terminal, dict)
    terminal["conservative_unique_capacity_bytes_after"] = 41

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="capacity"):
        carrier.derive_survivor_authority(report)


def test_two_clean_authority_requires_two_distinct_reproducible_receipts() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")

    authority = carrier.build_two_clean_authority(first, second)
    assert authority["run_receipt_identities"] == [
        first["receipt_identity_sha256"],
        second["receipt_identity_sha256"],
    ]
    assert authority["v3_report_sha256"] == "e" * 64
    assert authority["survivor_source_object_count"] == 2
    assert authority["canonical_capacity_credited"] == 0
    assert authority["authorized_optimized_target_exposure"] == 0
    assert authority["training_executed"] is False

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="run ids"):
        carrier.build_two_clean_authority(first, _receipt("clean-a"))


def test_two_clean_authority_rejects_report_drift_and_receipt_tampering() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b", report_sha="f" * 64)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="v3_report_sha256"):
        carrier.build_two_clean_authority(first, second)

    tampered = _receipt("clean-b")
    tampered["source_object_count"] = 4
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="self-hash"):
        carrier.build_two_clean_authority(first, tampered)


def test_two_clean_authority_requires_terminal_resource_measurement() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")
    second["resource_measurement_complete"] = False
    second["process_max_rss_kib"] = None
    unsigned = dict(second)
    unsigned.pop("receipt_identity_sha256")
    second["receipt_identity_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="max-RSS"):
        carrier.build_two_clean_authority(first, second)


def test_dependency_authority_binds_execution_work_limits(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    value["max_candidate_pairs"] = True
    identity = _write_json(path, value)

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="max_candidate_pairs"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_dependency_authority_binds_combined_inventory_and_base_map_roots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    value["combined_inventory_sha256"] = "not-a-hash"
    identity = _write_json(path, value)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="combined_inventory_sha256"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)

    value = _authority()
    value["incumbent_base_payload_map_sha256"] = "not-a-hash"
    identity = _write_json(path, value)
    with pytest.raises(
        carrier.RadaTwoCleanExecutionError,
        match="incumbent_base_payload_map_sha256",
    ):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_two_clean_authority_rejects_extra_receipt_fields_even_if_rehashed() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")
    second["text"] = "must never be accepted"
    unsigned = dict(second)
    unsigned.pop("receipt_identity_sha256")
    second["receipt_identity_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="receipt schema drift"):
        carrier.build_two_clean_authority(first, second)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("tokenizer_fit_authorized", True),
        ("learned_weights_created", True),
        ("final_test_outcomes_read", True),
        ("paid_compute_used", True),
    ],
)
def test_two_clean_authority_rejects_truth_widening_even_if_rehashed(
    field: str,
    bad_value: object,
) -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")
    second[field] = bad_value
    unsigned = dict(second)
    unsigned.pop("receipt_identity_sha256")
    second["receipt_identity_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="truth drift"):
        carrier.build_two_clean_authority(first, second)


def test_two_clean_authority_rejects_non_finite_telemetry() -> None:
    second = _receipt("clean-b")
    second["match_wall_clock_seconds"] = float("inf")
    unsigned = dict(second)
    unsigned.pop("receipt_identity_sha256")
    with pytest.raises(ValueError):
        carrier._canonical_bytes(unsigned)


def test_two_clean_authority_rejects_impossible_survivor_totals() -> None:
    first = _receipt("clean-a")
    second = _receipt("clean-b")
    second["survivor_source_object_count"] = 4
    unsigned = dict(second)
    unsigned.pop("receipt_identity_sha256")
    second["receipt_identity_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="exceeds input"):
        carrier.build_two_clean_authority(first, second)


def _physical_artifacts(
    run_id: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    report = _report()
    report.pop("report_sha256")
    report.update(
        {
            "schema_version": "synthetic-v3",
            "local_free_only": True,
            "model_training_executed": False,
            "source_count": 3,
            "raw_text_emitted": False,
        }
    )
    report["report_sha256"] = hashlib.sha256(
        runner._report_identity_bytes(report)
    ).hexdigest()
    survivor = carrier.derive_survivor_authority(report)
    receipt = _receipt(run_id, report_sha=str(report["report_sha256"]))
    receipt["survivor_authority_sha256"] = survivor["survivor_authority_sha256"]
    unsigned = dict(receipt)
    unsigned.pop("receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return report, survivor, receipt


def test_dependency_authority_requires_positive_exact_worker_timeout(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    value = _authority()
    value["worker_timeout_seconds"] = True
    identity = _write_json(path, value)

    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="worker_timeout_seconds"):
        carrier.validate_dependency_authority(path, expected_raw_sha256=identity)


def test_max_rss_uses_windows_backend_when_resource_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(carrier, "resource", None)
    monkeypatch.setattr(carrier, "_windows_peak_working_set_kib", lambda: 321)

    assert carrier._max_rss_kib() == 321


def test_windows_peak_working_set_normalizes_bytes_to_kib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKernel32:
        def GetCurrentProcess(self) -> int:
            return 123

        def K32GetProcessMemoryInfo(
            self,
            handle: object,
            counters: object,
            size: object,
        ) -> int:
            del handle, size
            counters._obj.PeakWorkingSetSize = 4097
            return 1

    monkeypatch.setattr(
        carrier.ctypes,
        "windll",
        SimpleNamespace(kernel32=FakeKernel32()),
        raising=False,
    )

    assert carrier._windows_peak_working_set_kib() == 5


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e400}',
    ],
)
def test_parent_readback_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
    raw: bytes,
) -> None:
    path = tmp_path / "generated.json"
    path.write_bytes(raw)

    with pytest.raises(carrier.RadaTwoCleanExecutionError):
        runner._read_json(path)


def test_parent_crossbinds_report_and_survivor_artifacts_to_receipt() -> None:
    report, survivor, receipt = _physical_artifacts("clean-a")

    assert runner._verify_report_artifact(report, receipt) == report["report_sha256"]
    assert (
        runner._verify_survivor_artifact(survivor, receipt)
        == survivor["survivor_authority_sha256"]
    )

    resealed_survivor = dict(survivor)
    resealed_survivor["duplicate_cluster_count"] = 99
    survivor_core = dict(resealed_survivor)
    survivor_core.pop("survivor_authority_sha256")
    resealed_survivor["survivor_authority_sha256"] = hashlib.sha256(
        _canonical(survivor_core)
    ).hexdigest()
    with pytest.raises(
        carrier.RadaTwoCleanExecutionError,
        match="identity differs from receipt",
    ):
        runner._verify_survivor_artifact(resealed_survivor, receipt)

    resealed_report = dict(report)
    resealed_report["algorithm"] = "post-child-resealed"
    report_core = dict(resealed_report)
    report_core.pop("report_sha256")
    resealed_report["report_sha256"] = hashlib.sha256(_canonical(report_core)).hexdigest()
    with pytest.raises(
        carrier.RadaTwoCleanExecutionError,
        match="identity differs from receipt",
    ):
        runner._verify_report_artifact(resealed_report, receipt)


def _runner_args(tmp_path: Path) -> argparse.Namespace:
    authority_path = tmp_path / "dependency-authority.json"
    authority = _authority()
    authority_sha = _write_json(authority_path, authority)
    return argparse.Namespace(
        dependency_authority=authority_path,
        expected_dependency_authority_sha256=authority_sha,
        inventory=tmp_path / "inventory.json",
        expected_inventory_sha256="6" * 64,
        base_payload_map=tmp_path / "base-payload-map.json",
        expected_base_payload_map_sha256="7" * 64,
        candidate_jsonl=tmp_path / "candidate.jsonl",
        quality_report=tmp_path / "quality.json",
        execution_evidence=tmp_path / "execution.json",
        output_root=tmp_path / "two-clean",
    )


def test_parent_worker_timeout_publishes_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runner_args(tmp_path)
    observed: list[int] = []

    def timeout(command: list[str], *, timeout_seconds: int) -> None:
        del command
        observed.append(timeout_seconds)
        raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout_seconds)

    monkeypatch.setattr(runner, "_run_worker", timeout)
    with pytest.raises(
        carrier.RadaTwoCleanExecutionError,
        match="authority deadline",
    ):
        runner._run_two_clean(args)

    incomplete, _ = runner._read_json(args.output_root / "incomplete.json")
    assert observed == [3600]
    assert incomplete["reason"] == "worker_timeout"
    assert incomplete["completed_run_ids"] == []
    assert not (args.output_root / "two-clean-authority.json").exists()


def test_parent_second_worker_timeout_preserves_completed_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runner_args(tmp_path)
    calls = 0

    def second_timeout(command: list[str], *, timeout_seconds: int) -> None:
        nonlocal calls
        del command
        calls += 1
        if calls == 2:
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout_seconds)

    monkeypatch.setattr(runner, "_run_worker", second_timeout)
    with pytest.raises(carrier.RadaTwoCleanExecutionError, match="authority deadline"):
        runner._run_two_clean(args)

    incomplete, _ = runner._read_json(args.output_root / "incomplete.json")
    assert incomplete["completed_run_ids"] == ["clean-a"]
    assert incomplete["reason"] == "worker_timeout"


def test_parent_operator_interrupt_publishes_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _runner_args(tmp_path)

    def interrupt(command: list[str], *, timeout_seconds: int) -> None:
        del command, timeout_seconds
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run_worker", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner._run_two_clean(args)

    incomplete, _ = runner._read_json(args.output_root / "incomplete.json")
    assert incomplete["reason"] == "operator_interrupt"
    assert incomplete["completed_run_ids"] == []
    assert not (args.output_root / "two-clean-authority.json").exists()


@pytest.mark.parametrize("failure", ["timeout", "interrupt"])
def test_worker_helper_kills_and_reaps_on_abort(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    events: list[object] = []

    class FakeProcess:
        returncode = None

        def wait(self, timeout: int | None = None) -> int:
            events.append(("wait", timeout))
            if len([event for event in events if isinstance(event, tuple)]) == 1:
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)
                raise KeyboardInterrupt
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            events.append("kill")

    monkeypatch.setattr(runner.subprocess, "Popen", lambda command: FakeProcess())
    exception = subprocess.TimeoutExpired if failure == "timeout" else KeyboardInterrupt
    with pytest.raises(exception):
        runner._run_worker(["worker"], timeout_seconds=17)

    assert events == [("wait", 17), "kill", ("wait", None)]
