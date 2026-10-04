#!/usr/bin/env python3
"""Fail-closed validator and deterministic evidence builder for optional MLflow tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

SCHEMA = "12-6.mlflow-local-tracking-qualification.v1"
EXPECTED_BASE = "5020afd671a3885c1b738c8b4eafe7525f630546"
EXPECTED_UPSTREAM = "0572b16ac9e9c98a02df9df40ad3e48ce3b7c588"
EXPECTED_LICENSE_BLOB = "db7cb10b5e330d56b40370bc178974ccabe71458"
EXPECTED_TRACKING_BLOB = "1a672b170a49b800d420127de63cfff7b394065c"
EXPECTED_REGISTRY_BLOB = "d80a60357c56eacac135f948b8a72556bb849e5a"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SECRET_KEY = re.compile(r"(?i)(token|secret|password|passwd|api[_-]?key|credential|authorization)")
SECRET_VALUE = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|authorization)\s*[:=]"
)


class ContractError(ValueError):
    """Raised when a qualification or evidence payload violates the contract."""


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs_no_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractError("top-level JSON value must be an object")
    return data


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256_json(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_contract(contract: dict[str, object]) -> None:
    _require(contract.get("schema_version") == 1, "schema_version must be 1")
    _require(contract.get("qualification_id") == SCHEMA, "qualification_id drift")
    _require(contract.get("status") == "CANDIDATE_LOCAL_ONLY_NOT_ADOPTED", "unsafe status")

    project = contract.get("project_authority")
    _require(isinstance(project, dict), "project_authority missing")
    _require(project.get("repository") == "Oleksii-debug/12-6-ai.", "repository drift")
    _require(project.get("base_git_sha") == EXPECTED_BASE, "base Git SHA drift")
    _require(project.get("parent_issue") == 720, "parent issue drift")
    _require(project.get("worker_issue") == 740, "worker issue drift")
    _require(project.get("registry_blob_sha") == EXPECTED_REGISTRY_BLOB, "registry blob drift")
    _require(project.get("registry_component_id") == "MLFLOW", "registry component drift")
    _require(
        project.get("registry_decision") == "P1_LOCAL_TRACKING_CANDIDATE",
        "registry decision drift",
    )

    upstream = contract.get("upstream")
    _require(isinstance(upstream, dict), "upstream missing")
    _require(upstream.get("repository") == "mlflow/mlflow", "upstream repository drift")
    _require(upstream.get("default_branch") == "master", "upstream branch drift")
    _require(upstream.get("git_sha") == EXPECTED_UPSTREAM, "upstream Git SHA drift")
    _require(upstream.get("license_spdx") == "Apache-2.0", "license must be Apache-2.0")
    _require(upstream.get("license_blob_sha") == EXPECTED_LICENSE_BLOB, "license blob drift")
    _require(
        upstream.get("tracking_source_blob_sha") == EXPECTED_TRACKING_BLOB,
        "tracking source blob drift",
    )
    semantics = upstream.get("observed_tracking_semantics")
    _require(isinstance(semantics, dict), "tracking semantics missing")
    for key in (
        "supports_explicit_local_file_uri",
        "supports_http_remote_uri",
        "project_must_set_tracking_uri_explicitly",
        "project_must_not_rely_on_upstream_default",
    ):
        _require(semantics.get(key) is True, f"upstream semantics gate failed: {key}")

    authority = contract.get("authority_policy")
    _require(isinstance(authority, dict), "authority_policy missing")
    _require(authority.get("execution_profile") == "LOCAL_FREE", "non-local execution profile")
    _require(authority.get("mlflow_role") == "OPTIONAL_METADATA_SINK_ONLY", "MLflow role widened")
    _require(
        authority.get("canonical_lineage_authority") == "GIT_AND_PROJECT_MANIFESTS",
        "canonical lineage authority weakened",
    )
    for key in (
        "mlflow_may_authorize_training",
        "mlflow_may_authorize_paid_compute",
        "mlflow_may_select_or_reselect_from_final_test",
        "mlflow_may_replace_project_checkpoint_manifest",
        "mlflow_may_replace_git_identity",
    ):
        _require(authority.get(key) is False, f"forbidden MLflow authority enabled: {key}")

    tracking = contract.get("tracking_policy")
    _require(isinstance(tracking, dict), "tracking_policy missing")
    _require(tracking.get("explicit_tracking_uri_required") is True, "explicit URI required")
    _require(tracking.get("allowed_uri_schemes") == ["file", "sqlite"], "URI scheme drift")
    _require(tracking.get("remote_tracking_forbidden") is True, "remote tracking must be forbidden")
    _require(tracking.get("network_tracking_forbidden") is True, "network tracking must be forbidden")
    _require(
        tracking.get("credentials_in_tracking_uri_forbidden") is True,
        "credential-bearing URI must be forbidden",
    )
    _require(
        tracking.get("artifact_payload_logging_policy") == "HASH_REFERENCES_ONLY",
        "artifact payload policy widened",
    )
    _require(tracking.get("secret_bearing_metadata_forbidden") is True, "secret policy weakened")
    _require(tracking.get("export_required_for_project_evidence") is True, "export must be required")
    _require(tracking.get("deterministic_export_required") is True, "determinism must be required")

    _require(
        contract.get("required_run_bindings")
        == [
            "source_git_sha",
            "run_manifest_sha256",
            "checkpoint_manifest_sha256",
            "checkpoint_id",
        ],
        "required run bindings drift",
    )
    promotion = contract.get("promotion")
    _require(isinstance(promotion, dict), "promotion missing")
    _require(promotion.get("current_state") == "CANDIDATE", "promotion state widened")
    _require(promotion.get("adopted") is False, "contract cannot self-adopt")
    _require(promotion.get("adoption_allowed_by_this_contract") is False, "self-adoption enabled")
    for key in (
        "runtime_import_tested",
        "dependency_lock_complete",
        "project_parity_proven",
        "measured_benefit_proven",
    ):
        _require(promotion.get(key) is False, f"unsupported promotion evidence claimed: {key}")

    truth = contract.get("truth_boundary")
    _require(isinstance(truth, dict), "truth_boundary missing")
    _require(truth.get("model_training_executed") is False, "training overclaim")
    _require(truth.get("optimizer_updates") == 0, "optimizer update overclaim")
    for key in (
        "checkpoint_mutated",
        "evaluation_payload_read",
        "final_test_payload_read",
        "gpu_provisioned",
        "paid_compute_authorized",
        "foreign_base_weights_used",
        "mlflow_runtime_installed_or_executed",
    ):
        _require(truth.get(key) is False, f"truth boundary overclaim: {key}")


def validate_tracking_uri(uri: object) -> str:
    _require(isinstance(uri, str) and uri, "tracking_uri must be an explicit non-empty string")
    parsed = urlparse(uri)
    _require(parsed.scheme in {"file", "sqlite"}, "only local file/sqlite tracking URIs are allowed")
    _require(not parsed.netloc, "tracking URI must not contain a remote host or credentials")
    _require(parsed.username is None and parsed.password is None, "tracking URI credentials forbidden")
    _require(not parsed.query and not parsed.fragment, "tracking URI query/fragment forbidden")
    _require(bool(parsed.path), "tracking URI must identify local storage")
    _require(not SECRET_VALUE.search(uri), "secret-like value in tracking URI")
    return uri


def _validate_scalar_map(name: str, value: object, *, numeric: bool = False) -> dict[str, object]:
    _require(isinstance(value, dict), f"{name} must be an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        _require(isinstance(key, str) and key, f"{name} key must be a non-empty string")
        _require(not SECRET_KEY.search(key), f"secret-like {name} key forbidden: {key}")
        if numeric:
            _require(
                isinstance(item, (int, float)) and not isinstance(item, bool),
                f"{name}.{key} must be numeric",
            )
        else:
            _require(
                isinstance(item, (str, int, float, bool)) or item is None,
                f"{name}.{key} invalid scalar",
            )
        if isinstance(item, str):
            _require(not SECRET_VALUE.search(item), f"secret-like {name} value forbidden")
        out[key] = item
    return out


def validate_run_input(payload: dict[str, object]) -> None:
    for key in (
        "source_git_sha",
        "run_manifest_sha256",
        "checkpoint_manifest_sha256",
        "checkpoint_id",
    ):
        value = payload.get(key)
        pattern = HEX40 if key == "source_git_sha" else HEX64
        _require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"invalid {key}")
    validate_tracking_uri(payload.get("tracking_uri"))
    _validate_scalar_map("params", payload.get("params", {}))
    _validate_scalar_map("metrics", payload.get("metrics", {}), numeric=True)
    _validate_scalar_map("tags", payload.get("tags", {}))

    artifacts = payload.get("artifact_references", [])
    _require(isinstance(artifacts, list), "artifact_references must be a list")
    for item in artifacts:
        _require(isinstance(item, dict), "artifact reference must be an object")
        _require(
            set(item) == {"logical_name", "sha256", "byte_size"},
            "artifact reference keys drift",
        )
        _require(
            isinstance(item["logical_name"], str) and item["logical_name"],
            "bad logical_name",
        )
        _require(not SECRET_KEY.search(item["logical_name"]), "secret-like artifact name forbidden")
        _require(
            isinstance(item["sha256"], str) and HEX64.fullmatch(item["sha256"]),
            "bad artifact sha256",
        )
        _require(
            isinstance(item["byte_size"], int) and not isinstance(item["byte_size"], bool),
            "bad byte_size",
        )
        _require(item["byte_size"] >= 0, "negative byte_size")

    _require(
        payload.get("canonical_lineage_authority") == "GIT_AND_PROJECT_MANIFESTS",
        "lineage authority drift",
    )
    _require(payload.get("mlflow_role") == "OPTIONAL_METADATA_SINK_ONLY", "MLflow role drift")


def build_evidence(contract: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    validate_contract(contract)
    validate_run_input(payload)
    normalized = json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    evidence: dict[str, object] = {
        "schema": "12-6.mlflow-local-tracking-evidence.v1",
        "qualification_id": SCHEMA,
        "qualification_sha256": _sha256_json(contract),
        "status": "LOCAL_TRACKING_EVIDENCE_ONLY_NO_AUTHORITY_PROMOTION",
        "upstream_git_sha": EXPECTED_UPSTREAM,
        "project_base_git_sha": EXPECTED_BASE,
        "run": normalized,
        "authority": {
            "canonical_lineage": "GIT_AND_PROJECT_MANIFESTS",
            "canonical_checkpoint": "D05_PROJECT_CHECKPOINT_MANIFEST",
            "mlflow": "OPTIONAL_METADATA_SINK_ONLY",
            "training_authorized_by_evidence": False,
            "paid_compute_authorized_by_evidence": False,
        },
    }
    evidence["evidence_sha256"] = _sha256_json(evidence)
    return evidence


def validate_evidence(contract: dict[str, object], evidence: dict[str, object]) -> None:
    validate_contract(contract)
    _require(
        evidence.get("schema") == "12-6.mlflow-local-tracking-evidence.v1",
        "evidence schema drift",
    )
    _require(evidence.get("qualification_id") == SCHEMA, "evidence qualification drift")
    _require(
        evidence.get("qualification_sha256") == _sha256_json(contract),
        "qualification hash drift",
    )
    run = evidence.get("run")
    _require(isinstance(run, dict), "evidence run missing")
    validate_run_input(run)
    claimed = evidence.get("evidence_sha256")
    _require(isinstance(claimed, str) and HEX64.fullmatch(claimed), "evidence hash missing")
    without_hash = dict(evidence)
    without_hash.pop("evidence_sha256", None)
    _require(claimed == _sha256_json(without_hash), "evidence self-hash mismatch")
    authority = evidence.get("authority")
    _require(isinstance(authority, dict), "evidence authority missing")
    _require(
        authority.get("training_authorized_by_evidence") is False,
        "evidence self-authorized training",
    )
    _require(
        authority.get("paid_compute_authorized_by_evidence") is False,
        "evidence self-authorized compute",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-contract")
    build = sub.add_parser("build-evidence")
    build.add_argument("run_input", type=Path)
    build.add_argument("output", type=Path)
    check = sub.add_parser("validate-evidence")
    check.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)

    try:
        contract = load_json(args.contract)
        if args.command == "validate-contract":
            validate_contract(contract)
        elif args.command == "build-evidence":
            run_input = load_json(args.run_input)
            evidence = build_evidence(contract, run_input)
            _write_json(args.output, evidence)
            validate_evidence(contract, evidence)
        else:
            evidence = load_json(args.evidence)
            validate_evidence(contract, evidence)
    except ContractError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
