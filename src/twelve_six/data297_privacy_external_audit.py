"""DATA-297 external-real audit built on the incumbent DATA-33 privacy authority.

This module measures DATA-33 unchanged. It never adds detectors, never persists
matched values or source text, and uses synthetic runtime-only challenge fixtures.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .data.privacy_filter import (
    assert_no_secret_values_in_manifest,
    detect_privacy_findings,
    privacy_policy_manifest,
    scan_record,
)

WORKER_ID = "DATA-297-PRIVACY-EXTERNAL-AUDIT"
SCHEMA_VERSION = "12-6.data297-privacy-external-audit.v1"
DEFAULT_CONFIG = Path("configs/data/data297_privacy_external_audit_v1.json")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class Data297AuditError(ValueError):
    """Raised when authority, source identity, or evidence contracts drift."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Data297AuditError(f"expected JSON object: {path}")
    return value


def load_config(path: Path = DEFAULT_CONFIG) -> Mapping[str, Any]:
    config = _load_json(path)
    if config.get("worker_id") != WORKER_ID:
        raise Data297AuditError("wrong DATA-297 worker identity")
    if config.get("local_free_only") is not True:
        raise Data297AuditError("DATA-297 is LOCAL_FREE only")
    expected = config["expected_training_inventory"]
    inventory = list(config["admitted_inventory"])
    if len(inventory) != int(expected["source_count"]):
        raise Data297AuditError("admitted source-count drift")
    if len({item["source_family"] for item in inventory}) != int(expected["family_count"]):
        raise Data297AuditError("admitted family-count drift")
    if sum(int(item["expected_input_utf8_bytes"]) for item in inventory) != int(
        expected["input_utf8_bytes"]
    ):
        raise Data297AuditError("admitted byte-capacity drift")
    for item in inventory:
        if not _SHA256_RE.fullmatch(str(item["expected_input_sha256"])):
            raise Data297AuditError("every admitted source must bind an exact lowercase SHA-256")
    return config


def assert_incumbent_privacy_authority(config: Mapping[str, Any]) -> str:
    retained = config["privacy_authority"]
    runtime = privacy_policy_manifest()
    if runtime["schema_version"] != retained["policy_schema"]:
        raise Data297AuditError("DATA-33 privacy schema drift")
    if runtime["policy_sha256"] != retained["policy_sha256"]:
        raise Data297AuditError("DATA-33 privacy policy identity drift")
    return str(runtime["policy_sha256"])


@dataclass(frozen=True)
class LabeledFixture:
    fixture_id: str
    category: str
    modality: str
    expected_sensitive: bool
    build: Callable[[], str]


def _synthetic_secret(label: str, length: int = 28) -> str:
    digest = hashlib.sha256(("DATA297:" + label).encode("ascii")).hexdigest()
    core = (digest * 4)[: max(8, length - 2)]
    return "A" + core + "!"


def _fixture_specs() -> tuple[LabeledFixture, ...]:
    return (
        LabeledFixture(
            "positive-pii-email", "pii", "natural", True,
            lambda: "Contact " + "fixture.owner" + "@" + "example.invalid" + " for review.",
        ),
        LabeledFixture(
            "positive-pii-phone", "pii", "natural", True,
            lambda: "Synthetic contact +1 202 555 0100.",
        ),
        LabeledFixture(
            "positive-private-key-marker", "code-secret", "code", True,
            lambda: "-----BEGIN " + "PRIVATE KEY-----\nFIXTURE\n-----END PRIVATE KEY-----",
        ),
        LabeledFixture(
            "positive-github-token-shape", "api-key", "code", True,
            lambda: 'token = "' + "gh" + "p_" + "A" * 36 + '"',
        ),
        LabeledFixture(
            "positive-aws-access-id-shape", "api-key", "code", True,
            lambda: 'key = "' + "AK" + "IA" + "A" * 16 + '"',
        ),
        LabeledFixture(
            "positive-dotenv-api-key-assignment", "code-secret", "code", True,
            lambda: "api_" + "key=" + _synthetic_secret("dotenv-api-key", 30),
        ),
        LabeledFixture(
            "positive-bearer-jwt-shape", "code-secret", "code", True,
            lambda: "Authorization: Bearer " + "eyJ" + "A" * 12 + "." + "B" * 12 + "." + "C" * 12,
        ),
        LabeledFixture(
            "positive-private-unix-ssh-path", "private-path", "code", True,
            lambda: "/home/" + "fixture-user" + "/.ssh/" + "id_rsa",
        ),
        LabeledFixture(
            "positive-private-windows-cloud-path", "private-path", "code", True,
            lambda: "C:\\Users\\" + "fixture-user" + "\\.aws\\credentials",
        ),
        LabeledFixture(
            "positive-basic-auth-header", "code-secret", "code", True,
            lambda: "Authorization: Basic " + base64.b64encode(
                ("fixture-user:" + _synthetic_secret("basic-auth", 22)).encode("ascii")
            ).decode("ascii"),
        ),
        LabeledFixture(
            "positive-credential-bearing-database-uri", "code-secret", "code", True,
            lambda: "postgresql://fixture-user:" + _synthetic_secret("database-uri", 22)
            + "@db.example.invalid/app",
        ),
        LabeledFixture(
            "positive-npm-auth-token-shape", "api-key", "code", True,
            lambda: "//registry.npmjs.org/:_auth" + "Token=" + _synthetic_secret("npm", 34),
        ),
        LabeledFixture(
            "positive-gitlab-token-shape", "api-key", "code", True,
            lambda: "gl" + "pat-" + _synthetic_secret("gitlab", 30).replace("!", "A"),
        ),
        LabeledFixture(
            "positive-huggingface-token-shape", "api-key", "code", True,
            lambda: "h" + "f_" + _synthetic_secret("hf", 34).replace("!", "A"),
        ),
        LabeledFixture(
            "positive-openai-style-key-shape", "api-key", "code", True,
            lambda: "s" + "k-" + _synthetic_secret("generic-sk", 40).replace("!", "A"),
        ),
        LabeledFixture(
            "negative-env-placeholder", "benign-code", "code", False,
            lambda: "api_" + "key=${API_KEY}",
        ),
        LabeledFixture(
            "negative-password-placeholder", "benign-code", "code", False,
            lambda: "pass" + "word=change" + "me",
        ),
        LabeledFixture(
            "negative-token-example-placeholder", "benign-code", "code", False,
            lambda: "token=example" + "123",
        ),
        LabeledFixture(
            "negative-checksum-literal", "benign-code", "code", False,
            lambda: 'checksum = "' + hashlib.sha256(b"public fixture").hexdigest() + '"',
        ),
        LabeledFixture(
            "negative-generic-documentation-path", "benign-path", "code", False,
            lambda: "/home/user/project/README.md",
        ),
        LabeledFixture(
            "negative-iso-date", "benign-natural", "natural", False,
            lambda: "Release date 2026-08-26.",
        ),
        LabeledFixture(
            "negative-public-url", "benign-natural", "natural", False,
            lambda: "See https://example.invalid/docs for the public example.",
        ),
    )


def audit_labeled_fixtures() -> Mapping[str, Any]:
    confusion = Counter(
        {"true_positive": 0, "false_negative": 0, "true_negative": 0, "false_positive": 0}
    )
    by_category: dict[str, Counter[str]] = defaultdict(
        lambda: Counter(
            {"true_positive": 0, "false_negative": 0, "true_negative": 0, "false_positive": 0}
        )
    )
    false_negative_ids: list[str] = []
    false_positive_ids: list[str] = []
    detector_counts: Counter[str] = Counter()
    payload_hashes: list[str] = []

    for fixture in _fixture_specs():
        payload = fixture.build()
        payload_hashes.append(_sha256_bytes(payload.encode("utf-8")))
        findings = detect_privacy_findings(payload)
        fired = bool(findings)
        detector_counts.update(item.detector_id for item in findings)
        if fixture.expected_sensitive and fired:
            outcome = "true_positive"
        elif fixture.expected_sensitive:
            outcome = "false_negative"
            false_negative_ids.append(fixture.fixture_id)
        elif fired:
            outcome = "false_positive"
            false_positive_ids.append(fixture.fixture_id)
        else:
            outcome = "true_negative"
        confusion[outcome] += 1
        by_category[fixture.category][outcome] += 1

    positive_total = confusion["true_positive"] + confusion["false_negative"]
    negative_total = confusion["true_negative"] + confusion["false_positive"]
    return {
        "fixture_count": sum(confusion.values()),
        "positive_fixture_count": positive_total,
        "negative_fixture_count": negative_total,
        "confusion": dict(sorted(confusion.items())),
        "false_negative_rate": confusion["false_negative"] / positive_total if positive_total else 0.0,
        "false_positive_rate": confusion["false_positive"] / negative_total if negative_total else 0.0,
        "false_negative_fixture_ids": sorted(false_negative_ids),
        "false_positive_fixture_ids": sorted(false_positive_ids),
        "detector_counts": dict(sorted(detector_counts.items())),
        "by_category": {
            key: dict(sorted(value.items())) for key, value in sorted(by_category.items())
        },
        "fixture_payload_set_sha256": _sha256_json(sorted(payload_hashes)),
        "payload_retention": "NONE",
    }


def _candidate_normalized_payloads(root: Path) -> Iterable[bytes]:
    for path in sorted(root.rglob("*.txt")):
        if path.is_file():
            yield path.read_bytes()


def _resolve_data229_text(root: Path, expected: Mapping[str, Any]) -> str:
    expected_sha = str(expected["expected_input_sha256"])
    expected_bytes = int(expected["expected_input_utf8_bytes"])
    for payload in _candidate_normalized_payloads(root):
        variants = (payload, payload[:-1]) if payload.endswith(b"\n") else (payload,)
        for candidate in variants:
            if len(candidate) == expected_bytes and _sha256_bytes(candidate) == expected_sha:
                return candidate.decode("utf-8", errors="strict")
    raise Data297AuditError(
        "DATA-229 normalized payload not found for immutable expected hash " + expected_sha
    )


def _find_unique_json(root: Path, filename: str) -> Mapping[str, Any]:
    paths = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(paths) != 1:
        raise Data297AuditError(f"expected exactly one {filename}, found {len(paths)}")
    return _load_json(paths[0])


def _resolve_data227_code(root: Path, expected: Mapping[str, Any]) -> tuple[str, str]:
    report = _find_unique_json(root, "data227-real-code-source-admission.json")
    objects = [
        item for item in report.get("objects", [])
        if item.get("source_id") == expected["source_id"]
    ]
    registry_sources = [
        item for item in report.get("registry", {}).get("sources", [])
        if item.get("source_id") == expected["source_id"]
    ]
    if len(objects) != 1 or len(registry_sources) != 1:
        raise Data297AuditError(
            "DATA-227 source object/registry entry missing or duplicated: "
            + str(expected["source_id"])
        )

    item = objects[0]
    registry_source = registry_sources[0]
    snapshot = registry_source.get("snapshot", {})
    raw_sha = str(item.get("raw_sha256", ""))
    expected_sha = str(expected["expected_input_sha256"])
    expected_bytes = int(expected["expected_input_utf8_bytes"])
    if not _SHA256_RE.fullmatch(raw_sha):
        raise Data297AuditError("DATA-227 object does not bind raw SHA-256")
    if item.get("normalization_sha256") != raw_sha:
        raise Data297AuditError("DATA-227 raw/normalization identity drift")
    if raw_sha != expected_sha or snapshot.get("sha256") != expected_sha:
        raise Data297AuditError("DATA-227 configured/report/registry SHA-256 drift")
    if int(snapshot.get("size_bytes", -1)) != expected_bytes:
        raise Data297AuditError("DATA-227 registry snapshot byte count drift")

    matches: list[bytes] = []
    for path in sorted(root.rglob("payload")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if len(payload) == expected_bytes and _sha256_bytes(payload) == expected_sha:
            matches.append(payload)
    if len(matches) != 1:
        raise Data297AuditError("DATA-227 exact snapshot payload missing or duplicated")
    return matches[0].decode("utf-8", errors="strict"), expected_sha


def _record_public_row(
    expected: Mapping[str, Any], result: Any, input_bytes: int
) -> Mapping[str, Any]:
    retained = (
        len(result.sanitized_text.encode("utf-8"))
        if result.train_eligible_after_privacy and result.sanitized_text is not None
        else 0
    )
    return {
        **result.evidence_record(),
        "source_family": expected["source_family"],
        "language": expected["language"],
        "input_utf8_bytes": input_bytes,
        "retained_training_utf8_bytes": retained,
        "excluded_input_utf8_bytes": 0 if result.train_eligible_after_privacy else input_bytes,
    }


def _aggregate_families(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row["source_family"])
        bucket = out.setdefault(
            family,
            {
                "source_count": 0,
                "input_utf8_bytes": 0,
                "retained_training_utf8_bytes": 0,
                "excluded_input_utf8_bytes": 0,
                "action_counts": Counter(),
                "detector_counts": Counter(),
            },
        )
        bucket["source_count"] += 1
        bucket["input_utf8_bytes"] += int(row["input_utf8_bytes"])
        bucket["retained_training_utf8_bytes"] += int(row["retained_training_utf8_bytes"])
        bucket["excluded_input_utf8_bytes"] += int(row["excluded_input_utf8_bytes"])
        bucket["action_counts"][row["action"]] += 1
        bucket["detector_counts"].update(row["detector_counts"])

    public: dict[str, Any] = {}
    for family, bucket in sorted(out.items()):
        public[family] = {
            "source_count": bucket["source_count"],
            "input_utf8_bytes": bucket["input_utf8_bytes"],
            "retained_training_utf8_bytes": bucket["retained_training_utf8_bytes"],
            "excluded_input_utf8_bytes": bucket["excluded_input_utf8_bytes"],
            "retention_ratio": (
                bucket["retained_training_utf8_bytes"] / bucket["input_utf8_bytes"]
                if bucket["input_utf8_bytes"] else 0.0
            ),
            "action_counts": dict(sorted(bucket["action_counts"].items())),
            "detector_counts": dict(sorted(bucket["detector_counts"].items())),
        }
    return public


def build_external_audit_report(
    *,
    data229_root: Path,
    data227_root: Path,
    config_path: Path = DEFAULT_CONFIG,
) -> Mapping[str, Any]:
    config = load_config(config_path)
    policy_sha = assert_incumbent_privacy_authority(config)
    rows: list[Mapping[str, Any]] = []

    for expected in config["admitted_inventory"]:
        if expected["authority"] == "DATA-229":
            text = _resolve_data229_text(data229_root, expected)
        elif expected["authority"] == "DATA-227":
            text, _ = _resolve_data227_code(data227_root, expected)
        else:
            raise Data297AuditError("unexpected admitted authority")

        payload = text.encode("utf-8")
        if len(payload) != int(expected["expected_input_utf8_bytes"]):
            raise Data297AuditError("materialized input-byte drift")
        if _sha256_bytes(payload) != expected["expected_input_sha256"]:
            raise Data297AuditError("materialized input SHA-256 drift")

        result = scan_record(
            record_id=str(expected["source_id"]),
            source_id=str(expected["source_id"]),
            source_version=str(expected["source_version"]),
            modality=str(expected["modality"]),
            text=text,
        )
        rows.append(_record_public_row(expected, result, len(payload)))

    expected_inventory = config["expected_training_inventory"]
    observed_input = sum(int(row["input_utf8_bytes"]) for row in rows)
    if observed_input != int(expected_inventory["input_utf8_bytes"]):
        raise Data297AuditError("complete admitted inventory byte total drift")
    observed_families = _aggregate_families(rows)
    for family, expected_bytes in expected_inventory["by_family_input_utf8_bytes"].items():
        if int(observed_families[family]["input_utf8_bytes"]) != int(expected_bytes):
            raise Data297AuditError("per-family admitted input-byte drift")

    fixture_metrics = audit_labeled_fixtures()
    core = {
        "schema_version": SCHEMA_VERSION,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "privacy_authority": {
            "worker": config["privacy_authority"]["worker"],
            "source_sha": config["privacy_authority"]["source_sha"],
            "policy_schema": config["privacy_authority"]["policy_schema"],
            "policy_sha256": policy_sha,
            "framework_action": "REUSED_UNCHANGED",
        },
        "inventory_authorities": config["inventory_authorities"],
        "claim_boundary": {
            "admitted_terminal_sources_scanned": len(rows),
            "admitted_terminal_families_scanned": len(observed_families),
            "nonterminal_data228_promoted": False,
            "coverage": (
                "DATA-33 high-confidence-patterns-only; false negatives remain possible "
                "outside labeled fixtures."
            ),
        },
        "fixture_metrics": fixture_metrics,
        "inventory": {
            "input_utf8_bytes": observed_input,
            "retained_training_utf8_bytes": sum(
                int(row["retained_training_utf8_bytes"]) for row in rows
            ),
            "excluded_input_utf8_bytes": sum(
                int(row["excluded_input_utf8_bytes"]) for row in rows
            ),
            "by_family": observed_families,
            "records": sorted(rows, key=lambda row: str(row["source_id"])),
        },
        "public_evidence": {
            "matched_values_retained": False,
            "text_previews_retained": False,
            "raw_source_text_retained": False,
            "fixture_payloads_retained": False,
        },
    }
    report = {**core, "report_identity_sha256": _sha256_json(core)}
    assert_no_secret_values_in_manifest(report)

    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True)
    for fixture in _fixture_specs():
        if fixture.build() in serialized:
            raise Data297AuditError("public report leaked a fixture payload")
    return report


def write_public_report(report: Mapping[str, Any], path: Path) -> None:
    assert_no_secret_values_in_manifest(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fixtures = sub.add_parser("fixtures")
    fixtures.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    fixtures.add_argument("--output", type=Path)

    audit = sub.add_parser("audit")
    audit.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit.add_argument("--data229-root", type=Path, required=True)
    audit.add_argument("--data227-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fixtures":
        config = load_config(args.config)
        core = {
            "schema_version": "12-6.data297-privacy-fixtures.v1",
            "worker_id": WORKER_ID,
            "local_free_only": True,
            "privacy_policy_sha256": assert_incumbent_privacy_authority(config),
            "fixture_metrics": audit_labeled_fixtures(),
        }
        report = {**core, "report_identity_sha256": _sha256_json(core)}
        assert_no_secret_values_in_manifest(report)
        if args.output:
            write_public_report(report, args.output)
        else:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0

    report = build_external_audit_report(
        data229_root=args.data229_root,
        data227_root=args.data227_root,
        config_path=args.config,
    )
    write_public_report(report, args.output)
    print(
        json.dumps(
            {
                "report_identity_sha256": report["report_identity_sha256"],
                "input_utf8_bytes": report["inventory"]["input_utf8_bytes"],
                "retained_training_utf8_bytes": report["inventory"]["retained_training_utf8_bytes"],
                "false_negative_rate": report["fixture_metrics"]["false_negative_rate"],
                "false_positive_rate": report["fixture_metrics"]["false_positive_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
