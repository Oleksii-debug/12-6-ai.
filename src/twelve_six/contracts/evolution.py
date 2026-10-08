"""Declarative-only schema evolution policy; never mutates an accepted v1 contract."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PLANS = frozenset(range(1, 11))


class ContractEvolutionError(ValueError):
    """A schema change lacks the version, test, migration, or impact evidence."""


def _schema(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict or not value:
        raise ContractEvolutionError("schema must be a nonempty exact mapping")
    for key, wire_type in value.items():
        if (type(key) is not str or _NAME.fullmatch(key) is None
                or type(wire_type) is not str or _NAME.fullmatch(wire_type) is None):
            raise ContractEvolutionError("invalid schema field/type")
    return dict(sorted(value.items()))


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def propose_version_change(
    *, contract_kind: str, previous_version: int, next_version: int,
    previous_fields: dict[str, str], next_fields: dict[str, str],
    migration_fixture_sha256: str | None,
    deprecation_notice: str | None,
    affected_plans: tuple[int, ...],
    compatibility_test_sha256: str,
) -> dict[str, Any]:
    """Produce a REVIEW_REQUIRED change record; no migration is executed."""
    from .baseline_v1 import baseline_manifest_v1

    if type(contract_kind) is not str or not _NAME.fullmatch(contract_kind):
        raise ContractEvolutionError("invalid contract kind")
    if (type(previous_version) is not int or type(next_version) is not int
            or previous_version < 1 or next_version <= previous_version):
        raise ContractEvolutionError("schema changes require a higher integer version")
    old = _schema(previous_fields)
    new = _schema(next_fields)
    if old == new:
        raise ContractEvolutionError("no schema change to qualify")
    breaking = any(name not in new or new[name] != wire_type for name, wire_type in old.items())
    if (type(compatibility_test_sha256) is not str
            or _SHA.fullmatch(compatibility_test_sha256) is None):
        raise ContractEvolutionError("missing exact compatibility-test evidence digest")
    if (type(affected_plans) is not tuple or not affected_plans
            or any(type(number) is not int or number not in _PLANS for number in affected_plans)
            or tuple(sorted(set(affected_plans))) != affected_plans):
        raise ContractEvolutionError("incomplete/ambiguous affected-plan impact analysis")
    if breaking and (
        type(migration_fixture_sha256) is not str
        or _SHA.fullmatch(migration_fixture_sha256) is None
        or type(deprecation_notice) is not str
        or not deprecation_notice.strip()
    ):
        raise ContractEvolutionError("breaking changes require migration fixture and deprecation")
    if migration_fixture_sha256 is not None and (
        type(migration_fixture_sha256) is not str
        or _SHA.fullmatch(migration_fixture_sha256) is None
    ):
        raise ContractEvolutionError("invalid migration fixture digest")
    if deprecation_notice is not None and (
        type(deprecation_notice) is not str or not deprecation_notice.strip()
    ):
        raise ContractEvolutionError("invalid deprecation notice")
    payload = {
        "schema": "12-6.contract-evolution-declaration.v1",
        "baseline_sha256": baseline_manifest_v1()["baseline_sha256"],
        "contract_kind": contract_kind,
        "previous_version": previous_version,
        "next_version": next_version,
        "previous_fields": old,
        "next_fields": new,
        "breaking": breaking,
        "migration_fixture_sha256": migration_fixture_sha256,
        "deprecation_notice": deprecation_notice,
        "affected_plans": list(affected_plans),
        "compatibility_test_sha256": compatibility_test_sha256,
        "decision": "REVIEW_REQUIRED",
        "auto_migrate_existing_artifacts": False,
        "reopen_terminal_plans": False,
    }
    payload["declaration_sha256"] = _sha256(payload)
    return payload


def validate_version_change(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject resealed approval, missing provenance, malformed impact or altered fixture."""
    if type(value) is not dict or set(value) != {
        "schema", "baseline_sha256", "contract_kind", "previous_version",
        "next_version", "previous_fields", "next_fields", "breaking",
        "migration_fixture_sha256", "deprecation_notice", "affected_plans",
        "compatibility_test_sha256", "decision", "auto_migrate_existing_artifacts",
        "reopen_terminal_plans", "declaration_sha256",
    }:
        raise ContractEvolutionError("unknown evolution declaration fields")
    if value.get("schema") != "12-6.contract-evolution-declaration.v1":
        raise ContractEvolutionError("unsupported evolution schema")
    if (value.get("decision") != "REVIEW_REQUIRED"
            or value.get("auto_migrate_existing_artifacts") is not False
            or value.get("reopen_terminal_plans") is not False):
        raise ContractEvolutionError("declaration cannot assert approval or mutation")
    rebuilt = propose_version_change(
        contract_kind=value["contract_kind"],
        previous_version=value["previous_version"],
        next_version=value["next_version"],
        previous_fields=value["previous_fields"],
        next_fields=value["next_fields"],
        migration_fixture_sha256=value["migration_fixture_sha256"],
        deprecation_notice=value["deprecation_notice"],
        affected_plans=tuple(value["affected_plans"]),
        compatibility_test_sha256=value["compatibility_test_sha256"],
    )
    if rebuilt != value:
        raise ContractEvolutionError("evolution declaration identity mismatch")
    return rebuilt
