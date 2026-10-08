"""Read-only Plan-1 S3 inventory of *observed*, never admitted, dependencies.

An installed wheel RECORD is self-declared. Its observed SHA-256 is a review
candidate, NOT an independently authenticated pin or proof of license/security.
No imports of the target packages, installation, or network operations occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

MAX_PROJECT_BYTES = 1024 * 1024
MAX_RECORD_BYTES = 4 * 1024 * 1024
NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{1,99}")
REQUIREMENT = re.compile(r"([A-Za-z][A-Za-z0-9._-]{1,99})(>=|==)([A-Za-z0-9][A-Za-z0-9.+_-]*)")


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def inspect(pyproject: Path) -> dict[str, Any]:
    """Inventory mandatory runtime requirements, fail closed on ambiguity.

    This deliberately does not compare environment state to an approved lock:
    Plan 1 Section 4 owns deterministic dependency locks and compatibility.
    """
    raw = pyproject.read_bytes()
    if not raw or len(raw) > MAX_PROJECT_BYTES:
        raise ValueError("project metadata size invalid")
    project = tomllib.loads(raw.decode("utf-8"))["project"]
    declared = project["dependencies"]
    if type(declared) is not list or not declared:
        raise ValueError("mandatory runtime dependencies required")
    assets: list[dict[str, Any]] = []
    observed: set[str] = set()
    for requirement in declared:
        if type(requirement) is not str:
            raise ValueError("unsupported requirement entry")
        match = REQUIREMENT.fullmatch(requirement.strip())
        if match is None:
            raise ValueError("unreviewable requirement expression")
        raw_name, operator, minimum = match.groups()
        name = normalized_name(raw_name)
        if name in observed:
            raise ValueError("duplicate declared runtime dependency")
        observed.add(name)
        entry: dict[str, Any] = {
            "distribution": name,
            "declared_requirement": f"{raw_name}{operator}{minimum}",
            "installed_version": None,
            "record_sha256_observed_untrusted": None,
            # Observational signals only; even an SPDX field is NOT legal review.
            "license_metadata_sha256_observed_untrusted": None,
            "license_metadata_field_observed_untrusted": None,
            "availability": "NOT_INSTALLED",
            "admission": "DENIED_UNQUALIFIED",
            "external_source_sha256": None,
            "independently_reviewed_license": False,
            "independently_reviewed_security": False,
            "data_rights_granted": False,
            "foreign_model_weights_admitted": False,
        }
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            assets.append(entry)
            continue
        actual_name = dist.metadata.get("Name")
        if type(actual_name) is not str or normalized_name(actual_name) != name:
            raise ValueError(f"installed distribution identity mismatch: {name}")
        version = dist.version
        if type(version) is not str or not version or len(version) > 128:
            raise ValueError(f"unreadable installed version: {name}")
        entry["installed_version"] = version
        # License text and SPDX expressions may be incomplete or asserted by an
        # untrusted installation. Digest the exact observed header bytes for an
        # independent review to compare; NEVER turn it into code admission.
        expression = dist.metadata.get("License-Expression")
        legacy_license = dist.metadata.get("License")
        if expression is not None and type(expression) is not str:
            raise ValueError(f"invalid License-Expression metadata: {name}")
        if legacy_license is not None and type(legacy_license) is not str:
            raise ValueError(f"invalid License metadata: {name}")
        for field, value in (("License-Expression", expression), ("License", legacy_license)):
            if value is None:
                continue
            encoded_license = value.encode("utf-8")
            if not encoded_license or len(encoded_license) > MAX_RECORD_BYTES:
                raise ValueError(f"unbounded installed license metadata: {name}")
            # Prefer the unambiguous PEP 639 field; neither field is authority.
            entry["license_metadata_field_observed_untrusted"] = field
            entry["license_metadata_sha256_observed_untrusted"] = hashlib.sha256(
                encoded_license
            ).hexdigest()
            if field == "License-Expression":
                break
        record = dist.read_text("RECORD")
        if record is None:
            entry["availability"] = "INSTALLED_RECORD_ABSENT"
        else:
            if type(record) is not str:
                raise ValueError(f"invalid RECORD: {name}")
            record_bytes = record.encode("utf-8")
            if not record_bytes or len(record_bytes) > MAX_RECORD_BYTES:
                raise ValueError(f"RECORD size invalid: {name}")
            entry["record_sha256_observed_untrusted"] = hashlib.sha256(
                record_bytes
            ).hexdigest()
            entry["availability"] = "INSTALLED_UNVERIFIED"
        assets.append(entry)
    return {
        "schema_version": 1,
        "purpose": "PLAN1_S3_OBSERVATION_ONLY",
        "source_pyproject_sha256": hashlib.sha256(raw).hexdigest(),
        "all_dependencies_admitted": False,
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    result = inspect(args.pyproject)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0  # Success means observational scan completed, NOT license admission.


if __name__ == "__main__":
    raise SystemExit(main())
