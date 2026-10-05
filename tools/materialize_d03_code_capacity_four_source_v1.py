#!/usr/bin/env python3
"""Physical exact-blob verifier for the four-source D03 code-capacity candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

MAX_CONFIG_BYTES = 1_048_576
MAX_LICENSE_BYTES = 262_144
MAX_JSON_INTEGER_DIGITS = 64
EXPECTED_RAW_BYTES = 336_947
REPORT_SCHEMA = "12-6.d03-code4-physical-materialization.v1"

CONFIGS = {
    "pydantic": Path("configs/data/next100_048_pydantic_code_rights_v1.json"),
    "scipy": Path("configs/data/scipy_v118_source_authority_v1.json"),
    "pandas": Path("configs/data/next100_050_pandas_source_authority_v2.json"),
    "typer": Path("configs/data/next100_052_typer_source_authority_v2.json"),
}

EXPECTED = {
    "pydantic": {
        "repository": "pydantic/pydantic",
        "commit": "cf67d4b3193cfe43ede18612ed62785eee11382",
        "family": "github:pydantic/pydantic",
        "license": ("LICENSE", "488c6260c10f2e88fa1fae58a63fccec8d600cd1", "MIT"),
        "files": (
            ("pydantic/main.py", "2ed62e5f3eded5d1e6b0052130e8d5f8627a60c6", 85_334),
            ("pydantic/fields.py", "ede966fa8010c070fede69d8f048b618a1b38f17", 82_023),
            ("pydantic/type_adapter.py", "d962305f2c923a6046b8234a334bdca0e644b159", 36_123),
            (
                "pydantic/functional_validators.py",
                "558e99c1e7a09e3c894d769c96cb28e25dc537ca",
                31_724,
            ),
        ),
    },
    "scipy": {
        "repository": "scipy/scipy",
        "commit": "54ef5423f2e4376230ec3bfda6912a07a50958e3",
        "family": "code.scipy.project",
        "license": ("LICENSE.txt", "1032aece8a8109d67f238de6865d8b08e750a8c0", "BSD-3-Clause"),
        "additional_license": (
            "LICENSES_bundled.txt",
            "68d75f62d9395b91f430df687fae2e278af6d0c4",
        ),
        "files": (
            (
                "scipy/optimize/_constraints.py",
                "75f81735dfd0feaff3865025467c859fd3b98ff6",
                25_257,
            ),
            (
                "scipy/optimize/_minimize.py",
                "91d736990aaebe2dc119c26b59782466fa314fde",
                53_050,
            ),
        ),
    },
    "pandas": {
        "repository": "pandas-dev/pandas",
        "commit": "0b54092c19202b579a1114e2d74f4b8028d5d573",
        "family": "github:pandas-dev/pandas",
        "license": ("LICENSE", "bd1cc2a30c626d0bbe43ab6ec1c4744c2e2df831", "BSD-3-Clause"),
        "files": (
            (
                "pandas/core/accessor.py",
                "26560f84000f24aee377bc761db0d2686452f813",
                15_837,
            ),
        ),
    },
    "typer": {
        "repository": "fastapi/typer",
        "commit": "fe2aa0e2f9c853de378e60ca24ec3b256144decf",
        "family": "github:fastapi/typer",
        "license": ("LICENSE", "a7694736cf37716aafec14b24aa8d6316ebe07a3", "MIT"),
        "files": (
            ("typer/utils.py", "addf9334d4210a9cddc9e5608ae446417d372eb0", 7_599),
        ),
    },
}

SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_style_secret": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
}


class PhysicalSourceError(RuntimeError):
    """Fail-closed physical source verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalSourceError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhysicalSourceError("duplicate JSON object member")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise PhysicalSourceError("non-finite JSON number")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PhysicalSourceError("non-finite JSON number")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise PhysicalSourceError("nonzero JSON number underflowed to zero")
    return parsed


def _bounded_int(value: str) -> int:
    if len(value.removeprefix("-")) > MAX_JSON_INTEGER_DIGITS:
        raise PhysicalSourceError("JSON integer exceeds digit limit")
    return int(value)


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PhysicalSourceError("cannot read source authority config") from exc
    require(len(raw) <= MAX_CONFIG_BYTES, "source authority config exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_object,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
            parse_int=_bounded_int,
        )
    except PhysicalSourceError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise PhysicalSourceError("invalid source authority config") from exc
    require(isinstance(value, dict), "source authority config root must be object")
    return value


def validate_configs(configs: dict[str, dict[str, Any]]) -> None:
    pydantic = configs["pydantic"]
    expected = EXPECTED["pydantic"]
    require(pydantic.get("source_family") == expected["family"], "Pydantic family drift")
    require(
        pydantic.get("upstream_commit") == expected["commit"],
        "Pydantic commit drift: "
        f"actual={pydantic.get('upstream_commit')!r} expected={expected['commit']!r}",
    )
    license_info = pydantic.get("license")
    require(isinstance(license_info, dict), "Pydantic license missing")
    require(
        (
            license_info.get("path"),
            license_info.get("blob_sha1"),
            license_info.get("license_id"),
        )
        == expected["license"],
        "Pydantic license drift",
    )
    decisions = pydantic.get("decisions")
    require(isinstance(decisions, list), "Pydantic decisions missing")
    observed = tuple(
        (row.get("path"), row.get("blob_sha1"), row.get("size_bytes"))
        for row in decisions
        if isinstance(row, dict)
    )
    require(observed == expected["files"], "Pydantic source set drift")

    scipy = configs["scipy"]
    expected = EXPECTED["scipy"]
    upstream = scipy.get("upstream")
    require(isinstance(upstream, dict), "SciPy upstream missing")
    require(upstream.get("repository") == expected["repository"], "SciPy repository drift")
    require(upstream.get("commit_sha") == expected["commit"], "SciPy commit drift")
    require(scipy.get("source_family") == expected["family"], "SciPy family drift")
    scipy_license = scipy.get("license")
    require(isinstance(scipy_license, dict), "SciPy license missing")
    require(
        scipy_license.get("root_path") == expected["license"][0],
        "SciPy root license path drift",
    )
    require(
        scipy_license.get("bundled_license_path") == expected["additional_license"][0],
        "SciPy bundled license path drift",
    )
    allowlist = scipy.get("allowlist")
    require(isinstance(allowlist, list), "SciPy allowlist missing")
    observed = tuple(
        (row.get("path"), row.get("git_blob_sha1"), row.get("raw_bytes"))
        for row in allowlist
        if isinstance(row, dict)
    )
    require(observed == expected["files"], "SciPy source set drift")
    capacity = scipy.get("capacity")
    require(isinstance(capacity, dict), "SciPy capacity missing")
    require(capacity.get("canonical_credit_bytes") == 0, "SciPy credit must remain zero")
    require(
        capacity.get("materialization_status") == "not_materialized",
        "SciPy authority must remain pre-materialization",
    )

    for name in ("pandas", "typer"):
        config = configs[name]
        expected = EXPECTED[name]
        source = config.get("bounded_source")
        license_info = config.get("license")
        composition = config.get("current_composition")
        truth = config.get("truth_boundary")
        require(isinstance(source, dict), f"{name} bounded source missing")
        require(isinstance(license_info, dict), f"{name} license missing")
        require(isinstance(composition, dict), f"{name} composition missing")
        require(isinstance(truth, dict), f"{name} truth boundary missing")
        require(source.get("repository") == expected["repository"], f"{name} repository drift")
        require(source.get("commit") == expected["commit"], f"{name} commit drift")
        require(source.get("source_family") == expected["family"], f"{name} family drift")
        expected_path, expected_blob, expected_size = expected["files"][0]
        require(
            (
                source.get("path"),
                source.get("git_blob_sha1"),
                source.get("size_bytes"),
            )
            == (expected_path, expected_blob, expected_size),
            f"{name} source object drift",
        )
        require(
            (
                license_info.get("path"),
                license_info.get("git_blob_sha1"),
                license_info.get("license_id"),
            )
            == expected["license"],
            f"{name} license drift",
        )
        require(
            composition.get("canonical_capacity_credit_bytes") == 0,
            f"{name} credit must remain zero",
        )
        require(truth.get("tokenizer_fit_authorized") is False, f"{name} tokenizer gate drift")
        require(truth.get("training_executed") is False, f"{name} training truth drift")
        require(truth.get("final_test_outcomes_read") is False, f"{name} final-test truth drift")
        require(truth.get("paid_compute_used") is False, f"{name} compute truth drift")


def download_exact(
    repository: str,
    commit: str,
    path: str,
    *,
    max_bytes: int,
) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-d03-code4-physical/1.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(max_bytes + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise PhysicalSourceError("bounded upstream download failed") from exc
    require(len(raw) <= max_bytes, "bounded upstream download exceeded byte limit")
    return raw


def verify_license(
    repository: str,
    commit: str,
    path: str,
    expected_blob: str,
    license_id: str,
) -> dict[str, Any]:
    raw = download_exact(repository, commit, path, max_bytes=MAX_LICENSE_BYTES)
    require(git_blob_sha1(raw) == expected_blob, "license Git blob identity drift")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PhysicalSourceError("license is not strict UTF-8") from exc
    semantic = " ".join(text.split())
    if license_id == "MIT":
        require("Permission is hereby granted" in semantic, "MIT grant text missing")
        require("without restriction" in semantic, "MIT grant scope missing")
    elif license_id == "BSD-3-Clause":
        require(
            "Redistribution and use in source and binary forms" in semantic,
            "BSD redistribution grant missing",
        )
        require(
            "Neither the name" in semantic,
            "BSD non-endorsement condition missing",
        )
    else:
        raise PhysicalSourceError("unsupported license family")
    return {
        "path": path,
        "license_id": license_id,
        "git_blob_sha1": expected_blob,
        "sha256": sha256(raw),
        "utf8_bytes": len(raw),
    }


def verify_source(
    repository: str,
    commit: str,
    path: str,
    expected_blob: str,
    expected_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    raw = download_exact(repository, commit, path, max_bytes=expected_bytes)
    require(len(raw) == expected_bytes, "source byte length drift")
    require(git_blob_sha1(raw) == expected_blob, "source Git blob identity drift")
    require(b"\x00" not in raw, "source contains NUL byte")
    hits = sorted(name for name, pattern in SECRET_PATTERNS.items() if pattern.search(raw))
    require(not hits, "source contains high-confidence credential pattern")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PhysicalSourceError("source is not strict UTF-8") from exc
    require("\ufffd" not in text, "source contains replacement character")
    require(
        not any(0xD800 <= ord(char) <= 0xDFFF for char in text),
        "source contains Unicode surrogate",
    )
    try:
        ast.parse(text, filename=path)
    except SyntaxError as exc:
        raise PhysicalSourceError("source does not parse as Python") from exc
    return (
        {
            "path": path,
            "git_blob_sha1": expected_blob,
            "sha256": sha256(raw),
            "utf8_bytes": len(raw),
            "strict_utf8": True,
            "python_ast_parse": "PASS",
            "credential_scan": "PASS",
        },
        raw,
    )


def materialize(source_head: str) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{40}", source_head) is not None, "invalid source head")
    configs = {name: load_config(path) for name, path in CONFIGS.items()}
    validate_configs(configs)

    families: list[dict[str, Any]] = []
    observed_hashes: set[str] = set()
    raw_total = 0
    for name in ("pydantic", "scipy", "pandas", "typer"):
        expected = EXPECTED[name]
        repository = expected["repository"]
        commit = expected["commit"]
        license_path, license_blob, license_id = expected["license"]
        license_record = verify_license(
            repository,
            commit,
            license_path,
            license_blob,
            license_id,
        )
        additional_license = expected.get("additional_license")
        additional_records: list[dict[str, Any]] = []
        if additional_license is not None:
            extra_path, extra_blob = additional_license
            raw = download_exact(repository, commit, extra_path, max_bytes=MAX_LICENSE_BYTES)
            require(git_blob_sha1(raw) == extra_blob, "additional license blob drift")
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PhysicalSourceError("additional license is not strict UTF-8") from exc
            additional_records.append(
                {
                    "path": extra_path,
                    "git_blob_sha1": extra_blob,
                    "sha256": sha256(raw),
                    "utf8_bytes": len(raw),
                }
            )

        source_records: list[dict[str, Any]] = []
        for path, expected_blob, expected_bytes in expected["files"]:
            record, raw = verify_source(
                repository,
                commit,
                path,
                expected_blob,
                expected_bytes,
            )
            digest = record["sha256"]
            require(digest not in observed_hashes, "exact duplicate across four-source set")
            observed_hashes.add(digest)
            source_records.append(record)
            raw_total += len(raw)

        families.append(
            {
                "name": name,
                "repository": repository,
                "family_id": expected["family"],
                "commit": commit,
                "license": license_record,
                "additional_license_files": additional_records,
                "source_files": source_records,
                "raw_candidate_bytes": sum(row["utf8_bytes"] for row in source_records),
            }
        )

    require(raw_total == EXPECTED_RAW_BYTES, "four-source raw byte total drift")
    core = {
        "schema_version": REPORT_SCHEMA,
        "source_head_sha": source_head,
        "execution_profile": "LOCAL_FREE",
        "physical_source_family_count": 4,
        "physical_source_file_count": 8,
        "physical_raw_candidate_bytes": raw_total,
        "families": families,
        "cross_four_exact_duplicate_count": 0,
        "claim_boundary": {
            "canonical_capacity_credit_bytes": 0,
            "canonical_family_credit": 0,
            "current_whole_graph_global_dedup_complete": False,
            "current_reserved_evaluation_decontamination_complete": False,
            "post_composition_quality_privacy_complete": False,
            "balance_and_family_caps_complete": False,
            "cluster_safe_split_complete": False,
            "deterministic_pack_two_clean_complete": False,
            "positive_exact_unique_loss_ledger": False,
            "tokenizer_fit_authorized": False,
            "authorized_optimized_target_exposure": 0,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    report = dict(core)
    report["report_sha256"] = sha256(canonical_bytes(core))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = materialize(args.source_head)
        payload = canonical_bytes(report) + b"\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            if handle.write(payload) != len(payload):
                raise PhysicalSourceError("incomplete report write")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, PhysicalSourceError) as exc:
        print(
            "D03_CODE4_PHYSICAL_BLOCKED "
            f"{type(exc).__name__} reason={str(exc)}"
        )
        return 2

    print(
        "D03_CODE4_PHYSICAL_PASS "
        f"head={report['source_head_sha']} "
        f"families={report['physical_source_family_count']} "
        f"files={report['physical_source_file_count']} "
        f"bytes={report['physical_raw_candidate_bytes']} "
        f"report_sha256={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
