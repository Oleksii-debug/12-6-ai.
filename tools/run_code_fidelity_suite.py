#!/usr/bin/env python3
"""Execute DATA-28 code-normalization fidelity evidence on pinned real samples."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from twelve_six.data.code_normalization import (
    CODE_NORMALIZATION_POLICY,
    CODE_NORMALIZATION_SCHEMA,
    CodeNormalizationError,
    decode_code_bytes,
)

REPORT_SCHEMA = "12-6.data28-code-fidelity-report.v1"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _download_sample(sample: dict[str, Any]) -> bytes:
    repository = sample["repository"]
    commit = sample["commit"]
    path = urllib.parse.quote(sample["path"], safe="/")
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-data28-fidelity/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _parser_evidence(parser: str, before: str, after: str) -> dict[str, Any]:
    if parser == "python_ast":
        before_tree = ast.parse(before)
        after_tree = ast.parse(after)
        return {
            "parser": parser,
            "available": True,
            "before_success": True,
            "after_success": True,
            "structurally_equal": ast.dump(before_tree) == ast.dump(after_tree),
        }
    if parser == "json":
        before_value = json.loads(before)
        after_value = json.loads(after)
        return {
            "parser": parser,
            "available": True,
            "before_success": True,
            "after_success": True,
            "structurally_equal": before_value == after_value,
        }
    if parser == "optional_pyyaml":
        if importlib.util.find_spec("yaml") is None:
            return {
                "parser": parser,
                "available": False,
                "before_success": None,
                "after_success": None,
                "reason": "PyYAML not installed in locked DATA-28 environment",
            }
        import yaml  # type: ignore[import-not-found]

        before_value = yaml.safe_load(before)
        after_value = yaml.safe_load(after)
        return {
            "parser": parser,
            "available": True,
            "before_success": True,
            "after_success": True,
            "structurally_equal": before_value == after_value,
        }
    if parser.startswith("not_run_"):
        return {
            "parser": parser,
            "available": False,
            "before_success": None,
            "after_success": None,
            "reason": parser.removeprefix("not_run_").replace("_", " "),
        }
    raise ValueError(f"unsupported parser contract: {parser}")


def _assert_source_head(expected_source_sha: str) -> None:
    if len(expected_source_sha) != 40 or any(
        char not in "0123456789abcdef" for char in expected_source_sha
    ):
        raise ValueError("--source-sha must be a lowercase 40-character Git SHA")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != expected_source_sha:
        raise RuntimeError(
            f"source head mismatch: expected {expected_source_sha}, actual {actual}"
        )


def run(
    *,
    source_sha: str,
    registry_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    _assert_source_head(source_sha)
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if registry.get("schema_version") != "12-6.code-fidelity-real-samples.v1":
        raise ValueError("unsupported real-sample registry schema")
    samples = registry.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("real-sample registry must contain samples")

    results: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    parser_counts: Counter[str] = Counter()
    total_source_bytes = 0
    total_normalized_bytes = 0

    for sample in samples:
        payload = _download_sample(sample)
        expected_blob = sample["git_blob_sha1"]
        actual_blob = _git_blob_sha1(payload)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"{sample['sample_id']}: git blob mismatch "
                f"expected={expected_blob} actual={actual_blob}"
            )
        normalized, evidence = decode_code_bytes(
            payload,
            language=sample["language"],
            path=sample["path"],
        )
        normalized_bytes = normalized.encode("utf-8")
        if normalized_bytes != payload:
            raise RuntimeError(f"{sample['sample_id']}: normalization changed source bytes")
        parser = _parser_evidence(sample["parser"], payload.decode("utf-8"), normalized)
        if parser.get("available"):
            parser_counts["available"] += 1
            if parser.get("before_success") and parser.get("after_success"):
                parser_counts["before_after_success"] += 1
            else:
                raise RuntimeError(f"{sample['sample_id']}: parser fidelity failed")
        else:
            parser_counts["unavailable_or_not_run"] += 1

        reason_counts.update(evidence.reasons)
        total_source_bytes += len(payload)
        total_normalized_bytes += len(normalized_bytes)
        results.append(
            {
                "sample_id": sample["sample_id"],
                "repository": sample["repository"],
                "commit": sample["commit"],
                "path": sample["path"],
                "language": sample["language"],
                "license_spdx_evidence": sample["license_spdx_evidence"],
                "license_evidence_kind": sample["license_evidence_kind"],
                "git_blob_sha1": actual_blob,
                "source_sha256": _sha256(payload),
                "normalized_sha256": _sha256(normalized_bytes),
                "source_bytes": len(payload),
                "normalized_bytes": len(normalized_bytes),
                "byte_identical": payload == normalized_bytes,
                "normalization": evidence.manifest(),
                "parser": parser,
            }
        )

    old_nfkc_probe = 'café = "① ﬁ K"\r\n\t# exact\r\n'
    import unicodedata

    old_nfkc_output = unicodedata.normalize(
        "NFKC",
        old_nfkc_probe.replace("\r\n", "\n").replace("\r", "\n"),
    ).strip("\n")
    probe_normalized, probe_evidence = decode_code_bytes(
        old_nfkc_probe.encode("utf-8"),
        language="python",
        path="regression_probe.py",
    )

    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "source_sha": source_sha,
        "normalization_schema": CODE_NORMALIZATION_SCHEMA,
        "normalization_policy": CODE_NORMALIZATION_POLICY,
        "registry_sha256": _sha256(registry_bytes),
        "authority_boundary": registry["authority_boundary"],
        "real_samples": results,
        "summary": {
            "samples": len(results),
            "languages": sorted({item["language"] for item in results}),
            "source_bytes": total_source_bytes,
            "normalized_bytes": total_normalized_bytes,
            "changed_bytes": total_normalized_bytes - total_source_bytes,
            "byte_identical_samples": sum(item["byte_identical"] for item in results),
            "normalization_reason_counts": dict(sorted(reason_counts.items())),
            "parser_counts": dict(sorted(parser_counts.items())),
        },
        "regression_probe": {
            "old_data10_nfkc_would_change_source": old_nfkc_output != old_nfkc_probe,
            "new_policy_byte_identical": probe_normalized == old_nfkc_probe,
            "source_sha256": probe_evidence.source_sha256,
            "normalized_sha256": probe_evidence.normalized_sha256,
        },
        "truth_boundary": {
            "semantic_rewriting_performed": False,
            "training_eligibility_inferred_from_license_evidence": False,
            "real_samples_used_for_training": False,
            "unavailable_parsers_are_reported_not_invented": True,
        },
    }
    core["report_sha256"] = _sha256(_canonical_json_bytes(core))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json_bytes(core))
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/data/code_fidelity_real_samples.v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data28-code-fidelity-evidence.json"),
    )
    args = parser.parse_args()
    try:
        report = run(
            source_sha=args.source_sha,
            registry_path=args.registry,
            output_path=args.output,
        )
    except (CodeNormalizationError, OSError, ValueError, RuntimeError) as exc:
        print(f"DATA-28 fidelity failure: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], sort_keys=True, ensure_ascii=False))
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
