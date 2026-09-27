from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tools.model341_current_main_resource_envelope_v1 import git_blob_sha1, validate_probe

RECEIPT_SCHEMA = "12-6.model341.current-main-local-free-resource-receipt.v1"
RECEIPT_STATUS = "CAPTURED_SYNTHETIC_MECHANICS_ONLY"
REPORT_RELATIVE_PATH = Path("reports/model341_current_main_resource_envelope_v1.json")
PROBE_TOOL_RELATIVE_PATH = Path("tools/model341_current_main_resource_envelope_v1.py")
EXPECTED_PROBE_REPORT_SHA256 = (
    "630c985cc09a550f2eca786c39540fa7b98e3b088e6eaded06422703ed631a9e"
)
EXPECTED_PROBE_TOOL_BLOB_SHA1 = "6a668c6885c8578b23cfc50d94c6add149b3d978"
EXPECTED_MEASUREMENT_TEST_BLOB_SHA1 = "6518994628abaab6bca302278ee8f3a02bde6e61"
MEASUREMENT_AUTHORITY_COMMENT_ID = 5851221028
MEASUREMENT_AUTHORITY_SHA256 = (
    "1eb6a61cd9eeec79ef557ff231873092ad289a60ccd17a616678332b27f7332f"
)
EXPECTED_CAPTURE = {
    "repository": "Oleksii-debug/12-6-ai.",
    "issue": 2166,
    "pull_request": 2171,
    "workflow_name": "CI",
    "workflow_run_id": 36281908773,
    "workflow_run_number": 4100,
    "job_id": 108515267981,
    "base_sha": "ac11574c3774c3a9aae52ac9543209b4fe8042cd",
    "probe_head_sha": "ccc007817b87e9f7d8c44fe32f58743cc8ad4251",
    "tested_merge_sha": "da47ada1132c2f2538d9d0e6a93eaa27ee33072d",
    "probe_tool_blob_sha1": EXPECTED_PROBE_TOOL_BLOB_SHA1,
    "ci_result": "success",
    "pytest_passed": 2134,
    "pytest_warnings": 1,
    "runner_os": "Ubuntu 24.04.5",
    "runner_image": "ubuntu-24.04",
    "runner_image_version": "20260920.314.1",
}
_EXPECTED_TOP_LEVEL_KEYS = {
    "capture",
    "probe_report",
    "probe_report_sha256",
    "schema",
    "status",
}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object member: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonfinite_constant,
    )


def measurement_authority_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority_schema": "12-6.model341.current-main-measurement-authority.v1",
        "claim_issue": EXPECTED_CAPTURE["issue"],
        "pr": EXPECTED_CAPTURE["pull_request"],
        "measurement_head_sha": EXPECTED_CAPTURE["probe_head_sha"],
        "measurement_merge_sha": EXPECTED_CAPTURE["tested_merge_sha"],
        "base_main_sha": EXPECTED_CAPTURE["base_sha"],
        "ci_run_id": EXPECTED_CAPTURE["workflow_run_id"],
        "ci_job_id": EXPECTED_CAPTURE["job_id"],
        "probe_tool_blob_sha1": EXPECTED_CAPTURE["probe_tool_blob_sha1"],
        "measurement_test_blob_sha1": EXPECTED_MEASUREMENT_TEST_BLOB_SHA1,
        "probe": receipt["probe_report"],
    }


def validate_probe_tool_blob(root: Path) -> None:
    path = root / PROBE_TOOL_RELATIVE_PATH
    actual = git_blob_sha1(path)
    if actual != EXPECTED_PROBE_TOOL_BLOB_SHA1:
        raise ValueError("captured probe tool blob mismatch")


def validate_receipt(receipt: dict[str, Any]) -> None:
    if set(receipt) != _EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("receipt top-level key set mismatch")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("receipt schema mismatch")
    if receipt.get("status") != RECEIPT_STATUS:
        raise ValueError("receipt status mismatch")

    capture = receipt.get("capture")
    if type(capture) is not dict or capture != EXPECTED_CAPTURE:
        raise ValueError("CI capture binding mismatch")

    stored_report_sha = receipt.get("probe_report_sha256")
    if stored_report_sha != EXPECTED_PROBE_REPORT_SHA256:
        raise ValueError("captured probe report identity mismatch")

    probe_report = receipt.get("probe_report")
    if type(probe_report) is not dict:
        raise ValueError("probe_report must be an object")
    actual_report_sha = canonical_json_sha256(probe_report)
    if actual_report_sha != EXPECTED_PROBE_REPORT_SHA256:
        raise ValueError("captured probe report content mismatch")

    validate_probe(probe_report)

    authority_sha = canonical_json_sha256(measurement_authority_payload(receipt))
    if authority_sha != MEASUREMENT_AUTHORITY_SHA256:
        raise ValueError("prepublished measurement authority mismatch")


def validate_receipt_file(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    receipt = strict_json_loads(path.read_text(encoding="utf-8"))
    if type(receipt) is not dict:
        raise ValueError("receipt must be an object")
    validate_receipt(receipt)
    if root is not None:
        validate_probe_tool_blob(root)
    return receipt


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validate_receipt_file(root / REPORT_RELATIVE_PATH, root=root)
    print("MODEL341_CURRENT_MAIN_RESOURCE_RECEIPT=VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
