from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

RECEIPT_SCHEMA = "12-6.model341.current-main-local-free-resource-receipt.v2"
RECEIPT_STATUS = "CAPTURED_SYNTHETIC_MECHANICS_ONLY"
PROBE_TOOL_RELATIVE_PATH = Path("tools/model341_current_main_resource_envelope_v2.py")
FOCUSED_TEST_RELATIVE_PATH = Path("tests/test_model341_current_main_resource_envelope_v2.py")
REPORT_RELATIVE_PATH = Path("reports/model341_current_main_resource_receipt_v2.json")

EXPECTED_PROBE_TOOL_BLOB_SHA1 = "f4476ea9bb7c9d400a133d56df5c311673bf3c85"
EXPECTED_FOCUSED_TEST_BLOB_SHA1 = "14088f9a996cbadbb8d9e443a68c18fdd43af1d6"

# Terminal v2 measurement authority is intentionally fail-closed until one exact
# shared-CI execution has been captured and independently prepublished.
CAPTURE_AUTHORITY_PUBLISHED = False
EXPECTED_CAPTURE: dict[str, Any] | None = None
EXPECTED_PROBE_REPORT_SHA256: str | None = None
PREPUBLISHED_MEASUREMENT_AUTHORITY_SHA256: str | None = None

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


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in content:
        raise ValueError("source text contains unsupported bare CR")
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object member: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _strict_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number rejected: {value}")
    return number


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_strict_json_float,
    )


def validate_probe_artifacts(root: Path) -> None:
    probe_path = root / PROBE_TOOL_RELATIVE_PATH
    if git_blob_sha1(probe_path) != EXPECTED_PROBE_TOOL_BLOB_SHA1:
        raise ValueError("v2 probe tool blob mismatch")
    test_path = root / FOCUSED_TEST_RELATIVE_PATH
    if git_blob_sha1(test_path) != EXPECTED_FOCUSED_TEST_BLOB_SHA1:
        raise ValueError("v2 focused test blob mismatch")


def _require_published_capture_authority() -> None:
    if CAPTURE_AUTHORITY_PUBLISHED is not True:
        raise ValueError("v2 capture authority is not published")
    if type(EXPECTED_CAPTURE) is not dict:
        raise ValueError("v2 expected capture authority is missing")
    for name, value in (
        ("expected report SHA-256", EXPECTED_PROBE_REPORT_SHA256),
        ("prepublished measurement authority SHA-256", PREPUBLISHED_MEASUREMENT_AUTHORITY_SHA256),
    ):
        if type(value) is not str or len(value) != 64:
            raise ValueError(f"{name} is missing")
        if any(character not in "0123456789abcdef" for character in value.lower()):
            raise ValueError(f"{name} is malformed")


def measurement_authority_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "authority_schema": "12-6.model341.current-main-measurement-authority.v2",
        "capture": receipt["capture"],
        "probe_tool_blob_sha1": EXPECTED_PROBE_TOOL_BLOB_SHA1,
        "focused_test_blob_sha1": EXPECTED_FOCUSED_TEST_BLOB_SHA1,
        "probe_report_sha256": receipt["probe_report_sha256"],
        "probe": receipt["probe_report"],
    }


def _load_verified_probe_module(root: Path) -> ModuleType:
    probe_path = (root / PROBE_TOOL_RELATIVE_PATH).resolve()
    spec = importlib.util.spec_from_file_location(
        "_model341_v2_receipt_verified_probe",
        probe_path,
    )
    if spec is None or spec.loader is None:
        raise ValueError("unable to load verified v2 probe module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_file = getattr(module, "__file__", None)
    if module_file is None or Path(module_file).resolve() != probe_path:
        raise ValueError("verified v2 probe import path mismatch")
    return module


def validate_receipt(receipt: dict[str, Any], *, root: Path) -> None:
    # Verify executable/test bytes before trusting any live v2 validator code.
    validate_probe_artifacts(root)
    _require_published_capture_authority()

    if type(receipt) is not dict:
        raise ValueError("receipt must be an object")
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
        raise ValueError("captured v2 probe report identity mismatch")

    probe_report = receipt.get("probe_report")
    if type(probe_report) is not dict:
        raise ValueError("probe_report must be an object")
    actual_report_sha = canonical_json_sha256(probe_report)
    if actual_report_sha != EXPECTED_PROBE_REPORT_SHA256:
        raise ValueError("captured v2 probe report content mismatch")

    authority_sha = canonical_json_sha256(measurement_authority_payload(receipt))
    if authority_sha != PREPUBLISHED_MEASUREMENT_AUTHORITY_SHA256:
        raise ValueError("prepublished v2 measurement authority mismatch")

    verified_probe = _load_verified_probe_module(root)
    validate_probe = getattr(verified_probe, "validate_probe", None)
    if not callable(validate_probe):
        raise ValueError("verified v2 probe validator missing")
    validate_probe(probe_report)


def validate_receipt_file(path: Path, *, root: Path) -> dict[str, Any]:
    # Artifact identity and publication state are checked before reading candidate
    # measurement JSON so an unpublished candidate cannot become authority.
    validate_probe_artifacts(root)
    _require_published_capture_authority()
    receipt = strict_json_loads(path.read_text(encoding="utf-8"))
    if type(receipt) is not dict:
        raise ValueError("receipt must be an object")
    validate_receipt(receipt, root=root)
    return receipt


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validate_receipt_file(root / REPORT_RELATIVE_PATH, root=root)
    print("MODEL341_CURRENT_MAIN_RESOURCE_RECEIPT_V2=VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
