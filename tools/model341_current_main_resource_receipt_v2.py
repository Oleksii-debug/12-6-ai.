from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

RECEIPT_SCHEMA = "12-6.model341.current-main-local-free-resource-receipt.v2"
RECEIPT_STATUS = "CAPTURED_SYNTHETIC_MECHANICS_ONLY"
PROBE_TOOL_RELATIVE_PATH = Path("tools/model341_current_main_resource_envelope_v2.py")
FOCUSED_TEST_RELATIVE_PATH = Path("tests/test_model341_current_main_resource_envelope_v2.py")
REPORT_RELATIVE_PATH = Path("reports/model341_current_main_resource_receipt_v2.json")
CURRENT_MODEL_RELATIVE_PATH = Path("src/twelve_six/model.py")
CURRENT_PYPROJECT_RELATIVE_PATH = Path("pyproject.toml")

EXPECTED_PROBE_TOOL_BLOB_SHA1 = "f4476ea9bb7c9d400a133d56df5c311673bf3c85"
EXPECTED_FOCUSED_TEST_BLOB_SHA1 = "d3d1a34b1bdbd0f38b4f9304852591933c7d5758"
EXPECTED_MODEL_BLOB_SHA1 = "c3879fe0ba9193d5a8176c284e1942f058ef7885"
EXPECTED_RUNTIME_PROJECT = {
    "requires-python": ">=3.11",
    "dependencies": ["numpy>=1.26", "safetensors>=0.5", "torch>=2.5"],
}
CAPTURE_PRECOMMIT_COMMENT_ID: int | None = 5887435637

# Terminal v2 measurement authority is intentionally fail-closed until one exact
# shared-CI execution has been captured and independently prepublished.
CAPTURE_AUTHORITY_PUBLISHED = True
EXPECTED_CAPTURE: dict[str, Any] | None = {
    "repository": "Oleksii-debug/12-6-ai.",
    "issue": 2280,
    "pull_request": 2281,
    "workflow_name": "CI",
    "workflow_run_id": 36549101208,
    "workflow_run_number": 4618,
    "job_id": 109342583358,
    "head_sha": "fd70099fe8a9bf44501762fa8c612624b597b132",
    "base_sha": "b8257d174d11bf1d61728229312659f5eb3305b8",
    "tested_merge_sha": "09f55ddc488e79a9d40d8b14f8e6aefd9e459f10",
    "probe_tool_blob_sha1": EXPECTED_PROBE_TOOL_BLOB_SHA1,
    "focused_test_blob_sha1": EXPECTED_FOCUSED_TEST_BLOB_SHA1,
    "workflow_conclusion": "success",
    "job_conclusion": "success",
    "capture_line_count": 1,
}
EXPECTED_PROBE_REPORT_SHA256: str | None = "f0e2b7024abaac32976ffe2d38b6dd15aedf1a7e7608be142ee599e690f2dbe8"
PREPUBLISHED_MEASUREMENT_AUTHORITY_SHA256: str | None = "a784ca37cef5b2aa419a9203665c7b4ee9000fc7deb1df05703bd2e89319bc68"

_EXPECTED_TOP_LEVEL_KEYS = {
    "capture",
    "probe_report",
    "probe_report_sha256",
    "schema",
    "status",
}
_EXPECTED_CAPTURE_KEYS = {
    "repository",
    "issue",
    "pull_request",
    "workflow_name",
    "workflow_run_id",
    "workflow_run_number",
    "job_id",
    "head_sha",
    "base_sha",
    "tested_merge_sha",
    "probe_tool_blob_sha1",
    "focused_test_blob_sha1",
    "workflow_conclusion",
    "job_conclusion",
    "capture_line_count",
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


def runtime_project_projection(pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject.get("project")
    if type(project) is not dict:
        raise ValueError("pyproject project table missing")
    requires_python = project.get("requires-python")
    dependencies = project.get("dependencies")
    if type(requires_python) is not str:
        raise ValueError("pyproject requires-python type mismatch")
    if type(dependencies) is not list or any(type(item) is not str for item in dependencies):
        raise ValueError("pyproject dependencies type mismatch")
    return {
        "requires-python": requires_python,
        "dependencies": list(dependencies),
    }


def validate_current_checkout_compatibility(root: Path) -> None:
    model_path = root / CURRENT_MODEL_RELATIVE_PATH
    if git_blob_sha1(model_path) != EXPECTED_MODEL_BLOB_SHA1:
        raise ValueError("current checkout model.py identity mismatch")

    pyproject_path = root / CURRENT_PYPROJECT_RELATIVE_PATH
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    projection = runtime_project_projection(pyproject)
    if projection != EXPECTED_RUNTIME_PROJECT:
        raise ValueError("current checkout runtime project dependency projection mismatch")


def _validate_capture(capture: dict[str, Any]) -> None:
    if set(capture) != _EXPECTED_CAPTURE_KEYS:
        raise ValueError("v2 capture key set mismatch")

    exact_int_fields = (
        "issue",
        "pull_request",
        "workflow_run_id",
        "workflow_run_number",
        "job_id",
        "capture_line_count",
    )
    for name in exact_int_fields:
        if type(capture.get(name)) is not int:
            raise ValueError(f"v2 capture {name} must be an exact integer")

    exact_string_fields = (
        "repository",
        "workflow_name",
        "head_sha",
        "base_sha",
        "tested_merge_sha",
        "probe_tool_blob_sha1",
        "focused_test_blob_sha1",
        "workflow_conclusion",
        "job_conclusion",
    )
    for name in exact_string_fields:
        if type(capture.get(name)) is not str:
            raise ValueError(f"v2 capture {name} must be an exact string")

    if capture["repository"] != "Oleksii-debug/12-6-ai.":
        raise ValueError("v2 capture repository mismatch")
    if capture["issue"] != 2280 or capture["pull_request"] != 2281:
        raise ValueError("v2 capture control identity mismatch")
    if capture["workflow_name"] != "CI":
        raise ValueError("v2 capture workflow mismatch")
    if capture["probe_tool_blob_sha1"] != EXPECTED_PROBE_TOOL_BLOB_SHA1:
        raise ValueError("v2 capture probe blob mismatch")
    if capture["focused_test_blob_sha1"] != EXPECTED_FOCUSED_TEST_BLOB_SHA1:
        raise ValueError("v2 capture focused test blob mismatch")
    if capture["workflow_conclusion"] != "success":
        raise ValueError("v2 capture workflow conclusion mismatch")
    if capture["job_conclusion"] != "success":
        raise ValueError("v2 capture job conclusion mismatch")
    if capture["capture_line_count"] != 1:
        raise ValueError("v2 capture line count must be exact integer one")


def _require_published_capture_authority() -> None:
    if CAPTURE_AUTHORITY_PUBLISHED is not True:
        raise ValueError("v2 capture authority is not published")
    if type(EXPECTED_CAPTURE) is not dict:
        raise ValueError("v2 expected capture authority is missing")
    if type(CAPTURE_PRECOMMIT_COMMENT_ID) is not int or CAPTURE_PRECOMMIT_COMMENT_ID <= 0:
        raise ValueError("v2 capture precommit comment id is missing")
    _validate_capture(EXPECTED_CAPTURE)
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
        "capture_precommit_comment_id": CAPTURE_PRECOMMIT_COMMENT_ID,
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
    # Authenticate all current executable/runtime bytes before dynamic import.
    validate_probe_artifacts(root)
    validate_current_checkout_compatibility(root)
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
