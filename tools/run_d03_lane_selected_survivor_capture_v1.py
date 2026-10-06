"""Capture exact selected post-G05/G06 survivors from a pinned execution lane.

The target lane executes unchanged. This wrapper temporarily intercepts its
already-materialized final-survivor return value, authenticates only the exact
missing IDs preregistered by the passed #3068 plan against the #3045 physical
composition, writes those raw rows as an ephemeral execution intermediate, and
emits a text-free zero-credit receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_d03_selected_raw_assembly_v1 as assembly_api
import run_d03_selected_raw_materialization_plan_v1 as plan_api

SCHEMA = "12-6.d03-lane-selected-survivor-capture.v1"
PLAN_IDENTITY = assembly_api.PLAN_IDENTITY
MAX_PLAN_BYTES = 2 * 1024 * 1024
MAX_COMPOSITION_BYTES = 128 * 1024 * 1024
RAW_KEYS = assembly_api.RAW_KEYS
_HEX40 = frozenset("0123456789abcdef")


class CaptureError(RuntimeError):
    """Raised when lane capture or selected-row authentication fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def blocked_constant(value: str) -> NoReturn:
    raise CaptureError(f"non-finite JSON constant: {value}")


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(value: bytes | Any) -> str:
    raw = value if isinstance(value, bytes) else canonical(value)
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def load_json(path: Path, *, max_bytes: int) -> dict[str, Any]:
    require(
        not path.is_symlink() and path.is_file(),
        f"JSON input is not a regular file: {path}",
    )
    raw = path.read_bytes()
    require(0 < len(raw) <= max_bytes, f"JSON input size invalid: {path}")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=strict_object,
        parse_constant=blocked_constant,
    )
    require(type(value) is dict, f"JSON root must be object: {path}")
    return value


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    require(completed.returncode == 0, "cannot resolve lane worktree HEAD")
    head = completed.stdout.strip()
    require(
        len(head) == 40 and set(head) <= _HEX40,
        "lane worktree HEAD is not a Git SHA-1",
    )
    return head


def _load_runner(
    runner_path: Path,
    *,
    expected_head: str,
    expected_blob: str,
) -> Any:
    require(
        len(expected_head) == 40 and set(expected_head) <= _HEX40,
        "expected lane head must be Git SHA-1",
    )
    require(
        len(expected_blob) == 40 and set(expected_blob) <= _HEX40,
        "expected runner blob must be Git SHA-1",
    )
    require(
        not runner_path.is_symlink() and runner_path.is_file(),
        "lane runner must be a regular file",
    )
    repo_root = runner_path.resolve().parents[1]
    require(_git_head(repo_root) == expected_head, "lane worktree HEAD drift")
    raw = runner_path.read_bytes()
    require(git_blob_sha1(raw) == expected_blob, "lane runner Git blob drift")
    tools_dir = str(runner_path.resolve().parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    module_name = f"_d03_lane_capture_{expected_blob}"
    require(module_name not in sys.modules, "lane runner namespace collision")
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    require(
        spec is not None and spec.loader is not None,
        "cannot load lane runner",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    require(callable(getattr(module, "main", None)), "lane runner main() missing")
    return module


def _resolve_hook(runner: Any, hook_path: str) -> tuple[Any, str, Any]:
    parts = hook_path.split(".")
    require(
        all(part.isidentifier() and not part.startswith("__") for part in parts),
        "hook path is invalid",
    )
    owner = runner
    for part in parts[:-1]:
        require(hasattr(owner, part), f"hook owner missing: {part}")
        owner = getattr(owner, part)
    name = parts[-1]
    require(hasattr(owner, name), f"hook target missing: {name}")
    original = getattr(owner, name)
    require(callable(original), "hook target is not callable")
    return owner, name, original


def _plan_target(
    plan: Mapping[str, Any],
    families: frozenset[str],
) -> tuple[frozenset[str], int]:
    assembly_api.verify_plan(plan)
    require(bool(families), "at least one required family is needed")
    family_plan = plan.get("family_materialization_plan")
    require(type(family_plan) is list, "family materialization plan missing")
    observed_families: set[str] = set()
    ids: set[str] = set()
    total_bytes = 0
    for raw in family_plan:
        require(type(raw) is dict, "family plan row must be object")
        family = raw.get("family")
        if family not in families:
            continue
        require(type(family) is str and family, "target family invalid")
        require(
            family not in observed_families,
            f"duplicate target family: {family}",
        )
        observed_families.add(family)
        record_ids = raw.get("missing_record_ids")
        require(
            type(record_ids) is list and record_ids,
            f"target IDs missing: {family}",
        )
        require(
            len(record_ids) == raw.get("missing_record_count"),
            f"target count drift: {family}",
        )
        for record_id in record_ids:
            require(
                type(record_id) is str and record_id,
                "target record ID invalid",
            )
            require(
                record_id not in ids,
                f"target record ID replay: {record_id}",
            )
            ids.add(record_id)
        family_bytes = raw.get("missing_payload_bytes")
        require(
            type(family_bytes) is int and family_bytes > 0,
            f"target bytes invalid: {family}",
        )
        total_bytes += family_bytes
    require(
        observed_families == set(families),
        "required family absent from passed plan",
    )
    require(bool(ids) and total_bytes > 0, "selected lane target is empty")
    return frozenset(ids), total_bytes


def _composition_authority(
    composition: Mapping[str, Any],
    target_ids: frozenset[str],
) -> dict[str, Mapping[str, Any]]:
    combined = composition.get("combined_inventory")
    require(
        type(combined) is dict,
        "composition combined inventory missing",
    )
    rows = combined.get("records")
    require(type(rows) is list and rows, "composition records missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        require(type(row) is dict, "composition row must be object")
        record_id = row.get("record_id")
        require(
            type(record_id) is str and record_id,
            "composition record ID invalid",
        )
        require(
            record_id not in by_id,
            f"composition record replay: {record_id}",
        )
        by_id[record_id] = row
    missing = target_ids - set(by_id)
    require(
        not missing,
        "target ID absent from composition: "
        + (min(missing) if missing else ""),
    )
    return {record_id: by_id[record_id] for record_id in target_ids}


def authenticate_capture(
    *,
    captured_rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    composition: Mapping[str, Any],
    families: frozenset[str],
    lane_head: str,
    runner_blob: str,
    hook_path: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    target_ids, target_bytes = _plan_target(plan, families)
    authority = _composition_authority(composition, target_ids)
    captured_by_id: dict[str, Mapping[str, Any]] = {}
    for number, row in enumerate(captured_rows, start=1):
        require(
            type(row) is dict,
            f"captured row {number} must be object",
        )
        require(
            set(row) == RAW_KEYS,
            f"captured row {number} schema drift",
        )
        record_id = row.get("record_id")
        require(
            type(record_id) is str and record_id,
            "captured record ID invalid",
        )
        require(
            record_id not in captured_by_id,
            f"captured record replay: {record_id}",
        )
        captured_by_id[record_id] = row
    absent = target_ids - set(captured_by_id)
    require(
        not absent,
        "selected target absent from lane survivors: "
        + (min(absent) if absent else ""),
    )

    output: list[dict[str, str]] = []
    family_counts: defaultdict[str, int] = defaultdict(int)
    family_bytes: defaultdict[str, int] = defaultdict(int)
    projection: list[dict[str, Any]] = []
    for record_id in sorted(target_ids):
        row = captured_by_id[record_id]
        expected = authority[record_id]
        for key in ("source_id", "family", "modality"):
            require(
                type(row.get(key)) is str
                and row[key] == expected.get(key),
                f"captured authority drift {record_id}:{key}",
            )
        require(
            row["family"] in families,
            f"captured family escaped lane target: {record_id}",
        )
        payload = row.get("normalized_payload")
        require(
            type(payload) is str and payload,
            f"captured payload invalid: {record_id}",
        )
        payload_raw = payload.encode("utf-8")
        require(
            len(payload_raw) == expected.get("payload_bytes"),
            f"captured payload byte drift: {record_id}",
        )
        payload_sha = sha256(payload_raw)
        require(
            payload_sha == expected.get("payload_sha256"),
            f"captured payload SHA drift: {record_id}",
        )
        clean = {key: str(row[key]) for key in RAW_KEYS}
        output.append(clean)
        family_counts[clean["family"]] += 1
        family_bytes[clean["family"]] += len(payload_raw)
        projection.append(
            {
                "record_id": record_id,
                "payload_sha256": payload_sha,
                "payload_bytes": len(payload_raw),
            }
        )

    observed_bytes = sum(item["payload_bytes"] for item in projection)
    require(
        observed_bytes == target_bytes,
        "selected lane payload bytes drift",
    )
    raw_jsonl = b"".join(canonical(row) + b"\n" for row in output)
    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "plan_identity_sha256": PLAN_IDENTITY,
        "lane_execution_head_sha": lane_head,
        "lane_runner_git_blob_sha1": runner_blob,
        "capture_hook_path": hook_path,
        "families": sorted(families),
        "selected_record_count": len(output),
        "selected_payload_bytes": observed_bytes,
        "selected_record_ids_sha256": sha256(sorted(target_ids)),
        "selected_payload_projection_sha256": sha256(projection),
        "family_record_counts": dict(sorted(family_counts.items())),
        "family_payload_bytes": dict(sorted(family_bytes.items())),
        "ephemeral_raw_jsonl_sha256": sha256(raw_jsonl),
        "raw_output_is_ephemeral": True,
        "durable_output_contains_raw_payload": False,
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    receipt = {
        **core,
        "capture_identity_sha256": sha256(core),
    }
    return output, receipt


def _write_create_only(path: Path, payload: bytes) -> None:
    require(
        not path.exists() and not path.is_symlink(),
        f"output already exists: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    require(
        not path.parent.is_symlink(),
        f"output parent is symlink: {path.parent}",
    )
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise CaptureError(
                f"output appeared concurrently: {path}"
            ) from exc
    finally:
        tmp.unlink(missing_ok=True)


def execute(
    *,
    runner_path: Path,
    expected_lane_head: str,
    expected_runner_blob: str,
    hook_path: str,
    survivor_index: int,
    families: frozenset[str],
    plan_json: Path,
    composition_json: Path,
    output_raw_jsonl: Path,
    output_receipt: Path,
    runner_args: Sequence[str],
) -> dict[str, Any]:
    require(
        type(survivor_index) is int and survivor_index >= 0,
        "survivor index invalid",
    )
    require(
        output_raw_jsonl != output_receipt,
        "raw and receipt outputs must differ",
    )
    runner = _load_runner(
        runner_path,
        expected_head=expected_lane_head,
        expected_blob=expected_runner_blob,
    )
    owner, name, original = _resolve_hook(runner, hook_path)
    captures: list[list[dict[str, Any]]] = []

    def intercepted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        require(
            isinstance(result, tuple),
            "hook result must be a tuple",
        )
        require(
            len(result) > survivor_index,
            "hook result lacks survivor slot",
        )
        rows = result[survivor_index]
        require(
            type(rows) is list and rows,
            "hook survivor slot must be non-empty list",
        )
        captures.append([dict(row) for row in rows])
        return result

    setattr(owner, name, intercepted)
    previous_argv = sys.argv
    try:
        sys.argv = [str(runner_path), *runner_args]
        try:
            status = runner.main()
        except SystemExit as exc:
            status = exc.code
    finally:
        sys.argv = previous_argv
        setattr(owner, name, original)
    require(
        type(status) is int and status == 0,
        f"lane runner failed with status {status!r}",
    )
    require(
        len(captures) == 1,
        f"expected one survivor capture, observed {len(captures)}",
    )

    plan = load_json(plan_json, max_bytes=MAX_PLAN_BYTES)
    composition = load_json(
        composition_json,
        max_bytes=MAX_COMPOSITION_BYTES,
    )
    output, receipt = authenticate_capture(
        captured_rows=captures[0],
        plan=plan,
        composition=composition,
        families=families,
        lane_head=expected_lane_head,
        runner_blob=expected_runner_blob,
        hook_path=hook_path,
    )
    raw_payload = b"".join(
        canonical(row) + b"\n" for row in output
    )
    receipt_payload = canonical(receipt) + b"\n"
    _write_create_only(output_raw_jsonl, raw_payload)
    try:
        _write_create_only(output_receipt, receipt_payload)
    except Exception:
        output_raw_jsonl.unlink(missing_ok=True)
        raise
    return receipt


def _split_cli(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    require("--" in argv, "lane runner arguments must follow --")
    index = argv.index("--")
    wrapper_args = list(argv[:index])
    runner_args = list(argv[index + 1 :])
    require(bool(runner_args), "lane runner arguments are empty")
    return wrapper_args, runner_args


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("--runner", type=Path, required=True)
    result.add_argument("--expected-lane-head", required=True)
    result.add_argument("--expected-runner-blob", required=True)
    result.add_argument("--hook-path", required=True)
    result.add_argument("--survivor-index", type=int, default=6)
    result.add_argument("--family", action="append", required=True)
    result.add_argument("--plan-json", type=Path, required=True)
    result.add_argument("--composition-json", type=Path, required=True)
    result.add_argument("--output-raw-jsonl", type=Path, required=True)
    result.add_argument("--output-receipt", type=Path, required=True)
    return result


def main() -> int:
    try:
        wrapper_argv, runner_argv = _split_cli(sys.argv[1:])
        args = parser().parse_args(wrapper_argv)
        require(
            len(args.family) == len(set(args.family)),
            "duplicate --family",
        )
        receipt = execute(
            runner_path=args.runner,
            expected_lane_head=args.expected_lane_head,
            expected_runner_blob=args.expected_runner_blob,
            hook_path=args.hook_path,
            survivor_index=args.survivor_index,
            families=frozenset(args.family),
            plan_json=args.plan_json,
            composition_json=args.composition_json,
            output_raw_jsonl=args.output_raw_jsonl,
            output_receipt=args.output_receipt,
            runner_args=runner_argv,
        )
    except (
        CaptureError,
        assembly_api.AssemblyError,
        plan_api.PlanError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("D03_LANE_SELECTED_SURVIVOR_CAPTURE=PASS_ZERO_CREDIT")
    print(
        "CAPTURE_IDENTITY_SHA256="
        f"{receipt['capture_identity_sha256']}"
    )
    print(
        "EPHEMERAL_RAW_JSONL_SHA256="
        f"{receipt['ephemeral_raw_jsonl_sha256']}"
    )
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
