#!/usr/bin/env python3
"""Executable fail-closed balance/diversity gate for NEXT100-106."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/data/next100_106_balance_gate_policy_v1.json"

INPUT_SCHEMA = "12-6.next100-106-post-dedup-family-vector.v1"
STRATA = ("ua", "en", "code")
MAX_BALANCE_JSON_BYTES = 1_048_576
MAX_BALANCE_JSON_DEPTH = 64
MAX_BALANCE_JSON_NODES = 10_000
MAX_BALANCE_INT_DIGITS = 64
EXPECTED_POLICY_IDENTITY_SHA256 = (
    "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
)


class GateError(ValueError):
    """Raised when an input cannot be trusted for balance evaluation."""


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Any:
    raise GateError(f"non-finite JSON constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GateError("JSON number is not finite")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise GateError("nonzero JSON number underflowed to zero")
    return parsed


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_BALANCE_INT_DIGITS:
        raise GateError("JSON integer exceeds digit limit")
    return int(value)


def load_json(path: Path) -> dict[str, Any]:
    """Bound and strictly decode policy or external family-vector evidence."""
    with path.open("rb") as source:
        raw = source.read(MAX_BALANCE_JSON_BYTES + 1)
    if len(raw) > MAX_BALANCE_JSON_BYTES:
        raise GateError("balance JSON exceeds byte limit")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_object,
            parse_constant=_reject_nonfinite_constant,
            parse_float=_parse_finite_float,
            parse_int=_parse_bounded_int,
        )
        pending = [(value, 0)]
        nodes = 0
        while pending:
            current, depth = pending.pop()
            nodes += 1
            if nodes > MAX_BALANCE_JSON_NODES or depth > MAX_BALANCE_JSON_DEPTH:
                raise GateError("balance JSON structure limit exceeded")
            if isinstance(current, dict):
                for key, child in current.items():
                    key.encode("utf-8")
                    pending.append((child, depth + 1))
            elif isinstance(current, list):
                pending.extend((child, depth + 1) for child in current)
            elif isinstance(current, str):
                current.encode("utf-8")
    except RecursionError as exc:
        raise GateError("balance JSON nesting limit exceeded") from exc
    except (UnicodeError, ValueError) as exc:
        raise GateError(f"invalid balance JSON: {exc}") from exc

    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def canonical_sha(data: dict[str, Any], identity_key: str) -> str:
    body = dict(data)
    body.pop(identity_key, None)
    payload = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{field} must be a non-negative integer")
    return value


def _hex(value: Any, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise GateError(f"{field} must be {length} lowercase hex characters")
    if value.lower() != value or any(ch not in "0123456789abcdef" for ch in value):
        raise GateError(f"{field} must be {length} lowercase hex characters")
    return value


def _require_exact_policy_int(value: Any, expected: int, field: str) -> None:
    if type(value) is not int or value != expected:
        raise GateError(f"{field} drift or invalid integer type")


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != "12-6.next100-106-balance-gate-policy.v1":
        raise GateError("unexpected policy schema")
    if policy.get("worker_id") != "NEXT100-106-EXECUTABLE-BALANCE-GATE-V3":
        raise GateError("unexpected worker id")
    if policy.get("execution_class") != "LOCAL_FREE":
        raise GateError("policy must remain LOCAL_FREE")
    if canonical_sha(policy, "policy_identity_sha256") != policy.get(
        "policy_identity_sha256"
    ):
        raise GateError("policy identity mismatch")

    cfg = policy.get("policy")
    expected_cfg_keys = {
        "target_total_source_bytes",
        "strata",
        "minimum_independent_families_per_stratum",
        "max_family_fraction_total",
        "max_family_fraction_own_stratum",
        "budget_quantum_bytes",
        "replay_or_duplication_to_meet_quota",
        "model_result_guided_mixture_retuning",
    }
    if not isinstance(cfg, dict) or set(cfg) != expected_cfg_keys:
        raise GateError("balance policy fields drift")
    _require_exact_policy_int(
        cfg.get("target_total_source_bytes"), 20_000_000, "20M source-byte planning target"
    )
    _require_exact_policy_int(
        cfg.get("minimum_independent_families_per_stratum"), 2, "minimum family count"
    )
    _require_exact_policy_int(cfg.get("budget_quantum_bytes"), 100, "budget quantum")
    if cfg["replay_or_duplication_to_meet_quota"] is not False:
        raise GateError("replay must remain forbidden")
    if cfg["model_result_guided_mixture_retuning"] is not False:
        raise GateError("model-result-guided mixture retuning must remain forbidden")

    strata = cfg.get("strata")
    if not isinstance(strata, dict) or set(strata) != set(STRATA):
        raise GateError("45/35/20 mixture drift")
    for stratum, (numerator, denominator) in {
        "ua": (9, 20),
        "en": (7, 20),
        "code": (1, 5),
    }.items():
        row = strata[stratum]
        if not isinstance(row, dict) or set(row) != {
            "target_numerator", "target_denominator"
        }:
            raise GateError("45/35/20 mixture drift")
        _require_exact_policy_int(
            row["target_numerator"], numerator, f"{stratum} mixture numerator"
        )
        _require_exact_policy_int(
            row["target_denominator"], denominator, f"{stratum} mixture denominator"
        )

    for key, numerator, denominator, label in (
        ("max_family_fraction_total", 1, 4, "global family cap"),
        ("max_family_fraction_own_stratum", 3, 5, "within-stratum family cap"),
    ):
        fraction = cfg.get(key)
        if not isinstance(fraction, dict) or set(fraction) != {
            "numerator", "denominator"
        }:
            raise GateError(f"{label} drift")
        _require_exact_policy_int(
            fraction["numerator"], numerator, f"{label} numerator"
        )
        _require_exact_policy_int(
            fraction["denominator"], denominator, f"{label} denominator"
        )

    boundary = policy.get("claim_boundary")
    expected_boundary = {
        "computes_source_mixture_feasibility_only": True,
        "creates_corpus_identity": False,
        "creates_shard_identity": False,
        "authorizes_tokenizer_fit": False,
        "authorizes_model_training": False,
        "authorizes_paid_compute": False,
        "relabels_source_bytes_as_loss_positions": False,
    }
    if not isinstance(boundary, dict) or set(boundary) != set(expected_boundary):
        raise GateError("claim boundary drift")
    if any(
        type(boundary[key]) is not bool or boundary[key] is not expected
        for key, expected in expected_boundary.items()
    ):
        raise GateError("claim boundary drift")
    if policy.get("policy_identity_sha256") != EXPECTED_POLICY_IDENTITY_SHA256:
        raise GateError("policy identity differs from pinned authority")


def validate_vector(vector: dict[str, Any]) -> list[dict[str, Any]]:
    if vector.get("schema_version") != INPUT_SCHEMA:
        raise GateError(f"unexpected input schema; expected {INPUT_SCHEMA}")
    if vector.get("terminal") is not True:
        raise GateError("post-dedup vector is nonterminal")

    authority = vector.get("dedup_authority")
    if not isinstance(authority, dict):
        raise GateError("dedup_authority must be an object")
    worker_id = authority.get("worker_id")
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise GateError("dedup authority worker_id is required")
    _hex(authority.get("head_sha"), 40, "dedup_authority.head_sha")
    _hex(
        authority.get("evidence_identity_sha256"),
        64,
        "dedup_authority.evidence_identity_sha256",
    )
    if authority.get("terminal_verdict") != "PASS":
        raise GateError("dedup authority terminal verdict must be PASS")

    families = vector.get("families")
    if not isinstance(families, list) or not families:
        raise GateError("families must be a non-empty list")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    by_stratum = defaultdict(int)
    counts = defaultdict(int)
    for index, item in enumerate(families):
        if not isinstance(item, dict):
            raise GateError(f"families[{index}] must be an object")
        family_id = item.get("family_id")
        if not isinstance(family_id, str) or not family_id.strip():
            raise GateError(f"families[{index}].family_id is required")
        if family_id in seen:
            raise GateError(f"duplicate family_id would create replay-like credit: {family_id}")
        seen.add(family_id)

        stratum = item.get("stratum")
        if stratum not in STRATA:
            raise GateError(f"invalid stratum for {family_id}: {stratum!r}")
        capacity = _positive_int(
            item.get("unique_bytes"), f"families[{index}].unique_bytes"
        )
        normalized.append(
            {"family_id": family_id, "stratum": stratum, "unique_bytes": capacity}
        )
        by_stratum[stratum] += capacity
        counts[stratum] += 1

    declared = vector.get("totals")
    if not isinstance(declared, dict):
        raise GateError("totals must be an object")
    if _nonnegative_int(declared.get("total_unique_bytes"), "totals.total_unique_bytes") != sum(
        by_stratum.values()
    ):
        raise GateError("declared total_unique_bytes mismatch")
    by_stratum_claim = declared.get("by_stratum")
    if (
        not isinstance(by_stratum_claim, dict)
        or set(by_stratum_claim) != set(STRATA)
        or any(
            type(by_stratum_claim[key]) is not int
            or by_stratum_claim[key] != by_stratum[key]
            for key in STRATA
        )
    ):
        raise GateError("declared by_stratum totals mismatch")
    family_count_claim = declared.get("family_count")
    if (
        not isinstance(family_count_claim, dict)
        or set(family_count_claim) != set(STRATA)
        or any(
            type(family_count_claim[key]) is not int
            or family_count_claim[key] != counts[key]
            for key in STRATA
        )
    ):
        raise GateError("declared family_count mismatch")

    return sorted(normalized, key=lambda item: item["family_id"])


def _ratio(cfg: dict[str, Any], stratum: str) -> tuple[int, int]:
    row = cfg["strata"][stratum]
    return row["target_numerator"], row["target_denominator"]


def stratum_target(total: int, cfg: dict[str, Any], stratum: str) -> int:
    numerator, denominator = _ratio(cfg, stratum)
    product = total * numerator
    if product % denominator:
        raise GateError("total does not produce exact integer stratum targets")
    return product // denominator


def family_limit(total: int, stratum_bytes: int, cfg: dict[str, Any]) -> int:
    global_cap = cfg["max_family_fraction_total"]
    own_cap = cfg["max_family_fraction_own_stratum"]
    global_numerator = total * global_cap["numerator"]
    own_numerator = stratum_bytes * own_cap["numerator"]
    if global_numerator % global_cap["denominator"]:
        raise GateError("budget quantum does not preserve exact global family cap")
    if own_numerator % own_cap["denominator"]:
        raise GateError("budget quantum does not preserve exact within-stratum family cap")
    return min(
        global_numerator // global_cap["denominator"],
        own_numerator // own_cap["denominator"],
    )


def feasible(
    total: int, families: list[dict[str, Any]], cfg: dict[str, Any]
) -> bool:
    if total <= 0:
        return False
    quantum = cfg["budget_quantum_bytes"]
    if total % quantum:
        raise GateError("candidate total must align to budget quantum")

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATA}
    for family in families:
        grouped[family["stratum"]].append(family)

    minimum = cfg["minimum_independent_families_per_stratum"]
    for stratum in STRATA:
        if len(grouped[stratum]) < minimum:
            return False
        required = stratum_target(total, cfg, stratum)
        cap = family_limit(total, required, cfg)
        available_under_caps = sum(
            min(family["unique_bytes"], cap) for family in grouped[stratum]
        )
        if available_under_caps < required:
            return False
    return True


def maximum_feasible_total(
    families: list[dict[str, Any]], cfg: dict[str, Any]
) -> int:
    target = cfg["target_total_source_bytes"]
    quantum = cfg["budget_quantum_bytes"]
    high = target // quantum
    low = 0

    # Feasibility is monotone in this capped no-replay setting: once finite
    # family capacities cannot support a larger exact mixture, increasing the
    # requested budget cannot restore missing capacity.
    while low < high:
        mid = (low + high + 1) // 2
        candidate = mid * quantum
        if feasible(candidate, families, cfg):
            low = mid
        else:
            high = mid - 1
    return low * quantum


def deterministic_allocation(
    total: int, families: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    if not feasible(total, families, cfg):
        raise GateError("cannot allocate an infeasible mixture")

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in STRATA}
    for family in families:
        grouped[family["stratum"]].append(family)

    allocations: list[dict[str, Any]] = []
    for stratum in STRATA:
        required = stratum_target(total, cfg, stratum)
        per_family_cap = family_limit(total, required, cfg)
        remaining = required
        candidates = sorted(
            grouped[stratum],
            key=lambda item: (-min(item["unique_bytes"], per_family_cap), item["family_id"]),
        )
        for family in candidates:
            take = min(family["unique_bytes"], per_family_cap, remaining)
            if take:
                allocations.append(
                    {
                        "family_id": family["family_id"],
                        "stratum": stratum,
                        "allocated_bytes": take,
                        "available_unique_bytes": family["unique_bytes"],
                        "effective_family_cap_bytes": per_family_cap,
                    }
                )
                remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise GateError("allocation invariant failed despite feasibility proof")
    return sorted(allocations, key=lambda item: item["family_id"])


def evaluate(
    policy: dict[str, Any], vector: dict[str, Any]
) -> dict[str, Any]:
    validate_policy(policy)
    families = validate_vector(vector)
    cfg = policy["policy"]

    by_stratum_capacity = {
        stratum: sum(
            family["unique_bytes"]
            for family in families
            if family["stratum"] == stratum
        )
        for stratum in STRATA
    }
    family_count = {
        stratum: sum(1 for family in families if family["stratum"] == stratum)
        for stratum in STRATA
    }
    minimum = cfg["minimum_independent_families_per_stratum"]
    family_minimum_pass = all(family_count[key] >= minimum for key in STRATA)

    maximum = maximum_feasible_total(families, cfg) if family_minimum_pass else 0
    target = cfg["target_total_source_bytes"]
    if maximum == target:
        status = "TARGET_20M_SOURCE_MIX_FEASIBLE"
    elif maximum > 0:
        status = "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    else:
        status = "BLOCKED_NO_NONZERO_POLICY_COMPLIANT_MIXTURE"

    allocations = deterministic_allocation(maximum, families, cfg) if maximum else []
    max_targets = (
        {key: stratum_target(maximum, cfg, key) for key in STRATA}
        if maximum
        else {key: 0 for key in STRATA}
    )
    target_targets = {key: stratum_target(target, cfg, key) for key in STRATA}

    result: dict[str, Any] = {
        "schema_version": "12-6.next100-106-balance-gate-result.v1",
        "policy_identity_sha256": policy["policy_identity_sha256"],
        "dedup_authority": vector["dedup_authority"],
        "input_totals": vector["totals"],
        "family_minimum": {
            "required_per_stratum": minimum,
            "observed": family_count,
            "pass": family_minimum_pass,
        },
        "maximum_feasible_total_source_bytes": maximum,
        "maximum_feasible_stratum_bytes": max_targets,
        "target_total_source_bytes": target,
        "target_stratum_bytes": target_targets,
        "raw_capacity_by_stratum": by_stratum_capacity,
        "raw_gap_to_target_by_stratum": {
            key: max(0, target_targets[key] - by_stratum_capacity[key])
            for key in STRATA
        },
        "deterministic_maximum_allocation": allocations,
        "status": status,
        "next_step": (
            "IMMUTABLE_CORPUS_MATERIALIZATION_STILL_REQUIRES_QUALITY_PRIVACY_"
            "DECONTAMINATION_SPLIT_PACK_AND_TWO_CLEAN_BUILD_GATES"
        ),
        "claim_boundary": {
            "authorized_training_exposure_loss_positions": 0,
            "corpus_identity": None,
            "shard_identity": None,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "source_bytes_are_loss_positions": False,
        },
    }
    result["result_identity_sha256"] = canonical_sha(
        result, "result_identity_sha256"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate-policy")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("input", type=Path)
    evaluate_parser.add_argument("--output", type=Path)

    return parser.parse_args()


def _write_staged_bytes(destination: BinaryIO, payload: bytes) -> None:
    if destination.write(payload) != len(payload):
        raise OSError("incomplete staged balance output write")
    destination.flush()
    os.fsync(destination.fileno())


def _same_inode(path: Path, identity: tuple[int, int]) -> bool:
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (info.st_dev, info.st_ino) == identity


def _payload_matches(path: Path, identity: tuple[int, int], payload: bytes) -> bool:
    if not _same_inode(path, identity):
        return False
    try:
        with path.open("rb") as source:
            info = os.fstat(source.fileno())
            return (
                (info.st_dev, info.st_ino) == identity
                and source.read(len(payload) + 1) == payload
            )
    except OSError:
        return False


def _write_new_output(path: Path, payload: bytes, *, input_path: Path) -> None:
    """Publish complete bytes once; require a trusted and stable parent directory.

    Portable checks cannot protect against hostile simultaneous same-user writers
    or ancestor renames. Native Windows/NTFS validation remains outstanding.
    """
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to overwrite existing balance output: {path}")
    try:
        parent = path.parent.absolute()
        if parent != parent.resolve(strict=True):
            raise GateError("balance output parent must have no symlink aliases")
        final = parent / path.name
        protected = {POLICY_PATH.resolve(), input_path.resolve()}
        if final in protected:
            raise GateError("balance output must not alias policy or input")
    except (OSError, RuntimeError) as exc:
        raise GateError("balance output path cannot be resolved safely") from exc

    staged_path: Path | None = None
    identity: tuple[int, int] | None = None
    linked = False
    verified = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=parent, delete=False,
        ) as destination:
            staged_path = Path(destination.name)
            stat = os.fstat(destination.fileno())
            identity = (stat.st_dev, stat.st_ino)
            if not identity[1]:
                raise OSError("filesystem has no stable staged file identity")
            _write_staged_bytes(destination, payload)
        if not _payload_matches(staged_path, identity, payload):
            raise GateError("staged balance output failed byte verification")
        os.link(staged_path, final)  # Create-only; never replace a concurrent result.
        linked = True
        if (
            not _payload_matches(final, identity, payload)
            or not _payload_matches(staged_path, identity, payload)
            or parent != path.parent.absolute()
            or parent != path.parent.resolve(strict=True)
        ):
            raise GateError("published balance output failed byte/path verification")
        verified = True
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite existing balance output: {path}") from exc
    finally:
        # Keep the original publication failure when cleanup or rollback also fails.
        primary_failure = sys.exc_info()[1]
        rollback_error: OSError | None = None
        cleanup_error: OSError | None = None
        if linked and not verified and identity is not None and _same_inode(final, identity):
            try:
                final.unlink()
            except OSError as exc:
                rollback_error = exc
        if staged_path is not None and identity is not None and _same_inode(staged_path, identity):
            try:
                staged_path.unlink()
            except OSError as exc:
                cleanup_error = exc
        if rollback_error is not None:
            stranded_stage = (
                f"; staged cleanup also failed: {staged_path}: {cleanup_error}"
                if cleanup_error is not None else ""
            )
            initial = (
                f"; initial publication failure: {primary_failure}"
                if primary_failure is not None else ""
            )
            raise GateError(
                f"ROLLBACK_INCOMPLETE: invalid balance output may remain: {final}"
                f"; rollback failure: {rollback_error}{stranded_stage}{initial}"
            ) from (primary_failure if primary_failure is not None else rollback_error)
        if cleanup_error is not None:
            if verified:
                # The final report was byte-verified and is already committed.
                print(
                    "OUTPUT_COMMITTED_CLEANUP_PENDING: "
                    + json.dumps(
                        {"output": str(final), "stage": str(staged_path)},
                        ensure_ascii=True, sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            else:
                initial = (
                    f"; initial publication failure: {primary_failure}"
                    if primary_failure is not None else ""
                )
                raise GateError(
                    f"STAGING_CLEANUP_INCOMPLETE: unpublished stage may remain: "
                    f"{staged_path}; cleanup failure: {cleanup_error}{initial}"
                ) from (primary_failure if primary_failure is not None else cleanup_error)


def main() -> int:
    args = parse_args()
    try:
        policy = load_json(POLICY_PATH)
        validate_policy(policy)

        if args.command == "validate-policy":
            print(
                "NEXT100-106 policy PASS "
                f"identity={policy['policy_identity_sha256']}"
            )
            return 0

        vector = load_json(args.input)
        result = evaluate(policy, vector)
        payload = (
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )
        if args.output:
            _write_new_output(
                args.output, payload.encode("utf-8"), input_path=args.input
            )
        else:
            print(payload, end="")
        return 0
    except (
        OSError, ValueError, TypeError, UnicodeError, RecursionError,
        OverflowError, RuntimeError,
    ) as exc:
        print(json.dumps({"status": "BLOCKED_INVALID_INPUT", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
