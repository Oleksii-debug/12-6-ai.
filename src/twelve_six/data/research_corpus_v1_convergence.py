from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

MODALITIES = ("uk", "en", "code")
ALLOWED_VERDICTS = {"ADMIT", "ADMIT_PROSE_ONLY", "ADMIT_BOUNDED_PD_EDITION_SNAPSHOT"}
SHA256_LEN = 64
GIT_SHA_LEN = 40


class ConvergenceError(ValueError):
    """Raised when an authority-convergence contract is not internally sound."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ConvergenceError(f"{field} must be a {length}-character lowercase hex string")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise ConvergenceError(f"{field} must be lowercase hexadecimal")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConvergenceError(f"{field} must be a positive integer")
    return value


def _capacity_and_families(
    sources: list[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, set[str]]]:
    capacity = {modality: 0 for modality in MODALITIES}
    families = {modality: set() for modality in MODALITIES}
    seen_source_ids: set[str] = set()
    for index, source in enumerate(sources):
        prefix = f"base.sources[{index}]"
        source_id = source.get("source_id")
        family = source.get("source_family")
        modality = source.get("modality")
        if not isinstance(source_id, str) or not source_id:
            raise ConvergenceError(f"{prefix}.source_id must be non-empty")
        if source_id in seen_source_ids:
            raise ConvergenceError(f"duplicate base source_id: {source_id}")
        seen_source_ids.add(source_id)
        if modality not in MODALITIES:
            raise ConvergenceError(f"{prefix}.modality must be one of {MODALITIES}")
        if not isinstance(family, str) or not family:
            raise ConvergenceError(f"{prefix}.source_family must be non-empty")
        bytes_count = _positive_int(
            source.get("declared_capacity_bytes"),
            f"{prefix}.declared_capacity_bytes",
        )
        capacity[modality] += bytes_count
        families[modality].add(family)
    return capacity, families


def _authority_identity(authority: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "authority_id": authority["authority_id"],
        "pr_number": authority["pr_number"],
        "head_sha": authority["head_sha"],
        "workflow_run_id": authority["dedicated_workflow"]["run_id"],
        "workflow_name": authority["dedicated_workflow"]["name"],
        "verdict": authority["verdict"],
        "modality": authority["modality"],
        "source_family": authority["source_family"],
        "declared_capacity_bytes": authority["declared_capacity_bytes"],
        "authority_identity_kind": authority["authority_identity_kind"],
        "authority_identity_sha256": authority["authority_identity_sha256"],
    }
    if "normalized_sha256" in authority:
        identity["normalized_sha256"] = authority["normalized_sha256"]
    if "normalized_objects" in authority:
        identity["normalized_objects"] = authority["normalized_objects"]
    return identity


def validate_convergence(
    manifest: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_schema = "12-6.next100-078-research-corpus-v1-authority-convergence.v1"
    if manifest.get("schema_version") != expected_schema:
        raise ConvergenceError("unsupported manifest schema_version")
    if manifest.get("execution_class") != "LOCAL_FREE":
        raise ConvergenceError("execution_class must remain LOCAL_FREE")
    if manifest.get("training_executed") is not False:
        raise ConvergenceError("training_executed must remain false")
    if manifest.get("compute_authorized") is not False:
        raise ConvergenceError("compute_authorized must remain false")

    base = manifest.get("base_vector")
    if not isinstance(base, Mapping):
        raise ConvergenceError("base_vector must be an object")
    _require_hex(base.get("head_sha"), GIT_SHA_LEN, "base_vector.head_sha")
    sources = base_config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConvergenceError("base config must contain a non-empty sources array")

    base_capacity, base_families = _capacity_and_families(sources)
    actual_base_total = sum(base_capacity.values())
    if actual_base_total != base.get("expected_capacity_bytes"):
        raise ConvergenceError(
            "base capacity drift: "
            f"expected {base.get('expected_capacity_bytes')}, observed {actual_base_total}"
        )
    expected_family_counts = base.get("expected_family_counts")
    if not isinstance(expected_family_counts, Mapping):
        raise ConvergenceError("base_vector.expected_family_counts must be an object")
    actual_base_family_counts = {m: len(base_families[m]) for m in MODALITIES}
    expected_counts = {m: expected_family_counts.get(m) for m in MODALITIES}
    if actual_base_family_counts != expected_counts:
        raise ConvergenceError(
            f"base family-count drift: expected {expected_counts}, "
            f"observed {actual_base_family_counts}"
        )

    authorities = manifest.get("additive_authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ConvergenceError("additive_authorities must be a non-empty array")

    final_capacity = dict(base_capacity)
    final_families = {m: set(base_families[m]) for m in MODALITIES}
    seen_authorities: set[str] = set()
    seen_heads: set[str] = set()
    authority_identities: list[dict[str, Any]] = []

    for index, authority in enumerate(authorities):
        if not isinstance(authority, Mapping):
            raise ConvergenceError(f"additive_authorities[{index}] must be an object")
        prefix = f"additive_authorities[{index}]"
        authority_id = authority.get("authority_id")
        if not isinstance(authority_id, str) or not authority_id:
            raise ConvergenceError(f"{prefix}.authority_id must be non-empty")
        if authority_id in seen_authorities:
            raise ConvergenceError(f"duplicate authority_id: {authority_id}")
        seen_authorities.add(authority_id)

        head_sha = _require_hex(authority.get("head_sha"), GIT_SHA_LEN, f"{prefix}.head_sha")
        if head_sha in seen_heads:
            raise ConvergenceError(f"duplicate authority head: {head_sha}")
        seen_heads.add(head_sha)

        _positive_int(authority.get("pr_number"), f"{prefix}.pr_number")
        workflow = authority.get("dedicated_workflow")
        if not isinstance(workflow, Mapping):
            raise ConvergenceError(f"{prefix}.dedicated_workflow must be an object")
        if workflow.get("conclusion") != "success":
            raise ConvergenceError(f"{prefix} is not terminal dedicated-workflow success")
        if not isinstance(workflow.get("name"), str) or not workflow.get("name"):
            raise ConvergenceError(f"{prefix}.dedicated_workflow.name must be non-empty")
        _positive_int(workflow.get("run_id"), f"{prefix}.dedicated_workflow.run_id")

        verdict = authority.get("verdict")
        if verdict not in ALLOWED_VERDICTS:
            raise ConvergenceError(
                f"{prefix}.verdict is not an admitted training-source verdict: {verdict!r}"
            )
        modality = authority.get("modality")
        if modality not in MODALITIES:
            raise ConvergenceError(f"{prefix}.modality must be one of {MODALITIES}")
        family = authority.get("source_family")
        if not isinstance(family, str) or not family:
            raise ConvergenceError(f"{prefix}.source_family must be non-empty")
        if family in final_families[modality]:
            raise ConvergenceError(
                f"duplicate independent-family credit attempted: {modality}/{family}"
            )

        capacity = _positive_int(
            authority.get("declared_capacity_bytes"),
            f"{prefix}.declared_capacity_bytes",
        )
        _require_hex(
            authority.get("authority_identity_sha256"),
            SHA256_LEN,
            f"{prefix}.authority_identity_sha256",
        )
        if "normalized_sha256" in authority:
            _require_hex(
                authority.get("normalized_sha256"),
                SHA256_LEN,
                f"{prefix}.normalized_sha256",
            )
        if "normalized_objects" in authority:
            objects = authority.get("normalized_objects")
            if not isinstance(objects, list) or not objects:
                raise ConvergenceError(f"{prefix}.normalized_objects must be non-empty when present")
            object_bytes = 0
            seen_object_hashes: set[str] = set()
            for object_index, obj in enumerate(objects):
                if not isinstance(obj, Mapping):
                    raise ConvergenceError(
                        f"{prefix}.normalized_objects[{object_index}] must be an object"
                    )
                obj_hash = _require_hex(
                    obj.get("sha256"),
                    SHA256_LEN,
                    f"{prefix}.normalized_objects[{object_index}].sha256",
                )
                if obj_hash in seen_object_hashes:
                    raise ConvergenceError(f"{prefix} contains a duplicate normalized object hash")
                seen_object_hashes.add(obj_hash)
                object_bytes += _positive_int(
                    obj.get("bytes"),
                    f"{prefix}.normalized_objects[{object_index}].bytes",
                )
            if object_bytes != capacity:
                raise ConvergenceError(
                    f"{prefix} normalized object bytes {object_bytes} "
                    f"!= declared capacity {capacity}"
                )

        if authority.get("materialization_state") != "NOT_COMPOSED_REMOTE_AUTHORITY":
            raise ConvergenceError(f"{prefix}.materialization_state must remain fail-closed")

        final_capacity[modality] += capacity
        final_families[modality].add(family)
        authority_identities.append(_authority_identity(authority))

    minimum = manifest.get("minimum_independent_families")
    if not isinstance(minimum, Mapping):
        raise ConvergenceError("minimum_independent_families must be an object")
    family_counts = {m: len(final_families[m]) for m in MODALITIES}
    diversity_pass = all(
        family_counts[m]
        >= _positive_int(minimum.get(m), f"minimum_independent_families.{m}")
        for m in MODALITIES
    )

    mixture = manifest.get("mixture_policy")
    if not isinstance(mixture, Mapping):
        raise ConvergenceError("mixture_policy must be an object")
    if mixture.get("replay_allowed") is not False:
        raise ConvergenceError("mixture_policy.replay_allowed must remain false")
    weights: dict[str, float] = {}
    for modality in MODALITIES:
        value = mixture.get(modality)
        valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_number or not math.isfinite(value) or value <= 0:
            raise ConvergenceError(f"mixture_policy.{modality} must be a finite positive number")
        weights[modality] = float(value)
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ConvergenceError("mixture weights must sum exactly to 1 within 1e-12")

    planning_target = _positive_int(
        mixture.get("planning_target_source_bytes"),
        "mixture_policy.planning_target_source_bytes",
    )
    target_by_modality = {m: int(round(planning_target * weights[m])) for m in MODALITIES}
    if sum(target_by_modality.values()) != planning_target:
        raise ConvergenceError(
            "planning target cannot be represented exactly by configured mixture weights"
        )
    gap_by_modality = {
        m: max(0, target_by_modality[m] - final_capacity[m]) for m in MODALITIES
    }
    max_nonreplay_mixture_source_bytes = min(
        math.floor(final_capacity[m] / weights[m]) for m in MODALITIES
    )
    limiting_modalities = [
        m
        for m in MODALITIES
        if math.floor(final_capacity[m] / weights[m]) == max_nonreplay_mixture_source_bytes
    ]

    identity_payload = {
        "schema_version": manifest["schema_version"],
        "base": {
            "name": base.get("name"),
            "head_sha": base.get("head_sha"),
            "capacity_bytes": base_capacity,
            "family_counts": actual_base_family_counts,
        },
        "additive_authorities": sorted(
            authority_identities,
            key=lambda item: item["authority_id"],
        ),
        "mixture_policy": {m: weights[m] for m in MODALITIES},
        "minimum_independent_families": {m: minimum[m] for m in MODALITIES},
    }
    authority_set_identity = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()

    decision = (
        "PASS_DIVERSITY_ONLY_BLOCK_EXACT_CORPUS" if diversity_pass else "BLOCK_DIVERSITY"
    )

    return {
        "schema_version": (
            "12-6.next100-078-research-corpus-v1-authority-convergence-report.v1"
        ),
        "decision": decision,
        "authority_set_identity_sha256": authority_set_identity,
        "base_capacity_bytes": base_capacity,
        "additive_capacity_bytes": {
            m: final_capacity[m] - base_capacity[m] for m in MODALITIES
        },
        "composed_authority_capacity_bytes": final_capacity,
        "composed_authority_total_bytes": sum(final_capacity.values()),
        "independent_family_counts": family_counts,
        "minimum_independent_families": {m: int(minimum[m]) for m in MODALITIES},
        "authority_diversity_gate": "PASS" if diversity_pass else "BLOCKED",
        "planned_mixture": weights,
        "max_nonreplay_mixture_source_bytes": max_nonreplay_mixture_source_bytes,
        "limiting_modalities": limiting_modalities,
        "planning_target_source_bytes": planning_target,
        "planning_target_by_modality": target_by_modality,
        "remaining_source_capacity_gap_by_modality": gap_by_modality,
        "remaining_source_capacity_gap_total_bytes": sum(gap_by_modality.values()),
        "exact_record_inventory": (
            "BLOCKED_UNTIL_REMOTE_AUTHORITY_BYTES_ARE_COMPOSED_AND_HASH_VERIFIED"
        ),
        "decontamination": "BLOCKED_UNTIL_EXACT_RECORD_INVENTORY",
        "unique_loss_ledger": "BLOCKED_UNTIL_POST_PACK_MATERIALIZATION",
        "training_authorized": False,
        "training_executed": False,
    }


def load_and_validate(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_path = repo_root / manifest["base_vector"]["config_path"]
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    return validate_convergence(manifest, base_config)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "configs/data/next100_078_research_corpus_v1_authority_convergence.json"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = load_and_validate(args.repo_root, args.manifest)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
