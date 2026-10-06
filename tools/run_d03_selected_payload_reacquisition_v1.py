"""Reacquire exact raw payloads for the #3068 selected-20M materialization plan.

Execution-only coordinator. Source-specific materializers remain owned by their
exact historical execution heads. This adapter loads one pinned lane in a fresh
process, asks that lane to reproduce authenticated payload bytes, then emits only
the record IDs required by the selected-20M plan. Every emitted byte is checked
against the independently bound #3045 inventory before publication.

Raw fragments are ephemeral inputs to run_d03_selected_raw_assembly_v1.py. This
tool grants no corpus, tokenizer, training, evaluation-result, paid-compute, or
scale authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

PLAN_ID = "ddf42773bb7ac4856e07c7b8e22b137581f02b79e231f0740b12e867c0981400"
COMPOSITION_ID = "84741c6cd99d06dbc55fb413b7b150573af7b010729b23839f4fdabcf8f505ef"
EXPECTED_MISSING_RECORDS = 5_083
EXPECTED_MISSING_BYTES = 16_417_002
EXPECTED_SELECTED_RECORDS = 5_294
EXPECTED_SELECTED_BYTES = 20_000_000
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 64 * 1024 * 1024

LANES: dict[str, dict[str, Any]] = {
    "code": {
        "head": "62dc9f487e0b3dd9eb2a8367a01061f8f55799ef",
        "runner": "tools/run_d03_code_delta_decontam_qp_v1.py",
        "blob": "3e1d9a68622a3e709488a128967f7d83ff1074ea",
        "families": (
            "code.scipy.project",
            "github:Textualize/rich",
            "github:fastapi/fastapi",
            "github:fastapi/typer",
            "github:pallets/flask",
            "github:pandas-dev/pandas",
            "github:pydantic/pydantic",
        ),
    },
    "ubuntu": {
        "head": "fb9c3d9c137b0ccacf40dbee8d6dc75db7a2209f",
        "runner": "tools/run_d03_ubuntu_irc_decontam_qp_v1.py",
        "blob": "9a4d0829772b2b460eb4d895254f053380b799fa",
        "families": ("common-pile/ubuntu_irc",),
    },
    "loc": {
        "head": "0aeddc6c32917dee110a669ffea55d55a399c1c9",
        "runner": "tools/run_d03_pep_loc_decontam_qp_v1.py",
        "blob": "4d4a84b7dd30df80e26023aa23a688189212f7ca",
        "families": ("en.loc.selected-digitized-books",),
    },
    "languk": {
        "head": "f6a2a440aacebec6a44a5f2ba955902970ff4d50",
        "runner": "tools/run_d03_languk_court_postdedup_clean_v1.py",
        "blob": "b9708073baa411a16335e509a3a89ce0023cb100",
        "families": ("ua.languk.supreme-court-decisions",),
    },
    "lesia": {
        "head": "1d594e048365fd2121433210b83c0dc9069c50bf",
        "runner": "tools/run_d03_wikisource_lesia1892_decontam_qp_v1.py",
        "blob": "f66e9d27e729198984905d66d37c01b9a32d5ff6",
        "families": (
            "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
        ),
    },
    "nbu": {
        "head": "dc4822ec1365494507bca1d67bab1b334a3fa143",
        "runner": "tools/run_d03_nbu_current40_decontam_qp_v1.py",
        "blob": "5db14f0770921f5b817c3c58e4b70cf7be8c66dc",
        "families": ("ua.nbu.official-resolutions",),
    },
    "rada": {
        "head": "a63d7c88ccb5fa380b7af4b44bde32433cf35877",
        "runner": "tools/run_d03_rada_current_postdata232_g05_g06_v1.py",
        "blob": "0fb78d61e7209181b2bdc7b279d2351ad3a1931e",
        "families": ("ua.rada.open-data.laws-texts",),
    },
    "franko": {
        "head": "4280ac3a6b901906b0b38b5ff0ce0e74a90dd154",
        "runner": "tools/run_d03_franko_decontam_qp_v1.py",
        "blob": "b21546426f8777e9dd86a2a3c19af41dde6ff7f6",
        "families": ("ua.verba.public-domain.franko1901",),
    },
}

RADA_PARENT = {
    "artifact_id": 11415639104,
    "artifact_zip_sha256": "54f0c5729c2087e13400413acf63b1eabdaa7bd697c03bf967dc44978b05abb7",
    "inventory_file_sha256": "b8e0e6f12cdb91f8be3582911be12b152272e98fc75d25fdf2da0f4dd3c66441",
    "handoff_file_sha256": "c57e9f4cc8a646a99171e019b65d714239b6b2231d9bc15bfc8a349ac920cc23",
    "report_file_sha256": "9701182b9f298dc274c318055729c7163a6a7768f39d5bc3995a6cecc68e9041",
    "execution_evidence_file_sha256": (
        "087eecc51a2e8c127aea035834ce6f185090f1e86e03d8fdaf9d5186c6d219e8"
    ),
    "result_file_sha256": "8e372e44c4d3e428576460e2f4586f6bd6c49aa62c8986dc6fa832cc9d0c4fc4",
    "proof_file_sha256": "a61e06345f75922b4e4cc03c0c01d85898c5f307c79031e837fb16465c48f1c2",
    "inventory_identity_sha256": (
        "e15066e8dbcee53a99e43a2d13b821ac379ffc4daef67ce8f32517dbcc92707a"
    ),
    "handoff_identity_sha256": (
        "01de5b9bfa4f69b5d7fc40807b62e598239ba6fd30fea408b66eac665861be04"
    ),
    "result_identity_sha256": (
        "2896e5c1a0555a1636db0431c05a51143dad07c2ac97eb462b5ec35958ae5ca3"
    ),
    "proof_identity_sha256": (
        "37d7b7bf2b7871387b6bfe1b4920757e778a4678ff5bc15776d2575a91a06d4e"
    ),
}


class ReacquisitionError(RuntimeError):
    """Raised when selected raw payload authority cannot be reproduced safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReacquisitionError(message)


def blocked_constant(value: str) -> NoReturn:
    raise ReacquisitionError(f"non-finite JSON constant: {value}")


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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON file: {path}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_JSON_BYTES, f"JSON input exceeds bound: {path}")
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=strict_object,
        parse_constant=blocked_constant,
    )
    require(type(value) is dict, f"JSON root must be exact object: {path}")
    return value


def verify_self_hash(
    document: Mapping[str, Any],
    field: str,
    expected: str,
    *,
    label: str,
) -> None:
    require(type(document) is dict, f"{label} root must be exact object")
    claimed = document.get(field)
    require(
        type(claimed) is str and SHA64.fullmatch(claimed) is not None,
        f"{label} identity malformed",
    )
    require(claimed == expected, f"{label} identity drift")
    core = dict(document)
    core.pop(field, None)
    require(sha256(core) == claimed, f"{label} self-hash mismatch")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    require(
        result.returncode == 0,
        "git failed: " + " ".join(args) + ": " + (result.stderr.strip() or result.stdout.strip()),
    )
    return result.stdout.strip()


def verify_lane_root(lane: str, root: Path) -> dict[str, Any]:
    spec = LANES[lane]
    require(root.is_dir() and not root.is_symlink(), f"{lane} lane root invalid")
    require(_git(root, "rev-parse", "HEAD") == spec["head"], f"{lane} HEAD drift")
    require(
        _git(root, "rev-parse", f"HEAD:{spec['runner']}") == spec["blob"],
        f"{lane} runner Git blob drift",
    )
    require(
        _git(root, "hash-object", str(root / spec["runner"])) == spec["blob"],
        f"{lane} runner worktree blob drift",
    )
    return spec


def load_lane_module(lane: str, root: Path) -> Any:
    spec = verify_lane_root(lane, root)
    module_path = root / spec["runner"]
    sys.path.insert(0, str(root / "tools"))
    sys.path.insert(0, str(root / "src"))
    name = f"selected_raw_lane_{lane}"
    imported = importlib.util.spec_from_file_location(name, module_path)
    require(
        imported is not None and imported.loader is not None,
        f"cannot load {lane} runner",
    )
    module = importlib.util.module_from_spec(imported)
    sys.modules[name] = module
    imported.loader.exec_module(module)
    return module


def selected_context(
    plan_path: Path,
    composition_path: Path,
    lane: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], tuple[str, ...]]:
    plan = load_json(plan_path)
    composition = load_json(composition_path)
    verify_self_hash(
        plan,
        "plan_identity_sha256",
        PLAN_ID,
        label="selected raw materialization plan",
    )
    verify_self_hash(
        composition,
        "composition_identity_sha256",
        COMPOSITION_ID,
        label="#3045 composition",
    )
    require(
        plan.get("selected_record_count") == EXPECTED_SELECTED_RECORDS
        and plan.get("selected_source_bytes") == EXPECTED_SELECTED_BYTES,
        "selected authority totals drift",
    )
    require(
        plan.get("missing_record_count") == EXPECTED_MISSING_RECORDS
        and plan.get("missing_payload_bytes") == EXPECTED_MISSING_BYTES,
        "missing authority totals drift",
    )

    inventory = composition.get("combined_inventory")
    require(isinstance(inventory, Mapping), "#3045 inventory missing")
    rows = inventory.get("records")
    require(type(rows) is list and rows, "#3045 inventory records missing")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in rows:
        require(type(raw) is dict, "#3045 inventory row must be exact object")
        record_id = raw.get("record_id")
        require(type(record_id) is str and record_id, "#3045 record_id invalid")
        require(record_id not in by_id, f"#3045 duplicate record_id: {record_id}")
        by_id[record_id] = raw

    spec = LANES[lane]
    families = tuple(spec["families"])
    family_plan = plan.get("family_materialization_plan")
    require(type(family_plan) is list, "family plan missing")
    plan_by_family = {
        str(item["family"]): item
        for item in family_plan
        if type(item) is dict and type(item.get("family")) is str
    }
    selected: dict[str, dict[str, Any]] = {}
    for family in families:
        item = plan_by_family.get(family)
        require(type(item) is dict, f"family absent from plan: {family}")
        authority = item.get("materializer_authority")
        require(type(authority) is dict, f"materializer authority missing: {family}")
        require(
            authority.get("head_sha") == spec["head"],
            f"materializer head drift: {family}",
        )
        ids = item.get("missing_record_ids")
        require(type(ids) is list and ids, f"missing IDs absent: {family}")
        family_rows: list[dict[str, Any]] = []
        for record_id in ids:
            require(type(record_id) is str and record_id, f"invalid planned ID: {family}")
            row = by_id.get(record_id)
            require(type(row) is dict, f"planned ID absent from #3045: {record_id}")
            require(row.get("family") == family, f"planned family drift: {record_id}")
            selected[record_id] = row
            family_rows.append(row)
        require(
            len(family_rows) == item.get("missing_record_count"),
            f"planned count drift: {family}",
        )
        require(
            sum(int(row["payload_bytes"]) for row in family_rows)
            == item.get("missing_payload_bytes"),
            f"planned byte drift: {family}",
        )
        projection = [
            {
                "record_id": row["record_id"],
                "payload_sha256": row["payload_sha256"],
                "payload_bytes": row["payload_bytes"],
            }
            for row in family_rows
        ]
        require(
            sha256(projection) == item.get("missing_payload_projection_sha256"),
            f"planned payload projection drift: {family}",
        )
    return plan, selected, families


def materialize_fragment(
    *,
    selected: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_payload_keys: set[str] = set()
    for record_id in sorted(selected):
        authority = selected[record_id]
        source_id = authority.get("source_id")
        require(type(source_id) is str and source_id, f"selected source_id invalid: {record_id}")
        by_source = payloads.get(source_id)
        by_record = payloads.get(record_id)
        if by_source is not None and by_record is not None:
            require(by_source == by_record, f"ambiguous payload mapping: {record_id}")
        raw = by_source if by_source is not None else by_record
        require(type(raw) is bytes and raw, f"reacquired payload missing: {record_id}")
        seen_payload_keys.add(source_id if by_source is not None else record_id)
        require(
            len(raw) == authority.get("payload_bytes"),
            f"reacquired payload byte drift: {record_id}",
        )
        require(
            sha256(raw) == authority.get("payload_sha256"),
            f"reacquired payload SHA drift: {record_id}",
        )
        text = raw.decode("utf-8", errors="strict")
        require(text.encode("utf-8") == raw, f"UTF-8 round-trip drift: {record_id}")
        output.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": authority["family"],
                "modality": authority["modality"],
                "normalized_payload": text,
            }
        )
    require(len(output) == len(selected), "fragment selected cardinality drift")
    require(
        sum(len(row["normalized_payload"].encode("utf-8")) for row in output)
        == sum(int(row["payload_bytes"]) for row in selected.values()),
        "fragment payload byte total drift",
    )
    return output


def write_fragment(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    require(not path.is_symlink(), f"output path must not be symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical(dict(row)) + b"\n" for row in rows)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def execute_fragment(
    *,
    lane: str,
    plan_path: Path,
    composition_path: Path,
    lane_root: Path,
    output_path: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    _plan, selected, families = selected_context(plan_path, composition_path, lane)
    rows = materialize_fragment(selected=selected, payloads=payloads)
    observed_families = {str(row["family"]) for row in rows}
    require(observed_families == set(families), f"{lane} fragment family coverage drift")
    write_fragment(output_path, rows)
    return {
        "lane": lane,
        "record_count": len(rows),
        "payload_bytes": sum(
            len(row["normalized_payload"].encode("utf-8")) for row in rows
        ),
        "fragment_sha256": sha256(output_path.read_bytes()),
    }


def _common(args: argparse.Namespace) -> tuple[Any, dict[str, dict[str, Any]]]:
    module = load_lane_module(args.lane, args.lane_root)
    _plan, selected, _families = selected_context(
        args.plan_json,
        args.composition_json,
        args.lane,
    )
    return module, selected


def reproduce_post_qp_payloads(
    module: Any,
    source_rows: Sequence[Mapping[str, Any]],
    source_payloads: Mapping[str, bytes],
) -> dict[str, bytes]:
    """Reproduce G05/G06 survivor bytes without re-opening evaluation payloads.

    DATA-232 only removes complete records; it does not transform survivor text.
    For reacquisition, run the exact lane-owned G05/G06 implementation over the
    authenticated post-global-dedup source bytes, then accept only rows whose
    final bytes independently match the #3045 selected inventory. This function
    creates no replacement scientific authority: its generated Q/P identities
    are intentionally local to the reacquisition process.
    """

    built = module.build_training_authorities(
        list(source_rows),
        source_payloads,
    )
    require(
        isinstance(built, tuple) and len(built) >= 1 and type(built[0]) is list,
        "lane training-authority builder returned unexpected shape",
    )
    training_records = built[0]
    clean = module.clean

    quality_inputs, metadata, excluded = clean._post_decontamination_records(
        training_records,
        {"excluded_records": []},
    )
    require(excluded == 0, "reacquisition-only decontamination projection drift")
    projection = clean._input_projection(quality_inputs)
    projection_sha = clean._sha256(clean._cjson(projection))
    reacquisition_manifest = sha256(
        b"12-6.selected-payload-reacquisition.no-data232-transform.v1"
    )
    quality = clean.build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=reacquisition_manifest,
        expected_input_rows_sha256=projection_sha,
    )
    quality_identity = quality.get("execution_identity_sha256")
    require(
        type(quality_identity) is str
        and SHA64.fullmatch(quality_identity) is not None,
        "reacquisition quality identity malformed",
    )
    clean.verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=reacquisition_manifest,
        expected_input_rows_sha256=projection_sha,
        expected_execution_identity_sha256=quality_identity,
    )
    partial_materializer = getattr(
        module,
        "_materialize_quality_survivors_with_partial",
        None,
    )
    if partial_materializer is None:
        quality_survivors, _quality_stats = clean._materialize_quality_survivors(
            quality_inputs,
            metadata,
            quality,
        )
    else:
        quality_survivors, _quality_stats, _partial_detail = partial_materializer(
            quality_inputs,
            metadata,
            quality,
        )

    privacy_inputs = clean._quality_records_for_privacy(quality_survivors)
    privacy_projection = clean._input_projection(privacy_inputs)
    privacy_projection_sha = clean._sha256(clean._cjson(privacy_projection))
    privacy = clean.build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=privacy_projection_sha,
    )
    privacy_identity = privacy.get("execution_identity_sha256")
    require(
        type(privacy_identity) is str
        and SHA64.fullmatch(privacy_identity) is not None,
        "reacquisition privacy identity malformed",
    )
    clean.verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=privacy_projection_sha,
        expected_execution_identity_sha256=privacy_identity,
    )
    final_survivors, _privacy_stats = clean._materialize_privacy_survivors(
        quality_survivors,
        privacy,
    )

    result: dict[str, bytes] = {}
    for row in final_survivors:
        record_id = row.get("record_id")
        payload = row.get("normalized_payload")
        require(
            type(record_id) is str and record_id,
            "post-QP survivor record_id invalid",
        )
        require(
            type(payload) is str and payload,
            f"post-QP survivor payload missing: {record_id}",
        )
        require(record_id not in result, f"post-QP survivor replay: {record_id}")
        raw = payload.encode("utf-8")
        require(
            raw.decode("utf-8", errors="strict") == payload,
            f"post-QP survivor UTF-8 drift: {record_id}",
        )
        result[record_id] = raw
    require(result, "post-QP reacquisition produced no survivors")
    return result


def run_code(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["code"]["head"])
    rows, payloads, _authority = module.acquire_delta_sources()
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_ubuntu(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["ubuntu"]["head"])
    _survivors, _evidence, parent_rows = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.acquire_ubuntu_survivors(
        candidate_jsonl=args.candidate_jsonl,
        expected_survivors=parent_rows,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_loc(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["loc"]["head"])
    survivor_ids = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.reconstruct_exact_survivors(
        survivor_ids=survivor_ids,
        clean_training_records=args.clean_training_records,
        clean_training_handoff=args.clean_training_handoff,
        pep_candidate=args.pep_candidate,
        pep_report=args.pep_report,
        loc_candidate=args.loc_candidate,
        loc_report=args.loc_report,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_languk(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["languk"]["head"])
    parent, _evidence, survivor_ids = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.acquire_languk_survivors(
        source_parquet=args.source_parquet,
        parent_survivors=parent,
        expected_survivor_ids=survivor_ids,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_lesia(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["lesia"]["head"])
    _survivors, _evidence, parent_rows = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.acquire_lesia_survivors(
        candidate_jsonl=args.candidate_jsonl,
        materialization_report_json=args.materialization_report_json,
        parent_rows=parent_rows,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_nbu(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["nbu"]["head"])
    _survivors, _report, parent_rows = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.acquire_nbu_survivors(
        candidate_jsonl=args.candidate_jsonl,
        materialization_report_json=args.materialization_report_json,
        nbu_intake_path=args.nbu_intake_path,
        parent_rows=parent_rows,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_franko(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    module.verify_local_authority(LANES["franko"]["head"])
    _survivors, _evidence, survivor_ids = module.verify_parent_artifact(
        args.parent_survivors,
        args.parent_evidence,
        args.parent_proof,
    )
    rows, payloads, _authority = module.acquire_franko_survivors(
        candidate_jsonl=args.candidate_jsonl,
        historical_terminal_evidence_json=args.historical_terminal_evidence_json,
        fresh_execution_authority_json=args.fresh_execution_authority_json,
        expected_survivor_ids=survivor_ids,
    )
    return reproduce_post_qp_payloads(module, rows, payloads)


def run_rada(args: argparse.Namespace) -> Mapping[str, bytes]:
    module, _selected = _common(args)
    scratch = args.scratch_dir
    require(not scratch.exists(), "Rada scratch directory already exists")
    scratch.mkdir(parents=True)
    captured: dict[str, Any] = {}
    original_loader = module.load_bound_runtime

    def capturing_loader() -> tuple[Any, Any, Any]:
        clean, reserved, verify_report = original_loader()
        original = clean._materialize_privacy_survivors

        def capturing_materializer(*values: Any, **kwargs: Any) -> Any:
            result = original(*values, **kwargs)
            rows, stats = result
            captured["rows"] = [dict(row) for row in rows]
            captured["clean"] = clean
            captured["original"] = original
            return rows, stats

        clean._materialize_privacy_survivors = capturing_materializer
        return clean, reserved, verify_report

    module.load_bound_runtime = capturing_loader
    parsed = module.parser().parse_args(
        [
            "--expected-execution-head",
            LANES["rada"]["head"],
            "--parent-artifact-id",
            str(RADA_PARENT["artifact_id"]),
            "--expected-parent-artifact-zip-sha256",
            RADA_PARENT["artifact_zip_sha256"],
            "--training-records-jsonl",
            str(args.training_records_jsonl),
            "--parent-inventory-json",
            str(args.parent_inventory_json),
            "--parent-handoff-json",
            str(args.parent_handoff_json),
            "--parent-report-json",
            str(args.parent_report_json),
            "--parent-execution-evidence-json",
            str(args.parent_execution_evidence_json),
            "--parent-result-json",
            str(args.parent_result_json),
            "--parent-two-clean-proof-json",
            str(args.parent_two_clean_proof_json),
            "--expected-parent-inventory-file-sha256",
            RADA_PARENT["inventory_file_sha256"],
            "--expected-parent-handoff-file-sha256",
            RADA_PARENT["handoff_file_sha256"],
            "--expected-parent-report-file-sha256",
            RADA_PARENT["report_file_sha256"],
            "--expected-parent-execution-evidence-file-sha256",
            RADA_PARENT["execution_evidence_file_sha256"],
            "--expected-parent-result-file-sha256",
            RADA_PARENT["result_file_sha256"],
            "--expected-parent-proof-file-sha256",
            RADA_PARENT["proof_file_sha256"],
            "--expected-parent-inventory-identity-sha256",
            RADA_PARENT["inventory_identity_sha256"],
            "--expected-parent-handoff-identity-sha256",
            RADA_PARENT["handoff_identity_sha256"],
            "--expected-parent-result-identity-sha256",
            RADA_PARENT["result_identity_sha256"],
            "--expected-parent-proof-identity-sha256",
            RADA_PARENT["proof_identity_sha256"],
            "--output-evidence",
            str(scratch / "evidence.json"),
            "--output-quality",
            str(scratch / "quality.json"),
            "--output-privacy",
            str(scratch / "privacy.json"),
            "--output-survivor-inventory",
            str(scratch / "inventory.json"),
        ]
    )
    try:
        module.execute(parsed)
        rows = captured.get("rows")
        require(type(rows) is list and rows, "Rada final survivor capture missing")
        payloads: dict[str, bytes] = {}
        for row in rows:
            require(type(row) is dict, "Rada captured survivor must be exact object")
            record_id = row.get("record_id")
            payload = row.get("normalized_payload")
            require(type(record_id) is str and record_id, "Rada record ID invalid")
            require(type(payload) is str and payload, f"Rada payload missing: {record_id}")
            require(record_id not in payloads, f"Rada record replay: {record_id}")
            payloads[record_id] = payload.encode("utf-8")
        return payloads
    finally:
        module.load_bound_runtime = original_loader
        clean = captured.get("clean")
        original = captured.get("original")
        if clean is not None and original is not None:
            clean._materialize_privacy_survivors = original
        shutil.rmtree(scratch, ignore_errors=True)


def add_common(subparser: argparse.ArgumentParser, lane: str) -> None:
    subparser.set_defaults(lane=lane)
    subparser.add_argument("--plan-json", type=Path, required=True)
    subparser.add_argument("--composition-json", type=Path, required=True)
    subparser.add_argument("--lane-root", type=Path, required=True)
    subparser.add_argument("--output-jsonl", type=Path, required=True)


def add_parent_triplet(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--parent-survivors", type=Path, required=True)
    subparser.add_argument("--parent-evidence", type=Path, required=True)
    subparser.add_argument("--parent-proof", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = result.add_subparsers(dest="command", required=True)

    code = commands.add_parser("code", allow_abbrev=False)
    add_common(code, "code")

    ubuntu = commands.add_parser("ubuntu", allow_abbrev=False)
    add_common(ubuntu, "ubuntu")
    add_parent_triplet(ubuntu)
    ubuntu.add_argument("--candidate-jsonl", type=Path, required=True)

    loc = commands.add_parser("loc", allow_abbrev=False)
    add_common(loc, "loc")
    add_parent_triplet(loc)
    loc.add_argument("--clean-training-records", type=Path, required=True)
    loc.add_argument("--clean-training-handoff", type=Path, required=True)
    loc.add_argument("--pep-candidate", type=Path, required=True)
    loc.add_argument("--pep-report", type=Path, required=True)
    loc.add_argument("--loc-candidate", type=Path, required=True)
    loc.add_argument("--loc-report", type=Path, required=True)

    languk = commands.add_parser("languk", allow_abbrev=False)
    add_common(languk, "languk")
    add_parent_triplet(languk)
    languk.add_argument("--source-parquet", type=Path, required=True)

    lesia = commands.add_parser("lesia", allow_abbrev=False)
    add_common(lesia, "lesia")
    add_parent_triplet(lesia)
    lesia.add_argument("--candidate-jsonl", type=Path, required=True)
    lesia.add_argument("--materialization-report-json", type=Path, required=True)

    nbu = commands.add_parser("nbu", allow_abbrev=False)
    add_common(nbu, "nbu")
    add_parent_triplet(nbu)
    nbu.add_argument("--candidate-jsonl", type=Path, required=True)
    nbu.add_argument("--materialization-report-json", type=Path, required=True)
    nbu.add_argument("--nbu-intake-path", type=Path, required=True)

    franko = commands.add_parser("franko", allow_abbrev=False)
    add_common(franko, "franko")
    add_parent_triplet(franko)
    franko.add_argument("--candidate-jsonl", type=Path, required=True)
    franko.add_argument("--historical-terminal-evidence-json", type=Path, required=True)
    franko.add_argument("--fresh-execution-authority-json", type=Path, required=True)

    rada = commands.add_parser("rada", allow_abbrev=False)
    add_common(rada, "rada")
    rada.add_argument("--training-records-jsonl", type=Path, required=True)
    rada.add_argument("--parent-inventory-json", type=Path, required=True)
    rada.add_argument("--parent-handoff-json", type=Path, required=True)
    rada.add_argument("--parent-report-json", type=Path, required=True)
    rada.add_argument("--parent-execution-evidence-json", type=Path, required=True)
    rada.add_argument("--parent-result-json", type=Path, required=True)
    rada.add_argument("--parent-two-clean-proof-json", type=Path, required=True)
    rada.add_argument("--scratch-dir", type=Path, required=True)
    return result


RUNNERS = {
    "code": run_code,
    "ubuntu": run_ubuntu,
    "loc": run_loc,
    "languk": run_languk,
    "lesia": run_lesia,
    "nbu": run_nbu,
    "rada": run_rada,
    "franko": run_franko,
}


def main() -> int:
    args = parser().parse_args()
    try:
        payloads = RUNNERS[args.command](args)
        result = execute_fragment(
            lane=args.lane,
            plan_path=args.plan_json,
            composition_path=args.composition_json,
            lane_root=args.lane_root,
            output_path=args.output_jsonl,
            payloads=payloads,
        )
    except (
        ReacquisitionError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"BLOCKED: {exc}")
        return 2

    print("SELECTED_PAYLOAD_REACQUISITION=PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
