#!/usr/bin/env python3
"""Zero-credit exact role probe for Rada_Trees secondary archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import classify_d03_rada_trees_members as classify
import download_d03_rada_trees_from_hf_snapshot as transport
import inventory_d03_rada_trees_archive as inventory

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_rada_trees_secondary_role_probe_v1.json"
CLASSIFIER_CONFIG = ROOT / "configs/data/d03_rada_trees_member_classification_v1.json"
SCHEMA = "12-6.d03-rada-trees-secondary-role-probe-report.v1"
ARCHIVE = "rada_xtag_texts.7z"
REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
DATASET = "uacorpus/Rada_Trees"
USER_AGENT = "12-6-ai-rada-trees-secondary-role-probe/1"
CHUNK = 1024 * 1024


class SecondaryProbeError(RuntimeError):
    """Fail-closed secondary archive probe error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SecondaryProbeError(message)


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "12-6.d03-rada-trees-secondary-role-probe.v1", "config schema drift")
    require(value.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    source = value.get("source")
    require(isinstance(source, dict), "source binding missing")
    require(source.get("dataset") == DATASET, "dataset drift")
    require(source.get("dataset_revision") == REVISION, "revision drift")
    require(source.get("archive_path") == ARCHIVE, "archive path drift")
    require(source.get("expected_size_bytes") == 697768591, "archive size drift")
    require(
        source.get("git_blob_oid") == "2ddd106a0140b6980c9a6152406f40f1f9bd5c30",
        "archive Git object drift",
    )
    require(
        source.get("xet_hash")
        == "46c56a953551f3d33800c8b50fbd355104197abec45c366f245103e830b17d30",
        "archive Xet drift",
    )
    boundary = value.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    require(boundary.get("training_authorized_bytes") == 0, "training credit weakened")
    require(boundary.get("unique_causal_loss_positions_authorized") == 0, "loss credit weakened")
    require(boundary.get("tokenizer_fit_authorized") is False, "tokenizer gate weakened")
    require(boundary.get("model_training_executed") is False, "training boundary weakened")
    require(boundary.get("paid_compute_used") is False, "compute boundary weakened")
    return value


def resolve_url() -> str:
    return (
        "https://huggingface.co/datasets/uacorpus/Rada_Trees/resolve/"
        f"{REVISION}/{ARCHIVE}"
    )


def _resolve_handoff(config: dict[str, Any], timeout: float) -> str:
    request = urllib.request.Request(
        resolve_url(),
        method="GET",
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
    )
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
            return None

    no_redirect = urllib.request.build_opener(NoRedirect())
    try:
        response = no_redirect.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        require(exc.code == 302, f"expected exact HTTP 302 handoff, got {exc.code}")
        headers = exc.headers
    else:
        response.close()
        raise SecondaryProbeError("secondary resolve unexpectedly returned direct response")

    source = config["source"]
    require(headers.get("X-Xet-Hash") == source["xet_hash"], "resolve Xet mismatch")
    repo_commit = headers.get("X-Repo-Commit")
    if repo_commit is not None:
        require(repo_commit == REVISION, "resolve revision mismatch")
    linked_size = headers.get("X-Linked-Size")
    if linked_size is not None:
        require(int(linked_size) == source["expected_size_bytes"], "resolve size mismatch")
    location = headers.get("Location")
    require(isinstance(location, str), "resolve Location missing")
    require(transport._provider_https_url(location), "resolve target outside approved provider")

    final_request = urllib.request.Request(
        location,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    with opener.open(final_request, timeout=timeout) as response:
        require(getattr(response, "status", None) == 200, "secondary final status is not 200")
        require(transport._provider_https_url(response.geturl()), "final provider origin rejected")
    return location


def acquire(config: dict[str, Any], destination: Path, timeout: float) -> str:
    source = config["source"]
    location = _resolve_handoff(config, timeout)
    request = urllib.request.Request(
        location,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    expected = int(source["expected_size_bytes"])
    digest = hashlib.sha256()
    total = 0
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    opener = urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout) as response, temporary.open("xb") as output:
            require(getattr(response, "status", None) == 200, "secondary download status is not 200")
            require(transport._provider_https_url(response.geturl()), "secondary final origin rejected")
            encoding = response.headers.get("Content-Encoding")
            require(encoding in (None, "", "identity"), "secondary content encoding applied")
            length = response.headers.get("Content-Length")
            if length is not None:
                require(int(length) == expected, "secondary Content-Length mismatch")
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                require(total <= expected, "secondary download exceeded exact size")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    require(total == expected, f"secondary byte size mismatch expected={expected} actual={total}")
    os.replace(temporary, destination)
    return digest.hexdigest()


def classify_extracted(
    root: Path,
    members: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in members:
        path = item["path"]
        disk = root / Path(path)
        mode = disk.lstat().st_mode
        require(stat.S_ISREG(mode), f"non-regular classified member: {path}")
        data = disk.read_bytes()
        require(len(data) == item["size_bytes"], f"classified member size drift: {path}")
        require(hashlib.sha256(data).hexdigest() == item["sha256"], f"classified hash drift: {path}")
        decision = classify.classify_content(path, data, policy)
        result.append(
            {
                "path": path,
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "classification": decision["class"],
                "decoded_encoding": decision["encoding"],
                "text_metrics": decision["metrics"],
                "path_year_hints": classify.year_hints(path),
                "text_emitted": False,
            }
        )
    return result


def build_report(config: dict[str, Any], archive: Path, content_sha256: str) -> dict[str, Any]:
    policy = config["inventory_policy"]
    extractor = inventory.find_extractor(None, list(policy["accepted_extractors"]))
    listing = inventory.list_archive(extractor, archive)
    inventory.validate_bounds(
        listing,
        int(policy["max_single_member_bytes"]),
        int(policy["max_total_uncompressed_bytes"]),
    )
    with tempfile.TemporaryDirectory(prefix="rada-secondary-") as tmp:
        root = Path(tmp)
        inventory.extract_archive(extractor, archive, root)
        members = inventory.inventory_extracted_tree(
            root,
            listing,
            int(policy["max_single_member_bytes"]),
            int(policy["max_total_uncompressed_bytes"]),
        )
        classifier_config = json.loads(CLASSIFIER_CONFIG.read_text(encoding="utf-8"))
        classify.validate_config(classifier_config)
        classified = classify_extracted(root, members, classifier_config["classification_policy"])

    suffixes = Counter(Path(item["path"]).suffix.lower() for item in members)
    top_levels = Counter(item["path"].split("/", 1)[0] for item in members)
    class_counts = Counter(item["classification"] for item in classified)
    class_bytes: Counter[str] = Counter()
    for item in classified:
        class_bytes[item["classification"]] += int(item["size_bytes"])
    candidates = [item for item in classified if item["classification"] == "PLAIN_TEXT_CANDIDATE"]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_hash[item["sha256"]].append(item)
    unique_candidate_bytes = sum(group[0]["size_bytes"] for group in by_hash.values())

    source = config["source"]
    inventory_payload = {
        "dataset_head_sha": REVISION,
        "archive_path": ARCHIVE,
        "upstream_object_identity": source["xet_hash"],
        "archive_sha256": content_sha256,
        "archive_size_bytes": source["expected_size_bytes"],
        "members": members,
    }
    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": config["worker_id"],
        "execution_profile": "LOCAL_FREE",
        "source": {
            **source,
            "content_sha256_observed": content_sha256,
            "content_sha256_bound_to_exact_revision_xet_size": True,
        },
        "inventory": {
            "member_count": len(members),
            "uncompressed_bytes": sum(int(item["size_bytes"]) for item in members),
            "inventory_identity_sha256": inventory.sha256_bytes(
                inventory.canonical_json(inventory_payload)
            ),
            "suffix_counts": dict(sorted(suffixes.items())),
            "top_level_counts": dict(sorted(top_levels.items())),
        },
        "classification": {
            "class_counts": dict(sorted(class_counts.items())),
            "class_bytes": dict(sorted(class_bytes.items())),
            "plain_text_candidate_members": len(candidates),
            "plain_text_candidate_bytes_before_exact_duplicate_collapse": sum(
                int(item["size_bytes"]) for item in candidates
            ),
            "plain_text_candidate_bytes_after_exact_duplicate_collapse": unique_candidate_bytes,
            "member_metadata": classified,
            "raw_member_text_emitted": False,
        },
        "decision": {
            "training_admission_claimed": False,
            "content_sha256_is_first_observed_successor_authority": True,
            "rerun_with_pinned_sha_required_before_any_capacity_credit": True,
        },
        "claim_boundary": {
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_payload_accessed": False,
            "paid_compute_used": False,
        },
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    config = load_config()
    try:
        content_sha256 = acquire(config, args.archive_output, args.timeout)
        report = build_report(config, args.archive_output, content_sha256)
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print("D03_RADA_TREES_SECONDARY_ROLE_PROBE=PASS_ZERO_CREDIT")
        print("CONTENT_SHA256=" + content_sha256)
        print("PLAIN_TEXT_CANDIDATE_MEMBERS=" + str(report["classification"]["plain_text_candidate_members"]))
        print("PLAIN_TEXT_CANDIDATE_BYTES=" + str(report["classification"]["plain_text_candidate_bytes_after_exact_duplicate_collapse"]))
        return 0
    except (SecondaryProbeError, ValueError, RuntimeError, OSError, urllib.error.URLError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    finally:
        args.archive_output.unlink(missing_ok=True)
        args.archive_output.with_suffix(args.archive_output.suffix + ".partial").unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
