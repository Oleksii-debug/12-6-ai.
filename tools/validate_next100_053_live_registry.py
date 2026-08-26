#!/usr/bin/env python3
"""Fail-closed live DATA-287 registry seal for NEXT100-053 attrs admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_053_attrs_code_source_v1.json")
USER_AGENT = "12-6-NEXT100-053-live-registry/1"
MAX_BYTES = 250_000


class RegistryError(RuntimeError):
    pass


def cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise RegistryError(f"registry exceeded {MAX_BYTES} bytes")
    return data


def load_config(repo: Path) -> dict[str, Any]:
    cfg = json.loads((repo / CONFIG).read_bytes())
    if cfg.get("worker_id") != "NEXT100-053-CODE-ATTRS":
        raise RegistryError("worker identity drift")
    if cfg.get("execution_profile") != "LOCAL_FREE":
        raise RegistryError("LOCAL_FREE boundary drift")
    return cfg


def registry_urls(binding: dict[str, Any]) -> tuple[str, str]:
    root = "https://raw.githubusercontent.com/Oleksii-debug/12-6-ai."
    pinned = f"{root}/{binding['commit']}/{binding['path']}"
    live = f"{root}/{binding['branch']}/{binding['path']}"
    return pinned, live


def validate_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    binding = cfg["live_registry_binding"]
    pinned_url, live_url = registry_urls(binding)
    pinned_raw = download(pinned_url)
    live_raw = download(live_url)
    if live_raw != pinned_raw:
        raise RegistryError("live DATA-287 registry bytes moved after binding")

    registry = json.loads(pinned_raw)
    if registry.get("schema_version") != "12-6.external-snapshot-registry.v2":
        raise RegistryError("live registry schema drift")
    if registry.get("registry_identity_sha256") != binding["registry_identity_sha256"]:
        raise RegistryError("live registry semantic identity drift")
    if registry.get("source_count") != binding["source_count"]:
        raise RegistryError("live source count drift")
    if registry.get("independent_source_family_count") != binding["independent_source_family_count"]:
        raise RegistryError("live family count drift")
    if registry.get("claim_boundary", {}).get("evaluation_authorized_source_count") != binding["evaluation_authorized_source_count"]:
        raise RegistryError("live evaluation-authorized count drift")
    if registry.get("local_free_only") is not True:
        raise RegistryError("live registry LOCAL_FREE evidence drift")

    code_sources = [source for source in registry.get("sources", []) if source.get("modality") == "code"]
    code_families = sorted(source["independent_source_family"]["family_id"] for source in code_sources)
    if code_families != sorted(binding["code_families"]):
        raise RegistryError(f"live code family set drift: {code_families}")

    expected_objects = {
        (item["source_family"], item["commit"], item["path"], item["git_blob_sha1"])
        for item in cfg["incumbent_code_families_for_dedup"]
    }
    actual_objects = {
        (
            source["independent_source_family"]["family_id"],
            source["exact_upstream_identity"]["commit"],
            source["exact_upstream_identity"]["path"],
            source["exact_upstream_identity"]["git_blob_sha1"],
        )
        for source in code_sources
    }
    if actual_objects != expected_objects:
        raise RegistryError("live code object identities differ from the dedup incumbents")

    selected_blobs = {item["git_blob_sha1"] for item in cfg["files"]}
    registry_blobs = {
        source.get("exact_upstream_identity", {}).get("git_blob_sha1")
        for source in registry.get("sources", [])
        if source.get("exact_upstream_identity", {}).get("git_blob_sha1")
    }
    overlap = sorted(selected_blobs & registry_blobs)
    if overlap:
        raise RegistryError(f"attrs exact blob already present in live registry: {overlap}")

    registry_families = {
        source.get("independent_source_family", {}).get("family_id")
        for source in registry.get("sources", [])
    }
    if cfg["source_family"] in registry_families:
        raise RegistryError("attrs family already present in live registry")

    report: dict[str, Any] = {
        "schema_version": "12-6.next100-053-live-registry-seal.v1",
        "worker_id": cfg["worker_id"],
        "verdict": "PASS_LIVE_REGISTRY",
        "execution_profile": "LOCAL_FREE",
        "registry_worker": binding["worker"],
        "registry_branch": binding["branch"],
        "registry_commit": binding["commit"],
        "registry_identity_sha256": binding["registry_identity_sha256"],
        "registry_raw_sha256": sha256(pinned_raw),
        "live_registry_bytes_equal_pinned": True,
        "source_count": registry["source_count"],
        "independent_source_family_count": registry["independent_source_family_count"],
        "evaluation_authorized_source_count": registry["claim_boundary"]["evaluation_authorized_source_count"],
        "incumbent_code_objects": [
            {
                "source_family": family,
                "commit": commit,
                "path": path,
                "git_blob_sha1": blob,
            }
            for family, commit, path, blob in sorted(actual_objects)
        ],
        "attrs_family_already_registered": False,
        "selected_blob_overlap": [],
        "claim_boundary": [
            "This seal proves only the live external-snapshot registry remained byte-identical to the pinned DATA-287 authority during this exact-head run.",
            "Parallel source candidates are not promoted into the terminal registry by this seal.",
            "Evaluation use remains unauthorized for the attrs training objects."
        ],
    }
    report["seal_sha256"] = sha256(cjson(report))
    return report


def validate_seal(report: dict[str, Any]) -> None:
    if report.get("verdict") != "PASS_LIVE_REGISTRY":
        raise RegistryError("terminal live registry verdict is not PASS")
    seal = report.get("seal_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        raise RegistryError("missing seal identity")
    copy = dict(report)
    copy.pop("seal_sha256", None)
    if sha256(cjson(copy)) != seal:
        raise RegistryError("seal self-hash mismatch")
    if report.get("live_registry_bytes_equal_pinned") is not True:
        raise RegistryError("live registry byte equality not proven")
    if report.get("evaluation_authorized_source_count") != 0:
        raise RegistryError("unexpected evaluation-authorized registry source")
    if report.get("selected_blob_overlap"):
        raise RegistryError("selected attrs blob overlaps live registry")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "validate"))
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output")
    parser.add_argument("--report")
    args = parser.parse_args()

    if args.command == "run":
        report = validate_registry(load_config(Path(args.repo).resolve()))
        validate_seal(report)
        raw = cjson(report)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        print(raw.decode("utf-8"), end="")
        return

    if not args.report:
        raise SystemExit("--report is required for validate")
    report = json.loads(Path(args.report).read_bytes())
    validate_seal(report)
    print(report["seal_sha256"])


if __name__ == "__main__":
    main()
