from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

PORT_CONFIG = Path("configs/data/data324_kubernetes_ua_current_main_v1.json")
LEGACY_CONFIG = Path("configs/data/data324_kubernetes_ua_recovery_v1.json")
MANIFEST = Path("data/external/snapshots/data324-kubernetes-ua-v1/manifest.json")
REPORT = Path("reports/data324/kubernetes-ua-recovery-v1.json")
RAW = Path("data/external/snapshots/data324-kubernetes-ua-v1/raw/what-is-kubernetes.md")
NORMALIZED = Path(
    "data/external/snapshots/data324-kubernetes-ua-v1/normalized/what-is-kubernetes.uk.txt"
)
LICENSE = Path(
    "data/external/rights-evidence/data324/kubernetes-website-cc-by-4.0-25f3dcb.txt"
)
ATTRIBUTION = Path("data/external/snapshots/data324-kubernetes-ua-v1/ATTRIBUTION.txt")
MATERIALIZER = Path(__file__).with_name("materialize_data324_kubernetes_ua.py")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "worker_id",
        "local_free_only",
        "ported_from",
        "port_base_main_sha",
        "source_authority",
        "rights_authority",
        "evaluation_firewall",
        "current_main_truth_boundary",
    }
)
EXPECTED_WORKER_ID = "D03-DATA324-CURRENT-MAIN-CONVERGENCE-20260907"
EXPECTED_PORTED_FROM = {
    "pr": 446,
    "head_sha": "d2c5076e832d10721ceb333756bbbc1108b64658",
    "tree_sha": "f980b54d861b38b77ba6129f61da5c9d514c80f0",
    "verdict": "ADMIT",
}
EXPECTED_PORT_BASE_MAIN_SHA = "a53279292af68dc95e4e615542b6c8c2ea7f9ee5"
EXPECTED_SOURCE_AUTHORITY = {
    "source_family": "kubernetes.website.docs",
    "canonical_upstream": "github:kubernetes/website",
    "upstream_revision": "25f3dcbed7429ebe20174ccc7000428d0f0aedda",
    "source_path": "content/uk/docs/concepts/overview/what-is-kubernetes.md",
    "source_git_blob_sha1": "b3c52cab3be6a8efbc33e91893c653df5972a794",
    "raw_sha256": "5c35e78f0a5f14210734e85778f13bd658ee4ea0640e356c99bb0b88eddd6e75",
    "raw_bytes": 27134,
    "normalized_sha256": "12abd14eff9018602ebb8ebb76ee2f60d1a178a2a6b5648d77e08d5e16f2e0b1",
    "normalized_utf8_bytes": 17415,
    "manifest_identity_sha256": "b957ab7f8d628ff1e71ae6c49c2297866fdb8de32cd322e5d0ad4ebccdbc7c38",
    "report_identity_sha256": "a35cd748527f81675b106d07659446d489be9ac184fb2118e5b4fbf79cc125e7",
}
EXPECTED_RIGHTS_AUTHORITY = {
    "license_id": "CC-BY-4.0",
    "license_git_blob_sha1": "da6ab6cc8f333d7e89a99812866df8f24374d47c",
    "license_sha256": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    "model_training": "ALLOWED_WITH_ATTRIBUTION_RETAINED_IN_PROVENANCE",
    "evaluation": "NOT_GRANTED_BY_DATA324",
}
EXPECTED_EVALUATION_FIREWALL = {
    "authority": "EVAL-290-UA-SELECTION-VALIDATION-V1",
    "reservation_commit_sha": "8a393d98d37b2090d1ae3fe8be8d4e58f651159e",
    "reserved_raw_sha256": [
        "50a790e0ece091f13fe039b5e36a23431680dec0357379f29b0029502f9b3a31",
        "e44c27b6151a1fea68eeef1e73e4460391f82e0501571d8aa4d5a792fc448b12",
    ],
    "final_test_accessed": False,
}


class PortValidationError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_json(root: Path, relative: Path) -> dict[str, Any]:
    try:
        value = json.loads((root / relative).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortValidationError(f"cannot load {relative}") from exc
    if not isinstance(value, dict):
        raise PortValidationError(f"{relative} root must be an object")
    return value


def _load_materializer() -> Any:
    spec = importlib.util.spec_from_file_location("data324_materializer_current", MATERIALIZER)
    if spec is None or spec.loader is None:
        raise PortValidationError("cannot load DATA-324 materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_self_identity(
    value: dict[str, Any], identity_field: str, expected: str, label: str
) -> None:
    actual = value.get(identity_field)
    if actual != expected or not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise PortValidationError(f"{label} declared identity mismatch")
    core = dict(value)
    core.pop(identity_field, None)
    if _sha256(_canonical_bytes(core)) != actual:
        raise PortValidationError(f"{label} self-identity recomputation failed")


def validate_snapshot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    port = _load_json(root, PORT_CONFIG)
    legacy = _load_json(root, LEGACY_CONFIG)
    manifest = _load_json(root, MANIFEST)
    report = _load_json(root, REPORT)
    materializer = _load_materializer()

    if set(port) != EXPECTED_TOP_LEVEL_KEYS:
        raise PortValidationError("current-main port top-level schema drift")
    if port.get("schema_version") != "12-6.data324-kubernetes-ua-current-main-port.v1":
        raise PortValidationError("current-main port schema drift")
    if port.get("worker_id") != EXPECTED_WORKER_ID:
        raise PortValidationError("current-main worker identity drift")
    if port.get("local_free_only") is not True:
        raise PortValidationError("current-main port must remain LOCAL_FREE")
    if port.get("ported_from") != EXPECTED_PORTED_FROM:
        raise PortValidationError("ported-from provenance drift")
    if port.get("port_base_main_sha") != EXPECTED_PORT_BASE_MAIN_SHA:
        raise PortValidationError("historical port base drift")

    source = port.get("source_authority")
    rights = port.get("rights_authority")
    firewall = port.get("evaluation_firewall")
    truth = port.get("current_main_truth_boundary")
    if not all(isinstance(v, dict) for v in (source, rights, firewall, truth)):
        raise PortValidationError("current-main authority objects missing")
    if source != EXPECTED_SOURCE_AUTHORITY:
        raise PortValidationError("source authority drift")
    if rights != EXPECTED_RIGHTS_AUTHORITY:
        raise PortValidationError("rights authority drift")
    if firewall != EXPECTED_EVALUATION_FIREWALL:
        raise PortValidationError("evaluation firewall drift")

    for field in ("upstream_revision",):
        if not isinstance(source.get(field), str) or not HEX40.fullmatch(source[field]):
            raise PortValidationError(f"malformed {field}")
    for field in (
        "raw_sha256",
        "normalized_sha256",
        "manifest_identity_sha256",
        "report_identity_sha256",
    ):
        if not isinstance(source.get(field), str) or not HEX64.fullmatch(source[field]):
            raise PortValidationError(f"malformed source_authority.{field}")
    reserved = firewall.get("reserved_raw_sha256")
    if not isinstance(reserved, list) or not reserved:
        raise PortValidationError("evaluation reservation set missing")
    if any(not isinstance(v, str) or not HEX64.fullmatch(v) for v in reserved):
        raise PortValidationError("evaluation reservation identity malformed")

    raw = (root / RAW).read_bytes()
    normalized = (root / NORMALIZED).read_bytes()
    license_bytes = (root / LICENSE).read_bytes()
    attribution = (root / ATTRIBUTION).read_text(encoding="utf-8")

    if len(raw) != source.get("raw_bytes") or _sha256(raw) != source.get("raw_sha256"):
        raise PortValidationError("raw snapshot identity mismatch")
    if _git_blob_sha1(raw) != source.get("source_git_blob_sha1"):
        raise PortValidationError("raw snapshot Git-blob identity mismatch")
    if (
        len(normalized) != source.get("normalized_utf8_bytes")
        or _sha256(normalized) != source.get("normalized_sha256")
    ):
        raise PortValidationError("normalized snapshot identity mismatch")
    if materializer.normalize_markdown_uk(raw).encode("utf-8") != normalized:
        raise PortValidationError("normalized bytes do not reproduce from frozen raw bytes")
    if _sha256(license_bytes) != rights.get("license_sha256"):
        raise PortValidationError("license SHA-256 mismatch")
    if _git_blob_sha1(license_bytes) != rights.get("license_git_blob_sha1"):
        raise PortValidationError("license Git-blob identity mismatch")
    if "CC BY 4.0" not in attribution or "Kubernetes Authors" not in attribution:
        raise PortValidationError("required attribution evidence missing")

    _require_self_identity(
        manifest,
        "manifest_identity_sha256",
        source["manifest_identity_sha256"],
        "manifest",
    )
    _require_self_identity(
        report,
        "report_identity_sha256",
        source["report_identity_sha256"],
        "report",
    )

    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != 1 or not isinstance(objects[0], dict):
        raise PortValidationError("manifest must bind exactly one source object")
    obj = objects[0]
    expected_pairs = {
        "path": source.get("source_path"),
        "git_blob_sha1": source.get("source_git_blob_sha1"),
        "raw_sha256": source.get("raw_sha256"),
        "raw_bytes": source.get("raw_bytes"),
        "normalized_sha256": source.get("normalized_sha256"),
        "normalized_utf8_bytes": source.get("normalized_utf8_bytes"),
    }
    if any(obj.get(key) != expected for key, expected in expected_pairs.items()):
        raise PortValidationError("manifest source object drift")
    if manifest.get("source_family") != source.get("source_family"):
        raise PortValidationError("source family drift")
    if manifest.get("canonical_upstream") != source.get("canonical_upstream"):
        raise PortValidationError("canonical upstream drift")
    if manifest.get("upstream_revision") != source.get("upstream_revision"):
        raise PortValidationError("upstream revision drift")

    if rights.get("license_id") != "CC-BY-4.0":
        raise PortValidationError("license id drift")
    if rights.get("model_training") != "ALLOWED_WITH_ATTRIBUTION_RETAINED_IN_PROVENANCE":
        raise PortValidationError("model-training rights drift")
    if rights.get("evaluation") != "NOT_GRANTED_BY_DATA324":
        raise PortValidationError("evaluation rights drift")
    if manifest.get("evaluation") != "NOT_SEPARATELY_ADMITTED":
        raise PortValidationError("manifest evaluation firewall weakened")
    if report.get("evaluation_decision") != "NOT_ADMITTED_REQUIRE_SEPARATE_AUTHORITY":
        raise PortValidationError("report evaluation firewall weakened")
    if source["raw_sha256"] in set(reserved):
        raise PortValidationError("training source collides with evaluation reservation")
    if firewall.get("final_test_accessed") is not False:
        raise PortValidationError("final-test truth boundary weakened")

    expected_truth = {
        "source_level_training_eligible_snapshot": True,
        "current_global_dedup": "NOT_RUN_REQUIRE_CURRENT_CANONICAL_OWNER",
        "evaluation_decontamination": "NOT_RUN_REQUIRE_RESERVED_EVAL_AUTHORITY",
        "post_composition_quality_privacy": "NOT_RUN_AT_COMPOSED_CORPUS_LEVEL",
        "family_credit_authorized": False,
        "corpus_admitted": False,
        "training_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
        "learned_20m_promoted": False,
    }
    if truth != expected_truth:
        raise PortValidationError("current-main truth boundary drift")
    if manifest.get("training_executed") is not False or report.get("training_executed") is not False:
        raise PortValidationError("historical snapshot unexpectedly claims training")

    # Preserve the old acquisition contract as provenance, but do not promote its
    # then-current exact-dedup preview into a current-main composed-corpus claim.
    if legacy.get("family_identity", {}).get("source_family") != source["source_family"]:
        raise PortValidationError("legacy source-family provenance drift")

    lang = materializer.language_evidence(normalized.decode("utf-8"))
    privacy = materializer.privacy_evidence(normalized.decode("utf-8"))
    if lang != manifest.get("language_evidence"):
        raise PortValidationError("language evidence no longer reproduces")
    if privacy != manifest.get("privacy_evidence"):
        raise PortValidationError("privacy evidence no longer reproduces")

    return {
        "decision": "PASS_SOURCE_LEVEL_CURRENT_MAIN_PORT",
        "source_family": source["source_family"],
        "raw_sha256": source["raw_sha256"],
        "normalized_sha256": source["normalized_sha256"],
        "normalized_utf8_bytes": source["normalized_utf8_bytes"],
        "manifest_identity_sha256": source["manifest_identity_sha256"],
        "report_identity_sha256": source["report_identity_sha256"],
        "evaluation_reservation_collision": False,
        "current_global_dedup": truth["current_global_dedup"],
        "family_credit_authorized": False,
        "training_authorized_bytes": 0,
        "model_training_executed": False,
        "paid_compute_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(validate_snapshot(args.root), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
