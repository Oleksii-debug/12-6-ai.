"""DATA-227 external-real code admission through DATA-24 D03 rights gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.data.code_normalization import decode_code_bytes
from twelve_six.data.external_sources import (
    PROJECT_RIGHTS_POLICY_REF,
    RIGHTS_APPROVED,
    USE_ALLOWED,
    EligibilityResolver,
    ExternalSourceSpec,
    RightsDecision,
    RightsEvidenceRef,
    SnapshotSpec,
    UsePermissions,
    build_external_source_registry,
    verify_local_snapshot,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.data227-real-code-source-admission.v2"
REPOSITORY = "Oleksii-debug/12-6-ai."
POLICY_PATH = Path("configs/data/data227_code_rights_policy_v1.json")
REPORT_NAME = "data227-real-code-source-admission.json"
NORMALIZATION_POLICY = "STRICT_UTF8_IDENTITY_PRESERVE_V1"
NEAR_SHINGLE_SIZE = 5
NEAR_THRESHOLD = 0.85
BANNED_PATH_PARTS = frozenset({
    "vendor", "vendored", "vendors", "third_party", "third-party", "node_modules",
    "dist", "build", "generated", "gen", "deps", "dependencies", ".venv", "site-packages",
})
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


class Data227Error(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _load_policy(repo: Path) -> dict[str, Any]:
    value = json.loads((repo / POLICY_PATH).read_bytes())
    if value.get("schema_version") != "12-6.data227-code-rights-policy.v1":
        raise Data227Error("unsupported DATA-227 rights policy schema")
    if value.get("policy_ref") != PROJECT_RIGHTS_POLICY_REF:
        raise Data227Error("DATA-227 policy_ref does not match DATA-24 authority")
    decisions = value.get("decisions")
    if not isinstance(decisions, list) or len(decisions) < 2:
        raise Data227Error("at least two code-source rights decisions are required")
    return value


def _require_head(repo: Path, source_sha: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise Data227Error(f"exact-head mismatch: {actual} != {source_sha}")


def _download(url: str, *, max_bytes: int = 250_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-DATA-227-rights-provenance/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise Data227Error(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _repo_metadata(repo_url: str) -> dict[str, Any]:
    prefix = "https://github.com/"
    if not repo_url.startswith(prefix):
        raise Data227Error(f"repository_url is not canonical GitHub https: {repo_url}")
    slug = repo_url[len(prefix):].strip("/")
    if slug.count("/") != 1:
        raise Data227Error(f"invalid repository slug: {slug}")
    payload = json.loads(_download(f"https://api.github.com/repos/{slug}", max_bytes=100_000))
    if payload.get("html_url") != repo_url:
        raise Data227Error(f"canonical repository identity mismatch for {repo_url}")
    if payload.get("fork") is not False:
        raise Data227Error(f"fork repository excluded: {repo_url}")
    if payload.get("mirror_url") is not None:
        raise Data227Error(f"mirror repository excluded: {repo_url}")
    return {"repository_url": repo_url, "fork": payload.get("fork"), "mirror_url": payload.get("mirror_url"), "archived": payload.get("archived")}


def _assert_path_allowed(path: str) -> None:
    parts = [part.casefold() for part in Path(path).parts]
    if any(part in BANNED_PATH_PARTS for part in parts):
        raise Data227Error(f"vendored/generated/build path excluded: {path}")
    lowered = path.casefold()
    if lowered.endswith((".min.js", ".min.mjs", ".min.cjs", ".map", ".pyc", ".so", ".dll")):
        raise Data227Error(f"minified/binary path excluded: {path}")
    if not lowered.endswith((".py", ".pyi", ".c", ".h", ".cc", ".cpp", ".rs", ".go", ".java", ".js", ".ts")):
        raise Data227Error(f"unsupported source-code extension: {path}")


def _assert_no_secrets(data: bytes, path: str) -> None:
    for pattern in _SECRET_PATTERNS:
        if pattern.search(data):
            raise Data227Error(f"secret-like material excluded: {path}")


def _token_shingles(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", text.casefold())
    if len(tokens) < NEAR_SHINGLE_SIZE:
        return set()
    return {tuple(tokens[i : i + NEAR_SHINGLE_SIZE]) for i in range(len(tokens) - NEAR_SHINGLE_SIZE + 1)}


def _near_jaccard(left: str, right: str) -> float:
    a, b = _token_shingles(left), _token_shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _rights(decision: dict[str, Any], *, license_sha256: str, policy_sha256: str) -> RightsDecision:
    source_id = decision["source_id"]
    source_version = f"git:{decision['commit']}"
    captured_at = decision["captured_at"]
    uses = decision["uses"]
    permissions = UsePermissions(acquisition=uses["acquisition"], storage=uses["storage"], analysis=uses["analysis"], model_training=uses["model_training"], redistribution=uses["redistribution"])
    return RightsDecision(
        status=RIGHTS_APPROVED,
        license_id=decision["license_id"],
        terms_url=decision["license_url"],
        allows_model_training=True,
        allows_derivatives=True,
        allows_redistribution=permissions.redistribution == USE_ALLOWED,
        policy_ref=PROJECT_RIGHTS_POLICY_REF,
        reviewed_at=decision["reviewed_at"],
        reviewer_ref=decision["reviewer_ref"],
        uses=permissions,
        evidence_refs=(
            RightsEvidenceRef(evidence_id=f"{source_id}.license", evidence_kind="license_text", uri=decision["license_url"], sha256=license_sha256, captured_at=captured_at, source_id=source_id, source_version=source_version),
            RightsEvidenceRef(evidence_id=f"{source_id}.policy-decision", evidence_kind="policy_decision", uri=f"file:{POLICY_PATH.as_posix()}", sha256=policy_sha256, captured_at=captured_at, source_id=source_id, source_version=source_version),
        ),
    )


def _materialize_one(repo: Path, output: Path, decision: dict[str, Any], *, policy_sha256: str) -> tuple[ExternalSourceSpec, dict[str, Any], str]:
    _assert_path_allowed(decision["path"])
    metadata = _repo_metadata(decision["repository_url"])
    raw = _download(decision["raw_url"])
    if len(raw) != decision["size_bytes"]:
        raise Data227Error(f"{decision['source_id']}: raw size drift")
    if _git_blob_sha1(raw) != decision["blob_sha1"]:
        raise Data227Error(f"{decision['source_id']}: raw Git blob identity drift")
    _assert_no_secrets(raw, decision["path"])
    normalized, norm = decode_code_bytes(raw, language=decision["language"], path=decision["path"])
    normalized_bytes = normalized.encode("utf-8")
    if normalized_bytes != raw:
        raise Data227Error(f"{decision['source_id']}: code normalization mutated bytes")
    if norm.policy != NORMALIZATION_POLICY:
        raise Data227Error(f"{decision['source_id']}: unexpected code normalization policy")

    license_bytes = _download(decision["license_url"])
    if _git_blob_sha1(license_bytes) != decision["license_blob_sha1"]:
        raise Data227Error(f"{decision['source_id']}: license Git blob identity drift")
    license_sha256 = _sha256(license_bytes)
    license_out = output / "rights-evidence" / (decision["source_family"].replace(":", "_").replace("/", "_") + ".license.txt")
    license_out.parent.mkdir(parents=True, exist_ok=True)
    license_out.write_bytes(license_bytes)

    raw_sha256 = _sha256(raw)
    snapshot_rel = Path("data/external/snapshots/sha256") / raw_sha256 / "payload"
    snapshot_path = repo / snapshot_rel
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(raw)
    snapshot = SnapshotSpec(
        uri=f"file:{snapshot_rel.as_posix()}", sha256=raw_sha256, size_bytes=len(raw),
        retrieved_at=decision["captured_at"], upstream_version=f"git:{decision['commit']}",
        retrieval_method="DATA-227 pinned raw GitHub fetch + Git blob SHA-1 gate",
    )
    verify_local_snapshot(snapshot, snapshot_path)
    source = ExternalSourceSpec(
        source_id=decision["source_id"], source_version=f"git:{decision['commit']}", provider=decision["provider"],
        source_url=decision["raw_url"], source_kind="source_code", purpose="pretraining", synthetic=False,
        benchmark_material=False, held_out=False, snapshot=snapshot,
        rights=_rights(decision, license_sha256=license_sha256, policy_sha256=policy_sha256),
    )
    evidence = {
        "source_id": decision["source_id"], "source_family": decision["source_family"], "repository": metadata,
        "commit": decision["commit"], "path": decision["path"], "blob_sha1": decision["blob_sha1"],
        "raw_sha256": raw_sha256, "normalization_sha256": norm.normalized_sha256, "normalization_policy": norm.policy,
        "license_id": decision["license_id"], "license_path": decision["license_path"], "license_blob_sha1": decision["license_blob_sha1"],
        "license_sha256": license_sha256, "license_evidence_artifact": license_out.as_posix(),
        "training_purpose_decision": "ALLOWED", "redistribution_decision": decision["uses"]["redistribution"],
        "redistribution_conditions": decision["redistribution_conditions"], "source_manifest_sha256": source.source_manifest_sha256,
    }
    return source, evidence, normalized


def _trainer_proof(texts: list[tuple[str, str]]) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    records = [TextRecord(record_id=sid, text=text, split="train") for sid, text in texts]
    packed = list(iter_packed_examples(records, tokenizer, expected_split="train", sequence_length=64, cross_document=False))
    if len(packed) < 4:
        raise Data227Error("bounded Trainer proof requires at least four packed examples")
    torch.manual_seed(227)
    spec = ModelSpec(schema_version=1, vocab_size=256, max_seq_len=64, d_model=48, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=12, d_ff=128, rope_rotary_dim=12)
    model = TwelveSixDecoder(spec, InitSpec())
    config = TrainerConfig(learning_rate=3e-4, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8, max_steps=4, warmup_steps=0, scheduler="constant", gradient_accumulation_steps=1, gradient_clip_norm=1.0, precision="fp32", seed=227, deterministic_algorithms=True, deterministic_warn_only=False)
    trainer = Trainer(model, config, device="cpu")
    metrics = []
    source_ids: set[str] = set()
    selected = []
    for source_id, text in texts:
        source_records = [TextRecord(record_id=source_id, text=text, split="train")]
        source_examples = list(iter_packed_examples(source_records, tokenizer, expected_split="train", sequence_length=64, cross_document=False))
        selected.extend(source_examples[:2])
    if len(selected) < 4:
        raise Data227Error("Trainer proof could not take two streamed windows from each family")
    for example in selected[:4]:
        source_ids.update(example.record_ids)
        batch = {"input_ids": torch.tensor([example.input_ids], dtype=torch.long), "labels": torch.tensor([example.labels], dtype=torch.long)}
        step = trainer.train_microbatch(batch)
        metrics.append(asdict(step))
    if trainer.optimizer_step != 4:
        raise Data227Error("Trainer did not complete the bounded four-step proof")
    if len(source_ids) < 2:
        raise Data227Error("Trainer proof did not stream both independent source families")
    return {"passed": True, "optimizer_steps": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen, "source_ids_seen": sorted(source_ids), "model_spec_sha256": spec.identity_sha256(), "parameters": spec.parameter_count(), "packing_sequence_length": 64, "byte_tokenizer_vocab": tokenizer.vocab_size, "final_step": metrics[-1]}


def run(repo: Path, output: Path, source_sha: str) -> dict[str, Any]:
    _require_head(repo, source_sha)
    policy = _load_policy(repo)
    policy_sha256 = _sha256((repo / POLICY_PATH).read_bytes())
    sources: list[ExternalSourceSpec] = []
    objects: list[dict[str, Any]] = []
    normalized: list[tuple[str, str]] = []
    for decision in policy["decisions"]:
        source, evidence, text = _materialize_one(repo, output, decision, policy_sha256=policy_sha256)
        sources.append(source); objects.append(evidence); normalized.append((source.source_id, text))
    families = {item["source_family"] for item in objects}
    if len(families) < 2:
        raise Data227Error("fewer than two independent code source families survived")
    raw_hashes = [item["raw_sha256"] for item in objects]
    exact_duplicates = sorted({digest for digest in raw_hashes if raw_hashes.count(digest) > 1})
    near_pairs: list[dict[str, Any]] = []
    for i, (left_id, left_text) in enumerate(normalized):
        for right_id, right_text in normalized[i + 1:]:
            score = _near_jaccard(left_text, right_text)
            if score >= NEAR_THRESHOLD:
                near_pairs.append({"left": left_id, "right": right_id, "jaccard": score})
    if exact_duplicates or near_pairs:
        raise Data227Error("exact/near duplicate code objects are not admissible")
    registry = build_external_source_registry(sources)
    resolver = EligibilityResolver(registry)
    eligibility = [resolver.assert_model_training_eligible(source.source_id, source.source_version, source.source_manifest_sha256).to_dict() for source in sources]
    proof = _trainer_proof(normalized)
    core: dict[str, Any] = {
        "schema_version": SCHEMA, "authority": "EXTERNAL_REAL_CODE_D03_ADMISSION_LOCAL_FREE", "repository": REPOSITORY,
        "source_sha": source_sha, "policy_ref": PROJECT_RIGHTS_POLICY_REF, "policy_sha256": policy_sha256,
        "source_family_count": len(families), "source_families": sorted(families), "admitted_object_count": len(objects),
        "admitted_raw_bytes": sum(source.snapshot.size_bytes for source in sources), "objects": objects, "registry": registry,
        "eligibility_decisions": eligibility,
        "deduplication": {"exact_duplicate_sha256": exact_duplicates, "near_threshold": NEAR_THRESHOLD, "near_pairs": near_pairs},
        "trainer_streaming_proof": proof, "blocked_prior_sources_reinterpreted": False,
        "blocked_prior_sources": ["pallets/itsdangerous@672971d66a2ef9f85151e53283113f33d642dabd", "pytest-dev/pluggy@3b6d46ddfcef132e1e4edfc98d24ad1eb6c36b37"],
    }
    report = {**core, "report_sha256": _sha256(_cjson(core))}
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_NAME).write_bytes(_cjson(report))
    (output / "external-code-registry.json").write_bytes(_cjson(registry))
    return report


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    digest = report.pop("report_sha256", None)
    expected = _sha256(_cjson(report))
    if digest != expected:
        raise Data227Error("report self-hash mismatch")
    report["report_sha256"] = digest
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise Data227Error("report source SHA mismatch")
    if report.get("source_family_count", 0) < 2 or report.get("admitted_raw_bytes", 0) <= 0:
        raise Data227Error("report lacks authoritative external-real code diversity/bytes")
    if report["deduplication"]["exact_duplicate_sha256"] or report["deduplication"]["near_pairs"]:
        raise Data227Error("report contains duplicate code objects")
    if not report["trainer_streaming_proof"]["passed"]:
        raise Data227Error("Trainer streaming proof did not pass")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", default=".")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--source-sha", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report")
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run(Path(args.repo_root), Path(args.output_dir), args.source_sha)
    else:
        report = validate(Path(args.report), args.expected_source_sha)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
