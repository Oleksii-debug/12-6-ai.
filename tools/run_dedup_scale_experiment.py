"""Run the D03 DATA-12 LOCAL_FREE exact/MinHash/benchmark contamination experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
import tracemalloc
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

try:
    import resource
except ImportError:  # pragma: no cover - Windows portability
    resource = None

from twelve_six.data.corpus_foundation import (
    SQLiteExactDedupIndex,
    StreamingShardPlan,
    reserved_registry_from_d06_manifest,
)
from twelve_six.data.dedup_scale import (
    DataTroveMinhashExecutionPlan,
    build_dedup_output_manifest,
    build_training_eligibility_envelope,
    run_datatrove_candidate_dedup,
    run_datatrove_reference_index,
    validate_datatrove_runtime,
)
from twelve_six.data.pipeline import normalize_text

REPORT_SCHEMA = "12-6.dedup-scale-experiment.v1"
CONFIG_SCHEMA = "12-6.dedup-scale-experiment-config.v1"
RESERVED_SOURCE_ID = "synthetic/d06/reserved/v1"
TRAIN_SOURCE_ID = "synthetic/local-free/train/v1"
COMMON_EN = (
    "corpus provenance filtering deduplication benchmark validation training evidence "
    "deterministic manifest sharding restart language model data quality memory throughput "
    "identity source policy exact near duplicate contamination evaluation"
).split()
COMMON_UK = (
    "корпус походження фільтрація дедуплікація бенчмарк перевірка навчання доказ "
    "детермінований маніфест шард перезапуск мова модель дані якість пам'ять "
    "пропускна здатність ідентичність джерело політика точний близький дублікат"
).split()


def _canonical(value: Any, *, newline: bool = True) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + ("\n" if newline else "")).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha(text: str) -> str:
    return _sha(normalize_text(text).encode("utf-8"))


def _jsonl(folder: Path) -> Iterator[dict[str, Any]]:
    paths = [folder] if folder.is_file() else sorted(folder.rglob("*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _d06_manifest() -> dict[str, Any]:
    payload = {
        "schema_version": "12-6.benchmark-registry.v1",
        "benchmarks": [
            {
                "benchmark_id": "data12-controlled-reserved",
                "version": "v1",
                "source_id": RESERVED_SOURCE_ID,
                "held_out": True,
                "allowed_uses": ["evaluation"],
                "license_id": None,
                "source_url": None,
                "notes": "LOCAL_FREE synthetic contamination mechanics fixture",
            }
        ],
    }
    return {**payload, "manifest_sha256": _sha(_canonical(payload, newline=False))}


def _english(index: int, words: int) -> str:
    return " ".join(COMMON_EN + [f"eng{index:06d}_{i:03d}" for i in range(words)])


def _ukrainian(index: int, words: int) -> str:
    return " ".join(COMMON_UK + [f"укр{index:06d}_{i:03d}" for i in range(words)])


def _code(index: int) -> str:
    body = [
        f"    value_{index}_{line} = seed + {line}  # deterministic token {index}_{line}"
        for line in range(80)
    ]
    return "\n".join([f"def transform_{index}(seed):", *body, f"    return value_{index}_79"])


def _base(index: int, words: int) -> tuple[str, str]:
    if index % 10 == 0:
        return "code", _code(index)
    if index % 3 == 0:
        return "uk", _ukrainian(index, words)
    return "en", _english(index, words)


def _near(text: str, marker: str) -> str:
    tokens = text.split()
    for n, position in enumerate((len(tokens) // 4, len(tokens) // 2, 3 * len(tokens) // 4)):
        tokens[position] = f"{marker}_{n}"
    return " ".join(tokens)


def _code_copy(text: str, marker: str) -> str:
    lines = text.splitlines()
    lines[0] = lines[0].replace("transform_", f"transform_copy_{marker}_", 1)
    lines[20] = lines[20].split("#", 1)[0] + f"# copied implementation {marker}"
    return "\n".join(lines)


def _boilerplate(index: int, words: int) -> str:
    shared = " ".join(COMMON_EN * 2)
    unique = " ".join(f"boiler{index:06d}_{i:03d}" for i in range(words + 40))
    return f"{shared} {unique} standard footer terms conditions navigation contact archive"


def _reserved(index: int, words: int) -> tuple[str, str]:
    if index % 2:
        return "uk", " ".join(COMMON_UK + [f"резерв{index:04d}_{i:03d}" for i in range(words)])
    return "en", " ".join(COMMON_EN + [f"reserved{index:04d}_{i:03d}" for i in range(words)])


def _translation(index: int, source_language: str, words: int) -> str:
    if source_language == "uk":
        prefix = "evaluation reference translated concept preserving meaning across languages changing lexical surface"
        body = " ".join(f"translation_en_{index:04d}_{i:03d}" for i in range(words))
    else:
        prefix = "оцінювання еталон перекладене поняття зберігає значення між мовами змінює лексику"
        body = " ".join(f"переклад_ук_{index:04d}_{i:03d}" for i in range(words))
    return f"{prefix} {body}"


def _validate(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported experiment config schema")
    out = dict(config)
    for field in ("record_count", "candidate_shards", "workers", "base_words", "reserved_records"):
        value = out.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    injections = out.get("injections")
    if not isinstance(injections, dict):
        raise ValueError("injections must be an object")
    for field in (
        "exact_duplicate", "near_en", "near_uk", "code_copy", "boilerplate_negative",
        "benchmark_exact", "benchmark_source_id", "benchmark_near_en", "benchmark_near_uk",
        "benchmark_translation",
    ):
        value = injections.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"injections.{field} must be a non-negative integer")
    return out


def _parent_indices(base_count: int, injections: Mapping[str, int]) -> dict[str, list[int]]:
    code = [i for i in range(base_count) if i % 10 == 0]
    uk = [i for i in range(base_count) if i % 10 != 0 and i % 3 == 0]
    en = [i for i in range(base_count) if i % 10 != 0 and i % 3 != 0]
    exact_n = injections["exact_duplicate"]
    near_en_n = injections["near_en"]
    if len(en) < exact_n + near_en_n or len(uk) < injections["near_uk"] or len(code) < injections["code_copy"]:
        raise ValueError("record_count is too small for requested injection families")
    return {
        "exact_duplicate": en[:exact_n],
        "near_en": en[exact_n : exact_n + near_en_n],
        "near_uk": uk[: injections["near_uk"]],
        "code_copy": code[: injections["code_copy"]],
    }


def _generate(config: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, set[str]]]:
    injections = config["injections"]
    base_count = int(config["record_count"]) - sum(injections.values())
    if base_count <= 0:
        raise ValueError("record_count must exceed injections")
    parents = _parent_indices(base_count, injections)
    family_for_base: dict[int, list[str]] = defaultdict(list)
    for category in ("near_en", "near_uk", "code_copy"):
        for ordinal, base_index in enumerate(parents[category]):
            family_for_base[base_index].append(f"{category}:{ordinal:05d}")

    reserved_pairs = [_reserved(i, int(config["base_words"])) for i in range(int(config["reserved_records"]))]
    reserved_records = [
        {
            "id": f"reserved-{i:05d}",
            "text": normalize_text(text),
            "source_id": RESERVED_SOURCE_ID,
            "metadata": {"category": "reserved_reference", "language": language, "reserved": True},
        }
        for i, (language, text) in enumerate(reserved_pairs)
    ]
    reserved_path = root / "reserved" / "00000.jsonl"
    _write_jsonl(reserved_path, reserved_records)
    reserved_registry = reserved_registry_from_d06_manifest(
        _d06_manifest(), {RESERVED_SOURCE_ID: [_content_sha(item["text"]) for item in reserved_records]}
    )

    truth = {
        "internal_near_families": set(),
        "benchmark_lexical_ids": set(),
        "benchmark_translation_ids": set(),
        "boilerplate_ids": set(),
    }
    raw = root / "raw" / "candidates.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    stream_hash = hashlib.sha256()

    def emit(handle, record: dict[str, Any]) -> None:
        record["text"] = normalize_text(record["text"])
        record["content_sha256"] = _content_sha(record["text"])
        stream_hash.update(_canonical({
            "id": record["id"], "source_id": record["source_id"],
            "content_sha256": record["content_sha256"], "category": record["metadata"]["category"],
        }))
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with raw.open("w", encoding="utf-8") as handle:
        for index in range(base_count):
            language, text = _base(index, int(config["base_words"]))
            families = family_for_base.get(index, [])
            emit(handle, {"id": f"base-{index:07d}", "text": text, "source_id": TRAIN_SOURCE_ID,
                "metadata": {"category": "base", "language": language, "family_ids": families,
                             "expected_duplicate_member": bool(families)}})

        for ordinal, parent in enumerate(parents["exact_duplicate"]):
            _, text = _base(parent, int(config["base_words"]))
            emit(handle, {"id": f"inject-exact-{ordinal:05d}", "text": text, "source_id": TRAIN_SOURCE_ID,
                "metadata": {"category": "exact_duplicate", "language": "en", "family_ids": [],
                             "expected_duplicate_member": True}})

        for category in ("near_en", "near_uk"):
            for ordinal, parent in enumerate(parents[category]):
                language, text = _base(parent, int(config["base_words"]))
                family = f"{category}:{ordinal:05d}"
                truth["internal_near_families"].add(family)
                emit(handle, {"id": f"inject-{category}-{ordinal:05d}", "text": _near(text, f"{category}_{ordinal}"),
                    "source_id": TRAIN_SOURCE_ID, "metadata": {"category": category, "language": language,
                    "family_ids": [family], "expected_duplicate_member": True}})

        for ordinal, parent in enumerate(parents["code_copy"]):
            _, text = _base(parent, int(config["base_words"]))
            family = f"code_copy:{ordinal:05d}"
            truth["internal_near_families"].add(family)
            emit(handle, {"id": f"inject-code-{ordinal:05d}", "text": _code_copy(text, str(ordinal)),
                "source_id": TRAIN_SOURCE_ID, "metadata": {"category": "code_copy", "language": "code",
                "family_ids": [family], "expected_duplicate_member": True}})

        for ordinal in range(injections["boilerplate_negative"]):
            record_id = f"inject-boilerplate-{ordinal:05d}"
            truth["boilerplate_ids"].add(record_id)
            emit(handle, {"id": record_id, "text": _boilerplate(ordinal, int(config["base_words"])),
                "source_id": TRAIN_SOURCE_ID, "metadata": {"category": "boilerplate_negative", "language": "en",
                "family_ids": [], "expected_duplicate_member": False}})

        for ordinal in range(injections["benchmark_exact"]):
            language, text = reserved_pairs[ordinal % len(reserved_pairs)]
            emit(handle, {"id": f"inject-benchmark-exact-{ordinal:05d}", "text": text, "source_id": TRAIN_SOURCE_ID,
                "metadata": {"category": "benchmark_exact", "language": language, "family_ids": [],
                             "expected_duplicate_member": True}})

        for ordinal in range(injections["benchmark_source_id"]):
            language, text = _base(base_count + ordinal + 1, int(config["base_words"]))
            emit(handle, {"id": f"inject-benchmark-source-{ordinal:05d}", "text": text,
                "source_id": RESERVED_SOURCE_ID, "metadata": {"category": "benchmark_source_id", "language": language,
                "family_ids": [], "expected_duplicate_member": True}})

        for category, language in (("benchmark_near_en", "en"), ("benchmark_near_uk", "uk")):
            matching = [(i, text) for i, (lang, text) in enumerate(reserved_pairs) if lang == language]
            for ordinal in range(injections[category]):
                reserved_index, text = matching[ordinal % len(matching)]
                record_id = f"inject-{category}-{ordinal:05d}"
                truth["benchmark_lexical_ids"].add(record_id)
                emit(handle, {"id": record_id, "text": _near(text, f"{category}_{reserved_index}_{ordinal}"),
                    "source_id": TRAIN_SOURCE_ID, "metadata": {"category": category, "language": language,
                    "family_ids": [], "expected_duplicate_member": True,
                    "reserved_reference_id": f"reserved-{reserved_index:05d}"}})

        for ordinal in range(injections["benchmark_translation"]):
            reserved_index = ordinal % len(reserved_pairs)
            source_language, _ = reserved_pairs[reserved_index]
            record_id = f"inject-benchmark-translation-{ordinal:05d}"
            truth["benchmark_translation_ids"].add(record_id)
            emit(handle, {"id": record_id,
                "text": _translation(reserved_index, source_language, int(config["base_words"])),
                "source_id": TRAIN_SOURCE_ID, "metadata": {"category": "benchmark_translation",
                "language": "en" if source_language == "uk" else "uk", "family_ids": [],
                "expected_duplicate_member": True, "known_semantic_relation": True,
                "reserved_reference_id": f"reserved-{reserved_index:05d}"}})

    config_sha = _sha(_canonical(dict(config)))
    source_identity = _sha(_canonical({"schema_version": "12-6.local-free-synthetic-source.v1",
        "generator": "tools/run_dedup_scale_experiment.py", "config_sha256": config_sha,
        "external_source_approval": "NOT_APPLICABLE_LOCAL_FREE_SYNTHETIC"}))
    core = {"schema_version": "12-6.dedup-scale-input-manifest.v1", "execution_class": "LOCAL_FREE_SYNTHETIC",
        "config_sha256": config_sha, "synthetic_source_identity_sha256": source_identity,
        "candidate_records": int(config["record_count"]), "reserved_records": int(config["reserved_records"]),
        "candidate_metadata_stream_sha256": stream_hash.hexdigest(), "raw_candidate_sha256": _file_sha(raw),
        "reserved_jsonl_sha256": _file_sha(reserved_path),
        "d06_benchmark_manifest_sha256": _d06_manifest()["manifest_sha256"],
        "reserved_registry_sha256": reserved_registry["registry_identity_sha256"]}
    return {**core, "manifest_sha256": _sha(_canonical(core))}, reserved_registry, truth


def _exact_and_shard(raw: Path, output: Path, source_sha: str, reserved_registry: Mapping[str, Any], shards: int) -> dict[str, Any]:
    reserved_sources = {item["source_id"] for item in reserved_registry["sets"]}
    reserved_hashes = {digest for item in reserved_registry["sets"] for digest in item["normalized_sha256"]}
    plan = StreamingShardPlan(source_sha, reserved_registry["registry_identity_sha256"], output.resolve().as_uri(),
        "data12-dedup-scale-v1", shards, 512, 8 * 1024 * 1024)
    output.mkdir(parents=True, exist_ok=True)
    db = output.parent / "exact.sqlite3"
    counts: dict[str, int] = defaultdict(int)
    shard_counts = [0] * shards
    total = sum(1 for _ in raw.open("r", encoding="utf-8"))
    boundary = total // 2
    tracemalloc.start()
    started = time.perf_counter()
    with raw.open("r", encoding="utf-8") as source, ExitStack() as stack:
        handles = [stack.enter_context((output / f"{i:05d}.jsonl").open("w", encoding="utf-8")) for i in range(shards)]

        def process(index: SQLiteExactDedupIndex, line: str) -> None:
            record = json.loads(line)
            counts["input"] += 1
            if record["source_id"] in reserved_sources:
                counts["reserved_source_excluded"] += 1
                return
            if record["content_sha256"] in reserved_hashes:
                counts["reserved_content_excluded"] += 1
                return
            if index.seen_or_add(record["content_sha256"]):
                counts["exact_duplicates_removed"] += 1
                return
            shard = plan.assign(record["id"])
            handles[shard].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            shard_counts[shard] += 1
            counts["survivors"] += 1

        with SQLiteExactDedupIndex(db) as index:
            for _ in range(boundary):
                process(index, next(source))
            index.commit()
        first_half_sha = _file_sha(db)
        with SQLiteExactDedupIndex(db) as index:
            for line in source:
                process(index, line)
            index.commit()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"counts": dict(counts), "elapsed_seconds": elapsed, "records_per_second": counts["input"] / elapsed,
        "tracemalloc_peak_bytes": peak, "sqlite_bytes": db.stat().st_size, "first_half_sqlite_sha256": first_half_sha,
        "final_sqlite_sha256": _file_sha(db), "restart_exercised": True,
        "shard_plan_sha256": plan.manifest()["plan_sha256"], "shard_counts": shard_counts,
        "shard_min_records": min(shard_counts), "shard_max_records": max(shard_counts)}


def _metrics(removed: Path, output: Path, truth: Mapping[str, set[str]]) -> dict[str, Any]:
    removed_records = list(_jsonl(removed))
    output_records = list(_jsonl(output))
    removed_ids = {record["id"] for record in removed_records}
    output_ids = {record["id"] for record in output_records}
    removed_families: set[str] = set()
    unexpected = 0
    boilerplate = 0
    for record in removed_records:
        metadata = record.get("metadata", {})
        removed_families.update(metadata.get("family_ids", []))
        if metadata.get("category") == "boilerplate_negative":
            boilerplate += 1
        if not metadata.get("expected_duplicate_member") and metadata.get("category") != "benchmark_translation":
            unexpected += 1
    internal = truth["internal_near_families"]
    lexical = truth["benchmark_lexical_ids"]
    translations = truth["benchmark_translation_ids"]
    detected = internal & removed_families
    lexical_removed = lexical & removed_ids
    translated_removed = translations & removed_ids
    translated_survived = translations & output_ids
    denominator = len(output_ids | removed_ids)
    return {"removed_records": len(removed_records), "output_records": len(output_records),
        "internal_near_families_expected": len(internal), "internal_near_families_detected": len(detected),
        "internal_near_false_negatives": len(internal - detected),
        "internal_near_recall": len(detected) / len(internal) if internal else 1.0,
        "benchmark_lexical_expected": len(lexical), "benchmark_lexical_removed": len(lexical_removed),
        "benchmark_lexical_survived": len(lexical - lexical_removed),
        "benchmark_lexical_recall": len(lexical_removed) / len(lexical) if lexical else 1.0,
        "benchmark_translation_expected": len(translations), "benchmark_translation_removed": len(translated_removed),
        "benchmark_translation_survived": len(translated_survived),
        "benchmark_translation_detection_rate": len(translated_removed) / len(translations) if translations else 1.0,
        "boilerplate_negative_expected": len(truth["boilerplate_ids"]), "boilerplate_negative_removed": boilerplate,
        "unexpected_removed": unexpected, "unexpected_removal_rate": unexpected / denominator if denominator else 0.0}


def _hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): _file_sha(path) for path in sorted(folder.rglob("*")) if path.is_file()}


def run(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    config = _validate(config)
    input_manifest, reserved_registry, truth = _generate(config, root)
    exact = _exact_and_shard(root / "raw" / "candidates.jsonl", root / "candidate_exact",
        input_manifest["synthetic_source_identity_sha256"], reserved_registry, int(config["candidate_shards"]))
    minhash = config["minhash"]
    plan = DataTroveMinhashExecutionPlan(
        source_registry_sha256=input_manifest["synthetic_source_identity_sha256"],
        reserved_registry_sha256=reserved_registry["registry_identity_sha256"],
        input_manifest_sha256=input_manifest["manifest_sha256"], workspace_uri=root.resolve().as_uri(),
        candidate_shards=int(config["candidate_shards"]), workers=int(config["workers"]),
        n_grams=int(minhash["n_grams"]), num_buckets=int(minhash["num_buckets"]),
        hashes_per_bucket=int(minhash["hashes_per_bucket"]), minhash_seed=int(minhash["seed"]),
        hash_precision=int(minhash["hash_precision"]))
    runtime = validate_datatrove_runtime(plan)
    dt_started = time.perf_counter()
    reference = run_datatrove_reference_index(plan, reference_input=root / "reserved", workspace=root / "datatrove",
        index_name="d06_reserved_v1")
    candidate = run_datatrove_candidate_dedup(plan, candidate_input=root / "candidate_exact", workspace=root / "datatrove",
        reference_index=Path(reference["reference_index"]), exercise_restart=True)
    dt_elapsed = time.perf_counter() - dt_started
    family = _metrics(Path(candidate["removed"]), Path(candidate["output"]), truth)
    thresholds = config["thresholds"]
    acceptance = (family["internal_near_recall"] >= float(thresholds["internal_near_recall_min"])
        and family["benchmark_lexical_recall"] >= float(thresholds["benchmark_lexical_recall_min"])
        and family["unexpected_removal_rate"] <= float(thresholds["unexpected_removal_rate_max"])
        and bool(candidate["restart"].get("verified")))
    metrics_core = {"exact": exact, "minhash": family, "datatrove_elapsed_seconds": dt_elapsed,
        "datatrove_input_records_per_second": exact["counts"]["survivors"] / dt_elapsed if dt_elapsed else 0.0,
        "restart": candidate["restart"], "thresholds": thresholds, "experiment_acceptance_pass": acceptance}
    output_manifest = build_dedup_output_manifest(plan=plan, input_records=int(config["record_count"]),
        exact_survivors=exact["counts"]["survivors"], final_survivors=family["output_records"],
        output_files=_hashes(Path(candidate["output"])), metrics_sha256=_sha(_canonical(metrics_core)))
    eligibility = build_training_eligibility_envelope(output_manifest=output_manifest, source_rights_eligible=False,
        record_policy_eligible=False, exact_reserved_overlap_count=0,
        lexical_reserved_overlap_count=family["benchmark_lexical_survived"],
        known_semantic_overlap_count=family["benchmark_translation_survived"], experiment_acceptance_pass=acceptance)
    if resource is None:
        rss_self = rss_children = None
    else:
        rss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    core = {"schema_version": REPORT_SCHEMA, "execution_class": "LOCAL_FREE_SYNTHETIC", "paid_cost_usd": 0,
        "config": config, "config_sha256": _sha(_canonical(config)), "input_manifest": input_manifest,
        "reserved_registry_identity_sha256": reserved_registry["registry_identity_sha256"],
        "dedup_scale_plan": plan.manifest(), "runtime": runtime, "metrics": metrics_core,
        "output_manifest": output_manifest, "eligibility": eligibility,
        "memory": {"peak_rss_self_kib": rss_self, "peak_rss_children_kib": rss_children,
                   "exact_stage_tracemalloc_peak_bytes": exact["tracemalloc_peak_bytes"],
                   "measurement_note": "Linux ru_maxrss self/children are maxima, not an additive total."},
        "truth_boundary": {"external_sources_approved": 0, "semantic_universal_cleanliness_claimed": False,
            "known_cross_language_translation_cases_are_calibration_only": True, "training_eligible_expected": False,
            "reason": "Synthetic mechanics lacks source-rights/policy approval and injects known cross-language benchmark relations."},
        "next_corpus_scale_target": {"records": 1000000, "minimum_uncompressed_bytes": 1073741824, "shards": 64,
            "requirements": ["same config/plan and D06 reference-index semantics",
                "intentional interrupted-stage restart", "retained peak RSS and wall-clock throughput",
                "approved-source run only after explicit rights registry approval"]},
        "platform": {"python": platform.python_version(), "platform": platform.platform()}}
    return {**core, "report_sha256": _sha(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        report = run(config, args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="data12-dedup-") as tmp:
            report = run(config, Path(tmp))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["metrics"]["experiment_acceptance_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
