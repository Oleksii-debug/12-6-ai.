from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

DEFAULT_CONFIG = Path("configs/data/d03_franko1901_exact_materialization_v1.json")
USER_AGENT = "12-6-ai-D03-Franko1901/1.0"
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def fetch_bounded(url: str, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    with urlopen(request, timeout=60) as response:
        final_url = response.geturl()
        if final_url != url:
            raise RuntimeError(f"unexpected redirect: {final_url}")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > max_bytes:
            raise RuntimeError("source exceeds configured bound before read")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError("source exceeds configured bound")
    return payload


def normalize_source_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def _has_forbidden_control(text: str) -> bool:
    for char in text:
        category = unicodedata.category(char)
        if category == "Cc" and char not in "\t\n\r":
            return True
    return False


def classify_text(text: str, policy: dict[str, object]) -> str | None:
    if not text:
        return "empty"
    if "\ufffd" in text and policy["reject_replacement_character"] is True:
        return "replacement_character"
    if policy["reject_control_characters"] is True and _has_forbidden_control(text):
        return "control_character"
    encoded = text.encode()
    if len(encoded) > int(policy["max_text_utf8_bytes"]):
        return "oversize_text"
    alpha = [char for char in text if char.isalpha()]
    if len(alpha) < int(policy["min_alphabetic_chars"]):
        return "too_few_alphabetic"
    cyrillic = [char for char in alpha if "\u0400" <= char <= "\u052f"]
    if len(cyrillic) / len(alpha) < float(policy["min_cyrillic_share_of_alpha"]):
        return "low_cyrillic_share"
    if policy["reject_url_like"] is True and URL_RE.search(text):
        return "url_like"
    if policy["reject_email_like"] is True and EMAIL_RE.search(text):
        return "email_like"
    return None


def validate_truth_boundary(config: dict[str, object]) -> None:
    truth = config["truth_boundary"]
    required_zero_false = {
        "canonical_capacity_credit_bytes": 0,
        "family_credit_authorized": False,
        "corpus_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "learned_20m_promoted": False,
    }
    if truth.get("candidate_only") is not True:
        raise RuntimeError("candidate-only boundary is not asserted")
    for key, expected in required_zero_false.items():
        if truth.get(key) != expected:
            raise RuntimeError(f"truth boundary drift: {key}")
    for key in (
        "global_dedup",
        "evaluation_decontamination",
        "post_composition_quality_privacy",
    ):
        if truth.get(key) != "NOT_RUN":
            raise RuntimeError(f"downstream gate must remain NOT_RUN: {key}")


def validate_contract(config: dict[str, object]) -> None:
    if config.get("local_free_only") is not True:
        raise RuntimeError("LOCAL_FREE boundary is not asserted")
    source = config["source"]
    if source["upstream_revision"] != "34a2c10ac35e1febad6c270a88fc8b83790407da":
        raise RuntimeError("upstream revision drift")
    if source["source_path"] != "data/sources/franko.csv":
        raise RuntimeError("source path drift")
    if source["source_git_blob_sha1"] != "45f33ac620907e1d1ed727524975b3a0fd1a0994":
        raise RuntimeError("source blob drift")
    rights = config["rights_boundary"]
    if rights["payload_field_allowed"] != "prov_clean_only":
        raise RuntimeError("payload field expansion is forbidden")
    forbidden = (
        "modern_text_allowed",
        "category_allowed",
        "cleaned_explanation_allowed",
        "variant_group_allowed",
    )
    if any(rights[key] is not False for key in forbidden):
        raise RuntimeError("LLM/enrichment fields must remain excluded")
    acquisition = config["acquisition"]
    if acquisition["fetch_count_required"] != 2:
        raise RuntimeError("two-fetch requirement drift")
    if acquisition["byte_identical_fetches_required"] is not True:
        raise RuntimeError("byte identity requirement drift")
    validate_truth_boundary(config)


def parse_and_filter(
    config: dict[str, object],
    raw: bytes,
) -> tuple[bytes, dict[str, object]]:
    source = config["source"]
    acquisition = config["acquisition"]
    policy = config["filter"]
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("source is not strict UTF-8") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if reader.fieldnames is None:
        raise RuntimeError("CSV header is missing")
    required_columns = list(acquisition["required_csv_columns"])
    missing = [name for name in required_columns if name not in reader.fieldnames]
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    rows_seen = 0
    accepted: list[dict[str, object]] = []
    rejected = Counter()
    seen_text_hashes: set[str] = set()
    aggregate_alpha = 0
    aggregate_cyrillic = 0

    for row_index, row in enumerate(reader, start=1):
        rows_seen += 1
        if rows_seen > int(acquisition["max_rows"]):
            raise RuntimeError("CSV exceeds configured row bound")
        value = row.get("prov_clean")
        if not isinstance(value, str):
            raise RuntimeError("prov_clean is missing from a row")
        text = normalize_source_text(value)
        reason = classify_text(text, policy)
        text_bytes = text.encode()
        text_hash = sha256_bytes(text_bytes)
        if reason is None and policy["deduplicate_exact_normalized_text"] is True:
            if text_hash in seen_text_hashes:
                reason = "exact_duplicate"
        if reason is not None:
            rejected[reason] += 1
            continue

        seen_text_hashes.add(text_hash)
        alpha = [char for char in text if char.isalpha()]
        cyrillic = [char for char in alpha if "\u0400" <= char <= "\u052f"]
        aggregate_alpha += len(alpha)
        aggregate_cyrillic += len(cyrillic)
        accepted.append(
            {
                "language": "uk",
                "modality": "natural_text",
                "record_id": f"franko1901:{row_index:06d}:{text_hash[:16]}",
                "source_family": source["source_family"],
                "source_id": source["source_id"],
                "text": text,
                "text_sha256": text_hash,
                "text_utf8_bytes": len(text_bytes),
            }
        )

    if not accepted:
        raise RuntimeError("filter retained zero rows")

    jsonl = b"".join(canonical_json_bytes(record) for record in accepted)
    inventory_projection = [
        {
            "record_id": record["record_id"],
            "text_sha256": record["text_sha256"],
            "text_utf8_bytes": record["text_utf8_bytes"],
        }
        for record in accepted
    ]
    stats = {
        "rows_seen": rows_seen,
        "accepted_rows": len(accepted),
        "rejected_rows": rows_seen - len(accepted),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "accepted_text_utf8_bytes": sum(
            int(record["text_utf8_bytes"]) for record in accepted
        ),
        "accepted_payload_jsonl_bytes": len(jsonl),
        "accepted_payload_jsonl_sha256": sha256_bytes(jsonl),
        "record_inventory_identity_sha256": sha256_bytes(
            canonical_json_bytes(inventory_projection)
        ),
        "aggregate_alphabetic_chars": aggregate_alpha,
        "aggregate_cyrillic_chars": aggregate_cyrillic,
        "aggregate_cyrillic_share_of_alpha": (
            aggregate_cyrillic / aggregate_alpha if aggregate_alpha else 0.0
        ),
    }
    return jsonl, stats


def build_report(
    config: dict[str, object],
    raw: bytes,
    stats: dict[str, object],
) -> dict[str, object]:
    source = config["source"]
    core = {
        "schema_version": "12-6.d03-franko1901-materialization-report.v1",
        "worker_id": config["worker_id"],
        "decision": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_id": source["source_id"],
        "source_family": source["source_family"],
        "upstream_revision": source["upstream_revision"],
        "source_git_blob_sha1": git_blob_sha1(raw),
        "source_sha256": sha256_bytes(raw),
        "source_bytes": len(raw),
        "materialization": stats,
        "rights_boundary": config["rights_boundary"],
        "truth_boundary": config["truth_boundary"],
        "local_free_only": True,
    }
    return {
        **core,
        "report_identity_sha256": sha256_bytes(canonical_json_bytes(core)),
    }


def materialize_from_bytes(
    config: dict[str, object],
    raw: bytes,
) -> tuple[bytes, dict[str, object]]:
    validate_contract(config)
    source = config["source"]
    if len(raw) != int(source["source_bytes"]):
        raise RuntimeError("source byte-count mismatch")
    if git_blob_sha1(raw) != source["source_git_blob_sha1"]:
        raise RuntimeError("source Git-blob identity mismatch")
    jsonl, stats = parse_and_filter(config, raw)
    return jsonl, build_report(config, raw, stats)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_contract(config)
    source = config["source"]
    max_bytes = int(config["acquisition"]["max_source_bytes"])

    if args.input:
        raw_a = Path(args.input).read_bytes()
        raw_b = raw_a
    else:
        raw_a = fetch_bounded(source["raw_url"], max_bytes)
        raw_b = fetch_bounded(source["raw_url"], max_bytes)
    if raw_a != raw_b:
        raise RuntimeError("two exact source acquisitions are not byte-identical")

    candidate_jsonl, report = materialize_from_bytes(config, raw_a)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "franko1901-candidate.jsonl").write_bytes(candidate_jsonl)
    (output_dir / "franko1901-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "ATTRIBUTION.txt").write_text(
        config["rights_boundary"]["required_attribution"] + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
