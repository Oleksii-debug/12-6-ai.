#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

WORKER_ID = "NEXT100-039-DATA-EN-RUST-DOCS"
UPSTREAM_REPO = "https://github.com/rust-lang/book"
UPSTREAM_COMMIT = "917544888a55e4da7109bdba8c88c893c0da70f4"
UPSTREAM_TREE = "6a29569c5b742d9151391b38a725768de0110419"
UPSTREAM_COMMIT_DATE = "2026-07-14T01:25:25Z"
RAW_ROOT = f"https://raw.githubusercontent.com/rust-lang/book/{UPSTREAM_COMMIT}"
REGISTRY_PATH = Path("data/registry/external_snapshots.v2.json")
FAMILY_ID = "en.rust-lang.book.prose"
NORMALIZATION_POLICY = "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1"

LICENSE_OBJECTS = {
    "LICENSE-MIT": "25597d5838fa4cd7ff5c3c2bb1d1b4c3731eda7f",
    "LICENSE-APACHE": "38634daab005dcffd47d8a68bd958c809fe2b59a",
    "CONTRIBUTING.md": "23fea7c0a57ee18717037ac99d8403218a739a21",
}

SELECTED = {
    "src/ch10-00-generics.md": ("7e1055fdc6da7151589d119a142a8b9236782107", 5738),
    "src/ch10-01-syntax.md": ("8a13a252e0ef1e5b65c50be39afe6fd7bf977483", 14580),
    "src/ch10-02-traits.md": ("1698cda0d3eeba1fb6197887316d0fa62cab445e", 18682),
    "src/ch10-03-lifetime-syntax.md": ("fc38d7415106494f003db0518bf8b4565c5e231a", 30935),
    "src/ch16-00-concurrency.md": ("95b1562f1df7247d415a7a98b147f1675fe0c478", 3010),
    "src/ch16-01-threads.md": ("6a1e3d40633dc1803a7ff5e1118695d27b746f3b", 12564),
    "src/ch16-02-message-passing.md": ("75bb15724b89a474f3465ce4cab96ee420423b7f", 11867),
    "src/ch16-03-shared-state.md": ("621f74b4089ac68353b5264b2ebbe098a668e8e0", 12519),
    "src/ch16-04-extensible-concurrency-sync-and-send.md": ("a866b6174ff5d7a84eaa80c236f27634f3dea170", 5153),
}

THIRD_PARTY_MARKERS = (
    "all rights reserved",
    "reprinted with permission",
    "republished with permission",
    "used with permission",
    "used by permission",
    "adapted with permission",
)
ENGLISH_SIGNALS = {
    "the", "and", "to", "of", "a", "in", "is", "for", "that", "with", "as",
    "we", "you", "this", "are", "rust", "type", "value", "thread", "reference",
    "function", "compiler", "data", "code",
}
SECRET_MARKERS = (
    "-----begin private key-----",
    "aws_secret_access_key",
    "github_token=",
    "authorization: bearer ",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity is SHA-1 by design.


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def fetch(path: str) -> bytes:
    request = urllib.request.Request(
        f"{RAW_ROOT}/{path}",
        headers={"User-Agent": "12-6-ai-next100-039-rust-docs-qualification"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    data.decode("utf-8", errors="strict")
    return data


def strip_fenced_code(text: str) -> tuple[str, dict[str, int]]:
    out: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    fenced_blocks = 0
    fenced_lines = 0
    for line in text.splitlines():
        if fence_char is None:
            match = re.match(r"^\s*(`{3,}|~{3,})", line)
            if match:
                marker = match.group(1)
                fence_char = marker[0]
                fence_len = len(marker)
                fenced_blocks += 1
                fenced_lines += 1
                continue
            out.append(line)
            continue

        fenced_lines += 1
        if re.match(rf"^\s*{re.escape(fence_char)}{{{fence_len},}}\s*$", line):
            fence_char = None
            fence_len = 0

    if fence_char is not None:
        raise ValueError("unterminated fenced code block")
    return "\n".join(out), {
        "fenced_code_blocks_removed": fenced_blocks,
        "fenced_code_lines_removed": fenced_lines,
    }


def normalize_prose(raw: bytes) -> tuple[bytes, dict[str, int]]:
    text = raw.decode("utf-8", errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text, stats = strip_fenced_code(text)
    stats.update(
        {
            "mdbook_directive_lines_removed": 0,
            "reference_definition_lines_removed": 0,
            "image_lines_removed": 0,
            "inline_code_spans_removed": 0,
            "html_tags_removed": 0,
        }
    )

    prose_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"\{\{#[^{}]+\}\}", line):
            stats["mdbook_directive_lines_removed"] += 1
            continue
        if re.match(r"^\s*\[[^\]]+\]:\s+\S+", line):
            stats["reference_definition_lines_removed"] += 1
            continue
        if stripped.startswith("!["):
            stats["image_lines_removed"] += 1
            continue

        spans = re.findall(r"`+[^`\n]+`+", line)
        stats["inline_code_spans_removed"] += len(spans)
        line = re.sub(r"`+[^`\n]+`+", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[([^\]]+)\]\[[^\]]+\]", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]", r"\1", line)

        tags = re.findall(r"<[^>]+>", line)
        stats["html_tags_removed"] += len(tags)
        line = re.sub(r"<[^>]+>", " ", line)

        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = line.replace("**", "").replace("__", "").replace("~~", "")
        line = re.sub(r"\\([\\`*_[\]{}()#+\-.!>])", r"\1", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        prose_lines.append(line)

    normalized_lines: list[str] = []
    previous_blank = True
    for line in prose_lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = blank
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    normalized = ("\n".join(normalized_lines) + "\n").encode("utf-8")
    return normalized, stats


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def shingle_set(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = word_tokens(text)
    if len(tokens) < width:
        return set()
    return {tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def write_evidence_bytes(output: Path, license_payloads: dict[str, bytes]) -> None:
    rights_dir = output.parent / "rights"
    rights_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in license_payloads.items():
        (rights_dir / Path(path).name).write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "12-6.external-snapshot-registry.v2":
        raise SystemExit("unexpected live registry schema")
    live_hashes = {row["snapshot"]["normalized_sha256"] for row in registry["sources"]}
    live_families = {
        row["independent_source_family"]["family_id"] for row in registry["sources"]
    }

    license_payloads = {path: fetch(path) for path in LICENSE_OBJECTS}
    license_blob_checks = {
        path: {
            "expected_git_blob_sha1": expected,
            "observed_git_blob_sha1": git_blob_sha1(payload),
            "raw_sha256": sha256(payload),
            "raw_bytes": len(payload),
            "pass": git_blob_sha1(payload) == expected,
        }
        for path, expected in LICENSE_OBJECTS.items()
        for payload in [license_payloads[path]]
    }
    if not all(row["pass"] for row in license_blob_checks.values()):
        raise SystemExit(f"license Git object drift: {license_blob_checks}")

    mit_text = license_payloads["LICENSE-MIT"].decode("utf-8")
    apache_text = license_payloads["LICENSE-APACHE"].decode("utf-8")
    contributing_text = license_payloads["CONTRIBUTING.md"].decode("utf-8")
    rights_preconditions = {
        "repo_dual_license_statement_present": (
            "same license as Rust itself, MIT/Apache2" in contributing_text
            and "LICENSE-*" in contributing_text
        ),
        "mit_permission_grant_present": (
            "Permission is hereby granted, free of charge" in mit_text
            and "use, copy, modify, merge" in mit_text
            and "publish, distribute, sublicense" in mit_text
        ),
        "mit_notice_condition_present": (
            "copyright notice and this permission notice" in mit_text.lower()
            and "included in all copies or substantial portions" in mit_text.lower()
        ),
        "apache_2_present": (
            "Apache License" in apache_text
            and "Version 2.0, January 2004" in apache_text
            and "Grant of Copyright License" in apache_text
        ),
    }
    if not all(rights_preconditions.values()):
        raise SystemExit(f"rights precondition failed: {rights_preconditions}")

    pages: list[dict[str, object]] = []
    normalized_payloads: dict[str, bytes] = {}
    for path, (expected_blob, expected_bytes) in SELECTED.items():
        raw = fetch(path)
        observed_blob = git_blob_sha1(raw)
        if observed_blob != expected_blob or len(raw) != expected_bytes:
            raise SystemExit(
                f"source identity drift for {path}: "
                f"blob={observed_blob} bytes={len(raw)}"
            )
        raw_text = raw.decode("utf-8")
        lower = raw_text.lower()
        third_party_hits = sorted(marker for marker in THIRD_PARTY_MARKERS if marker in lower)
        secret_hits = sorted(marker for marker in SECRET_MARKERS if marker in lower)
        normalized, removal_stats = normalize_prose(raw)
        normalized_text = normalized.decode("utf-8")
        tokens = word_tokens(normalized_text)
        alpha_chars = sum(ch.isalpha() for ch in normalized_text)
        ascii_alpha = sum("a" <= ch.lower() <= "z" for ch in normalized_text)
        signal_hits = sum(token in ENGLISH_SIGNALS for token in tokens)
        controls = sum(
            ord(ch) < 32 and ch not in "\n\t" for ch in normalized_text
        )
        quality_pass = (
            len(normalized) >= 1200
            and len(tokens) >= 250
            and signal_hits >= 12
            and (ascii_alpha / alpha_chars if alpha_chars else 0.0) >= 0.97
            and controls == 0
            and "\ufffd" not in normalized_text
            and not third_party_hits
            and not secret_hits
        )
        page = {
            "path": path,
            "expected_git_blob_sha1": expected_blob,
            "raw_git_blob_sha1": observed_blob,
            "raw_sha256": sha256(raw),
            "raw_bytes": len(raw),
            "normalized_sha256": sha256(normalized),
            "normalized_bytes": len(normalized),
            "normalization_policy": NORMALIZATION_POLICY,
            "word_count": len(tokens),
            "english_signal_hits": signal_hits,
            "ascii_alpha_ratio": round(ascii_alpha / alpha_chars, 6) if alpha_chars else 0.0,
            "control_character_count": controls,
            "third_party_rights_marker_hits": third_party_hits,
            "secret_marker_hits": secret_hits,
            "code_separation": removal_stats,
            "quality_status": "PASS" if quality_pass else "REJECT",
        }
        pages.append(page)
        normalized_payloads[path] = normalized

    normalized_hashes = [str(page["normalized_sha256"]) for page in pages]
    internal_exact_duplicates = len(normalized_hashes) - len(set(normalized_hashes))
    cross_registry_collisions = sorted(set(normalized_hashes) & live_hashes)
    family_collision = FAMILY_ID in live_families

    shingles = {
        path: shingle_set(payload.decode("utf-8"))
        for path, payload in normalized_payloads.items()
    }
    near_pairs: list[dict[str, object]] = []
    max_jaccard = 0.0
    selected_paths = list(SELECTED)
    for index, left in enumerate(selected_paths):
        for right in selected_paths[index + 1 :]:
            score = jaccard(shingles[left], shingles[right])
            max_jaccard = max(max_jaccard, score)
            if score >= 0.85:
                near_pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "five_word_shingle_jaccard": round(score, 6),
                    }
                )

    family_material = {
        "canonical_repository": UPSTREAM_REPO,
        "canonical_content_root": "src",
        "work": "The Rust Programming Language",
        "language": "en",
        "modality": "natural-language-documentation-prose",
    }
    family_identity = sha256(canonical_bytes(family_material))
    snapshot_material = {
        "family_identity_sha256": family_identity,
        "upstream_commit": UPSTREAM_COMMIT,
        "normalization_policy": NORMALIZATION_POLICY,
        "members": [
            {
                "path": page["path"],
                "raw_sha256": page["raw_sha256"],
                "normalized_sha256": page["normalized_sha256"],
            }
            for page in pages
        ],
    }
    snapshot_identity = sha256(canonical_bytes(snapshot_material))

    all_quality_pass = all(page["quality_status"] == "PASS" for page in pages)
    terminal = (
        all_quality_pass
        and internal_exact_duplicates == 0
        and not near_pairs
        and not cross_registry_collisions
        and not family_collision
        and all(rights_preconditions.values())
    )

    report: dict[str, object] = {
        "schema_version": "12-6.next100-039-rust-book-source-authority.v1",
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "training_executed": False,
        "verdict": "ADMIT_PROSE_ONLY" if terminal else "REJECT",
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "tree": UPSTREAM_TREE,
            "commit_date_utc": UPSTREAM_COMMIT_DATE,
            "canonical_source_root": "src",
            "selected_file_count": len(SELECTED),
            "selected_paths": selected_paths,
            "expected_selected_raw_bytes": sum(size for _, size in SELECTED.values()),
        },
        "rights_evidence": {
            "license_expression": "MIT OR Apache-2.0",
            "license_objects": license_blob_checks,
            "preconditions": rights_preconditions,
            "prose": {
                "model_training": "ALLOWED_BY_BROAD_PERMISSIVE_LICENSE_GRANT",
                "redistribution": "ALLOWED_WITH_LICENSE_COMPLIANCE",
                "selected_compliance_path": "MIT",
                "redistribution_condition": (
                    "Retain the Rust Project Developers copyright notice and the "
                    "MIT permission notice in redistributed copies or substantial portions."
                ),
                "evaluation": "NOT_SEPARATELY_ADMITTED",
                "explicit_ai_training_clause": False,
            },
            "code": {
                "repository_license": "MIT OR Apache-2.0",
                "rights_status": "LICENSED_BUT_EXCLUDED_FROM_THIS_TEXT_AUTHORITY",
                "model_training_under_this_authority": "EXCLUDED",
                "redistribution_under_this_authority": "EXCLUDED",
                "admitted_code_bytes": 0,
            },
            "claim_boundary": (
                "This is a source-rights qualification under the repository's broad "
                "permissive license grant, not legal advice and not a claim that the "
                "license contains AI-specific wording."
            ),
        },
        "prose_code_boundary": {
            "source_markdown_only": True,
            "fenced_code_removed": True,
            "mdbook_includes_resolved": False,
            "mdbook_directives_removed": True,
            "inline_code_spans_removed": True,
            "listing_tree_included": False,
            "generated_rendered_formats_included": False,
            "excluded_paths_and_derivatives": [
                "listings/**",
                "2018-edition/**",
                "nostarch/**",
                "target/**",
                "generated mdBook HTML",
                "doc.rust-lang.org rendered/offline HTML",
                "PDF/ePub/print renderings",
                "mirrors, forks, and community translations",
            ],
        },
        "family": {
            "family_id": FAMILY_ID,
            "family_identity_sha256": family_identity,
            "snapshot_identity_sha256": snapshot_identity,
            "family_credit": 1,
            "independence_rule": (
                "All selected files and every generated rendering of this canonical "
                "rust-lang/book lineage count as one English family. File count, "
                "rendering count, mirrors, forks, legacy snapshots, and translations "
                "do not create additional English family credit."
            ),
        },
        "normalization": {
            "policy_id": NORMALIZATION_POLICY,
            "steps": [
                "strict UTF-8 decode",
                "CRLF/CR to LF",
                "HTML entity decode then Unicode NFKC",
                "remove HTML comments",
                "remove fenced code blocks without resolving mdBook includes",
                "remove standalone mdBook directives and Markdown reference definitions",
                "remove image lines and inline code spans",
                "preserve link labels while removing destinations",
                "remove HTML/Listing tags and Markdown presentation markers",
                "collapse horizontal whitespace and repeated blank lines",
                "emit exactly one terminal LF",
            ],
        },
        "documents": pages,
        "quality": {
            "all_documents_pass": all_quality_pass,
            "total_raw_bytes": sum(int(page["raw_bytes"]) for page in pages),
            "total_normalized_bytes": sum(int(page["normalized_bytes"]) for page in pages),
            "total_words": sum(int(page["word_count"]) for page in pages),
            "minimum_normalized_bytes_per_document": 1200,
            "minimum_words_per_document": 250,
            "minimum_english_signal_hits_per_document": 12,
            "minimum_ascii_alpha_ratio": 0.97,
            "third_party_marker_policy": "REJECT_DOCUMENT",
            "secret_marker_policy": "REJECT_DOCUMENT",
        },
        "dedup": {
            "registry_path": str(REGISTRY_PATH),
            "live_registry_identity_sha256": registry["registry_identity_sha256"],
            "live_registry_source_count": registry["source_count"],
            "live_registry_independent_family_count": registry[
                "independent_source_family_count"
            ],
            "family_id_already_present": family_collision,
            "internal_exact_normalized_duplicate_count": internal_exact_duplicates,
            "within_family_near_duplicate_threshold": 0.85,
            "within_family_near_duplicate_pairs": near_pairs,
            "maximum_within_family_five_word_shingle_jaccard": round(max_jaccard, 6),
            "cross_registry_exact_normalized_sha256_collisions": cross_registry_collisions,
            "generated_duplicate_rule": (
                "Rendered HTML/offline docs/print copies/listing mirrors are excluded "
                "before materialization and cannot add bytes or family credit."
            ),
            "successor_integration_requirement": (
                "A corpus-registry successor must still rerun the incumbent cross-source "
                "near-copy/decontamination graph on the exact materialized payload; this "
                "qualification does not replace DATA-232/DATA-298 integration gates."
            ),
        },
        "purpose_decisions": {
            "model_training": "ADMIT" if terminal else "REJECT",
            "redistribution": "ADMIT_WITH_MIT_NOTICE" if terminal else "REJECT",
            "evaluation": "NOT_SEPARATELY_ADMITTED",
            "code_training_from_this_authority": "NO",
            "corpus_freeze_or_representativeness": "NOT_CLAIMED",
        },
    }
    report["authority_identity_sha256"] = sha256(canonical_bytes(report))
    payload = canonical_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    write_evidence_bytes(output, license_payloads)
    print(payload.decode("utf-8"), end="")
    return 0 if terminal else 2


if __name__ == "__main__":
    raise SystemExit(main())
