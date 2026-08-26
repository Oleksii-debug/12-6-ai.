#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

UPSTREAM_REPO = "https://github.com/mdn/content"
UPSTREAM_COMMIT = "41ace2122a86ea89fee604ec0970c2328f8077f6"
RAW_ROOT = f"https://raw.githubusercontent.com/mdn/content/{UPSTREAM_COMMIT}"
LICENSE_PATH = "LICENSE.md"
ATTRIBUTION_POLICY_PATH = "files/en-us/mdn/writing_guidelines/attrib_copyright_license/index.md"
REGISTRY_PATH = Path("data/registry/external_snapshots.v2.json")
SELECTION_ROOT = "files/en-us/web/http/guides"
SELECTED = (
    "files/en-us/web/http/guides/authentication/index.md",
    "files/en-us/web/http/guides/caching/index.md",
    "files/en-us/web/http/guides/compression/index.md",
    "files/en-us/web/http/guides/conditional_requests/index.md",
    "files/en-us/web/http/guides/content_negotiation/index.md",
    "files/en-us/web/http/guides/cookies/index.md",
    "files/en-us/web/http/guides/cors/index.md",
)
NORMALIZATION_POLICY = "MDN_PROSE_ONLY_MARKDOWN_V1"
THIRD_PARTY_MARKERS = (
    "all rights reserved",
    "used with permission",
    "reprinted with permission",
    "republished with permission",
    "originally published",
    "adapted from",
)
ENGLISH_SIGNALS = {
    "the", "and", "to", "of", "a", "in", "is", "for", "that", "with",
    "http", "request", "response", "server", "browser", "client", "web",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def fetch(path: str) -> bytes:
    req = urllib.request.Request(f"{RAW_ROOT}/{path}", headers={"User-Agent": "12-6-ai-next100-038-mdn-qualification"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    data.decode("utf-8", errors="strict")
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    frontmatter: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter, text[end + 5 :]


def strip_fenced_code(text: str) -> tuple[str, int]:
    out: list[str] = []
    fence: str | None = None
    removed = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            removed += 1
            continue
        if fence is not None:
            removed += 1
            if stripped.startswith(fence):
                fence = None
            continue
        out.append(line)
    if fence is not None:
        raise ValueError("unterminated fenced code block")
    return "\n".join(out), removed


def normalize_prose(raw: bytes) -> tuple[bytes, dict[str, int], dict[str, str]]:
    text = unicodedata.normalize("NFKC", raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n"))
    frontmatter, text = parse_frontmatter(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text, fenced_lines = strip_fenced_code(text)

    stats = {
        "fenced_code_lines_removed": fenced_lines,
        "image_lines_removed": 0,
        "table_lines_removed": 0,
        "embed_macro_lines_removed": 0,
        "inline_code_spans_removed": 0,
    }
    prose_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            prose_lines.append("")
            continue
        if stripped.startswith("![") or "<img" in stripped.lower() or "<picture" in stripped.lower():
            stats["image_lines_removed"] += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            stats["table_lines_removed"] += 1
            continue
        if re.search(r"\{\{\s*(Embed|InteractiveExample|LiveSample|EmbedGHLiveSample)", line, flags=re.I):
            stats["embed_macro_lines_removed"] += 1
            continue
        spans = re.findall(r"`+[^`\n]*`+", line)
        stats["inline_code_spans_removed"] += len(spans)
        line = re.sub(r"`+[^`\n]*`+", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\{\{[^{}]*\}\}", " ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = line.replace("**", "").replace("__", "").replace("~~", "")
        line = re.sub(r"\s+", " ", line).strip()
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
    return normalized, stats, frontmatter


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def shingle_set(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = word_tokens(text)
    return {tuple(tokens[i : i + width]) for i in range(max(0, len(tokens) - width + 1))}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    live_hashes = {row["snapshot"]["normalized_sha256"] for row in registry["sources"]}
    license_raw = fetch(LICENSE_PATH)
    attribution_raw = fetch(ATTRIBUTION_POLICY_PATH)
    license_text = license_raw.decode("utf-8")
    attribution_text = attribution_raw.decode("utf-8")

    rights_preconditions = {
        "prose_cc_by_sa_2_5_present": "All prose content is available under" in license_text and "CC-BY-SA 2.5" in license_text,
        "code_cc0_post_2010_present": "Added on or after August 20, 2010" in license_text and "CC0" in license_text,
        "code_mit_pre_2010_present": "Added before August 20, 2010" in license_text and "MIT" in license_text,
        "yari_code_age_ambiguity_present": "currently no way to determine" in attribution_text and "December 14 2020" in attribution_text,
        "logos_trademarks_excluded_by_policy": "Logos, trademarks, service marks, and wordmarks" in attribution_text,
    }
    if not all(rights_preconditions.values()):
        raise SystemExit(f"rights precondition failed: {rights_preconditions}")

    pages = []
    normalized_payloads: dict[str, bytes] = {}
    for path in SELECTED:
        raw = fetch(path)
        raw_text = raw.decode("utf-8")
        lower = raw_text.lower()
        marker_hits = sorted(marker for marker in THIRD_PARTY_MARKERS if marker in lower)
        normalized, removal_stats, frontmatter = normalize_prose(raw)
        normalized_text = normalized.decode("utf-8")
        tokens = word_tokens(normalized_text)
        signal_hits = sum(1 for token in tokens if token in ENGLISH_SIGNALS)
        alpha_chars = sum(ch.isalpha() for ch in normalized_text)
        ascii_alpha = sum(("a" <= ch.lower() <= "z") for ch in normalized_text)
        page = {
            "path": path,
            "title": frontmatter.get("title"),
            "slug": frontmatter.get("slug"),
            "canonical_url": f"https://developer.mozilla.org/en-US/docs/{frontmatter.get('slug')}" if frontmatter.get("slug") else None,
            "raw_sha256": sha256(raw),
            "raw_bytes": len(raw),
            "normalized_sha256": sha256(normalized),
            "normalized_bytes": len(normalized),
            "normalization_policy": NORMALIZATION_POLICY,
            "word_count": len(tokens),
            "english_signal_hits": signal_hits,
            "ascii_alpha_ratio": round(ascii_alpha / alpha_chars, 6) if alpha_chars else 0.0,
            "third_party_marker_hits": marker_hits,
            "mixed_rights_removal": removal_stats,
            "quality_status": "PASS" if len(tokens) >= 400 and signal_hits >= 20 and (ascii_alpha / alpha_chars if alpha_chars else 0.0) >= 0.95 and not marker_hits else "REJECT",
            "attribution": {
                "credit": "Mozilla Contributors",
                "title": frontmatter.get("title"),
                "source_url": f"https://developer.mozilla.org/en-US/docs/{frontmatter.get('slug')}" if frontmatter.get("slug") else None,
                "license": "CC-BY-SA-2.5-or-later",
                "modification_note": "Frontmatter, code samples, inline code spans, tables, media/embed lines, macros, HTML markup, link destinations, and Markdown formatting removed; prose NFKC/whitespace normalized.",
            },
        }
        pages.append(page)
        normalized_payloads[path] = normalized

    hashes = [page["normalized_sha256"] for page in pages]
    exact_internal_duplicates = len(hashes) - len(set(hashes))
    registry_collisions = sorted(set(hashes) & live_hashes)
    shingles = {path: shingle_set(payload.decode("utf-8")) for path, payload in normalized_payloads.items()}
    near_pairs = []
    max_jaccard = 0.0
    for i, left in enumerate(SELECTED):
        for right in SELECTED[i + 1 :]:
            score = jaccard(shingles[left], shingles[right])
            max_jaccard = max(max_jaccard, score)
            if score >= 0.85:
                near_pairs.append({"left": left, "right": right, "five_word_shingle_jaccard": round(score, 6)})

    family_material = {
        "canonical_repository": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "selection_root": SELECTION_ROOT,
        "license": "CC-BY-SA-2.5-or-later",
        "modality": "natural-language-documentation-prose",
    }
    family_identity = sha256(canonical_bytes(family_material))
    terminal = (
        all(page["quality_status"] == "PASS" for page in pages)
        and exact_internal_duplicates == 0
        and not registry_collisions
        and not near_pairs
    )

    report = {
        "schema_version": "12-6.next100-038-mdn-source-authority.v1",
        "worker_id": "NEXT100-038-DATA-EN-MDN",
        "local_free_only": True,
        "verdict": "ADMIT_PROSE_ONLY" if terminal else "REJECT",
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "selection_root": SELECTION_ROOT,
            "selected_file_count": len(SELECTED),
            "selected_paths": list(SELECTED),
        },
        "rights_evidence": {
            "license_path": LICENSE_PATH,
            "license_raw_sha256": sha256(license_raw),
            "attribution_policy_path": ATTRIBUTION_POLICY_PATH,
            "attribution_policy_raw_sha256": sha256(attribution_raw),
            "preconditions": rights_preconditions,
            "prose": {
                "license": "CC-BY-SA-2.5-or-later",
                "model_training": "ALLOWED",
                "redistribution": "ALLOWED_WITH_ATTRIBUTION_AND_SHAREALIKE",
                "evaluation": "NOT_SEPARATELY_ADMITTED",
                "required_attribution": "Document title + source URL + Mozilla Contributors + CC-BY-SA 2.5 notice + modification description.",
            },
            "code_samples": {
                "license_boundary": "CC0-1.0 if added on/after 2010-08-20; MIT if added before 2010-08-20",
                "historical_license_resolvable_per_snippet": False,
                "model_training": "REJECTED_THIS_AUTHORITY",
                "redistribution": "REJECTED_THIS_AUTHORITY",
                "reason": "MDN states Yari currently cannot determine which historical code license applies; MIT attribution cannot be reconstructed reliably per snippet. All code is removed from the admitted payload.",
            },
            "excluded_mixed_rights": [
                "code examples and snippets",
                "images and other media",
                "interactive/live/GitHub embeds",
                "Mozilla logos, trademarks, service marks, wordmarks, and site look-and-feel",
                "any page with explicit third-party republication/permission markers",
            ],
            "model_output_license_implication": "NOT_ADJUDICATED_BY_THIS_SOURCE_AUTHORITY",
        },
        "family": {
            "family_id": "en.mdn.webdocs.prose",
            "family_identity_sha256": family_identity,
            "independence_rule": "All selected MDN pages count as one upstream family regardless of page count; mirrors/forks do not create new families.",
        },
        "pages": pages,
        "quality": {
            "all_pages_pass": all(page["quality_status"] == "PASS" for page in pages),
            "total_raw_bytes": sum(page["raw_bytes"] for page in pages),
            "total_normalized_bytes": sum(page["normalized_bytes"] for page in pages),
            "total_words": sum(page["word_count"] for page in pages),
            "minimum_words_per_page": 400,
            "minimum_ascii_alpha_ratio": 0.95,
            "minimum_english_signal_hits": 20,
        },
        "dedup": {
            "live_registry_identity_sha256": registry["registry_identity_sha256"],
            "live_registry_source_count": registry["source_count"],
            "internal_exact_duplicate_count": exact_internal_duplicates,
            "cross_registry_normalized_sha256_collisions": registry_collisions,
            "within_mdn_near_duplicate_threshold": 0.85,
            "within_mdn_near_duplicate_pairs": near_pairs,
            "maximum_within_mdn_five_word_shingle_jaccard": round(max_jaccard, 6),
            "downstream_requirement": "Rerun canonical corpus-level exact/near dedup and evaluation decontamination when composing this family with later registries; source admission is not a corpus freeze.",
        },
        "claim_boundary": {
            "training_source_authority_terminal": terminal,
            "prose_only": True,
            "code_admitted": False,
            "representative_corpus_claimed": False,
            "production_corpus_frozen": False,
            "evaluation_authorized": False,
            "third_party_ambiguous_material_admitted": False,
        },
    }
    report["authority_identity_sha256"] = sha256(canonical_bytes(report))
    output = canonical_bytes(report)
    output_path = Path(args.output)
    if args.verify:
        if not output_path.exists() or output_path.read_bytes() != output:
            raise SystemExit("committed authority does not match deterministic rebuild")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output)
    print(output.decode("utf-8"), end="")
    return 0 if terminal else 2


if __name__ == "__main__":
    raise SystemExit(main())
