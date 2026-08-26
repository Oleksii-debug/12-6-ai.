#!/usr/bin/env python3
"""Deterministic, LOCAL_FREE Ukrainian Wikibooks source qualification probe.

The probe never promotes the source into the training registry. It verifies one
pinned official Wikimedia dump, emits revision-level provenance without copying
page text into the artifact, and measures language/quality statistics needed for
NEXT100-023's RETEST decision.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import statistics
import tempfile
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

DUMP_URL = (
    "https://dumps.wikimedia.org/ukwikibooks/20260201/"
    "ukwikibooks-20260201-pages-articles.xml.bz2"
)
DUMP_SHA1 = "6975ba549f822ea2394567743fdb3564c36e048a"
USER_AGENT = "12-6-ai-NEXT100-023-source-audit/1.0"
UA_SPECIFIC = set("іїєґІЇЄҐ")
IMPORT_MARKERS = (
    "переклад",
    "translation",
    "attribution",
    "джерело",
    "source",
    "copied",
    "imported",
    "wikipedia",
    "вікіпеді",
    "wikisource",
    "вікіджерел",
)


def normalize_wikitext(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def language_metrics(text: str) -> tuple[int, int, int]:
    letters = [c for c in text if c.isalpha()]
    cyr = [c for c in letters if "\u0400" <= c <= "\u052f"]
    ua = sum(c in UA_SPECIFIC for c in text)
    return len(letters), len(cyr), ua


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


def children(page: ET.Element, name: str):
    return page.findall(f"{{*}}{name}")


def child_text(parent: ET.Element, name: str, default: str = "") -> str:
    node = parent.find(f"{{*}}{name}")
    return default if node is None or node.text is None else node.text


def page_url(title: str, revision_id: int) -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return f"https://uk.wikibooks.org/w/index.php?title={encoded}&oldid={revision_id}"


def history_url(title: str) -> str:
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return f"https://uk.wikibooks.org/w/index.php?title={encoded}&action=history"


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dump(dump_path: Path) -> tuple[dict, list[dict]]:
    records: list[dict] = []
    duplicate_map: dict[str, list[int]] = {}
    page_sizes: list[int] = []
    total_letters = total_cyr = total_ua_specific = 0
    redirects = empty = ge500 = ge2000 = marker_pages = ua_pages = 0

    with bz2.open(dump_path, "rb") as stream:
        for _, page in ET.iterparse(stream, events=("end",)):
            if not page.tag.endswith("}page") and page.tag != "page":
                continue
            ns = child_text(page, "ns")
            if ns != "0":
                page.clear()
                continue

            title = child_text(page, "title")
            page_id = int(child_text(page, "id", "0"))
            rev = page.find("{*}revision")
            if rev is None:
                page.clear()
                continue
            revision_id = int(child_text(rev, "id", "0"))
            timestamp = child_text(rev, "timestamp")
            revision_sha1 = child_text(rev, "sha1")
            text_node = rev.find("{*}text")
            text = "" if text_node is None or text_node.text is None else text_node.text
            normalized = normalize_wikitext(text)
            text_bytes = len(text.encode("utf-8"))
            normalized_bytes = len(normalized.encode("utf-8"))
            normalized_sha256 = sha256_text(normalized)
            letters, cyr, ua_specific = language_metrics(normalized)
            lower = normalized.lower()
            has_import_marker = any(marker in lower for marker in IMPORT_MARKERS)
            is_redirect = lower.startswith("#redirect") or lower.startswith("#перенаправлення")

            page_sizes.append(text_bytes)
            total_letters += letters
            total_cyr += cyr
            total_ua_specific += ua_specific
            redirects += int(is_redirect)
            empty += int(normalized_bytes == 0)
            ge500 += int(text_bytes >= 500)
            ge2000 += int(text_bytes >= 2000)
            marker_pages += int(has_import_marker)
            ua_pages += int(ua_specific > 0)
            duplicate_map.setdefault(normalized_sha256, []).append(page_id)

            records.append(
                {
                    "page_id": page_id,
                    "title": title,
                    "revision_id": revision_id,
                    "revision_timestamp": timestamp,
                    "revision_sha1": revision_sha1,
                    "wikitext_utf8_bytes": text_bytes,
                    "normalized_utf8_bytes": normalized_bytes,
                    "normalized_sha256": normalized_sha256,
                    "redirect": is_redirect,
                    "ukrainian_specific_letter_count": ua_specific,
                    "possible_import_or_attribution_marker": has_import_marker,
                    "source_url": page_url(title, revision_id),
                    "history_url": history_url(title),
                }
            )
            page.clear()

    records.sort(key=lambda x: (x["page_id"], x["revision_id"], x["title"]))
    page_sizes.sort()
    duplicate_groups = sorted(
        [ids for ids in duplicate_map.values() if len(ids) > 1 and ids],
        key=lambda ids: (ids[0], len(ids)),
    )
    total_bytes = sum(page_sizes)
    report = {
        "schema_version": "12-6.next100-023.ua-wikibooks-probe.v1",
        "worker_id": "NEXT100-023-DATA-UA-WIKIBOOKS",
        "verdict": "RETEST",
        "snapshot": {
            "url": DUMP_URL,
            "expected_sha1": DUMP_SHA1,
            "actual_sha1": file_sha1(dump_path),
            "compressed_bytes": dump_path.stat().st_size,
            "selection": "all namespace-0 pages from pinned pages-articles current-revision dump",
        },
        "quality": {
            "namespace0_pages": len(records),
            "redirect_pages": redirects,
            "nonredirect_pages": len(records) - redirects,
            "empty_pages": empty,
            "utf8_text_bytes": total_bytes,
            "min_page_bytes": page_sizes[0] if page_sizes else 0,
            "median_page_bytes": int(statistics.median(page_sizes)) if page_sizes else 0,
            "p90_page_bytes": percentile(page_sizes, 0.90),
            "max_page_bytes": page_sizes[-1] if page_sizes else 0,
            "pages_ge_500_bytes": ge500,
            "pages_ge_2000_bytes": ge2000,
            "exact_normalized_duplicate_groups": len(duplicate_groups),
            "exact_normalized_duplicate_pages": sum(len(x) for x in duplicate_groups),
        },
        "language": {
            "letter_count": total_letters,
            "cyrillic_letter_count": total_cyr,
            "cyrillic_letter_ratio": round(total_cyr / total_letters, 6) if total_letters else 0.0,
            "ukrainian_specific_letter_count": total_ua_specific,
            "pages_with_ukrainian_specific_letters": ua_pages,
            "pages_with_ukrainian_specific_letters_ratio": round(ua_pages / len(records), 6) if records else 0.0,
        },
        "provenance_risk": {
            "pages_with_possible_import_or_attribution_markers": marker_pages,
            "page_history_contributor_lists_materialized": false,
            "imported_text_additional_attribution_completeness": "NOT_PROVEN",
        },
        "family_lineage": {
            "lineage_namespace": "wikimedia.uk",
            "independent_family_credit": 0,
            "status": "UNRESOLVED_CROSS_WIKIMEDIA_LINEAGE",
        },
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "admission_blockers": [
            "PAGE_LEVEL_IMPORTED_TEXT_ATTRIBUTION_COMPLETENESS_NOT_YET_PROVEN",
            "CROSS_WIKIMEDIA_WIKIPEDIA_WIKISOURCE_LINEAGE_AND_DEDUP_NOT_YET_TERMINAL",
        ],
    }
    if report["snapshot"]["actual_sha1"] != DUMP_SHA1:
        raise SystemExit("pinned dump SHA-1 mismatch")
    return report, records


def self_test() -> None:
    assert normalize_wikitext("A  B\r\n\r\n\r\nВікі") == "A B\n\nВікі"
    letters, cyr, ua = language_metrics("Українська мова ґрунт")
    assert letters > 0 and cyr > 0 and ua >= 3
    assert len(sha256_text("x")) == 64
    print("SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/next100_023"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    args.out.mkdir(parents=True, exist_ok=True)
    if args.dump:
        dump_path = args.dump
        report, records = audit_dump(dump_path)
    else:
        with tempfile.TemporaryDirectory(prefix="next100-023-") as tmp:
            dump_path = Path(tmp) / "ukwikibooks-20260201-pages-articles.xml.bz2"
            download(DUMP_URL, dump_path)
            report, records = audit_dump(dump_path)

    report_path = args.out / "ua_wikibooks_probe_report_v1.json"
    manifest_path = args.out / "ua_wikibooks_revision_manifest_v1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "12-6.next100-023.ua-wikibooks-revision-manifest.v1",
                "snapshot_sha1": DUMP_SHA1,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
