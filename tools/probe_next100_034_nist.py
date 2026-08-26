#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

DOCS = [
    {
        "publication_id": "NIST.SP.800-204",
        "doi": "10.6028/NIST.SP.800-204",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-204.pdf",
        "pdf_start_page": 9,
        "author": "Ramaswamy Chandramouli (NIST)",
    },
    {
        "publication_id": "NIST.SP.800-204C",
        "doi": "10.6028/NIST.SP.800-204C",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-204C.pdf",
        "pdf_start_page": 11,
        "author": "Ramaswamy Chandramouli (NIST)",
    },
    {
        "publication_id": "NIST.SP.800-215",
        "doi": "10.6028/NIST.SP.800-215",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-215.pdf",
        "pdf_start_page": 11,
        "author": "Ramaswamy Chandramouli (NIST)",
    },
]

MAX_NORMALIZED_BYTES = 20_000
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-NEXT100-034/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF-"):
        raise RuntimeError(f"not a PDF: {url}")
    return data


def normalize_extracted(text: str) -> tuple[str, int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    email_count = len(EMAIL_RE.findall(text))
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text).strip() + "\n"

    encoded = text.encode("utf-8")
    if len(encoded) > MAX_NORMALIZED_BYTES:
        prefix = encoded[:MAX_NORMALIZED_BYTES]
        while True:
            try:
                candidate = prefix.decode("utf-8")
                break
            except UnicodeDecodeError:
                prefix = prefix[:-1]
        cut = candidate.rfind("\n\n")
        if cut < 12_000:
            cut = candidate.rfind("\n")
        if cut < 12_000:
            raise RuntimeError("cannot find safe deterministic truncation boundary")
        text = candidate[:cut].rstrip() + "\n"
    return text, email_count


def shingle_set(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    return {tuple(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}


def quality(text: str) -> dict[str, object]:
    chars = len(text)
    words = WORD_RE.findall(text)
    letters = sum(ch.isalpha() for ch in text)
    replacements = text.count("\ufffd")
    controls = sum(ord(ch) < 32 and ch not in "\n\t" for ch in text)
    stop = {"the", "and", "of", "to", "in", "for", "is", "that", "with", "as"}
    stop_hits = sum(w.lower() in stop for w in words)
    return {
        "utf8_bytes": len(text.encode("utf-8")),
        "characters": chars,
        "words": len(words),
        "alphabetic_char_ratio": round(letters / max(chars, 1), 6),
        "english_stopword_ratio": round(stop_hits / max(len(words), 1), 6),
        "unicode_replacement_chars": replacements,
        "unexpected_control_chars": controls,
    }


def main() -> None:
    version = subprocess.run(["pdftotext", "-v"], text=True, capture_output=True, check=True)
    version_line = (version.stderr or version.stdout).splitlines()[0]
    existing_hashes: set[str] = set()
    registry = Path("data/registry/real_snapshots.v1.json")
    if registry.exists():
        payload = json.loads(registry.read_text(encoding="utf-8"))
        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "normalized_sha256" and isinstance(v, str):
                        existing_hashes.add(v)
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(payload)

    out_docs = []
    normalized_texts: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="next100-034-nist-") as tmp:
        tmp_path = Path(tmp)
        for doc in DOCS:
            raw = download(doc["url"])
            pdf_path = tmp_path / f"{doc['publication_id']}.pdf"
            txt_path = tmp_path / f"{doc['publication_id']}.txt"
            pdf_path.write_bytes(raw)
            subprocess.run(
                [
                    "pdftotext", "-f", str(doc["pdf_start_page"]), "-nopgbrk", "-enc", "UTF-8",
                    str(pdf_path), str(txt_path),
                ],
                check=True,
            )
            extracted = txt_path.read_text(encoding="utf-8")
            normalized, redacted_emails = normalize_extracted(extracted)
            norm_b = normalized.encode("utf-8")
            norm_sha = sha256_bytes(norm_b)
            q = quality(normalized)
            if q["words"] < 1800 or q["alphabetic_char_ratio"] < 0.55 or q["english_stopword_ratio"] < 0.08:
                raise RuntimeError(f"quality gate failed for {doc['publication_id']}: {q}")
            if q["unicode_replacement_chars"] != 0 or q["unexpected_control_chars"] != 0:
                raise RuntimeError(f"text corruption gate failed for {doc['publication_id']}: {q}")
            if norm_sha in existing_hashes:
                raise RuntimeError(f"exact normalized collision with existing registry: {doc['publication_id']}")
            normalized_texts[doc["publication_id"]] = normalized
            out_docs.append({
                **doc,
                "raw_bytes": len(raw),
                "raw_sha256": sha256_bytes(raw),
                "normalized_utf8_bytes": len(norm_b),
                "normalized_sha256": norm_sha,
                "emails_redacted": redacted_emails,
                "quality": q,
            })

    overlaps = []
    ids = list(normalized_texts)
    for i, a in enumerate(ids):
        sa = shingle_set(normalized_texts[a])
        for b in ids[i+1:]:
            sb = shingle_set(normalized_texts[b])
            overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
            overlaps.append({"a": a, "b": b, "smaller_set_13word_shingle_overlap": round(overlap, 6)})
            if overlap >= 0.30:
                raise RuntimeError(f"near-duplicate gate failed: {a} {b} {overlap}")

    manifest = {
        "schema_version": "12-6.next100-034-nist-probe.v1",
        "worker_id": "NEXT100-034-DATA-EN-NIST",
        "family_id": "en.usgov.nist.technical-series",
        "family_count": 1,
        "normalization": {
            "extractor": version_line,
            "command": "pdftotext -f <pinned-start-page> -nopgbrk -enc UTF-8",
            "postprocess": "CRLF/CR->LF; FF->LF; Unicode NFKC; redact email as <EMAIL>; rstrip lines; collapse >=3 LF to 2; strip+terminal LF; truncate to <=20000 UTF-8 bytes at last blank-line boundary (fallback last LF >=12000 bytes)",
            "max_normalized_bytes_per_document": MAX_NORMALIZED_BYTES,
        },
        "documents": out_docs,
        "pairwise_dedup": overlaps,
        "existing_registry_exact_hashes_checked": len(existing_hashes),
        "total_normalized_utf8_bytes": sum(d["normalized_utf8_bytes"] for d in out_docs),
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    print("NIST_PROBE_JSON=" + encoded)
    print("::notice title=NIST_PROBE::" + encoded)


if __name__ == "__main__":
    main()

# Persistence retrigger: terminal evidence is committed by CI, not inferred from this marker.
