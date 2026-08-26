"""NEXT100-102: global cross-source dedup over the current converged authority vector."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

CONFIG_SCHEMA = "12-6.next100-102-cross-source-dedup-converged.v1"
REPORT_SCHEMA = "12-6.next100-102-cross-source-dedup-converged-report.v1"
NIST_POLICY = "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1"
MDN_POLICY = "MDN_PROSE_ONLY_MARKDOWN_V1"
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


class ConvergedDedupError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _raw_project(head: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./{head}/{path}"


def _load_bound_local(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    raw = (root / str(binding["path"])).read_bytes()
    if _git_blob_sha1(raw) != binding["git_blob_sha1"]:
        raise ConvergedDedupError(f"local authority blob drift: {binding['path']}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ConvergedDedupError("local authority root must be object")
    return value


def _fetch_authority(binding: Mapping[str, Any]) -> dict[str, Any]:
    raw = v1.fetch_exact_source(_raw_project(str(binding["head_sha"]), str(binding["path"])))
    if _git_blob_sha1(raw) != binding["git_blob_sha1"]:
        raise ConvergedDedupError(f"remote authority blob drift: {binding['worker_id']}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or value.get(binding["identity_key"]) != binding["identity_value"]:
        raise ConvergedDedupError(f"remote authority identity drift: {binding['worker_id']}")
    return value


def _common_row(source_id: str, family: str, origin: str, obj: str, modality: str,
                binding: Mapping[str, Any], capacity: int, expected_bytes: int,
                expected_sha: str, url: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_family": family,
        "stable_origin_id": origin,
        "stable_object_id": obj,
        "modality": modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"{binding['worker_id']} exact head {binding['head_sha']} workflow {binding['workflow_run']}",
        "declared_capacity_bytes": capacity,
        "expected_raw_bytes": expected_bytes,
        "expected_raw_sha256": expected_sha,
        "acquisition_url": url,
        "origin_key": obj,
    }


def _derive_kmu_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    if a.get("verdict") != "ADMIT" or a.get("rights", {}).get("training") != "ALLOWED_PRETRAINING":
        raise ConvergedDedupError("KMu authority is not training ADMIT")
    family = a.get("source_family", {}).get("family_id")
    records = a.get("records")
    if family != "ua.kmu.portal.secretariat-news" or not isinstance(records, list) or not records:
        raise ConvergedDedupError("KMu family/records drift")
    rows = []
    for r in records:
        if not isinstance(r, Mapping) or r.get("quality") != "PASS":
            raise ConvergedDedupError("KMu quality drift")
        rows.append(_common_row(
            f"ua.kmu.secretariat.{r['id']}", family, "publisher:kmu.gov.ua:secretariat-news",
            f"sha256:{r['raw_sha256']}", "uk", b, int(r["normalized_bytes"]), int(r["raw_bytes"]),
            str(r["raw_sha256"]), _raw_project(str(b["head_sha"]), str(r["raw_path"])),
        ))
    if sum(x["declared_capacity_bytes"] for x in rows) != a.get("aggregate", {}).get("normalized_bytes"):
        raise ConvergedDedupError("KMu aggregate drift")
    return rows


def _derive_nist_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    if a.get("terminal_status") != "ADMIT" or a.get("rights", {}).get("model_training") != "ALLOWED_WITH_NIST_SOURCE_PROVENANCE":
        raise ConvergedDedupError("NIST authority is not training ADMIT")
    detail = a.get("detailed_authority", {})
    raw = v1.fetch_exact_source(_raw_project(str(b["head_sha"]), str(detail.get("path"))))
    if _git_blob_sha1(raw) != detail.get("git_blob_sha1"):
        raise ConvergedDedupError("NIST detailed authority blob drift")
    d = json.loads(raw.decode("utf-8"))
    terminal = {x["publication_id"]: x for x in a.get("admit", []) if isinstance(x, Mapping)}
    rows = []
    for x in d.get("admit", []):
        pid = x["publication_id"]
        seal = terminal.get(pid)
        if seal is None or seal["normalized_sha256"] != x["normalized_sha256"] or int(seal["normalized_utf8_bytes"]) != int(x["normalized_utf8_bytes"]):
            raise ConvergedDedupError("NIST detail/seal mismatch")
        row = _common_row(
            "en.nist." + pid.lower().replace(".", "-"), "en.usgov.nist.technical-series",
            "publisher:nist:technical-series", f"publication:{pid}", "en", b,
            int(x["normalized_utf8_bytes"]), int(x["normalized_utf8_bytes"]), str(x["normalized_sha256"]), str(x["pdf_url"]),
        )
        row.update({
            "materialization_policy": NIST_POLICY,
            "materialization_source_expected_raw_bytes": int(x["raw_bytes"]),
            "materialization_source_expected_raw_sha256": str(x["raw_sha256"]),
            "materialization_pdf_start_page": int(x["pdf_start_page"]),
        })
        rows.append(row)
    if len(rows) != 3 or sum(x["declared_capacity_bytes"] for x in rows) != 59358:
        raise ConvergedDedupError("NIST bounded set drift")
    return rows


def _derive_mdn_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    claim = a.get("claim_boundary", {})
    pages = a.get("pages")
    if a.get("verdict") != "ADMIT_PROSE_ONLY" or claim.get("training_source_authority_terminal") is not True or claim.get("prose_only") is not True:
        raise ConvergedDedupError("MDN authority is not terminal prose-only ADMIT")
    if a.get("family", {}).get("family_id") != "en.mdn.webdocs.prose" or not isinstance(pages, list) or len(pages) != 1:
        raise ConvergedDedupError("MDN family/page drift")
    p = pages[0]
    if p.get("quality_status") != "PASS" or p.get("normalization_policy") != MDN_POLICY:
        raise ConvergedDedupError("MDN quality/normalization drift")
    commit = str(a.get("upstream", {}).get("commit"))
    path = str(p["path"])
    row = _common_row(
        "en.mdn.webdocs.http-compression", "en.mdn.webdocs.prose", "github:mdn/content",
        f"git-sha1:{p['git_blob_sha1']}", "en", b, int(p["normalized_bytes"]), int(p["normalized_bytes"]),
        str(p["normalized_sha256"]), f"https://raw.githubusercontent.com/mdn/content/{commit}/{path}",
    )
    row.update({
        "materialization_policy": MDN_POLICY,
        "materialization_source_expected_raw_bytes": int(p["raw_bytes"]),
        "materialization_source_expected_raw_sha256": str(p["raw_sha256"]),
        "materialization_source_expected_git_blob_sha1": str(p["git_blob_sha1"]),
    })
    return [row]


def _derive_verba_rows(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[dict[str, Any]]:
    if a.get("verdict") != "ADMIT" or a.get("scope", {}).get("training_admitted") is not True:
        raise ConvergedDedupError("Verba authority is not training ADMIT")
    family = a.get("family", {}).get("source_family")
    s = a.get("snapshot", {})
    if family != "ua.verba.public-domain.nomis1864":
        raise ConvergedDedupError("Verba family drift")
    return [_common_row(
        "ua.verba.nomis1864.bounded24", family, "edition:nomis1864:verba-v1.0.2",
        f"sha256:{s['normalized_sha256']}", "uk", b, int(s["normalized_bytes"]), int(s["normalized_bytes"]),
        str(s["normalized_sha256"]), _raw_project(str(b["head_sha"]), str(s["normalized_path"])),
    )]


def expand_inventory(config: Mapping[str, Any], repo_root: str | Path = ".") -> dict[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("local_free_only") is not True or config.get("model_training_executed") is not False:
        raise ConvergedDedupError("config safety/schema invariant failed")
    root = Path(repo_root)
    conv_b = config["convergence_authority"]
    conv = _load_bound_local(root, conv_b)
    if conv.get("claim_boundary", {}).get("safe_result") != "SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION":
        raise ConvergedDedupError("convergence handoff is not terminal")
    base = _load_bound_local(root, config["base_inventory"])
    inv = deepcopy(base)
    rows = list(inv["sources"])
    deriv = {"kmu_v1": _derive_kmu_rows, "nist_v2": _derive_nist_rows, "mdn_v1": _derive_mdn_rows, "verba_v1": _derive_verba_rows}
    for b in config["late_authority_bindings"]:
        rows.extend(deriv[str(b["kind"])](_fetch_authority(b), b))
    inv["sources"] = rows
    ids = [x["source_id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise ConvergedDedupError("source_id collision")
    cap = {m: sum(int(x["declared_capacity_bytes"]) for x in rows if x["modality"] == m) for m in ("uk", "en", "code")}
    cap["total"] = sum(cap.values())
    fam = {m: len({x["source_family"] for x in rows if x["modality"] == m}) for m in ("uk", "en", "code")}
    fam["total"] = len({x["source_family"] for x in rows})
    if len(rows) != int(conv_b["expected_numeric_source_object_count"]) or cap != conv_b["expected_numeric_capacity_bytes"] or fam != conv_b["expected_independent_family_counts"]:
        raise ConvergedDedupError(f"converged vector drift: objects={len(rows)} cap={cap} fam={fam}")
    return inv


def normalize_nist_extracted(text: str, max_bytes: int = 20_000) -> bytes:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n"))
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return raw
    prefix = raw[:max_bytes]
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
        raise ConvergedDedupError("no safe NIST truncation boundary")
    return (candidate[:cut].rstrip() + "\n").encode()


def _materialize_nist(row: Mapping[str, Any], pdf: bytes, version_prefix: str, max_bytes: int) -> bytes:
    if len(pdf) != int(row["materialization_source_expected_raw_bytes"]) or _sha256(pdf) != row["materialization_source_expected_raw_sha256"]:
        raise ConvergedDedupError(f"NIST source identity drift: {row['source_id']}")
    version = subprocess.run(["pdftotext", "-v"], text=True, capture_output=True, check=True)
    line = (version.stderr or version.stdout).splitlines()[0]
    if version_prefix not in line:
        raise ConvergedDedupError(f"unsupported pdftotext version: {line}")
    with tempfile.TemporaryDirectory(prefix="next100-102-nist-") as tmp:
        pdf_path, txt_path = Path(tmp)/"s.pdf", Path(tmp)/"s.txt"
        pdf_path.write_bytes(pdf)
        subprocess.run(["pdftotext", "-f", str(row["materialization_pdf_start_page"]), "-nopgbrk", "-enc", "UTF-8", str(pdf_path), str(txt_path)], check=True)
        out = normalize_nist_extracted(txt_path.read_text(encoding="utf-8"), max_bytes)
    if len(out) != int(row["expected_raw_bytes"]) or _sha256(out) != row["expected_raw_sha256"]:
        raise ConvergedDedupError(f"NIST normalized identity drift: {row['source_id']}")
    return out


def _mdn_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ConvergedDedupError("unterminated MDN frontmatter")
    return text[end + 5:]


def _mdn_strip_fences(text: str) -> str:
    out, fence = [], None
    for line in text.splitlines():
        s = line.lstrip()
        if fence is None and (s.startswith("```") or s.startswith("~~~")):
            fence = s[:3]
            continue
        if fence is not None:
            if s.startswith(fence):
                fence = None
            continue
        out.append(line)
    if fence is not None:
        raise ConvergedDedupError("unterminated MDN fence")
    return "\n".join(out)


def normalize_mdn_prose(raw: bytes) -> bytes:
    text = unicodedata.normalize("NFKC", raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"<!--.*?-->", "", _mdn_strip_fences(_mdn_frontmatter(text)), flags=re.S)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("![") or "<img" in s.lower() or "<picture" in s.lower():
            continue
        if s.startswith("|") and s.endswith("|"):
            continue
        if re.search(r"\{\{\s*(Embed|InteractiveExample|LiveSample|EmbedGHLiveSample)", line, flags=re.I):
            continue
        line = re.sub(r"`+[^`\n]*`+", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\{\{[^{}]*\}\}", " ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"\s+", " ", line.replace("**", "").replace("__", "").replace("~~", "")).strip()
        lines.append(line)
    out, prev_blank = [], True
    for line in lines:
        blank = not line
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    while out and not out[-1]:
        out.pop()
    return ("\n".join(out) + "\n").encode()


def _materialize_mdn(row: Mapping[str, Any], raw: bytes) -> bytes:
    if len(raw) != int(row["materialization_source_expected_raw_bytes"]) or _sha256(raw) != row["materialization_source_expected_raw_sha256"] or _git_blob_sha1(raw) != row["materialization_source_expected_git_blob_sha1"]:
        raise ConvergedDedupError("MDN source identity drift")
    out = normalize_mdn_prose(raw)
    if len(out) != int(row["expected_raw_bytes"]) or _sha256(out) != row["expected_raw_sha256"]:
        raise ConvergedDedupError("MDN normalized identity drift")
    return out


def materialize_payload(row: Mapping[str, Any], config: Mapping[str, Any]) -> bytes:
    raw = v1.fetch_exact_source(str(row["acquisition_url"]))
    policy = row.get("materialization_policy")
    if policy is None:
        return raw
    if policy == MDN_POLICY:
        return _materialize_mdn(row, raw)
    if policy == NIST_POLICY:
        m = config["materialization"]
        return _materialize_nist(row, raw, str(m["nist_pdftotext_version_prefix"]), int(m["nist_max_normalized_utf8_bytes"]))
    raise ConvergedDedupError(f"unsupported materialization policy: {policy}")


def audit_live(config: Mapping[str, Any], repo_root: str | Path = ".") -> dict[str, Any]:
    inv = expand_inventory(config, repo_root)
    report = v3.audit_payloads(inv, {r["source_id"]: materialize_payload(r, config) for r in inv["sources"]})
    v3.verify_report(report)
    core = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": str(config["worker_id"]),
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "convergence_head_sha": config["convergence_authority"]["head_sha"],
        "source_count": report["source_count"],
        "pre_dedup_declared_capacity_bytes": report["terminal_candidates"]["declared_capacity_bytes_before"],
        "post_dedup_conservative_unique_capacity_bytes": report["terminal_candidates"]["conservative_unique_capacity_bytes_after"],
        "duplicate_discount_bytes": report["terminal_candidates"]["duplicate_discount_bytes"],
        "dedup_report": report,
        "claim_boundary": {
            "global_cross_source_dedup_executed": True,
            "corpus_identity_created": False,
            "evaluation_decontamination_executed": False,
            "post_pack_loss_positions_authorized": False,
            "training_authorized": False,
            "next_gate": "BALANCE_DIVERSITY_RETEST_THEN_CORPUS_MATERIALIZATION"
        }
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    core = dict(report)
    expected = core.pop("report_sha256", None)
    if report.get("schema_version") != REPORT_SCHEMA or expected != _sha256(_canonical_bytes(core)):
        raise ConvergedDedupError("report schema/self-hash failed")
    if report.get("local_free_only") is not True or report.get("model_training_executed") is not False:
        raise ConvergedDedupError("report safety invariant failed")
    nested = report.get("dedup_report")
    if not isinstance(nested, Mapping):
        raise ConvergedDedupError("missing nested V3 report")
    v3.verify_report(nested)
    if int(report.get("source_count", -1)) != 22 or int(report.get("pre_dedup_declared_capacity_bytes", -1)) != 320632:
        raise ConvergedDedupError("report does not bind current 22-object/320632-byte vector")
    if int(report.get("post_dedup_conservative_unique_capacity_bytes", -1)) > 320632:
        raise ConvergedDedupError("dedup inflated capacity")
    if report.get("claim_boundary", {}).get("training_authorized") is not False or report.get("claim_boundary", {}).get("corpus_identity_created") is not False:
        raise ConvergedDedupError("downstream readiness overclaim")


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
