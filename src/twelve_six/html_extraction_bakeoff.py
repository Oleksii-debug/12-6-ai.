"""Fail-closed HTML extraction bake-off for SWARM-742.

This module is an independent research/verification surface. It does not select or
replace a production extractor and it never grants corpus or training authority.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class BakeoffError(RuntimeError):
    """Base error for invalid benchmark authority or execution."""


class ContractError(BakeoffError):
    """Raised when the frozen contract or fixture identity is invalid."""


class RuntimeIdentityError(BakeoffError):
    """Raised when an installed extractor does not match the pinned version."""


Extractor = Callable[[str], str | None]
_TOKEN_RE = re.compile(r"[\w’'-]+", re.UNICODE)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-serializable value with the benchmark canonical encoding."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def normalize_text(text: str | None) -> str:
    """Normalize extractor output for deterministic comparison."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(normalize_text(text))]


def _multiset_prf(candidate: str, gold: str) -> tuple[float, float, float]:
    candidate_counts = Counter(_tokens(candidate))
    gold_counts = Counter(_tokens(gold))
    overlap = sum((candidate_counts & gold_counts).values())
    candidate_n = sum(candidate_counts.values())
    gold_n = sum(gold_counts.values())
    precision = overlap / candidate_n if candidate_n else 0.0
    recall = overlap / gold_n if gold_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _contract_hash_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    return payload


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate immutable fixture identities and the non-authorizing boundary."""
    if contract.get("schema_version") != 1:
        raise ContractError("unsupported schema_version")
    if contract.get("contract_id") != "D03-HTML-EXTRACTION-BAKEOFF-V1":
        raise ContractError("unexpected contract_id")
    if contract.get("training_authorized_bytes") != 0:
        raise ContractError("benchmark cannot authorize training bytes")
    if contract.get("corpus_capacity_credited") != 0:
        raise ContractError("benchmark cannot grant corpus capacity")
    if contract.get("production_extractor_replacement_authorized") is not False:
        raise ContractError("benchmark cannot replace the production extractor")

    expected_contract_hash = contract.get("contract_sha256")
    actual_contract_hash = canonical_sha256(_contract_hash_payload(contract))
    if expected_contract_hash != actual_contract_hash:
        raise ContractError(
            f"contract hash mismatch: expected {expected_contract_hash}, got {actual_contract_hash}"
        )

    extractors = contract.get("extractors")
    if not isinstance(extractors, Mapping) or set(extractors) != {"trafilatura", "resiliparse"}:
        raise ContractError("contract must bind exactly trafilatura and resiliparse")
    for name, spec in extractors.items():
        if not isinstance(spec, Mapping):
            raise ContractError(f"extractor spec must be a mapping: {name}")
        version = spec.get("version")
        sdist_hash = spec.get("pypi_sdist_sha256")
        if not isinstance(version, str) or not version:
            raise ContractError(f"missing pinned version: {name}")
        if not isinstance(sdist_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", sdist_hash):
            raise ContractError(f"missing pinned PyPI sdist SHA-256: {name}")
        if spec.get("license") != "Apache-2.0":
            raise ContractError(f"unexpected extractor license: {name}")

    fixtures = contract.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) < 4:
        raise ContractError("at least four frozen fixtures are required")
    ids: set[str] = set()
    kinds: set[str] = set()
    languages: set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise ContractError("fixture must be a mapping")
        fixture_id = fixture.get("id")
        payload = fixture.get("payload")
        payload_hash = fixture.get("payload_sha256")
        kind = fixture.get("payload_kind")
        if not isinstance(fixture_id, str) or not fixture_id or fixture_id in ids:
            raise ContractError("fixture ids must be unique nonempty strings")
        if not isinstance(payload, str):
            raise ContractError(f"fixture payload must be UTF-8 text: {fixture_id}")
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != payload_hash:
            raise ContractError(f"fixture hash mismatch: {fixture_id}")
        if kind not in {"html", "warc_response"}:
            raise ContractError(f"unsupported fixture payload_kind: {fixture_id}")
        if not fixture.get("gold_text"):
            raise ContractError(f"fixture missing gold_text: {fixture_id}")
        if not fixture.get("required_anchors"):
            raise ContractError(f"fixture missing required anchors: {fixture_id}")
        ids.add(fixture_id)
        kinds.add(str(kind))
        languages.add(str(fixture.get("language")))
    if kinds != {"html", "warc_response"}:
        raise ContractError("fixture suite must cover HTML and WARC response envelopes")
    if not {"en", "uk"}.issubset(languages):
        raise ContractError("fixture suite must cover English and Ukrainian")

    boundary = contract.get("authority_boundary")
    if not isinstance(boundary, Mapping):
        raise ContractError("missing authority boundary")
    if boundary.get("model_training_executed") is not False:
        raise ContractError("model training must remain false")
    if boundary.get("paid_compute_used") is not False:
        raise ContractError("paid compute must remain false")
    forbidden = set(boundary.get("forbidden_terminal_states", []))
    if not {"ADOPTED", "TRAINING_AUTHORIZED", "CORPUS_RELEASED"}.issubset(forbidden):
        raise ContractError("authority boundary must forbid promotion states")


def load_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def extract_html_payload(fixture: Mapping[str, Any]) -> str:
    """Return HTML from a frozen fixture; WARC handling is fixture-only, not production parsing."""
    payload = fixture["payload"]
    if fixture["payload_kind"] == "html":
        return payload
    if fixture["payload_kind"] != "warc_response":
        raise ContractError(f"unsupported payload kind: {fixture['payload_kind']}")

    marker = "\r\n\r\nHTTP/"
    if not payload.startswith("WARC/1.1\r\n") or marker not in payload:
        raise ContractError(f"malformed frozen WARC response: {fixture['id']}")
    warc_headers, http_tail = payload.split(marker, 1)
    if "\r\nWARC-Type: response\r\n" not in f"\r\n{warc_headers}\r\n":
        raise ContractError(f"frozen WARC record is not a response: {fixture['id']}")
    http_message = "HTTP/" + http_tail
    if "\r\n\r\n" not in http_message:
        raise ContractError(f"missing HTTP body in frozen WARC response: {fixture['id']}")
    http_headers, body = http_message.split("\r\n\r\n", 1)
    status_line = http_headers.split("\r\n", 1)[0]
    if not re.fullmatch(r"HTTP/\d(?:\.\d)? 2\d\d(?: .*)?", status_line):
        raise ContractError(f"non-success HTTP status in frozen WARC response: {fixture['id']}")
    if "content-type: text/html" not in http_headers.casefold():
        raise ContractError(f"frozen WARC HTTP payload is not text/html: {fixture['id']}")
    if not body.strip():
        raise ContractError(f"empty frozen WARC HTTP body: {fixture['id']}")
    return body


def resolve_runtime_extractor(name: str, spec: Mapping[str, Any]) -> Extractor:
    """Load one extractor only after exact installed-version validation."""
    distribution = str(spec["distribution"])
    try:
        runtime_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeIdentityError(f"missing runtime distribution: {distribution}") from exc
    if runtime_version != spec["version"]:
        raise RuntimeIdentityError(
            f"runtime version mismatch for {distribution}: "
            f"expected {spec['version']}, got {runtime_version}"
        )

    if name == "trafilatura":
        from trafilatura import extract

        def trafilatura_adapter(html: str) -> str | None:
            return extract(
                html,
                include_comments=False,
                include_tables=False,
                include_links=False,
                include_images=False,
                output_format="txt",
            )

        return trafilatura_adapter

    if name == "resiliparse":
        from resiliparse.extract.html2text import extract_plain_text

        def resiliparse_adapter(html: str) -> str:
            return extract_plain_text(
                html,
                main_content=True,
                preserve_formatting=False,
                list_bullets=False,
                alt_texts=False,
                links=False,
                form_fields=False,
                noscript=False,
            )

        return resiliparse_adapter
    raise RuntimeIdentityError(f"unsupported extractor: {name}")


def _score_fixture(fixture: Mapping[str, Any], output: str) -> dict[str, Any]:
    normalized = normalize_text(output)
    precision, recall, f1 = _multiset_prf(normalized, str(fixture["gold_text"]))
    required = [normalize_text(str(v)).casefold() for v in fixture["required_anchors"]]
    forbidden = [normalize_text(str(v)).casefold() for v in fixture["forbidden_boilerplate"]]
    haystack = normalized.casefold()
    anchor_hits = sum(fragment in haystack for fragment in required)
    leakage_hits = sum(fragment in haystack for fragment in forbidden)
    return {
        "fixture_id": fixture["id"],
        "language": fixture["language"],
        "payload_kind": fixture["payload_kind"],
        "output_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "output_bytes": len(normalized.encode("utf-8")),
        "token_precision": round(precision, 8),
        "token_recall": round(recall, 8),
        "token_f1": round(f1, 8),
        "anchor_recall": round(anchor_hits / len(required), 8),
        "boilerplate_leakage": round(leakage_hits / len(forbidden), 8) if forbidden else 0.0,
        "empty": not bool(normalized),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row[key]) for row in rows) / len(rows), 8) if rows else 0.0


def run_extractor(
    name: str,
    spec: Mapping[str, Any],
    fixtures: list[Mapping[str, Any]],
    extractor: Extractor,
) -> dict[str, Any]:
    """Run an extractor twice per fixture and reject nondeterministic normalized output."""
    rows: list[dict[str, Any]] = []
    nondeterministic: list[str] = []
    total_duration_ns = 0
    for fixture in fixtures:
        html = extract_html_payload(fixture)
        start = time.perf_counter_ns()
        first = normalize_text(extractor(html))
        total_duration_ns += time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        second = normalize_text(extractor(html))
        total_duration_ns += time.perf_counter_ns() - start
        deterministic = first == second
        if not deterministic:
            nondeterministic.append(str(fixture["id"]))
        scored = _score_fixture(fixture, first)
        scored["deterministic"] = deterministic
        rows.append(scored)

    return {
        "extractor": name,
        "distribution": spec["distribution"],
        "configured_version": spec["version"],
        "release_identity": f"pypi-sdist-sha256:{spec['pypi_sdist_sha256']}",
        "fixtures": rows,
        "macro_token_precision": _mean(rows, "token_precision"),
        "macro_token_recall": _mean(rows, "token_recall"),
        "macro_token_f1": _mean(rows, "token_f1"),
        "macro_anchor_recall": _mean(rows, "anchor_recall"),
        "macro_boilerplate_leakage": _mean(rows, "boilerplate_leakage"),
        "all_deterministic": not nondeterministic,
        "nondeterministic_fixtures": nondeterministic,
        "non_authoritative_total_duration_ns": total_duration_ns,
    }


def _eligible(result: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    return (
        result["all_deterministic"]
        and result["macro_anchor_recall"] >= rule["minimum_macro_anchor_recall"]
        and result["macro_boilerplate_leakage"] <= rule["maximum_macro_boilerplate_leakage"]
    )


def select_candidate(
    results: Mapping[str, Mapping[str, Any]], rule: Mapping[str, Any]
) -> tuple[str, str]:
    """Apply the preregistered selection rule without producing an ADOPTED state."""
    if any(not result["all_deterministic"] for result in results.values()):
        return "RETEST_NONDETERMINISTIC", "at least one extractor was nondeterministic"

    eligible = {name: _eligible(result, rule) for name, result in results.items()}
    if eligible["trafilatura"] and not eligible["resiliparse"]:
        return "CANDIDATE_TRAFILATURA", "only Trafilatura passed quality gates"
    if eligible["resiliparse"] and not eligible["trafilatura"]:
        return "CANDIDATE_RESILIPARSE", "only Resiliparse passed quality gates"
    if not any(eligible.values()):
        return "NO_CLEAR_WINNER", "neither extractor passed all quality gates"

    left = results["trafilatura"]
    right = results["resiliparse"]
    advantage = float(rule["minimum_macro_token_f1_advantage"])
    f1_delta = float(left["macro_token_f1"]) - float(right["macro_token_f1"])
    if (
        f1_delta >= advantage
        and left["macro_boilerplate_leakage"] <= right["macro_boilerplate_leakage"]
    ):
        return "CANDIDATE_TRAFILATURA", "Trafilatura met the preregistered F1/leakage rule"
    if (
        -f1_delta >= advantage
        and right["macro_boilerplate_leakage"] <= left["macro_boilerplate_leakage"]
    ):
        return "CANDIDATE_RESILIPARSE", "Resiliparse met the preregistered F1/leakage rule"
    return str(rule["tie_state"]), "quality difference did not clear the preregistered margin"


def deterministic_report_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Remove telemetry that is intentionally not byte-reproducible."""
    projected = json.loads(json.dumps(report))
    projected.pop("deterministic_evidence_sha256", None)
    for result in projected.get("results", {}).values():
        result.pop("non_authoritative_total_duration_ns", None)
    return projected


def validate_report(report: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Fail closed on evidence that overstates the benchmark authority."""
    validate_contract(contract)
    allowed_states = set(contract["authority_boundary"]["allowed_terminal_states"])
    if report.get("terminal_state") not in allowed_states:
        raise ContractError(f"forbidden or unknown terminal state: {report.get('terminal_state')}")
    if report.get("training_authorized_bytes") != 0:
        raise ContractError("report cannot authorize training bytes")
    if report.get("corpus_capacity_credited") != 0:
        raise ContractError("report cannot grant corpus capacity")
    if report.get("production_extractor_replacement_authorized") is not False:
        raise ContractError("report cannot replace a production extractor")
    if report.get("base_sha") != contract["base_sha"]:
        raise ContractError("report base SHA drift")
    if report.get("contract_sha256") != contract["contract_sha256"]:
        raise ContractError("report contract identity drift")
    expected = canonical_sha256(deterministic_report_projection(report))
    if report.get("deterministic_evidence_sha256") != expected:
        raise ContractError("deterministic evidence hash mismatch")


def run_bakeoff(
    contract: Mapping[str, Any],
    adapters: Mapping[str, Extractor] | None = None,
) -> dict[str, Any]:
    """Execute the benchmark. Emit RETEST if pinned runtime identity is unavailable."""
    validate_contract(contract)
    fixtures = list(contract["fixtures"])
    results: dict[str, dict[str, Any]] = {}
    runtime_errors: dict[str, str] = {}

    for name, spec in contract["extractors"].items():
        try:
            extractor = (
                adapters[name]
                if adapters is not None
                else resolve_runtime_extractor(name, spec)
            )
            results[name] = run_extractor(name, spec, fixtures, extractor)
        except (KeyError, RuntimeIdentityError) as exc:
            runtime_errors[name] = str(exc)

    if runtime_errors:
        terminal_state = "RETEST_RUNTIME_REQUIRED"
        if any("version mismatch" in error for error in runtime_errors.values()):
            terminal_state = "RETEST_RUNTIME_IDENTITY"
        reason = "pinned extractor runtime unavailable or identity-invalid"
    else:
        terminal_state, reason = select_candidate(results, contract["selection_rule"])

    report: dict[str, Any] = {
        "schema_version": 1,
        "evidence_id": "D03-HTML-EXTRACTION-BAKEOFF-EVIDENCE-V1",
        "swarm_worker_id": contract["swarm_worker_id"],
        "swarm_lane_key": contract["swarm_lane_key"],
        "base_sha": contract["base_sha"],
        "contract_sha256": contract["contract_sha256"],
        "execution_profile": "LOCAL_FREE",
        "terminal_state": terminal_state,
        "reason": reason,
        "runtime_errors": runtime_errors,
        "results": results,
        "production_extractor_replacement_authorized": False,
        "training_authorized_bytes": 0,
        "corpus_capacity_credited": 0,
        "model_training_executed": False,
        "paid_compute_used": False,
        "final_test_payload_accessed": False,
    }
    report["deterministic_evidence_sha256"] = canonical_sha256(
        deterministic_report_projection(report)
    )
    validate_report(report, contract)
    return report
