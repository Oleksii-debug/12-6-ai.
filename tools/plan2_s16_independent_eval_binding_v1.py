"""Bind independent real EVAL-303 selection and EVAL-233 final custody for S16.

This does not admit the S15 training-source candidates to canonical S3--S9.
It reuses separate source-specific evaluation approvals and does not expose
heldout payload, final-test outcomes, model scores or optimizer targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools import materialize_d03_franko1901 as franko_materializer
from tools import materialize_next100_037_python_docs_accepted_chunks as cpython
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_real_eval233_decontamination_v1 as final_audit
from tools import plan2_s15_three_family_physical_v1 as physical
from tools import plan2_s15_three_family_split_probe_v1 as split_probe
from tools import plan2_s16_source_rights_privacy_v1 as rights
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from tools.validate_next100_037_python_docs_source_authority import EXPECTED_AUTHORITY_ID
from twelve_six import d03_arxiv_postrights_admission as arxiv_rights
from twelve_six.data import decontamination_authority_v2 as data232
from twelve_six.data import eval233_final_test_resolver_v1 as eval233
from twelve_six.data import eval303_selection_payload_resolver_v1 as eval303
from twelve_six.data import franko1901_dedup_intake as franko
from twelve_six.data import post_admission_dedup_intake_v2 as post_admission
from twelve_six.data import privacy_filter_v3 as privacy

SCHEMA = "12-6.plan2-s16-independent-real-eval-binding.v1"
OUTPUT = "independent-real-eval-binding.json"
OUTPUT_WITH_SELECTION = "independent-real-eval-binding-with-selection-scan.json"
OUTPUT_WITH_CPYTHON = "independent-real-eval-binding-with-cpython.json"
OUTPUT_WITH_FRANKO = "independent-real-eval-binding-with-cpython-franko.json"
OUTPUT_WITH_ARXIV = "independent-real-eval-binding-with-cpython-franko-arxiv.json"
ARXIV_AUTHORITY = "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json"
CPYTHON_AUTHORITY = "configs/data/next100_037_python_docs_source_authority_v1.json"
SELECTION_CONFIG = "configs/evaluation/eval303_selection_validation_composite_v1.json"
SELECTION_MEMBERS = "data/evaluation/eval303/selection-validation/composite-membership.jsonl"
SELECTION_GIT_BLOB = "c2e5b27bac541e55be2c807ea65b0e2ed77b7019"
SELECTION_MEMBER_GIT_BLOB = "04cf8b3fc1857f0d7b51e581aab8e4dc68f7f47d"
SELECTION_IDENTITY = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"
SELECTION_MEMBERS_SHA256 = "e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6"
ROLES = ("train", "validation", "test")
LANGS = ("uk", "en")


# EVAL-303 freezes different EN and UA metadata shapes. Reject extra
# heldout payload, outcome or forged grant fields before returning rows.
SELECTION_COMMON_MEMBER_KEYS = frozenset({
    "component_head_sha",
    "component_set_identity_sha256",
    "component_worker_id",
    "content_sha256",
    "final_reporting_eligible",
    "final_test_eligible",
    "future_training_prohibited",
    "hyperparameter_selection_eligible",
    "language",
    "license_id",
    "modality",
    "model_selection_eligible",
    "purpose",
    "record_id",
    "reservation_authority_ref",
    "rights_authority_ref",
    "selection_eligible",
    "selection_stratum",
    "source_family",
    "source_id",
    "source_version",
    "tokenizer_fit_eligible",
    "tokenizer_selection_eligible",
    "training_eligible",
    "utf8_bytes",
})
SELECTION_MEMBER_KEYS_BY_LANG = {
    "en": SELECTION_COMMON_MEMBER_KEYS | {"source_git_blob_sha1", "source_path"},
    "uk": SELECTION_COMMON_MEMBER_KEYS | {
        "candidate_locator", "reservation_commit_sha", "selector_rank_sha256",
        "source_kind", "source_raw_bytes", "source_raw_sha256",
    },
}


class IndependentEvalDenied(ValueError):
    """Physical selection/final membership or rights reservation drift."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise IndependentEvalDenied(reason)


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in values:
        need(k not in result, "duplicate heldout membership JSON key")
        result[k] = v
    return result


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "strict")).hexdigest()


def inspect_selection_rows(raw: bytes, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate *metadata only*: selected text cannot reenter training."""
    need(isinstance(raw, bytes) and bool(raw) and raw.endswith(b"\n"),
         "invalid selection membership bytes")
    need(config["schema_version"] == "12-6.eval303-selection-validation-composite.v1"
         and config["selection_identity_sha256"] == SELECTION_IDENTITY
         and config["purpose"] == "selection-validation"
         and config["rights_and_reservation"][
             "all_included_records_selection_validation_authorized"] is True
         and config["rights_and_reservation"][
             "all_included_records_training_prohibited"] is True
         and config["rights_and_reservation"][
             "all_included_records_tokenizer_fit_prohibited"] is True
         and config["rights_and_reservation"][
             "all_included_records_final_test_prohibited"] is True,
         "selection scope/rights/config drift")
    entries = raw.splitlines(keepends=True)
    need(len(entries) == config["composite_membership"]["documents"] == 10,
         "selection count drift")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    contents: set[str] = set()
    families: Counter[str] = Counter()
    langcounts: Counter[str] = Counter()
    for line in entries:
        need(line.endswith(b"\n"), "selection member lacks line terminator")
        try:
            row = json.loads(line.decode("utf-8", "strict"), object_pairs_hook=_pairs)
        except IndependentEvalDenied:
            raise  # Preserve precise duplicate-member refusal, not a generic parse error.
        except (UnicodeError, ValueError) as exc:
            raise IndependentEvalDenied("malformed selection row") from exc
        need(type(row) is dict, "invalid membership row")
        need(type(row.get("language")) is str
             and row["language"] in SELECTION_MEMBER_KEYS_BY_LANG
             and set(row) == SELECTION_MEMBER_KEYS_BY_LANG[row["language"]],
             "selection member contains unknown or missing metadata fields")
        rid = row.get("record_id")
        fam = row.get("source_family")
        sid = row.get("source_id")
        lang = row.get("language")
        digest = row.get("content_sha256")
        length = row.get("utf8_bytes")
        need(all(type(v) is str and bool(v) for v in (rid, sid, fam))
             and type(lang) is str and lang in LANGS
             and type(digest) is str and len(digest) == 64
             and all(ch in "0123456789abcdef" for ch in digest)
             and type(length) is int and length > 0,
             "selected record source/language/hash invalid")
        need(row.get("purpose") == "selection-validation"
             and row.get("selection_stratum") in ("ua", "en")
             and row["selection_stratum"] == ("ua" if lang == "uk" else "en")
             and row.get("selection_eligible") is True
             and row.get("future_training_prohibited") is True
             and row.get("training_eligible") is False
             and row.get("tokenizer_fit_eligible") is False
             and row.get("final_test_eligible") is False
             and type(row.get("rights_authority_ref")) is str
             and bool(row["rights_authority_ref"])
             and type(row.get("reservation_authority_ref")) is str
             and bool(row["reservation_authority_ref"]),
             "selected record rights/reservation grant changed")
        need(rid not in ids and digest not in contents,
             "duplicate selection record or content")
        ids.add(rid)
        contents.add(digest)
        families[fam] += 1
        langcounts[lang] += 1
        rows.append(row)
    need(dict(families) == config["source_families"]["records_by_family"]
         and len(families) == config["source_families"]["independent_count"] == 4
         and langcounts == {"uk": 8, "en": 2},
         "selection language/family composition drift")
    return rows


def _training_rows(root: Path, sources: dict[str, Any]) -> list[dict[str, str]]:
    by_id = {row["record_id"]: row for row in sources["source_members"]}
    rejected = set(sources["g06_rejected_record_ids"])
    return [
        {
            **row,
            "source_family": by_id[row["record_id"]]["source_family"],
            "lineage_family": by_id[row["record_id"]]["document_family"],
            "modality": by_id[row["record_id"]]["modality"],
        }
        for row in split_probe.physical_rows(root, sources)
        if row["record_id"] not in rejected
    ]


def _surviving_rows(
    training: list[dict[str, str]], report: dict[str, Any],
) -> tuple[list[dict[str, str]], int]:
    rows = {_sha_text(row["record_id"]): row for row in training}
    excluded = {row["record_id_sha256"] for row in report["excluded_records"]}
    need(len(rows) == len(training) and excluded <= rows.keys(),
         "DATA-232 projection record identity drift")
    parents = {key: key for key in rows if key not in excluded}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for match in report["match_evidence"]:
        if "peer_record_id_sha256" not in match:
            continue
        a, b = match["train_record_id_sha256"], match["peer_record_id_sha256"]
        need(a in rows and b in rows and a != b, "unknown DATA-232 peer")
        if a in parents and b in parents:
            a, b = root(a), root(b)
            parents[max(a, b)] = min(a, b)

    sizes = {key: len(row["text"].encode("utf-8")) for key, row in rows.items()}
    representatives: dict[str, str] = {}
    for key in sorted(parents, key=lambda value: (-sizes[value], value)):
        representatives.setdefault(root(key), key)
    return [rows[key] for key in sorted(representatives.values())], len(excluded)


def _summarize_survivors(
    survivors: list[dict[str, str]], training_count: int, excluded_count: int,
) -> dict[str, Any]:
    families: dict[str, Counter[str]] = {}
    modalities: Counter[str] = Counter()
    inventory = []
    for row in survivors:
        size = len(row["text"].encode("utf-8"))
        families.setdefault(row["source_family"], Counter())[row["modality"]] += size
        modalities[row["modality"]] += size
        inventory.append({
            "record_id_sha256": _sha_text(row["record_id"]), "source_id": row["source_id"],
            "source_family": row["source_family"], "modality": row["modality"],
            "content_sha256": _sha_text(row["text"]), "utf8_bytes": size,
        })
    return {
        "policy": "remove-reserved-exclusions-then-longest-per-peer-component-v1",
        "surviving_records": len(inventory),
        "excluded_for_reserved_sets": excluded_count,
        "removed_as_training_duplicates": training_count - excluded_count - len(inventory),
        "surviving_utf8_bytes_by_family_and_modality": {
            family: dict(sorted(counts.items())) for family, counts in sorted(families.items())
        },
        "surviving_utf8_bytes_by_modality": dict(sorted(modalities.items())),
        "survivor_inventory_sha256": books.sha(books.canonical(inventory)),
        "production_admission": False,
    }


def _candidate_projection(
    training: list[dict[str, str]], report: dict[str, Any],
) -> dict[str, Any]:
    """Project verified DATA-232 evidence onto this exact, candidate-only cohort."""
    survivors, excluded_count = _surviving_rows(training, report)
    return _summarize_survivors(survivors, len(training), excluded_count)


def _scan_selection(
    root: Path,
    sources: dict[str, Any],
    inputs: tuple[Path, Path, Path, Path],
    *,
    extra_rows: list[dict[str, str]] | None = None,
    final_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    receipt, _ = _scan_selection_with_survivors(
        root, sources, inputs, extra_rows=extra_rows, final_rows=final_rows,
    )
    return receipt


def _scan_selection_with_survivors(
    root: Path,
    sources: dict[str, Any],
    inputs: tuple[Path, Path, Path, Path],
    *,
    extra_rows: list[dict[str, str]] | None = None,
    final_rows: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    ua_data, ua_manifest, en_data, en_authority = inputs
    try:
        evaluation, authority, evidence = eval303.resolve_eval303_selection_payloads_from_reconstructed(
            eval290_data_jsonl=ua_data,
            eval290_manifest_json=ua_manifest,
            eval291_data_jsonl=en_data,
            eval291_authority_json=en_authority,
            eval303_membership_jsonl=root / SELECTION_MEMBERS,
        )
        training = _training_rows(root, sources)
        corpus_identity = sources["source_members_sha256"]
        if extra_rows is not None:
            training += extra_rows
            corpus_identity = books.sha(books.canonical({
                "s15_members": corpus_identity,
                "extra_members": [
                    {**{key: value for key, value in row.items() if key != "text"},
                     "content_sha256": _sha_text(row["text"])}
                    for row in extra_rows
                ],
            }))
        if final_rows is not None:
            need({row["source_family"] for row in training}.isdisjoint(
                row["source_family"] for row in evaluation + final_rows
            ), "extended training and reserved source families overlap")
        report = data232.build_report(
            training,
            evaluation + (final_rows or []),
            training_corpus_identity=corpus_identity,
            selection_validation_identity=SELECTION_IDENTITY,
            final_test_identity=eval233.DATA232_FINAL_TEST_ID,
            authorities={
                "schema": "12-6.data232-reserved-authorities.v1",
                "authorities": [{key: authority[key] for key in (
                    "authority_id", "identity_sha256", "role", "source_sha"
                )}, {
                    "authority_id": "eval233-final-test",
                    "identity_sha256": eval233.DATA232_FINAL_TEST_ID,
                    "role": "final_test",
                    "source_sha": eval233.EVAL233_HEAD,
                }],
            },
            quarantine_cross_source_families=True,
        )
        data232.verify_report(report)
    except (OSError, ValueError, RuntimeError) as exc:
        raise IndependentEvalDenied("EVAL-303 payload decontamination denied") from exc
    counts = report["counts"]
    peer_matches = Counter(
        match["match_type"] for match in report["match_evidence"]
        if "peer_record_id_sha256" in match
    )
    receipt = {
        "selection_resolver_sha256": evidence["resolver_identity_sha256"],
        "data232_report_sha256": report["report_sha256"],
        "training_documents_compared": len(training),
        "selection_documents_compared": len(evaluation),
        "excluded_training_documents": counts["excluded_training_records"],
        "quarantined_training_families": counts["quarantined_source_families"],
        "training_peer_match_evidence_by_type": dict(sorted(peer_matches.items())),
        "clean": not (counts["excluded_training_records"] or counts["quarantined_source_families"]),
    }
    if extra_rows is not None:
        survivors, excluded_count = _surviving_rows(training, report)
        receipt["candidate_dedup_projection"] = _summarize_survivors(
            survivors, len(training), excluded_count,
        )
        return receipt, survivors
    return receipt, []


def _load_cpython_rows(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    authority = json.loads(books.read_checked(root, CPYTHON_AUTHORITY), object_pairs_hook=_pairs)
    identity = authority.pop("authority_identity_sha256", None)
    need(identity == EXPECTED_AUTHORITY_ID == books.sha(books.canonical(authority)),
         "CPython source authority identity drift")
    source = authority["source"]
    rights = authority["rights"]
    need(rights["uses"]["model_training"] == "ALLOWED" and rights["license_id"] == "PSF-2.0",
         "CPython source has no pinned training grant")
    license_url = f"https://raw.githubusercontent.com/python/cpython/{source['upstream_commit']}/LICENSE"
    try:
        license_bytes = cpython.fetch_exact(license_url)
        need(books.sha(license_bytes) == rights["license_sha256"]
             and books.git_blob(license_bytes) == rights["license_git_blob_sha1"],
             "CPython license bytes differ from the pinned grant")
        texts = cpython.verified_accepted_texts(authority)
    except cpython.MaterializationError as exc:
        raise IndependentEvalDenied("CPython source materialization denied") from exc
    need(all(privacy.hash_safe_scan(text).action == "ALLOW" for text in texts),
         "canonical privacy scan rejected a CPython candidate chunk")
    return [
        {
            "record_id": f"{source['source_id']}:{_sha_text(text)}",
            "source_id": source["source_id"],
            "source_family": source["source_family"],
            "lineage_family": source["source_family"],
            "modality": "en",
            "text": text,
        }
        for text in texts
    ], {
        "source_authority_sha256": identity,
        "source_family": source["source_family"],
        "license_sha256": rights["license_sha256"],
        "accepted_chunks": len(texts),
        "accepted_utf8_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "rejected_chunks": authority["quality_privacy"]["rejected_chunk_count"],
    }


def _load_franko_rows(root: Path, candidate: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        config = json.loads(books.read_checked(root, franko_materializer.DEFAULT_CONFIG.as_posix()),
                            object_pairs_hook=_pairs)
        franko_materializer.validate_contract(config)
        source_dir = candidate.parent.resolve(strict=True)
        license_bytes = books.read_checked(source_dir, "LICENSE.cc0.txt")
        license_evidence = franko_materializer.validate_license_bytes(config, license_bytes)
        attribution = books.read_checked(source_dir, "ATTRIBUTION.txt")
        need(attribution == (config["rights_boundary"]["required_attribution"] + "\n").encode(),
             "Franko attribution bytes differ from the pinned grant")
        need(not candidate.is_symlink() and candidate.is_file()
             and candidate.stat().st_size == franko.CANDIDATE_JSONL_BYTES,
             "Franko candidate size or file type drift")
        projection = franko.validate_and_project_franko(
            candidate,
            root / "configs/data/d03_franko1901_terminal_execution_v1.json",
            root / "configs/data/d03_franko1901_fresh_execution_authority_v1.json",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise IndependentEvalDenied("Franko source or rights verification denied") from exc
    need(projection.payloads is not None and projection.sources is not None,
         "Franko projection did not retain verified source bytes")
    rows, rejections = _privacy_screen(
        projection.sources, projection.payloads, franko.SOURCE_ID, franko.SOURCE_FAMILY, "uk",
    )
    need(bool(rows), "no Franko source members survived the privacy scan")
    return rows, {
        "source_family": franko.SOURCE_FAMILY,
        "candidate_sha256": franko.CANDIDATE_SHA256,
        "verified_source_records": len(projection.sources),
        "accepted_records": len(rows),
        "accepted_utf8_bytes": sum(len(row["text"].encode("utf-8")) for row in rows),
        "privacy_policy_version": privacy.POLICY_VERSION,
        "privacy_rejections": rejections,
        "license_sha256": license_evidence["license_sha256"],
        "attribution_sha256": books.sha(attribution),
    }


def _privacy_screen(
    inventory: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    payloads: dict[str, bytes],
    source_id: str, family: str, modality: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = []
    privacy_rejections: Counter[str] = Counter()
    for member in inventory:
        payload = payloads[member["source_id"]]
        scan = privacy.hash_safe_scan(payload)
        if scan.action != "ALLOW":
            privacy_rejections[scan.action] += 1
            continue
        rows.append({
            "record_id": member["source_id"],
            "source_id": source_id,
            "source_family": family,
            "lineage_family": family,
            "modality": modality,
            "text": payload.decode("utf-8", "strict"),
        })
    return rows, dict(sorted(privacy_rejections.items()))


def _load_arxiv_rows(root: Path, candidate: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        authority_raw = books.read_checked(root, ARXIV_AUTHORITY)
        authority = json.loads(authority_raw, object_pairs_hook=_pairs)
        arxiv_rights.validate_repository_bindings(authority, root)
        need(not candidate.is_symlink() and candidate.is_file()
             and 0 < candidate.stat().st_size <= 2_000_000, "arXiv candidate size or file type drift")
        inventory, payloads, source = post_admission._prepare_source(
            post_admission.ARXIV, authority_raw=authority_raw,
            candidate_raw=books.read_checked(candidate.parent.resolve(strict=True), candidate.name),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise IndependentEvalDenied("arXiv source or rights verification denied") from exc
    rows, rejections = _privacy_screen(
        inventory, payloads, "arxiv_abstracts", post_admission.ARXIV.family, "en",
    )
    need(bool(rows), "no arXiv source members survived the privacy scan")
    return rows, {
        **source,
        "domain": "science_descriptive_metadata_and_abstracts",
        "verified_source_records": len(inventory),
        "accepted_records": len(rows),
        "accepted_utf8_bytes": sum(len(row["text"].encode("utf-8")) for row in rows),
        "privacy_policy_version": privacy.POLICY_VERSION,
        "privacy_rejections": rejections,
    }


def inspect(
    root: Path, *, selection_inputs: tuple[Path, Path, Path, Path] | None = None,
    include_cpython: bool = False,
    franko_candidate: Path | None = None,
    arxiv_candidate: Path | None = None,
) -> dict[str, Any]:
    if arxiv_candidate is not None and franko_candidate is None:
        raise IndependentEvalDenied("the arXiv extension requires the verified Franko extension")
    if franko_candidate is not None and not include_cpython:
        raise IndependentEvalDenied("Franko extension requires the CPython comparison")
    if include_cpython and selection_inputs is None:
        raise IndependentEvalDenied("CPython extension requires the frozen selection inputs")
    root = root.resolve(strict=True)
    try:
        permissions = rights.inspect(root)
        sources = physical.inspect(root)
        finale = final_audit.inspect(root)
        selection_config_raw = books.read_checked(root, SELECTION_CONFIG)
        selected_raw = books.read_checked(root, SELECTION_MEMBERS)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IndependentEvalDenied("real physical source or evaluation lineage denied") from exc
    need(books.git_blob(selection_config_raw) == SELECTION_GIT_BLOB,
         "frozen EVAL-303 configuration Git blob drift")
    need(books.git_blob(selected_raw) == SELECTION_MEMBER_GIT_BLOB
         and books.sha(selected_raw) == SELECTION_MEMBERS_SHA256,
         "frozen EVAL-303 membership bytes drift")
    try:
        config = json.loads(selection_config_raw.decode("utf-8", "strict"),
                            object_pairs_hook=_pairs)
    except (UnicodeError, ValueError) as exc:
        raise IndependentEvalDenied("invalid EVAL-303 configuration") from exc
    need(type(config) is dict
         and config["composite_membership"]["sha256"] == SELECTION_MEMBERS_SHA256
         and config["final_test_firewall"]["data232_final_test_identity_sha256"]
         == eval233.DATA232_FINAL_TEST_ID,
         "selection/final firewalls disconnected")
    selected = inspect_selection_rows(selected_raw, config)
    need(permissions["physical_manifest_sha256"] == sources["manifest_sha256"]
         and permissions["source_level_rights_verified"] is True
         and permissions["physical_privacy_executed"] is True
         and permissions["s3_s9_corpus_admitted"] is False
         and permissions["training_corpus_authorized"] is False,
         "unverified original training candidate rights/privacy")
    need(finale["physical_three_family_sha256"] == sources["manifest_sha256"]
         and finale["selection_identity_sha256"] == SELECTION_IDENTITY
         and finale["real_final_test_custody_verified"] is True
         and finale["real_final_test_outcomes_read"] is False
         and finale["real_final_decontamination_clean"] is True
         and finale["physical_s3_s9_admitted"] is False,
         "real DATA232 final reserve/decontamination not cleared")
    try:
        final_rows, _authority, final_receipt = eval233.resolve_eval233_final_test(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise IndependentEvalDenied("real final set cannot be independently resolved") from exc
    need(final_receipt["source_final_set_identity_sha256"] ==
         finale["real_eval233_final_membership_identity_sha256"]
         and final_receipt["final_test_outcomes_read"] is False
         and final_receipt["documents"] == 16,
         "real final-test source identity drift")
    train_families = {row["source_family"] for row in sources["source_members"]}
    val_families = {row["source_family"] for row in selected}
    test_families = {row["source_family"] for row in final_rows}
    need(len(train_families) == 3 and len(val_families) == 4
         and len(test_families) == 2
         and train_families.isdisjoint(val_families)
         and train_families.isdisjoint(test_families)
         and val_families.isdisjoint(test_families),
         "train, validation and final source-family overlap")
    train_ids = {row["source_id"] for row in sources["source_members"]}
    val_ids = {row["source_id"] for row in selected}
    test_ids = {row["source_id"] for row in final_rows}
    need(train_ids.isdisjoint(val_ids)
         and train_ids.isdisjoint(test_ids)
         and val_ids.isdisjoint(test_ids),
         "train, validation and final source identity overlap")
    train_hashes = {row["normalized_sha256"] for row in sources["source_members"]}
    val_hashes = {row["content_sha256"] for row in selected}
    test_hashes = {_sha_text(row["text"]) for row in final_rows}
    need(not train_hashes & val_hashes
         and not train_hashes & test_hashes
         and not val_hashes & test_hashes,
         "train, selection or final exact content leakage")
    by_role: dict[str, Counter[str]] = {
        "train": Counter(x["modality"] for x in sources["source_members"]),
        "validation": Counter(x["language"] for x in selected),
        "test": Counter("uk" if x["modality"] == "ua" else x["modality"]
                        for x in final_rows),
    }
    need(all(by_role[role][lang] > 0 for role in ROLES for lang in LANGS)
         and by_role["train"] == {"uk": 12, "en": 3}
         and by_role["validation"] == {"uk": 8, "en": 2}
         and by_role["test"] == {"uk": 8, "en": 8},
         "real independent UA/EN role coverage changed")
    byte_counts: dict[str, Counter[str]] = {
        "train": Counter(), "validation": Counter(), "test": Counter(),
    }
    for row in sources["source_members"]:
        byte_counts["train"][row["modality"]] += row["normalized_bytes"]
    for row in selected:
        byte_counts["validation"][row["language"]] += row["utf8_bytes"]
    for row in final_rows:
        lang = "uk" if row["modality"] == "ua" else "en"
        byte_counts["test"][lang] += len(row["text"].encode("utf-8"))
    core = {
        "schema_version": SCHEMA,
        "decision": "INDEPENDENT_REAL_UA_EN_EVAL_REUSED_NO_PRODUCTION_CORPUS",
        "physical_source_rights_sha256": permissions["manifest_sha256"],
        "physical_training_candidate_sha256": sources["manifest_sha256"],
        "eval303_selection_identity_sha256": SELECTION_IDENTITY,
        "eval303_selection_membership_sha256": SELECTION_MEMBERS_SHA256,
        "eval233_final_set_identity_sha256":
            final_receipt["source_final_set_identity_sha256"],
        "eval233_data232_report_sha256": finale["data232_report_sha256"],
        "role_language_records": {
            role: {lang: by_role[role][lang] for lang in LANGS}
            for role in ROLES
        },
        "role_language_utf8_bytes": {
            role: {lang: byte_counts[role][lang] for lang in LANGS}
            for role in ROLES
        },
        "role_family_counts": {
            "train": len(train_families), "validation": len(val_families),
            "test": len(test_families),
        },
        "all_three_roles_physical_ua_en": True,
        "all_source_families_disjoint_across_roles": True,
        "role_exact_content_overlap_count": 0,
        "evaluation_use_rights_separately_admitted": True,
        "heldout_fitted_to_tokenizer": False,
        "real_final_test_outcomes_read": False,
        "current_s15_train_sources_are_corpus_admitted": False,
        "production_balance_approved": False,
        "train_selection_cross_source_near_decontamination_verified": False,
        "production_s3_s9_dataset_authorized": False,
        "training_authorized": False,
        "plan2_evaluation_release_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "plan9_binding_authorized": False,
        "terminal_done": False,
    }
    if selection_inputs is not None:
        scan = _scan_selection(root, sources, selection_inputs)
        core["train_selection_cross_source_near_decontamination_verified"] = scan["clean"]
        core["selection_decontamination"] = scan
        if include_cpython:
            extra_rows, source = _load_cpython_rows(root)
            need(source["source_family"] not in train_families, "CPython family already in S15")
            extended = _scan_selection(
                root, sources, selection_inputs, extra_rows=extra_rows, final_rows=final_rows
            )
            core["cpython_extension"] = {
                **source,
                "decontamination": extended,
                "final_documents_compared": len(final_rows),
                "combined_training_families": len(train_families) + 1,
                "combined_training_records": {
                    **by_role["train"], "en": by_role["train"]["en"] + len(extra_rows),
                },
                "combined_training_utf8_bytes": {
                    **byte_counts["train"],
                    "en": byte_counts["train"]["en"] + source["accepted_utf8_bytes"],
                },
            }
            if franko_candidate is not None:
                franko_rows, franko_source = _load_franko_rows(root, franko_candidate)
                need(franko_source["source_family"] not in train_families | {source["source_family"]},
                     "Franko family already in the candidate cohort")
                franko_scan = _scan_selection(
                    root, sources, selection_inputs,
                    extra_rows=extra_rows + franko_rows, final_rows=final_rows,
                )
                core["franko_extension"] = {
                    **franko_source,
                    "decontamination": franko_scan,
                    "final_documents_compared": len(final_rows),
                    "combined_training_families": len(train_families) + 2,
                    "combined_training_records": {
                        "uk": by_role["train"]["uk"] + len(franko_rows),
                        "en": by_role["train"]["en"] + len(extra_rows),
                    },
                    "combined_training_utf8_bytes": {
                        "uk": byte_counts["train"]["uk"] + franko_source["accepted_utf8_bytes"],
                        "en": byte_counts["train"]["en"] + source["accepted_utf8_bytes"],
                    },
                }
                if arxiv_candidate is not None:
                    arxiv_rows, arxiv_source = _load_arxiv_rows(root, arxiv_candidate)
                    need(arxiv_source["source_family"] not in train_families | {
                        source["source_family"], franko_source["source_family"],
                    }, "arXiv family already in the candidate cohort")
                    core["arxiv_extension"] = {
                        **arxiv_source,
                        "decontamination": _scan_selection(
                            root, sources, selection_inputs,
                            extra_rows=extra_rows + franko_rows + arxiv_rows, final_rows=final_rows,
                        ),
                        "final_documents_compared": len(final_rows),
                        "coverage": {
                            "science_candidate_records": len(arxiv_rows),
                            "math": "UNKNOWN_NO_EXACT_SUBDOMAIN_LABELS",
                            "code": "UNKNOWN_NO_STANDALONE_CODE_STRATUM_IN_THIS_COHORT",
                            "sk": "UNKNOWN_NO_VERIFIED_COHORT",
                            "cz": "UNKNOWN_NO_VERIFIED_COHORT",
                            "science_eval": "UNKNOWN_NO_DEDICATED_EVAL_COHORT",
                        },
                    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(
    root: Path,
    destination: Path,
    *,
    selection_inputs: tuple[Path, Path, Path, Path] | None = None,
    include_cpython: bool = False,
    franko_candidate: Path | None = None,
    arxiv_candidate: Path | None = None,
) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "unsafe independent evaluation destination")
    report = inspect(root, selection_inputs=selection_inputs, include_cpython=include_cpython,
                     franko_candidate=franko_candidate, arxiv_candidate=arxiv_candidate)
    destination.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_WITH_SELECTION if selection_inputs is not None else OUTPUT
    if include_cpython:
        output = OUTPUT_WITH_FRANKO if franko_candidate is not None else OUTPUT_WITH_CPYTHON
    if arxiv_candidate is not None:
        output = OUTPUT_WITH_ARXIV
    target = destination / output
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable independent-evaluation readback changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-reconstructed", type=Path, nargs=4,
        metavar=("UA_JSONL", "UA_MANIFEST", "EN_JSONL", "EN_AUTHORITY"),
        help="compare the original frozen selection documents in memory via DATA-232",
    )
    parser.add_argument("--include-cpython", action="store_true",
                        help="verify the admitted CPython source against both reserved sets")
    parser.add_argument("--franko-candidate", type=Path,
                        help="extend the comparison with the verified Franko1901 candidate JSONL")
    parser.add_argument("--arxiv-candidate", type=Path,
                        help="extend the comparison with the verified arXiv abstract candidate JSONL")
    args = parser.parse_args()
    try:
        inputs = tuple(args.selection_reconstructed) if args.selection_reconstructed else None
        report = stage(args.root, args.out_dir, selection_inputs=inputs,
                       include_cpython=args.include_cpython,
                       franko_candidate=args.franko_candidate, arxiv_candidate=args.arxiv_candidate)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "decision": report["decision"],
        "manifest_sha256": report["manifest_sha256"],
        "role_language_records": report["role_language_records"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
