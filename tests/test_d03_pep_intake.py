from __future__ import annotations

import json
from copy import deepcopy

import pytest

from twelve_six.data.pep_intake import (
    PepIntakeError,
    PepTreeEntry,
    classify_license,
    git_blob_sha1,
    materialize,
    self_identity,
    validate_config,
    verify_payload,
)

CONFIG_PATH = "configs/data/d03_pep_public_domain_intake_v1.json"


def load_config() -> dict:
    return json.loads(open(CONFIG_PATH, encoding="utf-8").read())


def with_identity(config: dict) -> dict:
    config = deepcopy(config)
    config["contract_identity_sha256"] = self_identity(
        config,
        "contract_identity_sha256",
    )
    return config


def pep(text: str) -> bytes:
    return text.encode("utf-8")


def accepted(title: str = "Accepted") -> bytes:
    return pep(
        f"{title}\n{'=' * len(title)}\n\n"
        "Copyright\n=========\n\n"
        "This document has been placed in the public domain.\n"
    )


def opl() -> bytes:
    return pep(
        "Excluded\n========\n\n"
        "Copyright\n=========\n\n"
        "This document is licensed under the Open Publication License.\n"
    )


def entry(path: str, raw: bytes) -> dict:
    return {
        "path": path,
        "mode": "100644",
        "type": "blob",
        "sha": git_blob_sha1(raw),
        "size": len(raw),
    }


def fixture(config: dict) -> tuple[dict, dict[str, bytes]]:
    sentinels = config["known_opl_sentinels"]
    payloads = {path: opl() for path in sentinels}
    payloads["peps/pep-0001.rst"] = accepted("One")
    payloads["peps/pep-0002.rst"] = accepted("Two") + b"\nextra\n"
    payloads["peps/pep-0003.rst"] = accepted("Three") + b"\nmore extra\n"
    tree = {
        "sha": config["upstream"]["tree_sha1"],
        "truncated": False,
        "tree": [entry(path, raw) for path, raw in payloads.items()],
    }
    return tree, payloads


def fetcher(config: dict, payloads: dict[str, bytes]):
    prefix = config["upstream"]["raw_url_template"].split("{path}", 1)[0]

    def fetch(url: str) -> bytes:
        assert url.startswith(prefix)
        path = url[len(prefix) :]
        return payloads[path]

    return fetch


def test_checked_in_config_is_fail_closed() -> None:
    validate_config(load_config())


def test_opl_only_has_authority_inside_copyright_section() -> None:
    outside = (
        "PEP\n===\n\n"
        "Copyright\n=========\n\n"
        "This document has been placed in the public domain.\n\n"
        "References\n==========\n\n"
        "Historical text mentions the Open Publication License.\n"
    )
    assert classify_license(outside).status == "ACCEPT"
    assert classify_license(opl().decode()).status == "EXCLUDE"


def test_missing_or_ambiguous_copyright_is_quarantined() -> None:
    assert classify_license("PEP\n===\n\nNo rights section.\n").status == "QUARANTINE"
    duplicate = (
        "Copyright\n=========\n\nThis document is in the public domain.\n\n"
        "Copyright\n=========\n\nThis document is in the public domain.\n"
    )
    assert classify_license(duplicate).status == "QUARANTINE"


def test_tree_requires_all_five_opl_sentinels() -> None:
    config = load_config()
    tree, payloads = fixture(config)
    missing = config["known_opl_sentinels"][0]
    tree["tree"] = [item for item in tree["tree"] if item["path"] != missing]
    payloads.pop(missing)
    with pytest.raises(PepIntakeError, match="missing OPL sentinels"):
        materialize(config, tree, fetcher(config, payloads))


def test_blob_or_size_drift_fails_closed() -> None:
    raw = accepted()
    item = PepTreeEntry("peps/pep-0001.rst", "0" * 40, len(raw))
    with pytest.raises(PepIntakeError, match="Git blob mismatch"):
        verify_payload(item, raw)
    item = PepTreeEntry("peps/pep-0001.rst", git_blob_sha1(raw), len(raw) + 1)
    with pytest.raises(PepIntakeError, match="size mismatch"):
        verify_payload(item, raw)


def test_materialization_is_deterministic_and_zero_credit() -> None:
    config = load_config()
    tree, payloads = fixture(config)
    first_records, first_report = materialize(
        config,
        tree,
        fetcher(config, payloads),
    )
    second_records, second_report = materialize(
        config,
        deepcopy(tree),
        fetcher(config, payloads),
    )
    assert first_records == second_records
    assert first_report == second_report
    assert first_report["accepted_documents"] == 3
    assert first_report["training_authorized_bytes"] == 0
    assert first_report["authorized_unique_loss_positions"] == 0
    assert first_report["corpus_admitted"] is False
    assert first_report["privacy_gate"] == "NOT_RUN"
    assert first_report["global_dedup_gate"] == "NOT_RUN"
    assert first_report["model_training_executed"] is False
    assert all(record["training_eligible"] is False for record in first_records)
    assert first_report["known_opl_sentinels_verified_excluded"] == sorted(
        config["known_opl_sentinels"]
    )


def test_bounded_selection_prefers_larger_documents_then_path() -> None:
    config = load_config()
    config["selection_policy"]["max_documents"] = 2
    config["selection_policy"]["max_examined_documents"] = 16
    config = with_identity(config)
    tree, payloads = fixture(config)
    records, report = materialize(config, tree, fetcher(config, payloads))
    assert report["accepted_documents"] == 2
    assert [record["source_path"] for record in records] == [
        "peps/pep-0003.rst",
        "peps/pep-0002.rst",
    ]


def test_training_credit_or_gate_removal_invalidates_contract() -> None:
    config = load_config()
    config["training_authorized_bytes"] = 1
    config = with_identity(config)
    with pytest.raises(PepIntakeError, match="training_authorized_bytes drift"):
        validate_config(config)

    config = load_config()
    config["required_downstream_gates"].remove("privacy")
    config = with_identity(config)
    with pytest.raises(PepIntakeError, match="required downstream gates drift"):
        validate_config(config)
