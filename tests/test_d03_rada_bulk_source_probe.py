from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.probe_d03_rada_bulk_source as probe_mod
from tools.probe_d03_rada_bulk_source import (
    DEFAULT_CONFIG,
    ProbeError,
    _load_config,
    inventory_archive,
    observe_archive_inventory,
)


def _config(min_entries: int = 2) -> dict:
    return {
        "schema_version": "12-6.d03-rada-bulk-source-probe.v1",
        "worker_id": "TEST-RADA-BULK",
        "local_free_only": True,
        "model_training_executed": False,
        "training_authorized_bytes": 0,
        "parent_authority": {
            "head_sha": "f" * 40,
            "registry_identity_sha256": "e" * 64,
        },
        "source": {
            "family_id": "ua.rada.open-data.laws-texts",
            "dataset_id": "laws-texts",
            "archive_url": "https://example.invalid/texts.zip",
        },
        "probe_policy": {
            "canonical_entry_regex": r"d[0-9]+\.htm",
            "min_canonical_entries": min_entries,
            "max_archive_bytes": 1_000_000,
            "max_entry_bytes": 100_000,
            "max_total_uncompressed_bytes": 1_000_000,
        },
    }


def _archive(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _production_config() -> dict:
    return json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_inventory_is_deterministic_and_keeps_training_closed() -> None:
    archive = _archive(
        {
            "d2.htm": b"<html>two</html>",
            "nested/d1.htm": b"<html>one</html>",
            "README.txt": b"ignored",
        }
    )
    md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    report_a = inventory_archive(
        archive,
        _config(),
        expected_md5=md5,
        expected_bytes=len(archive),
    )
    report_b = inventory_archive(
        archive,
        _config(),
        expected_md5=md5,
        expected_bytes=len(archive),
    )

    assert report_a == report_b
    assert report_a["inventory"]["canonical_entry_count"] == 2
    assert report_a["inventory"]["ignored_file_count"] == 1
    assert [row["basename"] for row in report_a["inventory"]["entries"]] == [
        "d1.htm",
        "d2.htm",
    ]
    assert report_a["training_authorized_bytes"] == 0
    assert report_a["corpus_admitted"] is False
    assert report_a["gates"]["canonical_normalization"] == "NOT_RUN"
    assert report_a["gates"]["discovery_capacity_threshold"] == "PASS"
    assert report_a["gates"]["exact_archive_identity"] == (
        "PASS_PINNED_DISCOVERY_REVALIDATED"
    )
    assert report_a["safe_result"] == (
        "PINNED_BULK_ARCHIVE_INVENTORIED_DOWNSTREAM_GATES_REQUIRED"
    )
    assert len(report_a["config_identity_sha256"]) == 64


def test_unpinned_observation_is_machine_distinct_from_strict_pass() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})

    report = inventory_archive(archive, _config())

    assert report["gates"]["exact_archive_identity"] == "OBSERVED_UNPINNED"
    assert report["gates"]["discovery_capacity_threshold"] == "PASS"
    assert report["safe_result"] == "CURRENT_UPSTREAM_OBSERVED_SUCCESSOR_PIN_REQUIRED"
    assert report["training_authorized_bytes"] == 0


def test_forensic_observation_retains_below_minimum_without_waiving_gate() -> None:
    archive = _archive({"d1.htm": b"a", "readme.txt": b"x"})

    first = observe_archive_inventory(archive, _config(min_entries=2))
    second = observe_archive_inventory(archive, _config(min_entries=2))

    assert first == second
    assert first["inventory"]["canonical_entry_count"] == 1
    assert first["gates"]["safe_zip_inventory"] == "PASS"
    assert first["gates"]["discovery_capacity_threshold"] == "FAIL_BELOW_MINIMUM"
    assert first["gates"]["exact_archive_identity"] == "OBSERVED_UNPINNED"
    assert first["training_authorized_bytes"] == 0
    assert first["corpus_admitted"] is False
    assert first["tokenizer_fit_authorized"] is False
    assert first["model_training_executed"] is False
    assert first["safe_result"] == (
        "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_SUCCESSOR_TRIAGE_REQUIRED"
    )
    assert len(first["archive"]["sha256"]) == 64
    assert len(first["inventory"]["entry_identity_sha256"]) == 64


def test_forensic_observation_does_not_bypass_structural_safety() -> None:
    archive = _archive({"../d1.htm": b"a"})

    with pytest.raises(ProbeError, match="unsafe archive path"):
        observe_archive_inventory(archive, _config(min_entries=2))


def test_rejects_archive_identity_drift() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})
    md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    with pytest.raises(ProbeError, match="byte identity drift"):
        inventory_archive(
            archive,
            _config(),
            expected_md5=md5,
            expected_bytes=len(archive) + 1,
        )


def test_rejects_partial_expected_identity() -> None:
    archive = _archive({"d1.htm": b"a", "d2.htm": b"b"})
    with pytest.raises(ProbeError, match="must be supplied together"):
        inventory_archive(archive, _config(), expected_bytes=len(archive))


def test_rejects_duplicate_canonical_basenames() -> None:
    archive = _archive({"a/d1.htm": b"a", "b/d1.htm": b"b"})
    with pytest.raises(ProbeError, match="duplicate canonical basename"):
        inventory_archive(archive, _config())


def test_rejects_path_traversal() -> None:
    archive = _archive({"../d1.htm": b"a", "d2.htm": b"b"})
    with pytest.raises(ProbeError, match="unsafe archive path"):
        inventory_archive(archive, _config())


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "docs//README.txt",
        "docs/./README.txt",
        "./docs/README.txt",
        "docs///README.txt",
    ],
)
def test_rejects_noncanonical_ignored_zip_path_aliases(unsafe_path: str) -> None:
    archive = _archive(
        {"d1.htm": b"a", "d2.htm": b"b", unsafe_path: b"ignored"}
    )
    with pytest.raises(ProbeError, match="unsafe archive path"):
        observe_archive_inventory(archive, _config(min_entries=2))


def test_safe_archive_name_allows_one_directory_trailing_slash() -> None:
    assert probe_mod._safe_archive_name("docs/")
    assert probe_mod._safe_archive_name(r"docs\README.txt")
    assert not probe_mod._safe_archive_name("docs//")
    assert not probe_mod._safe_archive_name("docs/./")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "C:/zak/perv/text/d3.htm",
        r"C:\zak\perv\text\d3.htm",
        "C:d3.htm",
        "docs:stream/d3.htm",
    ],
)
def test_rejects_windows_drive_and_ads_zip_paths(unsafe_path: str) -> None:
    archive = _archive(
        {"d1.htm": b"a", "d2.htm": b"b", unsafe_path: b"unsafe"}
    )
    with pytest.raises(ProbeError, match="unsafe archive path"):
        observe_archive_inventory(archive, _config(min_entries=2))


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "docs/CON",
        "docs/aux.txt",
        "docs/LPT1.txt",
        "docs/COM¹",
        "docs/report.",
        "docs/report ",
        "docs/question?.txt",
        "docs/control\x01.txt",
    ],
)
def test_rejects_windows_reserved_and_invalid_ignored_paths(
    unsafe_path: str,
) -> None:
    assert not probe_mod._safe_archive_name(unsafe_path)
    archive = _archive(
        {"d1.htm": b"a", "d2.htm": b"b", unsafe_path: b"unsafe"}
    )
    with pytest.raises(ProbeError, match="unsafe archive path"):
        observe_archive_inventory(archive, _config(min_entries=2))


def test_rejects_casefold_aliases_in_ignored_zip_entries() -> None:
    archive = _archive(
        {
            "d1.htm": b"a",
            "d2.htm": b"b",
            "docs/README.txt": b"first",
            "docs/readme.txt": b"second",
        }
    )
    with pytest.raises(ProbeError, match="case-folded ZIP path collision"):
        observe_archive_inventory(archive, _config(min_entries=2))


def test_portable_zip_names_still_accepted() -> None:
    assert probe_mod._safe_archive_name("docs/")
    assert probe_mod._safe_archive_name("docs/README.txt")
    assert probe_mod._safe_archive_name("zak/perv/text/d100.htm")


def test_rejects_too_few_canonical_entries() -> None:
    archive = _archive({"d1.htm": b"a", "readme.txt": b"x"})
    with pytest.raises(ProbeError, match="below minimum"):
        inventory_archive(archive, _config(min_entries=2))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda cfg: cfg["parent_authority"].__setitem__("head_sha", "0" * 40),
            "parent_authority.head_sha drifted",
        ),
        (
            lambda cfg: cfg["parent_authority"].__setitem__(
                "registry_identity_sha256", "0" * 64
            ),
            "parent_authority.registry_identity_sha256 drifted",
        ),
        (
            lambda cfg: cfg["source"].__setitem__(
                "archive_url", "https://example.invalid/other.zip"
            ),
            "source.archive_url drifted",
        ),
        (
            lambda cfg: cfg["probe_policy"].__setitem__("canonical_entry_regex", ".*"),
            "probe_policy.canonical_entry_regex drifted",
        ),
        (
            lambda cfg: cfg["probe_policy"].__setitem__("max_archive_bytes", 9_999_999_999),
            "probe_policy.max_archive_bytes drifted",
        ),
        (
            lambda cfg: cfg["rights_boundary"].__setitem__(
                "bulk_extension_status", "ADMITTED"
            ),
            r"rights_boundary\.bulk_extension_status drifted from the pinned v1 authority",
        ),
        (
            lambda cfg: cfg["claim_boundary"].__setitem__(
                "training_exposure_authorized", True
            ),
            r"claim_boundary\.training_exposure_authorized drifted from the pinned v1 authority",
        ),
    ],
)
def test_production_config_authority_drift_fails_closed(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    config = copy.deepcopy(_production_config())
    mutator(config)

    with pytest.raises(ProbeError, match=message):
        _load_config(_write_config(tmp_path, config))


def test_production_config_loads_under_exact_v1_authority() -> None:
    config = _load_config(DEFAULT_CONFIG)

    assert config["source"]["dataset_id"] == "laws-texts"
    assert config["training_authorized_bytes"] == 0


def test_live_probe_can_retain_exact_safe_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive({"d1.htm": b"a"})
    config = _config(min_entries=1)
    retained = tmp_path / "retained.zip"
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(probe_mod, "_load_config", lambda _: config)
    monkeypatch.setattr(
        probe_mod,
        "_download",
        lambda _url, *, max_bytes: (archive, {"etag": "fixture"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_d03_rada_bulk_source.py",
            "--config",
            str(tmp_path / "ignored.json"),
            "--accept-current-upstream",
            "--archive-output",
            str(retained),
            "--output",
            str(report_path),
        ],
    )

    probe_mod.main()

    assert retained.read_bytes() == archive
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["archive"]["sha256"] == hashlib.sha256(archive).hexdigest()
    assert report["gates"]["safe_zip_inventory"] == "PASS"
    assert report["training_authorized_bytes"] == 0
    assert report["corpus_admitted"] is False


def test_archive_output_rejects_preexisting_local_archive_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "input.zip"
    archive_path.write_bytes(_archive({"d1.htm": b"a"}))
    monkeypatch.setattr(probe_mod, "_load_config", lambda _: _config(min_entries=1))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_d03_rada_bulk_source.py",
            "--archive",
            str(archive_path),
            "--archive-output",
            str(tmp_path / "retained.zip"),
            "--accept-current-upstream",
        ],
    )

    with pytest.raises(ProbeError, match="only valid for a live source acquisition"):
        probe_mod.main()



@pytest.mark.parametrize("variant", ["top", "nested", "escaped_equivalent"])
def test_probe_config_rejects_duplicate_json_members(
    tmp_path: Path, variant: str
) -> None:
    config = _production_config()
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True)
    if variant == "top":
        raw = '{"schema_version": ' + json.dumps(config["schema_version"]) + "," + raw[1:]
    elif variant == "nested":
        raw = raw.replace(
            '"source": {',
            '"source": {"dataset_id": "laws-texts", ',
            1,
        )
    else:
        raw = raw[:-1] + f', "{chr(92)}u0073chema_version": ' + json.dumps(
            config["schema_version"]
        ) + "}"
    path = tmp_path / "ambiguous.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ProbeError, match="duplicate probe config JSON key"):
        _load_config(path)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_probe_config_rejects_nonfinite_json(
    tmp_path: Path, number: str
) -> None:
    raw = json.dumps(_production_config(), ensure_ascii=False)
    path = tmp_path / "nonfinite.json"
    path.write_text(
        raw[:-1] + ', "untrusted_number": ' + number + "}",
        encoding="utf-8",
    )
    with pytest.raises(
        ProbeError,
        match="non-standard probe config JSON constant|non-finite probe config JSON number",
    ):
        _load_config(path)


def test_probe_config_rejects_excessive_nesting_as_controlled_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "deep.json"
    path.write_text(
        '{"schema_version":' + "[" * 10000 + "0" + "]" * 10000 + "}",
        encoding="utf-8",
    )
    with pytest.raises(ProbeError, match="cannot load config"):
        _load_config(path)


def test_probe_config_rejects_invalid_utf8_and_non_object_root(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-utf8.json"
    invalid.write_bytes(bytes([255]))
    with pytest.raises(ProbeError, match="cannot load config"):
        _load_config(invalid)
    root = tmp_path / "array.json"
    root.write_text("[]", encoding="utf-8")
    with pytest.raises(ProbeError, match="root must be an object"):
        _load_config(root)


def test_probe_cli_rejects_ambiguous_config_before_archive_publication(
    tmp_path: Path,
) -> None:
    config = _production_config()
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True)
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version": ' + json.dumps(config["schema_version"]) + "," + raw[1:],
        encoding="utf-8",
    )
    retained = tmp_path / "should-not-exist.zip"
    report = tmp_path / "should-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "probe_d03_rada_bulk_source.py"),
            "--config",
            str(path),
            "--archive-output",
            str(retained),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.count("\n") == 1
    assert "duplicate probe config JSON key" in result.stderr
    assert "Traceback" not in result.stderr
    assert not retained.exists()
    assert not report.exists()
