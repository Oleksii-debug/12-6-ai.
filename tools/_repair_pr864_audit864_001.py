from pathlib import Path


source_path = Path("tools/normalize_d03_rada_bulk_html.py")
source = source_path.read_text(encoding="utf-8")

anchor = '''def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json'''
replacement = '''def _canonical_json_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _serialized_probe_report_bytes(data: Mapping[str, Any]) -> bytes:
    """Reproduce the exact deterministic bytes emitted by the pinning CLI."""
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\\n"
    ).encode("utf-8")


def _load_json'''
if anchor not in source:
    raise SystemExit("source anchor 1 not found")
source = source.replace(anchor, replacement, 1)

anchor = '''    if inventory.get("canonical_raw_bytes") != computed_raw_total:
        raise NormalizationError("probe canonical raw-byte total drift")
    return by_basename


def materialize_normalized_records('''
replacement = '''    if inventory.get("canonical_raw_bytes") != computed_raw_total:
        raise NormalizationError("probe canonical raw-byte total drift")
    return by_basename


def _validate_successor_observation_pin(
    probe: Mapping[str, Any],
    config: Mapping[str, Any],
    archive: bytes,
    *,
    probe_report_sha256: str,
) -> None:
    pin = config.get("successor_observation_pin")
    if not isinstance(pin, Mapping):
        raise NormalizationError("successor observation pin missing")
    if pin.get("source_pr") != 864:
        raise NormalizationError("successor observation source PR drift")
    source_head = pin.get("source_head_sha")
    if not isinstance(source_head, str) or not re.fullmatch(r"[0-9a-f]{40}", source_head):
        raise NormalizationError("successor observation head must be an exact commit SHA")
    if pin.get("config_path") != "configs/data/d03_rada_bulk_observation_pin_v1.json":
        raise NormalizationError("successor observation config path drift")
    for field in (
        "config_identity_sha256",
        "source_observation_report_sha256",
        "pinned_probe_report_sha256",
        "archive_sha256",
        "entry_identity_sha256",
    ):
        value = pin.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            raise NormalizationError(f"successor observation {field} must be SHA-256")
    if pin.get("pin_status") != "PASS_EXACT_EXECUTION":
        raise NormalizationError("successor observation pin is not exact-execution PASS")
    if pin.get("discovery_capacity_threshold") != "FAIL_BELOW_MINIMUM":
        raise NormalizationError("successor observation discovery threshold drift")
    if pin.get("normalization_input_only") is not True:
        raise NormalizationError("successor observation must remain normalization-input-only")
    for field in ("normalized_capacity_credited", "training_authorized_bytes"):
        value = pin.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise NormalizationError(
                f"successor observation {field} must be exact integer zero"
            )

    if probe_report_sha256 != pin["pinned_probe_report_sha256"]:
        raise NormalizationError(
            "probe report identity does not match successor observation pin"
        )
    archive_sha256 = _sha256(archive)
    if archive_sha256 != pin["archive_sha256"]:
        raise NormalizationError(
            "archive SHA-256 does not match successor observation pin"
        )
    archive_meta = probe.get("archive")
    if (
        not isinstance(archive_meta, Mapping)
        or archive_meta.get("sha256") != pin["archive_sha256"]
    ):
        raise NormalizationError(
            "probe archive identity does not match successor observation pin"
        )
    inventory = probe.get("inventory")
    if (
        not isinstance(inventory, Mapping)
        or inventory.get("entry_identity_sha256") != pin["entry_identity_sha256"]
    ):
        raise NormalizationError(
            "probe inventory identity does not match successor observation pin"
        )


def materialize_normalized_records('''
if anchor not in source:
    raise SystemExit("source anchor 2 not found")
source = source.replace(anchor, replacement, 1)

anchor = '''    if not isinstance(probe_report_sha256, str) or not SHA256_RE.fullmatch(
        probe_report_sha256
    ):
        raise NormalizationError("probe report identity must be SHA-256")
    expected = _validate_probe(probe, config, archive)
    normalizer = config["normalization"]'''
replacement = '''    if not isinstance(probe_report_sha256, str) or not SHA256_RE.fullmatch(
        probe_report_sha256
    ):
        raise NormalizationError("probe report identity must be SHA-256")
    observed_probe_report_sha256 = _sha256(_serialized_probe_report_bytes(probe))
    if probe_report_sha256 != observed_probe_report_sha256:
        raise NormalizationError(
            "probe report identity does not match deterministic report bytes"
        )
    expected = _validate_probe(probe, config, archive)
    _validate_successor_observation_pin(
        probe,
        config,
        archive,
        probe_report_sha256=observed_probe_report_sha256,
    )
    normalizer = config["normalization"]'''
if anchor not in source:
    raise SystemExit("source anchor 3 not found")
source = source.replace(anchor, replacement, 1)
source_path.write_text(source, encoding="utf-8")

test_path = Path("tests/test_d03_rada_bulk_normalization.py")
tests = test_path.read_text(encoding="utf-8")
anchor = '''def _report_sha(report: dict) -> str:
    encoded = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()'''
replacement = '''def _report_sha(report: dict) -> str:
    encoded = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()'''
if anchor not in tests:
    raise SystemExit("test anchor 1 not found")
tests = tests.replace(anchor, replacement, 1)

anchor = '''    config["parent_probe"]["probe_config_identity_sha256"] = report[
        "config_identity_sha256"
    ]
    return config'''
replacement = '''    config["parent_probe"]["probe_config_identity_sha256"] = report[
        "config_identity_sha256"
    ]
    successor = config["successor_observation_pin"]
    successor["pinned_probe_report_sha256"] = _report_sha(report)
    successor["archive_sha256"] = report["archive"]["sha256"]
    successor["entry_identity_sha256"] = report["inventory"]["entry_identity_sha256"]
    return config'''
if anchor not in tests:
    raise SystemExit("test anchor 2 not found")
tests = tests.replace(anchor, replacement, 1)

marker = "def test_frozen_successor_pin_rejects_coherent_report_substitution()"
if marker not in tests:
    tests += '''


def test_frozen_successor_pin_rejects_coherent_report_substitution() -> None:
    archive = _archive({"d1.htm": b"<p>coherent substitute</p>"})
    report = _strict_probe(archive, 1)
    config = _normalizer_config_for_report(report)
    config["successor_observation_pin"] = copy.deepcopy(
        CONFIG["successor_observation_pin"]
    )
    with pytest.raises(
        NormalizationError,
        match="probe report identity does not match successor observation pin",
    ):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=_report_sha(report),
        )


def test_successor_pin_rejects_archive_substitution_after_report_binding() -> None:
    archive = _archive({"d1.htm": b"<p>coherent substitute</p>"})
    report = _strict_probe(archive, 1)
    config = _normalizer_config_for_report(report)
    config["successor_observation_pin"]["archive_sha256"] = "0" * 64
    with pytest.raises(
        NormalizationError,
        match="archive SHA-256 does not match successor observation pin",
    ):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=_report_sha(report),
        )


def test_successor_pin_rejects_inventory_substitution_after_report_and_archive_binding() -> None:
    archive = _archive({"d1.htm": b"<p>coherent substitute</p>"})
    report = _strict_probe(archive, 1)
    config = _normalizer_config_for_report(report)
    config["successor_observation_pin"]["entry_identity_sha256"] = "0" * 64
    with pytest.raises(
        NormalizationError,
        match="probe inventory identity does not match successor observation pin",
    ):
        materialize_normalized_records(
            archive,
            report,
            config,
            probe_report_sha256=_report_sha(report),
        )


def test_direct_api_cannot_forge_report_digest_for_changed_report_object() -> None:
    archive = _archive({"d1.htm": b"<p>one</p>"})
    report = _strict_probe(archive, 1)
    config = _normalizer_config_for_report(report)
    forged = copy.deepcopy(report)
    forged["ignored_attacker_field"] = "coherent-rehash-substitute"
    with pytest.raises(
        NormalizationError,
        match="probe report identity does not match deterministic report bytes",
    ):
        materialize_normalized_records(
            archive,
            forged,
            config,
            probe_report_sha256=_report_sha(report),
        )
'''
test_path.write_text(tests, encoding="utf-8")
