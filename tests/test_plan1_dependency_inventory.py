"""Plan-1 S3 no-admission dependency observation and negative/restart checks."""
import hashlib

import pytest

from tools.plan1_dependency_inventory import inspect


def project(tmp_path, deps):
    path = tmp_path / "pyproject.toml"
    content = "[project]\nname='fixture'\ndependencies = [" + ",".join(
        repr(d) for d in deps
    ) + "]\n"
    path.write_text(content, encoding="utf-8")
    return path


def test_installed_record_is_observation_never_authority(monkeypatch, tmp_path):
    from importlib import metadata

    class FakeDist:
        version = "2.5.1"

        def __init__(self):
            self.metadata = {"Name": "torch", "License": "MIT"}

        def read_text(self, name):
            assert name == "RECORD"
            return "torch/__init__.py,sha256=untrusted,10\n"

    monkeypatch.setattr(metadata, "distribution", lambda name: FakeDist())
    path = project(tmp_path, ["torch>=2.5"])
    first = inspect(path)
    assert first == inspect(path)  # restart/observational determinism
    asset = first["assets"][0]
    assert asset["record_sha256_observed_untrusted"] == hashlib.sha256(
        b"torch/__init__.py,sha256=untrusted,10\n"
    ).hexdigest()
    assert asset["installed_version"] == "2.5.1"
    assert asset["availability"] == "INSTALLED_UNVERIFIED"
    assert asset["admission"] == "DENIED_UNQUALIFIED"
    assert first["all_dependencies_admitted"] is False
    assert asset["independently_reviewed_license"] is False
    assert asset["data_rights_granted"] is False
    assert asset["foreign_model_weights_admitted"] is False


def test_missing_or_unrecorded_wheel_not_admitted(monkeypatch, tmp_path):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    assert inspect(project(tmp_path, ["numpy>=1.26"]))["assets"][0][
        "availability"
    ] == "NOT_INSTALLED"

    class Unrecorded:
        version = "1.26.0"

        def __init__(self):
            self.metadata = {"Name": "numpy"}

        def read_text(self, name):
            return None

    monkeypatch.setattr(metadata, "distribution", lambda name: Unrecorded())
    result = inspect(project(tmp_path, ["numpy>=1.26"]))
    assert result["assets"][0]["availability"] == "INSTALLED_RECORD_ABSENT"
    assert result["all_dependencies_admitted"] is False


@pytest.mark.parametrize("bad", [
    ["numpy>=1.26", "NumPy>=1.26"],
    ["torch>=2.5;python_version>='3.11'"],
    ["torch[extra]>=2.5"],
    ["torch~=2.5"],
    ["torch"],
    ["https://example.org/repo"],
    [],
])
def test_ambiguous_requirements_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        inspect(project(tmp_path, bad))


def test_wrong_installed_identity_and_oversize_record_fail_closed(monkeypatch, tmp_path):
    from importlib import metadata

    class Fake:
        version = "2.5.0"
        record = "bytes"

        def __init__(self):
            self.metadata = {"Name": "foreign"}

        def read_text(self, name):
            return self.record

    fake = Fake()
    monkeypatch.setattr(metadata, "distribution", lambda name: fake)
    path = project(tmp_path, ["torch>=2.5"])
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect(path)
    fake.metadata = {"Name": "torch"}
    fake.record = "x" * (4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="RECORD size invalid"):
        inspect(path)


def test_license_metadata_hash_never_grants_rights(monkeypatch, tmp_path):
    """Observed SPDX declarations help independent review, but cannot self-authorize."""
    from importlib import metadata

    class Distribution:
        version = "2.5.1"

        def __init__(self):
            self.metadata = {
                "Name": "torch",
                "License-Expression": "BSD-3-Clause",
                "License": "Additional bundled notices for native libraries",
            }

        def read_text(self, name):
            assert name == "RECORD"
            return "torch/__init__.py,sha256=untrusted,10\n"

    dist = Distribution()
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)
    manifest = project(tmp_path, ["torch>=2.5"])
    observed = inspect(manifest)["assets"][0]
    assert observed["license_metadata_field_observed_untrusted"] == "License-Expression"
    assert observed["license_metadata_sha256_observed_untrusted"] == hashlib.sha256(
        b"BSD-3-Clause"
    ).hexdigest()
    assert observed["admission"] == "DENIED_UNQUALIFIED"
    assert observed["independently_reviewed_license"] is False
    assert observed["foreign_model_weights_admitted"] is False
    assert inspect(manifest)["assets"][0] == observed  # readback determinism

    dist.metadata["License-Expression"] = "Apache-2.0"
    mutated = inspect(manifest)["assets"][0]
    assert mutated["license_metadata_sha256_observed_untrusted"] != (
        observed["license_metadata_sha256_observed_untrusted"]
    )
    assert mutated["admission"] == "DENIED_UNQUALIFIED"

    dist.metadata["License-Expression"] = ""
    with pytest.raises(ValueError, match="unbounded installed license"):
        inspect(manifest)
    dist.metadata["License-Expression"] = "X" * (4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="unbounded installed license"):
        inspect(manifest)


def test_license_notice_path_inventory_and_traversal_rejection(monkeypatch, tmp_path):
    """Real dist-info LICENSE paths are observed, never accepted as legal authority."""
    from importlib import metadata

    class Distribution:
        version = "0.7.0"

        def __init__(self):
            self.metadata = {"Name": "safetensors"}
            self.files = [
                "safetensors-0.7.0.dist-info/licenses/LICENSE",
                "safetensors-0.7.0.dist-info/licenses/NOTICE",
                "safetensors/__init__.py",
            ]

        def read_text(self, name):
            assert name == "RECORD"
            return "safetensors/__init__.py,sha256=untrusted,10\n"

    dist = Distribution()
    monkeypatch.setattr(metadata, "distribution", lambda name: dist)
    manifest = project(tmp_path, ["safetensors>=0.5"])
    entry = inspect(manifest)["assets"][0]
    assert entry["license_metadata_sha256_observed_untrusted"] is None
    assert entry["license_notice_paths_observed_untrusted"] == [
        "safetensors-0.7.0.dist-info/licenses/LICENSE",
        "safetensors-0.7.0.dist-info/licenses/NOTICE",
    ]
    assert entry["independently_reviewed_license"] is False
    assert entry["admission"] == "DENIED_UNQUALIFIED"
    assert inspect(manifest)["assets"][0] == entry

    for unsafe in ("../escape/LICENSE", "pkg" + chr(92) + "LICENSE", "/root/LICENSE", "pkg//LICENSE"):
        dist.files = [unsafe]
        with pytest.raises(ValueError, match="unsafe installed file inventory"):
            inspect(manifest)
