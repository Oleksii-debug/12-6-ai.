"""The source seal and Plan 2 handoff for the native training component."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from twelve_six.training.launcher import NativeTrainingEndpoint, TrainingLauncher
from twelve_six.training.portable_packet import PortableTrainingPacket, PortableTrainingRun
from twelve_six.training.runtime_release import (
    RELEASE_PATH,
    RELEASE_SHA256,
    RELEASE_V2_PATH,
    RELEASE_V2_SHA256,
    TrainingRuntimeReleaseError,
    verify_plan2_handoff,
    verify_runtime_release,
)

ROOT = Path(__file__).resolve().parents[1]
V1_ACCEPTED = "08f3d9454819a7f47406b5b4eb2758480633f5c7"
RUN_CONTROL = "src/twelve_six/training/run_control.py"


def test_runtime_release_binds_the_accepted_data_and_native_entrypoints():
    release = verify_runtime_release(ROOT, release_version=2)
    assert release["schema"] == "12-6.plan3.training-runtime-release.v2"
    assert release["qualification"] == "LOCAL_COMPONENT"
    assert release["full_profile_measurement_status"] == "OPEN_EXTERNAL"
    assert release["campaign_authorized"] is False
    handoff = release["plan2_handoff"]
    assert handoff["tensor_fields"] == ["input_ids", "target_ids", "loss_mask"]
    assert handoff["first_batch_shape"] == [2, 32]
    assert handoff["training_targets"] == 1_985_122
    assert handoff["first_batch_targets"] == 45
    assert callable(PortableTrainingPacket.checkpoint_identity)
    assert callable(PortableTrainingRun.run)
    assert callable(NativeTrainingEndpoint.__call__)
    assert callable(TrainingLauncher.invoke)


@pytest.mark.parametrize(("version", "relative"), [(1, RELEASE_PATH), (2, RELEASE_V2_PATH)])
def test_runtime_release_refuses_a_locally_rewritten_descriptor(tmp_path, version, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    raw = (ROOT / relative).read_bytes()
    path.write_bytes(raw.replace(b'"campaign_authorized": false', b'"campaign_authorized": true'))
    with pytest.raises(TrainingRuntimeReleaseError, match="descriptor changed"):
        verify_runtime_release(tmp_path, release_version=version)


@pytest.fixture
def copied_release(tmp_path):
    release = verify_runtime_release(ROOT, release_version=2)
    for relative in (RELEASE_V2_PATH, *release["runtime_source_sha256_lf"]):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    assert verify_runtime_release(tmp_path, release_version=2) == release
    return tmp_path


def test_runtime_release_refuses_changed_native_recovery_source(copied_release):
    relative = "src/twelve_six/checkpoint/recovery_lock.py"
    with (copied_release / relative).open("ab") as stream:
        stream.write(b"\n# changed recovery source\n")
    with pytest.raises(TrainingRuntimeReleaseError, match=f"release source changed: {relative}"):
        verify_runtime_release(copied_release, release_version=2)


def test_physical_handoff_requires_runtime_from_the_verified_checkout(copied_release):
    with pytest.raises(TrainingRuntimeReleaseError, match="load the handoff runtime"):
        verify_plan2_handoff(copied_release, copied_release / "unused", release_version=2)


def test_v1_default_reopens_the_accepted_snapshot_without_accepting_current_runtime(tmp_path):
    historical = subprocess.check_output(["git", "show", f"{V1_ACCEPTED}:{RELEASE_PATH}"], cwd=ROOT)
    assert historical == (ROOT / RELEASE_PATH).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(historical).hexdigest() == RELEASE_SHA256
    release = json.loads(historical)
    descriptor = tmp_path / RELEASE_PATH
    descriptor.parent.mkdir(parents=True)
    descriptor.write_bytes(historical)
    for relative in release["runtime_source_sha256_lf"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "show", f"{V1_ACCEPTED}:{relative}"], cwd=ROOT))

    assert verify_runtime_release(tmp_path) == verify_runtime_release(tmp_path, release_version=1) == release
    with pytest.raises(TrainingRuntimeReleaseError, match=f"release source changed: {RUN_CONTROL}"):
        verify_runtime_release(ROOT)
    with pytest.raises(TrainingRuntimeReleaseError, match="unsupported training runtime release version"):
        verify_runtime_release(ROOT, release_version=3)

    v2 = verify_runtime_release(ROOT, release_version=2)
    assert v2["predecessor"] == {
        "accepted_git_sha": V1_ACCEPTED,
        "descriptor_sha256": RELEASE_SHA256,
        "schema": release["schema"],
    }
    assert v2["plan2_handoff"] == release["plan2_handoff"]
    old_hashes, new_hashes = release["runtime_source_sha256_lf"], v2["runtime_source_sha256_lf"]
    assert old_hashes.keys() == new_hashes.keys()
    assert {path for path in old_hashes if old_hashes[path] != new_hashes[path]} == {RUN_CONTROL}
    assert hashlib.sha256((ROOT / RELEASE_V2_PATH).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == RELEASE_V2_SHA256


def test_cli_can_explicitly_verify_v2():
    output = subprocess.check_output(
        [sys.executable, "-m", "twelve_six.training.runtime_release", "--root", str(ROOT), "--release-version", "2"],
        cwd=ROOT,
    )
    assert json.loads(output) == verify_runtime_release(ROOT, release_version=2)
