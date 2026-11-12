"""The source seal and Plan 2 handoff for the native training component."""

from pathlib import Path

import pytest

from twelve_six.training.launcher import NativeTrainingEndpoint, TrainingLauncher
from twelve_six.training.portable_packet import PortableTrainingPacket, PortableTrainingRun
from twelve_six.training.runtime_release import (
    RELEASE_PATH,
    TrainingRuntimeReleaseError,
    verify_plan2_handoff,
    verify_runtime_release,
)

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_release_binds_the_accepted_data_and_native_entrypoints():
    release = verify_runtime_release(ROOT)
    assert release["schema"] == "12-6.plan3.training-runtime-release.v1"
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


def test_runtime_release_refuses_a_locally_rewritten_descriptor(tmp_path):
    path = tmp_path / RELEASE_PATH
    path.parent.mkdir(parents=True)
    raw = (ROOT / RELEASE_PATH).read_bytes()
    path.write_bytes(raw.replace(b'"campaign_authorized": false', b'"campaign_authorized": true'))
    with pytest.raises(TrainingRuntimeReleaseError, match="descriptor changed"):
        verify_runtime_release(tmp_path)


@pytest.fixture
def copied_release(tmp_path):
    release = verify_runtime_release(ROOT)
    for relative in (RELEASE_PATH, *release["runtime_source_sha256_lf"]):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    assert verify_runtime_release(tmp_path) == release
    return tmp_path


def test_runtime_release_refuses_changed_native_recovery_source(copied_release):
    relative = "src/twelve_six/checkpoint/recovery_lock.py"
    with (copied_release / relative).open("ab") as stream:
        stream.write(b"\n# changed recovery source\n")
    with pytest.raises(TrainingRuntimeReleaseError, match=f"release source changed: {relative}"):
        verify_runtime_release(copied_release)


def test_physical_handoff_requires_runtime_from_the_verified_checkout(copied_release):
    with pytest.raises(TrainingRuntimeReleaseError, match="load the handoff runtime"):
        verify_plan2_handoff(copied_release, copied_release / "unused")
