import json

from twelve_six.run_control.hardware import collect_hardware_profile


def test_hardware_profile_is_json_serializable_and_non_authorizing(tmp_path):
    profile = collect_hardware_profile(working_directory=tmp_path)

    assert profile["schema_version"] == 1
    assert profile["cpu"]["logical_cores"] is None or profile["cpu"]["logical_cores"] > 0
    assert profile["disk"]["total_bytes"] > 0
    assert profile["disk"]["free_bytes"] >= 0
    assert profile["authorization"]["cost_inferred"] is False
    assert profile["torch"]["cuda_device_count"] == len(profile["torch"]["cuda_devices"])
    json.dumps(profile)
