"""LOCAL_FREE Plan 5 Section 9 media/provenance/authority gates."""
from dataclasses import replace

import pytest

from twelve_six_agent_runtime.multimodal import (
    MediaAdmission, MediaArtifact, MediaBoundaryError, MediaPolicy, MediaTransform,
    admit_media, media_tool_descriptor, prepare_media_transfer,
)
from twelve_six_agent_runtime.task_state import PendingEffect, TaskStore
from twelve_six_agent_runtime.tools import ToolBoundaryError, ToolRegistry, make_result


def artifact(**kwargs):
    payload = dict(artifact_id="media-1", kind="image", direction="input",
                   mime_type="image/png", byte_size=1024,
                   content_sha256="a" * 64, source_sha256="a" * 64,
                   source_id="host-camera", evidence_id="source-receipt",
                   timestamp_ms=1000, privacy="private", locality="local")
    payload.update(kwargs)
    return MediaArtifact(**payload)


def policy(**kwargs):
    values = dict(allowed_kinds=("asr", "audio", "image", "tts", "vision"),
                  allowed_privacy=("private", "public"), max_bytes=2048,
                  allow_remote=False)
    values.update(kwargs)
    return MediaPolicy(**values)


def admit(item=None, pol=None, **kwargs):
    args = dict(destination="local", model_id="fixture-model",
                model_capabilities=("image",), host_evidence_id="host-admission",
                verify_content=lambda _: True, verify_provenance=lambda _: True,
                authorize=lambda *_: True)
    args.update(kwargs)
    return admit_media(item or artifact(), pol or policy(), **args)


def test_typed_media_image_audio_asr_tts_vision_and_model_neutrality():
    for kind, mime, direction in (
        ("image", "image/png", "input"), ("audio", "audio/wav", "input"),
        ("asr", "text/plain", "output"), ("tts", "audio/ogg", "output"),
        ("vision", "application/json", "output"),
    ):
        item = artifact(kind=kind, mime_type=mime, direction=direction)
        receipt = admit(item, model_capabilities=(kind,))
        assert receipt.artifact_identity == item.identity
        assert receipt.model_id == "fixture-model"
    assert admit(model_id="replacement-model").artifact_identity == artifact().identity


def test_privacy_size_locality_and_unsupported_capability_fail_closed():
    denied = [
        (artifact(privacy="restricted"), policy()),
        (artifact(byte_size=2049), policy()),
        (artifact(kind="vision", mime_type="text/plain"), policy()),
    ]
    for item, rule in denied:
        with pytest.raises(MediaBoundaryError):
            admit(item, rule)
    for overrides in (
        dict(destination="remote"), dict(model_capabilities=("text",)),
        dict(verify_content=lambda _: False),
        dict(verify_provenance=lambda _: False),
        dict(authorize=lambda *_: False),
    ):
        with pytest.raises(MediaBoundaryError):
            admit(**overrides)
    assert admit(pol=policy(allow_remote=True), destination="remote").destination == "remote"


def test_bad_artifact_types_and_bounded_fields_rejected():
    for values in (
        {"kind": "video"}, {"mime_type": "application/octet-stream"},
        {"byte_size": -1}, {"byte_size": 0}, {"byte_size": 2**30},
        {"content_sha256": "invalid"}, {"source_sha256": "0" * 64},
        {"privacy": "unknown"}, {"direction": "duplex"},
        {"timestamp_ms": True}, {"source_id": ""},
    ):
        with pytest.raises(MediaBoundaryError):
            artifact(**values).validate()
    with pytest.raises(MediaBoundaryError):
        admit(pol=policy(max_bytes=-1))
    with pytest.raises(MediaBoundaryError):
        admit(pol=policy(allowed_kinds=("image", "image")))


def test_transform_chain_timestamps_evidence_and_identity():
    step = MediaTransform("resize", "a" * 64, "b" * 64, "resize-evidence", 1001)
    item = artifact(content_sha256="b" * 64, transforms=(step,))
    assert admit(item).artifact_identity == item.identity
    assert item.identity != artifact().identity
    for bad in (
        replace(step, input_sha256="c" * 64),
        replace(step, output_sha256="c" * 64),
        replace(step, evidence_id=""),
        replace(step, timestamp_ms=999),
    ):
        with pytest.raises(MediaBoundaryError):
            admit(artifact(content_sha256="b" * 64, transforms=(bad,)))
    with pytest.raises(MediaBoundaryError):
        admit(artifact(transforms=tuple(step for _ in range(17))))


def harness(tmp_path):
    tasks = TaskStore(tmp_path / "tasks.db")
    tasks.create(task_id="task", plan_id="plan5", step_id="media")
    init = tasks.load("task")
    tasks.checkpoint("task", expected_epoch=init.control_epoch,
                     expected_revision=init.revision, step_id="transfer",
                     checkpoint_id="reserve", pending_effects=(
                         PendingEffect("effect", "media transfer"),))
    registry = ToolRegistry(tasks)
    registry.register(media_tool_descriptor(), verify_descriptor=lambda _: True)
    return tasks, registry


def test_tool_descriptor_is_closed_and_sorted():
    descriptor = media_tool_descriptor()
    descriptor.validate()
    assert descriptor.side_effect == "external"
    assert descriptor.permissions == ("media.transfer",)
    assert descriptor.input_schema["additionalProperties"] is False


def test_registry_task_state_canonical_effect_and_host_receipt(tmp_path):
    tasks, registry = harness(tmp_path)
    call = prepare_media_transfer(registry, admit(), task_id="task", effect_id="effect",
                                  expected_epoch=0, expected_revision=1,
                                  grant_evidence_id="grant", authorize_tool=lambda *_: True,
                                  verify_admission=lambda _: True)
    assert "artifact_identity" in call.request_json
    assert tasks.issue_effect("task", "effect", expected_epoch=0,
                              expected_revision=1).pending_effects[0].status == "unknown"
    result = make_result(call, receipt_id="host-receipt", evidence_id="verified-result",
                         outcome="success", output={
                             "artifact_identity": artifact().identity,
                             "accepted": True,
                         })
    final = registry.accept_result(call, result, verify_call=lambda _: True,
                                   verify_result=lambda *_: True)
    assert final.pending_effects[0].status == "resolved"
    assert TaskStore(tasks.path).load("task") == final
    with pytest.raises(ToolBoundaryError):
        registry.accept_result(call, result, verify_call=lambda _: True,
                               verify_result=lambda *_: True)


def test_forged_admission_and_missing_grant_rejected(tmp_path):
    tasks, registry = harness(tmp_path)
    good = admit()
    bad = replace(good, artifact_identity="0" * 64)
    for admission in (bad, MediaAdmission("untyped", good.artifact_identity,
                                         "local", "fixture", "host")):
        with pytest.raises(MediaBoundaryError):
            prepare_media_transfer(registry, admission, task_id="task", effect_id="effect",
                                   expected_epoch=0, expected_revision=1,
                                   grant_evidence_id="grant", authorize_tool=lambda *_: True,
                                   verify_admission=lambda _: True)
    with pytest.raises(ToolBoundaryError):
        prepare_media_transfer(registry, good, task_id="task", effect_id="effect",
                               expected_epoch=0, expected_revision=1,
                               grant_evidence_id="grant", authorize_tool=lambda *_: False,
                               verify_admission=lambda _: True)
    with pytest.raises(MediaBoundaryError):
        prepare_media_transfer(registry, good, task_id="task", effect_id="effect",
                               expected_epoch=0, expected_revision=1,
                               grant_evidence_id="grant", authorize_tool=lambda *_: True,
                               verify_admission=lambda _: False)
    assert tasks.load("task").pending_effects[0].status == "pending"


def test_restart_unknown_effect_never_retries_blindly(tmp_path):
    tasks, registry = harness(tmp_path)
    good = admit()
    call = prepare_media_transfer(registry, good, task_id="task", effect_id="effect",
                                  expected_epoch=0, expected_revision=1,
                                  grant_evidence_id="grant", authorize_tool=lambda *_: True,
                                  verify_admission=lambda _: True)
    tasks.issue_effect("task", "effect", expected_epoch=0, expected_revision=1)
    restarted = TaskStore(tasks.path)
    resumed = restarted.resume("task")
    assert resumed.pending_effects[0].status == "unknown"
    with pytest.raises(ToolBoundaryError):
        prepare_media_transfer(registry, good, task_id="task", effect_id="effect",
                               expected_epoch=1, expected_revision=resumed.revision,
                               grant_evidence_id="grant", authorize_tool=lambda *_: True,
                               verify_admission=lambda _: True)
    result = make_result(call, receipt_id="postrestart", evidence_id="proof",
                         outcome="success", output={
                             "artifact_identity": good.artifact_identity, "accepted": True,
                         })
    with pytest.raises(ToolBoundaryError):
        registry.accept_result(call, result, verify_call=lambda _: True,
                               verify_result=lambda *_: True)
    assert restarted.load("task").pending_effects[0].status == "unknown"
