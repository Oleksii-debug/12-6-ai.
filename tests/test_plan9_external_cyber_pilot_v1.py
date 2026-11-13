from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import plan9_external_cyber_pilot_v1 as pilot
from twelve_six.training.resource_envelope import DISK_FLOOR_DENIAL, HostResources


def test_pilot_cursor_recovers_exact_position_without_opening_training():
    class Stream:
        @staticmethod
        def cursor(position):
            return {
                "next_microbatch_index": position, "optimizer_step": position // 12,
                "consumed_loss_positions": 1303 if position == 24 else 1340,
                "cursor_sha256": "cursor",
            }

        @staticmethod
        def verify_cursor(cursor, digest):
            if cursor != Stream.cursor(cursor["next_microbatch_index"]) or digest != "cursor":
                raise ValueError("bad cursor")

        @staticmethod
        def iter_batches(cursor, digest):
            Stream.verify_cursor(cursor, digest)
            assert cursor["next_microbatch_index"] == 24
            yield {"input_ids": [[2, 3]], "target_ids": [[3, 4]], "loss_mask": [[1, 1]]}

    hooks = pilot.PilotCursor(Stream())
    with pytest.raises(ValueError, match="bad cursor"):
        hooks.restore({**Stream.cursor(24), "consumed_loss_positions": 10})
    hooks.restore(Stream.cursor(24))
    with pytest.raises(ValueError, match="receipt and physical cursor disagree"):
        next(hooks.batches(SimpleNamespace(committed_step=1, tokens_seen=1303)))
    batch = next(hooks.batches(SimpleNamespace(committed_step=2, tokens_seen=1303)))
    assert batch["target_ids"].tolist() == [[3, 4]]
    assert hooks.read(SimpleNamespace(micro_step=25, tokens_seen=1340)) == Stream.cursor(25)


def test_pilot_rejects_unsupported_host_before_training(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot.sys, "platform", "win32")
    monkeypatch.setattr(pilot, "Trainer", lambda *_a, **_kw: pytest.fail("trainer was constructed"))
    args = SimpleNamespace(scratch=tmp_path, action="prepare")
    with pytest.raises(ValueError, match="Linux on an external CUDA host"):
        pilot.execute(Path.cwd(), {"run_manifest_sha256": "a" * 64}, None, args)


def test_pilot_storage_denial_is_deferred_only_for_pending_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "observe_host_resources", lambda *_: HostResources("cuda:0", 1, 1, 0))
    monkeypatch.setattr(pilot, "assess_resource_envelope", lambda *_a, **_kw: {
        "denials": [DISK_FLOOR_DENIAL], "unpruned_checkpoint_volume_floor_bytes": 10,
    })
    cfg = SimpleNamespace(model=object())
    with pytest.raises(RuntimeError, match=DISK_FLOOR_DENIAL):
        pilot._admit(cfg, tmp_path, remaining=2, prepare=False)
    pilot._admit(cfg, tmp_path, remaining=2, prepare=False, defer_disk=True)
