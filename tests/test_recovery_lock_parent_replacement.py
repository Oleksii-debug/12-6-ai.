from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from twelve_six.checkpoint.recovery_lock import exclusive_recovery_lock


def _parent_replacement_holder(
    root: str,
    entered: multiprocessing.synchronize.Event,
    mutate: multiprocessing.synchronize.Event,
    wrote: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    try:
        with exclusive_recovery_lock(root) as mutation_root:
            entered.set()
            if not mutate.wait(20):
                raise RuntimeError("timed out waiting for parent replacement")
            (mutation_root / "pinned-write.txt").write_text(
                "old-parent\n", encoding="utf-8"
            )
            wrote.set()
            if not release.wait(20):
                raise RuntimeError("timed out waiting to release holder")
    except OSError as exc:
        result_queue.put(("drift", str(exc)))
    else:
        result_queue.put(("unexpected-success", ""))


def _parent_replacement_contender(
    root: str,
    acquired: multiprocessing.synchronize.Event,
    result_queue: multiprocessing.queues.Queue,
) -> None:
    with exclusive_recovery_lock(root):
        acquired.set()
    result_queue.put(("ok", ""))


def test_parent_replacement_cannot_create_second_logical_lock_lane(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("parent replacement regression requires POSIX directory-fd aliases")

    ctx = multiprocessing.get_context("spawn")
    base = tmp_path / "base"
    root = base / "workspace" / "recovery"
    moved_parent = base / "moved-workspace"
    base.mkdir()

    holder_entered = ctx.Event()
    mutate = ctx.Event()
    holder_wrote = ctx.Event()
    holder_release = ctx.Event()
    contender_acquired = ctx.Event()
    results = ctx.Queue()

    holder = ctx.Process(
        target=_parent_replacement_holder,
        args=(
            str(root),
            holder_entered,
            mutate,
            holder_wrote,
            holder_release,
            results,
        ),
    )
    contender = ctx.Process(
        target=_parent_replacement_contender,
        args=(str(root), contender_acquired, results),
    )

    holder.start()
    assert holder_entered.wait(20), "holder did not acquire recovery lock"

    os.replace(root.parent, moved_parent)
    root.mkdir(parents=True)
    mutate.set()
    assert holder_wrote.wait(20), "holder did not write through pinned root"

    assert not (root / "pinned-write.txt").exists()
    assert (
        moved_parent / "recovery" / "pinned-write.txt"
    ).read_text(encoding="utf-8") == "old-parent\n"

    contender.start()
    assert not contender_acquired.wait(
        0.75
    ), "contender entered through recreated parent before holder release"

    holder_release.set()
    assert contender_acquired.wait(20), "contender did not proceed after holder release"

    holder.join(30)
    contender.join(30)
    assert holder.exitcode == 0
    assert contender.exitcode == 0

    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    assert ("ok", "") in outcomes
    drift = [message for status, message in outcomes if status == "drift"]
    assert len(drift) == 1, outcomes
    assert "recovery parent changed during publication critical section" in drift[0]
