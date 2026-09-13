"""Repository-wide test dependencies for durable bounded-pilot attempts."""

from __future__ import annotations

import tempfile
import uuid

from twelve_six.training.bounded_pilot import _set_test_recovery_store_factory
from twelve_six.training.resilience import RecoveryPolicy, RecoveryStore


def bounded_pilot_test_recovery_store_factory(gate):
    run_id = f"pytest-{uuid.uuid4().hex}"
    manifest = {
        "schema_version": "12-6.pytest-bounded-pilot-run-manifest.v1",
        "run_id": run_id,
        "candidate": {"git_sha": "1" * 40},
        "recovery": {
            "topology": {
                "backend": "single-process-cpu",
                "world_size": 1,
                "rank_count": 1,
                "resume_policy": "exact_topology",
            }
        },
        "bounded_pilot": gate._bounded_start_projection(),
    }
    store = RecoveryStore(
        tempfile.mkdtemp(prefix="12-6-bounded-pilot-pytest-"),
        run_manifest=manifest,
        policy=RecoveryPolicy(
            checkpoint_every_steps=1,
            retain_last=2,
            max_restarts=0,
            max_preemptions=0,
        ),
    )
    return store, store.run_manifest_sha256, store.run_id


_set_test_recovery_store_factory(bounded_pilot_test_recovery_store_factory)
