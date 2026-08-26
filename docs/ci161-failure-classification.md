# CI-161 experimental failure classification

CI-161 separates workflow failure from model/training failure. A red GitHub Actions run is not evidence that training diverged unless the retained report identifies an actual training-phase signal.

## Contract

Every migrated experiment workflow emits `12-6.experiment-failure-report.v1` for classified phases. The report contains a stable `failure_class`, `phase`, `experiment_started`, `optimizer_steps_completed`, source identity, bounded diagnostic codes, command/output hashes and byte counts. It does not retain raw stdout/stderr, environment variables, training text or secrets.

`experiment_started` means that this process committed at least one optimizer step. It is deliberately independent of `failure_class`. For example, a non-finite loss on the first forward pass is `NONFINITE_TRAINING` with `experiment_started=false`; a missing pytest is `BOOTSTRAP_DEPENDENCY_MISSING` with `experiment_started=false`.

## Taxonomy

Required classes are `BOOTSTRAP_DEPENDENCY_MISSING`, `LOCK_PROFILE_STALE`, `STATIC_CHECK_FAILED`, `FOCUSED_TEST_FAILED`, `EXPERIMENT_NOT_STARTED`, `RESOURCE_OOM`, `TIMEOUT`, `NONFINITE_TRAINING`, `CHECKPOINT_FAILURE`, `EVALUATION_FAILURE`, `SCIENTIFIC_REJECTION` and `SUCCESS`.

Additional CI-161 classes are `CONFIGURATION_ERROR`, `DATA_INPUT_FAILURE`, `CANCELLED` and `UNCLASSIFIED_FAILURE`. They exist so configuration/data/operator/unknown failures are not coerced into model divergence. `UNCLASSIFIED_FAILURE` is intentionally conservative and is preferable to an unsupported scientific claim.

Classification precedence is specific infrastructure evidence first, then cancellation/resource failures, then numerical/checkpoint/evaluation/scientific signals, then static/test phases, then generic pre-optimizer failure. Missing dependency evidence therefore outranks `FOCUSED_TEST_FAILED`.

## Mandatory bootstrap integration

`tools/run_classified_phase.py` is stdlib-safe and loads the classifier directly from its file so it can operate before the ML package/runtime is importable. Run it with the same Python interpreter whose environment is being tested.

Example after creating the exact locked environment:

```bash
.ci/bin/python tools/run_classified_phase.py \
  --phase focused_test \
  --report evidence/focused-test.json \
  --source-sha "$EXPECTED_SOURCE_SHA" \
  --workflow "$GITHUB_WORKFLOW" \
  --run-id "$GITHUB_RUN_ID" \
  --required-module pytest \
  -- .ci/bin/python -m pytest -q tests/test_target.py
```

If `pytest` is absent, the command is not executed and the report is `BOOTSTRAP_DEPENDENCY_MISSING`, `experiment_started=false`. Lock validation commands should use `--phase lock_validation`; a non-zero lock-validation result is `LOCK_PROFILE_STALE`. Static checks use `static_check`; selected tests use `focused_test`.

The canonical purpose-environment verifier remains authoritative for exact lock/profile validation. CI-161 wraps its invocation; it does not replace `tools/verify_purpose_environment.py` or the purpose-profile lock model.

## Trainer integration without replacement

`RunFailureTracker` composes with the existing Trainer metrics contract. No Trainer implementation change is required.

```python
from twelve_six.experiment_failure import RunFailureTracker

tracker = RunFailureTracker(start_optimizer_step=trainer.optimizer_step)

def on_metrics(metrics):
    tracker.observe_metrics(metrics)
    tracker.write_start_marker(output_dir / "experiment-start.json")
    existing_metrics_callback(metrics)

trainer.run(batches, on_metrics=on_metrics, ...)
```

For experiment subprocesses, pass the marker to the classified wrapper:

```bash
.ci/bin/python tools/run_classified_phase.py \
  --phase training \
  --report evidence/training.json \
  --start-marker evidence/experiment-start.json \
  --require-experiment-start \
  -- .ci/bin/python -m twelve_six.some_experiment ...
```

The marker counts optimizer steps completed by the current process. For resume runs initialize the tracker with the loaded optimizer step; a checkpoint loaded at step 500 is not itself evidence that the resumed process started training.

## Phase mapping for future experiment workflows

Use `bootstrap` for dependency/tool installation and import probes; `lock_validation` for purpose-profile/lock identity checks; `static_check` for compile/lint/type/static contracts; `focused_test` for selected pytest contracts; `prepare` for corpus/config/setup work before the first optimizer step; `training` for optimizer-bearing execution; `checkpoint` for durable checkpoint save/load/verification; `evaluation` for held-out or generation verification; `scientific_gate` only for explicit scientific acceptance/rejection criteria; and `finalize` for report assembly.

Do not collapse multiple phases into one shell step if the distinction matters. In particular, install/import checks and pytest execution should be separate from training. This is the migration that prevents a missing test dependency from appearing as a failed experiment.

## Historical migration boundary

`reports/ci161/historical_failure_sample.json` retro-classifies a bounded sample using only retained Actions run/job/step conclusions, committed workflow structure and exact dependency locks. Historical raw logs are not copied into the repository. Where retained evidence only proves that execution never reached training, CI-161 reports `EXPERIMENT_NOT_STARTED` rather than guessing a model failure.

The MILESTONE-100 and SCALE-141 samples are classified as `BOOTSTRAP_DEPENDENCY_MISSING`: their historical workflows installed runtime/toolchain locks without the dev lock, invoked `python -m pytest`, failed at the contract-test step, and skipped all training steps. Future workflows must install the declared dev/test layer or use a purpose profile that explicitly includes it before invoking pytest.
