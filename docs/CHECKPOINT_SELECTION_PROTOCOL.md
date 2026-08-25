# EVAL-139 checkpoint-selection protocol

Status: canonical experiment-level v1 policy for future learned-Base campaigns, including the planned ~10M scale. This policy is LOCAL_FREE and does not authorize paid compute.

## Purpose boundary

Every metric is bound to one immutable `EvaluationPurpose` before use. The four purposes are intentionally non-interchangeable:

- `training_metrics`: optimizer/training telemetry. Examples include training loss, gradient norm, throughput, update ratio, and other quantities produced while fitting. These metrics are never selector inputs.
- `selection_validation`: the only selector-eligible purpose. The canonical v1 metric is held-out BPB.
- `final_test`: a physically distinct held-out suite used only after the selected checkpoint is frozen. In v1 it may be evaluated only on that frozen checkpoint, so a test sweep across candidate checkpoints is a protocol error.
- `diagnostic_only`: reserved mechanistic or capability suites. They can explain behavior but cannot change checkpoint selection.

Each purpose binds an exact suite identity SHA-256 and a preregistered metric-name set. The purpose identity is the SHA-256 of the purpose, suite ID, suite hash, metric names, and eligibility bit. Changing a purpose label, suite, or allowed metric changes the identity.

Selection-validation, final-test, training, and diagnostic suites must have distinct suite hashes. Relabeling the same physical held-out split from `selection_validation` to `final_test` is rejected.

For current first-party held-out infrastructure, use `BenchmarkRegistry.manifest()["manifest_sha256"]` as the suite identity where the registry manifest fully identifies the exact reserved split/suite. For checkpoint-v1 artifacts, retain the incumbent `checkpoint_id`; optional artifact SHA-256 binding can be recorded alongside it.

## Canonical preregistered selector

The frozen v1 rule is:

- metric: BPB;
- direction: minimize;
- smoother: causal trailing median;
- smoothing window: 3 checkpoints;
- minimum improvement: 0.01 BPB.

The rule must be recorded, including `rule_identity_sha256`, before the first optimizer step of a campaign. Evaluation suite identities must also be frozen before selection results are inspected.

Checkpoints are processed in chronological experiment order. The first checkpoint completing a three-point window becomes the incumbent. At each later checkpoint, the median of that checkpoint and the two immediately preceding selection-validation BPB measurements is computed. The later checkpoint replaces the incumbent only when the smoothed BPB improves by at least 0.01. A trailing window is used deliberately: centered smoothing would use future validation measurements to score an earlier checkpoint.

The selector accepts only `SelectionValidationObservation`. Passing a training, final-test, or diagnostic observation type is a hard error. A selection observation must match the exact selection-purpose identity, preregistered metric name, and a registered checkpoint ID.

This is a checkpoint-selection policy, not an early-stopping policy. Training may continue to its preregistered budget. If an earlier checkpoint wins, every later registered checkpoint remains retained. The machine report explicitly emits the complete checkpoint registry, all retained IDs, and `delete_unselected_checkpoints: false`.

## Post-selection evidence

After the decision is frozen, final-test, training, and diagnostic observations can be attached to the experiment report. They are not part of `selection_decision_sha256`. Therefore changing any final-test value can change the full report hash but cannot change either the selected checkpoint or the selection-decision hash.

For transparency the report also identifies:

1. the checkpoint selected by the preregistered rule;
2. the absolute post-hoc raw validation minimum;
3. the final chronological checkpoint.

The latter two are comparison fields only. They never feed the selector.

## Historical trajectory check

EVAL-139 inspected the retained evidence available at implementation time rather than assuming every named workflow had executed.

The successful LEARN-03 ~500K fixed-control artifact (467,808 parameters, workflow run `32861195699`, artifact `9569471346`) contains real checkpoint-v1 manifests for two seeds through 262,332 optimized tokens. Under the frozen v1 rule:

| trajectory | selected | post-hoc raw best validation | final |
| --- | ---: | ---: | ---: |
| seed 1337 | 131,292 tokens, BPB 3.074960 | 65,772 tokens, BPB 3.057125 | 262,332 tokens, BPB 3.798958 |
| seed 1338 | 131,292 tokens, BPB 3.057962 | 131,292 tokens, BPB 3.057962 | 262,332 tokens, BPB 4.021088 |

Seed 1337 is the useful adversarial case: the protocol does not chase the single lowest raw validation point, and it also does not default to the final checkpoint after clear degradation. Seed 1338 independently selects the same token region while its raw minimum happens to agree with the selector.

The dedicated long ~100K exact-head workflow (`TRAIN-41 Long 100K Base`, run `32862102098`) did not reach training; it failed at locked-environment verification. The learned ~1M long matrix exact-head workflow inspected (`RESEARCH41 Learned-Base Scaling Experiment`, run `32861604025`) failed its focused control-test step before matrix execution. No long-run result is fabricated for either case. An older successfully executed controlled matrix does contain ~95K and ~1.04M prefixes through 65,772 optimized tokens; both are monotonic over their three recorded points and therefore select their final available prefix point. The evidence file labels these explicitly as `EXECUTED_PREFIX_NOT_LONG_TRAJECTORY` rather than presenting them as the missing long campaigns.

The historical runs predate EVAL-139 and did not reserve a physically distinct final-test suite. Historical analysis therefore tests selection behavior only; it does not rebrand their validation split as final test.

Machine-readable historical evidence is in `evidence/eval139_historical_selection_evidence.json`.

## Future ~10M campaign use

Before training starts, the experiment manifest should persist:

1. exact training-data/run identities;
2. exact selection-validation suite identity;
3. exact final-test suite identity, physically distinct from selection validation;
4. each diagnostic-suite identity;
5. checkpoint/evaluation schedule;
6. the frozen EVAL-139 rule identity.

At each scheduled checkpoint, save the checkpoint first, retain its checkpoint-v1 identity, then record selection-validation BPB bound to that identity. Complete the preregistered run unless another independently preregistered safety/instability rule stops it. Run the selector once the candidate registry is closed. Only then evaluate the final-test suite on the frozen winner.

The canonical CLI is `tools/eval139_checkpoint_selection.py`. It consumes a frozen JSON manifest containing purposes, checkpoint registry, selection observations, optional non-selection observations, and provenance; it emits a self-hashed machine report. Omitting `selection_rule` uses the frozen v1 defaults.

## Limitations of v1

The three-point trailing median deliberately trades responsiveness for robustness and can lag a fast validation improvement or degradation by one checkpoint. The 0.01 threshold is an absolute BPB threshold, not a statistical confidence interval. These choices are now fixed for v1 so they cannot be tuned to the observed trajectory. A future v2 may replace them only by explicit preregistration before looking at the campaign it will govern.
