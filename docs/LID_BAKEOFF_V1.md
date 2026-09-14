# D03 Language-ID Bakeoff V1

Status: `PREPARED_NOT_EXECUTED`.

This package implements the independent-verification seam requested by issue #720 for comparing
`FASTTEXT_LID176`, `OPENLID_V3`, `GLOTLID`, and `LINGUA` on one frozen, project-authored set of
Ukrainian, English, code, mixed-language, and noise cases. It does not import, install, download, or
run any of those systems. The checked-in state therefore contains no candidate quality result.

## Authority and scientific boundary

`configs/research/lid_bakeoff_v1.json` binds the exact Git blob identity of
`configs/research/open_source_reuse_registry_v2.json` and copies the registry's candidate identity,
upstream, license-context, and decision fields. The validator recomputes the Git blob SHA-1 from the
registry bytes and fails if either the registry or the copied candidate metadata drifts.

The same contract explicitly excludes `NLLB_LID218E` from unrestricted adoption because the live
registry records CC-BY-NC-4.0 model-family context and the decision
`DO_NOT_USE_AS_HIDDEN_UNRESTRICTED_DEPENDENCY`.

The fixture in `tests/fixtures/lid_bakeoff_v1.jsonl` is original project text created only for this
LID calibration package. Every record is marked `training_allowed=false`,
`tokenizer_fit_allowed=false`, `final_test=false`, and `benchmark_material=false`. It is not corpus
capacity, tokenizer material, a final test, or evidence of model capability.

## Evidence format

A real bakeoff evidence JSON must use `evidence_kind=EXTERNAL_CANDIDATE_EXECUTION`, bind the exact
contract fixture SHA-256 and registry Git blob SHA-1, and contain exactly one execution for each of
the four candidates. Each execution must record:

- an immutable upstream ref;
- an artifact identity for the actually executed model/package;
- the project adapter identity that maps raw output to `uk`, `en`, `code`, `mixed`, `noise`, or
  `unknown`;
- a command/environment identity sufficient to reproduce the invocation;
- a durable license-review reference with status `REVIEWED_FOR_BAKEOFF`;
- exactly one prediction for every frozen case, retaining both normalized and raw labels;
- confidence only when the upstream exposes it, bounded to finite `[0, 1]` when present;
- `automatic_adoption_requested=false`.

Missing candidates, missing/duplicate cases, unknown labels, non-finite confidence, authority drift,
or incomplete runtime/license identity fail closed.

## Running the verifier

Preflight the checked-in contract before executing any candidate:

```text
python -m twelve_six.lid_bakeoff \
  --contract configs/research/lid_bakeoff_v1.json \
  --registry configs/research/open_source_reuse_registry_v2.json \
  --fixture tests/fixtures/lid_bakeoff_v1.jsonl
```

After a separate, reviewed runner produces a complete evidence file:

```text
python -m twelve_six.lid_bakeoff \
  --contract configs/research/lid_bakeoff_v1.json \
  --registry configs/research/open_source_reuse_registry_v2.json \
  --fixture tests/fixtures/lid_bakeoff_v1.jsonl \
  --evidence path/to/lid-evidence.json \
  --output path/to/lid-report.json
```

`--allow-test-evidence` exists only so unit tests can exercise scoring with synthetic records.
Production comparison rejects that evidence kind.

## Interpretation

A complete report contains total accuracy, per-category accuracy, and a confusion table for each
candidate. `COMPARABLE_EVIDENCE_READY` means only that the evidence is structurally complete and
bound to the expected authorities. The report always leaves `selected_candidate=null` and
`automatic_adoption_allowed=false`. D03 must separately review the measured error profile,
short-text/mixed/noise behavior, exact runtime cost, license/notice obligations, and downstream
integration before choosing any production LID path.

No arbitrary accuracy threshold is encoded as a scientific law. No upstream benchmark claim is
accepted as 12-6 evidence. No corpus mutation, tokenizer fit, Base training, final-test access, GPU
provisioning, or paid compute is authorized or performed by this package.
