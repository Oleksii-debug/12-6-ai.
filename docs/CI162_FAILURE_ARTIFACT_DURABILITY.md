# CI-162 failure artifact durability

Authority: `LOCAL_FREE_DIAGNOSTIC_EVIDENCE_NOT_CHECKPOINT_PROMOTION`.

This contract preserves useful experiment diagnostics across workflow failure without turning an incomplete training run or partial checkpoint into valid learned-model evidence.

## Live audit motivating the contract

The learned-model workflows historically placed `actions/upload-artifact` after the success path. That is safe against accidental publication of incomplete state, but it also discards already-valid diagnostics and committed checkpoints when a later step fails.

A concrete exact-head MILESTONE-150 run (`32902234495`, source `c6e8ac2784920cb96b9b34d26cd68cf9468bd5f0`) completed focused tests, prepared the common DATA-25/tokenizer/evaluation truth and trained the 100K rung through the verified step-500 checkpoint. A fresh-process resume then failed on a Python tuple/list representation mismatch in the persisted run manifest. The artifact upload step was skipped, so bootstrap state, the train curve and D05-verified checkpoints that already existed were not retained by Actions.

SCALE-141 exact-head run `32902872519` failed in its focused contract tests before training and likewise skipped its artifact upload. MILESTONE-100 and SCALE-02 also used success-only artifact upload in their inherited workflows. The failure modes differ, but the durability gap is the same.

## Common finalization contract

`tools/evidence_finalizer.py` is stdlib-only at process startup. This matters when environment installation itself is the failure. It provides four operations:

- `bootstrap`: write a self-hashed, allowlisted environment/bootstrap manifest before the ML runtime is installed;
- `mark-phase`: persist the exact phase most recently entered;
- `test-status`: retain a self-hashed focused-test result even when pytest exits non-zero;
- `finalize`: build the only directory that an `if: always()` artifact-upload step is allowed to publish.

The finalizer never declares checkpoint validity itself. It recognizes only existing checkpoint candidates and delegates validity to the incumbent verifier:

- D05 checkpoint-v1: `twelve_six.checkpoint.verify_checkpoint`;
- DCP scale checkpoint: `twelve_six.distributed.dcp_checkpoint.verify_scale_checkpoint`, which includes the existing `COMMITTED` control-plane requirement.

A checkpoint is retained only when the existing verifier accepts the source directory, the directory is copied, the same verifier accepts the staged copy, the checkpoint identity is unchanged, and the complete source/staged SHA-256 plus byte-size inventories are identical.

Unknown or incomplete checkpoint directories remain in the working directory and are recorded as invalid/excluded. They are never copied into the upload bundle.

A `train-curve.jsonl` file is retained only when the same run directory contains at least one D05/DCP-verified retained checkpoint. This prevents a partial curve from being presented without a committed state anchor.

## Failure semantics

The finalization report records the exact last-entered phase, the prior job status, every valid or rejected checkpoint candidate, retained/excluded training evidence, metadata rejections, retention period, payload hashes/sizes and a self-hash.

On a failed job its interpretation is exactly `FAILURE_DIAGNOSTICS_ONLY_NO_COMPLETION_CLAIM`. A successful finalizer does not convert the experiment job to PASS and does not grant model, stage, audit or promotion authority.

Workflow upload uses `if: always()` only together with `steps.evidence_finalizer.outcome == 'success'`. Therefore an experiment failure can retain a valid diagnostic bundle, while a failure inside finalization itself cannot publish a partially staged directory as authoritative evidence.

## Privacy boundary

The finalizer is allowlist-based for ordinary metadata. Raw/private corpus directory names are denied, symlinks are denied, and retained text metadata is screened for common secret/token/private-key signatures. The bootstrap manifest records only a fixed environment-variable allowlist and never dumps the runner environment.

Corpus bytes are not upload candidates. A corpus manifest may be retained because it contains identity/provenance metadata rather than raw training text.

## Retention

Migrated workflows explicitly set 30-day Actions artifact retention. The same value is stored in the finalization report.

## Failure injection coverage

`tests/test_evidence_finalizer_ci162.py` covers:

1. failure before training: bootstrap and focused-test evidence survives;
2. failure during training: an unanchored training curve is excluded;
3. failure during checkpoint save: a valid D05 checkpoint survives while an incomplete sibling remains invalid;
4. failure during report generation: a previously committed checkpoint survives without manufacturing a final report;
5. a DCP-shaped candidate is rejected unless the existing DCP verifier accepts its committed control plane and payload inventory;
6. raw-corpus paths and secret-like metadata are rejected.

## Initial migrations

The reusable contract is wired into:

- `MILESTONE-100 First Learned ~1M Base`;
- `SCALE-02 S2 1M Executable Preflight`.

These migrations are intentionally narrow and do not change model geometry, optimizer semantics, dataset/tokenizer identities, checkpoint formats, evaluation metrics or learned-model claims. MILESTONE-150 is under active convergence ownership; this CI-162 package is stacked on it rather than rewriting that workflow concurrently.
