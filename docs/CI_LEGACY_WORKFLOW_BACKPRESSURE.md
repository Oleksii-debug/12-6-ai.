# CI legacy-workflow backpressure

This control-plane package is stacked on PR #646 and addresses issue #652 without deleting scientific checks or narrowing dependency scopes by guesswork.

## Contract

Every active GitHub Actions workflow that triggers on `pull_request` must have a PR-scoped `concurrency` group with `cancel-in-progress: true`. When a new commit supersedes an older head of the same pull request, the older workflow run may be cancelled. The newest exact-head run remains required and is not treated as PASS until it finishes successfully.

This package does not use cancellation as a substitute for path scoping. Existing path scopes are preserved. Broad specialist workflows remain explicitly marked `BROAD_PENDING_DEPENDENCY_AUDIT` until their full dependency closure is demonstrated.

## Machine authority

`configs/ci/legacy_workflow_backpressure_v1.json` inventories every active workflow and records whether its pull-request trigger is path-scoped.

`python tools/validate_ci_backpressure.py` fails closed when:

- an active workflow is not represented in the inventory;
- the inventory references a removed workflow;
- a pull-request trigger or path-scope state drifts without review;
- a pull-request workflow loses its concurrency block;
- `cancel-in-progress` is not true;
- the concurrency group is not keyed by pull-request number with a ref fallback.

The repository test suite also validates the live inventory through `tests/test_ci_backpressure.py`.

## Scientific boundary

No model, optimizer, tokenizer, corpus, evaluation, checkpoint format, training recipe, stage gate, or paid-compute authority is changed. Cancellation removes only superseded duplicate work for the same pull request; it does not waive the latest exact-head evidence requirement.

## Next remediation

After this package is terminal and the convergence parent is accepted, audit the broad specialist workflows one by one. Only add `paths` when the complete dependency surface is known. Prefer moving cheap static validators into shared CI where doing so preserves their exact scientific authority.
