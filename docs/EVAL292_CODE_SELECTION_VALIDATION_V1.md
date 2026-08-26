# EVAL-292 Code Selection-Validation V1

Worker: `EVAL-292-CODE-SELECTION-VALIDATION-V1`

## Verdict

`BLOCKED`.

At the immutable cutoff `2026-08-26T12:02:04Z`, no Wave-1 external-real
code object satisfies both mandatory selection conditions:

1. explicit authorization for evaluation use; and
2. reservation of the exact object from all training.

The selection-validation set therefore contains zero documents, zero source
families and zero selected bytes. This is intentional fail-closed behavior.

## Terminal evidence consumed

DATA-227 exact head
`8ebdb2e132ed7bae5245e9d4c140752640ab9885` has a successful dedicated
workflow run `32956209865`. Its rights-policy Git blob is
`0ce5223a1cade10031899bf27348a1a65121d4c6`. It admits two independent
repositories for D03/model-training use:

- `github:encode/httpx`, exact object
  `b5addb64f0161ff6bfe94c124ef76f6a1fba5254:httpx/_content.py`,
  Git blob `6f479a0885f723b7395843d41164a87041820776`;
- `github:psf/requests`, exact object
  `5460f467b02e49471c0fd6cfc9ca0adab6351f98:src/requests/_internal_utils.py`,
  Git blob `0466a7d347db4ed34a37db51b75fc8e80bc06055`.

DATA-227 does not separately authorize evaluation use and does not reserve
either object from training.

EVAL-233 exact head
`b5512b4648cb09dd052b08884dc53f291e1ce935` has successful dedicated run
`32957254139`. Its authority Git blob
`2008570890819f32c356677e1e250707d339b53a` records:

- `evaluation_use_explicitly_authorized=false`;
- `reserved_from_training=false`;
- `BLOCKED_DATA227_TRAINING_ONLY_NO_EVALUATION_RESERVATION`.

This is the governing EVAL rule: model-training permission must not be
reinterpreted as evaluation permission.

DATA-295 exact head
`6ab35f8f0f68f1943ff612f4ab529d2d970db1d6` has successful dedicated run
`32966394993`. Its preregistered future-corpus policy input inventory includes
both DATA-227 code families and 9,703 code bytes. DATA-295 is not relabelled as
a frozen corpus identity here; it is consumed only as terminal evidence that
the current future-training plan includes the same two candidates.

## Separation guarantees

No final-test record or byte is copied into EVAL-292. No final-test outcome is
read for selection construction. Because the selected set is empty, overlap
with the current future-training code inventory is exactly zero.

The immutable authority binds exact repository identities, commits, paths,
source Git blob hashes, license Git blob hashes, upstream rights-policy blob,
terminal upstream heads and dedicated workflow runs.

## Deterministic rebuild

`python -m twelve_six.eval292_code_selection_validation build` reconstructs the
manifest from fixed source-cut facts after verifying the exact terminal
EVAL-233 authority blob. `verify` rejects any change that weakens the blocked
state, fabricates authorization/reservation, admits source bytes, exposes
final-test data, or introduces training overlap.

The dedicated workflow rebuilds the manifest and byte-compares it to
`evidence/eval292/code-selection-validation-v1.json`.

## Unblock conditions

A successor may publish a non-empty set only after terminal evidence exists
for exact code objects that are explicitly authorized for evaluation,
reserved from every training inventory before selection construction, drawn
from multiple independent repositories when available, and proven
source/content-disjoint from the frozen training inventory.

LOCAL_FREE only.
