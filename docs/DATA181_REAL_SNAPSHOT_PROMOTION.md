# DATA-181 — Real snapshot promotion

DATA-181 closes the narrow promotion gap between the already accepted DATA-21/22
bounded real-source intake and the DATA-24 D03 rights/eligibility contract. It does
not perform a new rights review and does not infer authorization from a license label.

## Accepted identities retained

The promotion is pinned to DATA-21/22 PR #204, exact head
`dcc7dfc39299487bca5bdbfe5e6c70eaa6706278`, successful workflow run
`32900602711`, intake manifest
`9d50c0baf98247c1babc5fca8dead5b1fa87264ad92ea62527c34e342a7dd735`,
and candidate-registry identity
`678d250ac9910f58ab1b9113cf713a2fea52a6a21e7a8434e6434d95a8045214`.

Three accepted objects are promoted independently: one bounded Ukrainian Rada
HTML object and two files from the pinned Standard Ebooks manual commit. Each has
an exact expected raw SHA-256/size and the DATA-21/22 extraction/normalization
SHA-256/UTF-8 byte count.

## Rights semantics

The canonical source registry remains DATA-24 schema
`12-6.external-source-registry.v2`. Acquisition, storage, analysis, model training,
and redistribution are recorded independently. Every promoted object carries:
1. immutable source-rights evidence bound to the exact promoted source/version; and
2. the exact DATA-21/22 reviewed candidate registry as policy-decision evidence.

The source-rights evidence substantiates the already accepted review. The
DATA-24 `EligibilityResolver` remains the only training-eligibility resolver.
A public URL or license label cannot grant eligibility by itself.

## Snapshot semantics

Large/raw source bytes are intentionally not committed to Git, per project
bootstrap rules. The committed registry declares content-addressed `file:` snapshot
URIs plus exact SHA-256 and size. The DATA-181 workflow acquires every object twice,
fails closed unless both acquisitions equal the accepted raw identity, materializes
the raw payload at its declared snapshot URI, and runs `verify_local_snapshot`.

The exact-head workflow uploads those raw payloads as the
`data181-canonical-source-snapshots-<SHA>` artifact together with the canonical
registry, rights evidence, normalized outputs, promotion report, and deterministic
small-corpus outputs.

## Normal corpus gates

After full-object extraction reproduces the DATA-21/22 normalized identity, a
generic deterministic chunker creates bounded natural-text records. There is no
source-specific admission bypass. Every chunk is offered to:

`DATA-24 EligibilityResolver -> admit_for_pretraining -> incumbent D03 build_dataset`

Rejected chunks are recorded. At least one chunk from every promoted object must
pass. The small corpus is built twice from the same admitted inputs and both dataset
identity and output hashes must match.

This package is a canonical source-snapshot promotion for these three exact objects.
It is not a representative corpus, not a production corpus freeze, and not a
universal benchmark-clean claim. It makes no claim about intelligence, production
readiness, alignment, or instruction following. All execution is `LOCAL_FREE`.
