# Plan 8 Section 10 exact integration

`freeze_candidate` checks the clean candidate checkout and records its base,
head, complete tree and changed paths. Declare the owned paths and the
dependency, test, workflow and authority paths before qualification. Keep the
freeze SHA-256 alongside the independent exact-head qualification receipt.
`verify_frozen_checkout` checks the candidate before a gate executes.

Before integration, supply the independently qualified head, tree and
qualification digest to `prepare_integration`. Use the observed PR head,
mergeability and freshly fetched `origin/main` SHA. The preview binds that
tracking ref. A local integration fixture may use `refs/heads/main`. It
rejects a changed head and any main
movement that overlaps declared paths. For a disjoint advance it predicts the
combined tree using an isolated Git index. An explicit exact-base authority
requirement still applies.

Persist the preview and its SHA-256. Merge through the repository's normal
expected-head protection using the preview's frozen head. The verifier covers
squash and two-parent merge topology. After fetching accepted main,
`verify_accepted_integration` checks the actual commit parents, complete tree,
main ref, ancestry and preservation of declared paths. Record DONE after that
readback and the required independent qualification.

The record loaders compare the serialized freeze and preview with independently
retained expected digests. They support a fresh process after handoff.
`pytest -q tests/test_plan8_integration_qualification.py` exercises real local
Git commits, squash and merge, disjoint main movement and rejected readbacks.
