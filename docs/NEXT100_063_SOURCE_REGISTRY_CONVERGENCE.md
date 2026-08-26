# NEXT100-063 — scoped source-registry convergence

## Purpose

This package consumes the exact-green `NEXT100-065-CROSSSOURCE-DEDUP-V3` vector and one later exact-green terminal English source authority, `NEXT100-037-DATA-EN-PYTHON-DOCS`.

The narrow objective is to remove the hard diversity blocker `EN terminal family count 1 < 2` for this exact source vector without claiming that the complete 20 MB research corpus, decontamination, packing, tokenizer fitting, or model training is ready.

## Exact authorities

Base dedup authority:

- head `efc278cec0e4773eb4ff405bf4b4d24ee63b5d13`
- dedicated workflow run `32999969398` — success
- inherited source objects: 11
- inherited family vector: UK 2, EN 1, code 4

Added terminal source authority:

- worker `NEXT100-037-DATA-EN-PYTHON-DOCS`
- PR #467
- head `5a6a495a24bce449334cbc5126d0114f61a9f57c`
- dedicated workflow run `32998356906` — success
- authority identity `46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d`
- source family `python.cpython.documentation`
- exact upstream object `python/cpython@7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480:Doc/tutorial/introduction.rst`

## Privacy/quality-preserving materialization

The terminal source authority does not authorize the complete normalized CPython object for training. DATA-228 produced 16 deterministic chunks; 14 were accepted and two were rejected by the `pii_phone` predicate.

NEXT100-063 therefore reconstructs the exact DATA-228 normalization and chunking rules, verifies the whole upstream raw object, verifies the whole normalized object, reproduces all 16 chunks, applies the same quality/privacy policy, and requires the exact ordered set of 14 accepted chunk SHA-256 identities from NEXT100-037.

Only those 14 accepted chunks are joined into an ephemeral comparison payload. The two rejected chunks are absent from dedup input and from declared training capacity. The materialized text itself is never written to the public report.

## Cross-source gate

The accepted-only payload is passed through the inherited NEXT100-065 / DATA-298 exact, normalized, near-copy, fragment, code-copy, and lineage-aware accounting. The convergence fails closed if the added source develops a capacity-collapsing cross-family match.

Expected exact family vector after the scoped addition:

- UK: 2
- EN: 2
- code: 4

This resolves the minimum-two-family English blocker only for this exact terminal vector.

## Truth boundary

A successful NEXT100-063 run means only that this scoped source vector can safely include the terminal CPython documentation family after accepted-chunk materialization and cross-source dedup.

It does not mean:

- the canonical global source registry has absorbed every concurrent terminal source;
- the 20 MB source-byte target is complete;
- evaluation reservations have been applied;
- an immutable train/validation/test corpus has been materialized;
- decontamination has passed;
- post-pack unique loss positions are known;
- tokenizer fitting is authorized;
- 20M model training is authorized;
- paid compute is authorized or used.

`LOCAL_FREE` only. No model training, optimizer updates, final-test consumption, or paid compute are performed by this package.
