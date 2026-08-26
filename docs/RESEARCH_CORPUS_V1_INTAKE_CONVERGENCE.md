# Research Corpus V1 intake convergence

## Purpose

This package implements the first ordered data action from the live ~20M readiness controller: compose a successor Research Corpus V1 intake from terminal source authorities.

It is deliberately an intake authority, not a final corpus build. It does not freeze the exact candidate record inventory, run global cross-source near-dedup, run evaluation decontamination, create train/validation shards, prove two clean builds, authorize optimized targets, or authorize training.

## Bound authorities

The intake keeps the terminal DATA-287 baseline families and adds bounded terminal authorities discovered after that registry snapshot:

- English: CPython documentation and the bounded NIST technical-series subset.
- Ukrainian: KMu Secretariat news, the bounded Nomis1864 public-domain snapshot, and the terminal Ukrainian Wikisource snapshot that remains conditioned on standard near-match decontamination.
- Code: the bounded Starlette implementation snapshot.

Every candidate is bound to an exact PR and head SHA. Evaluation use remains fail-closed. Each candidate still requires standard cross-source dedup and evaluation decontamination before corpus inclusion.

## Deterministic lower-bound projection

The manifest currently accounts for a known terminal intake lower bound of 277,885 payload bytes:

- Ukrainian: 100,856 bytes.
- English: 162,052 bytes.
- Code: 14,977 bytes.

If global dedup does not collapse any listed lineages, the intake projects to four Ukrainian, three English, and three code families. This is only `PROJECTED_PASS_NOT_TERMINAL`; it is not a terminal family-diversity pass.

Against the acquisition-planning proxy of 9M Ukrainian / 7M English / 4M code bytes, the known terminal lower bound still leaves 19,722,115 bytes of aggregate proxy gap. These byte targets are planning capacity only and must not be interpreted as optimized-token counts.

The live controller also records a separate token-budget truth: 20M requested optimized targets for the 20,613,440-parameter model are a pipeline pilot, not a science-complete quality baseline. The reference planning budget is much larger and must remain separate from this byte-capacity proxy.

## Truth boundary

The validator fails closed if this intake attempts to claim any of the following early:

- exact candidate inventory frozen;
- global dedup passed;
- evaluation decontamination passed;
- privacy/quality pipeline passed;
- train/validation split frozen;
- two clean builds matched;
- unique no-replay loss ledger terminal;
- nonzero authorized optimized targets;
- final corpus ready;
- training authorized;
- paid compute authorized.

The current authorized unique optimized-target count remains exactly zero.

## Validation

Run locally from the repository root:

```bash
python tools/validate_research_corpus_v1_intake.py
pytest -q tests/test_research_corpus_intake.py
```

No dedicated GitHub Actions workflow is added by this package. The repository is already experiencing Actions queue pressure, so this change intentionally avoids adding another workflow fan-out surface. Existing aggregate CI can exercise the tests when the convergence branch is integrated into an appropriate CI path.

## Next ordered work

1. Freeze the exact pre-decontamination candidate record inventory and deterministic identity using the controller's existing candidate-identity builder once every selected record has exact raw/normalized identities and authority bindings.
2. Run standard exact/near-match evaluation decontamination.
3. Run global cross-source dedup and family-collapse checks on the composed inventory.
4. Expand terminal source capacity while preserving rights, provenance, privacy, quality, and evaluation firewalls.
5. Run split, two-clean-build, and unique no-replay loss-ledger gates.
6. Only after terminal data readiness, rebind 20M checkpoint/optimizer mechanics and refresh the learning preregistration.
7. Request explicit material-compute authorization only after the campaign is data-ready.
