# D03 HTML extraction bake-off V1

Status: `INDEPENDENT_VERIFICATION / LOCAL_FREE / NON-AUTHORIZING`

Worker: `SWARM-742`

Canonical key: `D03|HTML-EXTRACTION|INDEPENDENT-VERIFY|TRAFILATURA-VS-RESILIPARSE`

Base: `main@5020afd671a3885c1b738c8b4eafe7525f630546`

Parents: #720 and D03 lane #4.

## Purpose

This package implements the P0-B extraction bake-off requested by #720 without replacing any
production extraction path. It compares pinned Trafilatura and Resiliparse releases on small,
project-authored, frozen English/Ukrainian HTML fixtures plus one frozen WARC-response envelope.
It is deliberately separate from the active Rada bulk extraction/normalization work in PR #641.

A benchmark winner is only a candidate for later integration work. It is never equivalent to
`ADOPTED`, source admission, corpus capacity, tokenizer authority, or model-training permission.

## Pinned candidates

- Trafilatura `2.2.0`, released 2026-07-31, Apache-2.0. The contract binds PyPI source archive
  SHA-256 `8c2cabb84066465228d03183fb698ce0b1245b81c58140b8ae0de57fddf3aae7`.
- Resiliparse `1.0.9`, released 2026-07-20, Apache-2.0. The contract binds PyPI source archive
  SHA-256 `872e4e37f0dd24b383feb3c112ccf1b8328eb77256279a137080e4a65fc36c20`.

Runtime execution also requires the installed distribution version to equal the configured version.
A missing distribution or version drift produces `RETEST_RUNTIME_REQUIRED` or
`RETEST_RUNTIME_IDENTITY`; it cannot produce a candidate recommendation.

The adapter calls are intentionally narrow:

- Trafilatura: plain-text `extract`, comments/tables/links/images disabled.
- Resiliparse: `extract_plain_text`, `main_content=True`, formatting/bullets/alt text/links/form
  fields/noscript disabled.

This configuration compares main-text extraction rather than feature richness.

## Frozen fixtures and scoring

`configs/research/html_extraction_bakeoff_v1.json` is self-identified and binds every fixture by
SHA-256. The suite covers:

1. English article with header/navigation/sidebar/footer noise.
2. Ukrainian article with cookie/navigation/aside/footer noise.
3. Malformed English HTML with intentionally unclosed tags.
4. Ukrainian HTML inside a minimal frozen WARC response envelope.

The WARC helper is only a strict decoder for this controlled fixture shape. It is not a production
WARC parser and must not be reused as one.

Each extractor runs twice per fixture. Normalized outputs must be identical. Metrics are:

- multiset token precision, recall and F1 against project-authored gold main text;
- required-content anchor recall;
- forbidden-boilerplate leakage;
- output byte count and SHA-256;
- execution duration as explicitly non-authoritative telemetry.

The deterministic evidence identity excludes timing, so two semantically identical runs retain the
same evidence hash.

## Preregistered selection rule

An extractor is eligible only if all fixture outputs are deterministic, macro anchor recall is at
least `0.80`, and macro boilerplate leakage is at most `0.25`.

If only one extractor is eligible, it becomes the candidate. If both are eligible, a candidate must
beat the other by at least `0.02` macro token F1 while having no worse boilerplate leakage.
Otherwise the terminal state is `NO_CLEAR_WINNER`. Any nondeterminism forces
`RETEST_NONDETERMINISTIC`.

Allowed terminal states are limited to RETEST states, `NO_CLEAR_WINNER`,
`CANDIDATE_TRAFILATURA`, or `CANDIDATE_RESILIPARSE`. `ADOPTED`, `TRAINING_AUTHORIZED`, and
`CORPUS_RELEASED` are forbidden by both contract and report validation.

## Operator execution

Install the exact pinned extractor releases in a disposable LOCAL_FREE environment, then run:

```text
PYTHONPATH=src python tools/run_html_extraction_bakeoff.py \
  --output reports/research/html_extraction_bakeoff_swarm742.json
```

The command exits non-zero for RETEST states. `--allow-retest` exists only so constrained
environments can materialize truthful blocker evidence; it does not turn RETEST into PASS.

Focused contract/mechanics tests:

```text
PYTHONPATH=src pytest -q tests/test_html_extraction_bakeoff.py
```

## Current execution truth boundary

The SWARM-742 local execution environment could not resolve external package indexes, and neither
pinned extractor was preinstalled. Therefore no actual Trafilatura-versus-Resiliparse quality
winner is claimed from that environment. The committed code is designed to emit
`RETEST_RUNTIME_REQUIRED` under exactly this condition instead of substituting a mock benchmark.

Test doubles are used only to validate scoring, determinism, selection, evidence hashing, and
fail-closed authority rules. They are not extractor evidence.

## Explicitly not authorized

- no production extractor replacement;
- no Rada PR #641 mutation or result reinterpretation;
- no external source admission or rights determination;
- no corpus/tokenizer capacity credit;
- no benchmark/final-test payload access;
- no model/optimizer/checkpoint mutation;
- no paid compute or long training;
- no claim that upstream speed/quality statements are 12-6 measurements.

A downstream adoption decision requires an exact-runtime execution of this contract, review of the
resulting machine report, integration-specific tests on the real production acquisition path, and
normal D03/D10 authority gates.
