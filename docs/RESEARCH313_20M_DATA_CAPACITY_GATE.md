# RESEARCH-313 — 20M Data Capacity Gate

Worker: `RESEARCH-313-20M-DATA-CAPACITY-GATE`

Mode: `LOCAL_FREE`.

Verdict: **BLOCKED_NO_TERMINAL_FINAL_CORPUS_LEDGER**

Evidence identity: `cd6d80f2e31631723c7f513327082845b3edfcd8a90768809da74aaf8476ba99`

## Scope

This gate asks one narrow question: how many **actual unique nonignored causal loss positions** may a future approximately 20M-parameter Base optimize before any replay, using the final external-real corpus candidate.

It does not equate source bytes, padded slots, packed windows, epochs, optimizer steps, or a token/parameter heuristic with unique training exposure. It does not claim a universal compute-optimal token/parameter ratio.

## Authority cutoff

The strongest frozen corpus contract is DATA-300 PR #392, exact head `8ea7f830e50a23754d189dd4134f4afad76a7ee9`, contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5`.

DATA-300 explicitly says the corpus state is `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`. Its exact candidate inventory has five training-admitted source objects, four independent families and 183,061 source bytes:

- UK: 88,565 bytes / 1 family;
- EN: 84,793 bytes / 1 family;
- code: 9,703 bytes / 2 families.

The selected DATA-295 policy remains 45% UK / 35% EN / 20% code with a 20,000,000 source-byte target, but its hard minimum is two independent families in every stratum. Current family counts are 1/1/2, so the family-constrained no-replay source-byte budget is **0**.

## Actual unique-loss evidence

DATA-294 is the only terminal exact causal-position ledger at this cutoff. It covers the three DATA-229 text objects only, under exact `s0-byte-v1`, document isolation, no BOS/EOS, and padding excluded.

Its one-pass total is exactly **173,355 unique nonignored causal loss positions**:

| scope | positions |
|---|---:|
| UK | 88,564 |
| EN | 84,791 |
| code | 0 |
| total text | 173,355 |

That ledger identity is `9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0`.

This is real measured exposure capacity for its exact three-object text scope. It is **not** authority for the DATA-300 five-source candidate. The full post-split, post-dedup, post-reservation Wave-3 train ledger does not exist, so this gate deliberately does not convert 183,061 source bytes into a guessed loss-position count.

## Budget classes

### Mechanics

Mechanics qualification may continue on bounded fixtures or synthetic data because it is a trainer/runtime/checkpoint claim, not a learned-quality claim. If the run is required to use the final external-real candidate under DATA-300 release rules, the currently authorized exposure is **0 positions** because the candidate is not a released corpus.

### Short science

The project has a prior no-replay short-science precedent of **2,000,060 actual nonignored causal loss positions** on the 10M Base (RESEARCH-251 consuming LEARN-217). That is evidence that a bounded learned trajectory can be scientifically useful; it is not an optimum and is not automatically portable to 20M.

The current external-real candidate cannot authorize that budget. The only exact external-real ledger available here is 173,355 text-only positions, and DATA-300 does not authorize using it as the final five-source corpus.

### Meaningful learned campaign

RESEARCH-251 used a project-specific decision band of roughly 0.5–2 unique positions per parameter for 10M/100M scale comparisons, explicitly as a research sufficiency envelope rather than a universal law. Applied only as a local planning reference, a nominal 20M Base would correspond to **10M–40M unique positions**.

This gate does **not** use that band to set the no-replay ceiling. The ceiling must come from the final corpus ledger. The current DATA-300 candidate therefore authorizes **0 positions** for a meaningful learned campaign.

## Maximum safe preregistered exposure before replay

**0 unique nonignored causal loss positions, now.**

The zero is a fail-closed authorization result, not a claim that the source material contains zero learnable text. DATA-300 still has unresolved hard gates:

- `G05_QUALITY`;
- `G06_PRIVACY`;
- `G09_BALANCE_DIVERSITY`;
- `G10_SELECTION_VALIDATION`;
- `G12_UNIQUE_LOSS`;
- `G14_TWO_CLEAN_BUILDS`.

In particular, there is no terminal full five-source train ledger and the hard family gate fails.

Future unlock rule:

> Once a terminal final corpus exists, the maximum preregistered no-replay exposure is exactly the immutable train ledger's one-pass count of unique nonignored causal loss positions.

Nothing else may raise that number: not source-byte totals, padding, epochs, sampling with replacement, duplicated documents, source aliases, or a token/parameter ratio.

## 20M implication

A future ~20M Base can be mechanically qualified before the corpus gate opens, but a learned external-real campaign must remain blocked. To make a meaningful 20M learned campaign plausible, the final corpus should first clear all DATA-300 release gates and publish an exact train ledger. The existing project planning reference says 10M unique positions is the lower edge of a meaningful 20M comparison envelope, but the actual authorized maximum remains whatever the terminal corpus ledger proves, capped at one pass.

No training was executed by RESEARCH-313.
