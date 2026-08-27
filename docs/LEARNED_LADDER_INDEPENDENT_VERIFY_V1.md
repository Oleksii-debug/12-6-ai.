# Independent learned-ladder verification v1

This package closes the independent-verification evidence gap named in issue #548 for the existing LEARN-191 ~3M and LEARN-217 ~10M producer artifacts. It is an independent evidence audit. It does not retrain either model, mutate producer branches, authorize learned 20M training, or convert generic release CI into a PASS.

## Exact verified producer authorities

VERIFY-219 binds PR #348 at `a75920cef8bde37a8c590e34095be83c97b75f1d`, dedicated run `32940842372`, artifact `9597788382`, and artifact SHA-256 `f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee`.

VERIFY-218-LEARNED-10M-INDEPENDENT binds PR #355 at `c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`, dedicated run `32952787070`, artifact `9602650341`, and artifact SHA-256 `8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee`.

Both artifacts bind DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8` and the project byte tokenizer. The verifier checks the downloaded ZIP digest, exact source SHA, model identity, random-init / no-foreign-weight / no-paid-compute boundary, fresh-process resume evidence, checkpoint identities, and every retained checkpoint payload hash declared by its checkpoint manifest.

## Verdicts

VERIFY-219 is `PASS_SCIENTIFIC_EVIDENCE_NOT_RELEASE_GATE`. The dedicated LEARN-191 workflow completed successfully and its artifact is internally consistent. The nominal targets `16,632`, `65,772`, and `131,292` are threshold labels; the authoritative actual trainer exposures are `17,125`, `66,417`, and `131,938` optimized tokens. Selection-validation BPB falls from `7.9952104052` at random init to `2.2859499700` at the final retained checkpoint. Fresh phase-boundary resume and a separate final fresh load are present. Generic CI run `32940842230` nevertheless concluded `failure` in job `locked-x86-64`, step `Locked clean install, package smoke and full checks`; this package does not relabel that release signal.

VERIFY-218-LEARNED-10M-INDEPENDENT is `PASS_SCIENTIFIC_EVIDENCE_NOT_RELEASE_GATE`. The dedicated LEARN-217 terminal workflow completed successfully. The retained phase-one checkpoint has `1,000,133` actual optimized tokens and the chronological final checkpoint has `2,000,060`. Fresh D05 verification reports PASS for checkpoint load/identity, evaluation non-mutation, first-party logits/generation, M150 common-evaluation identity, reproducibility-manifest validation, and best/final retention. The retained checkpoint payloads and both retained recovery generations pass independent manifest hash checks. The recovery preflight reproduces the historical immutable-overwrite failure and proves corruption rejection, optimizer-state restore, RNG restore, and older-generation byte preservation.

## Scientific comparison boundary

The two rungs are not a matched-token experiment. The 10M artifact has about `15.1591x` the optimized-token exposure of the 3M artifact (`2,000,060` versus `131,938`). Therefore this package does not rank model scale from the two endpoint metrics and does not fit a scaling law from them. Nominal target labels are never substituted for actual trainer token counts.

For a learned 20M recipe, the defensible carry-forward is procedural: random initialization, exact immutable identity binding, a fresh-process phase boundary, preregistered held-out selection, best plus chronological-final retention, checkpoint payload verification, evaluation non-mutation, first-party inference checks, an explicitly scoped train-probe exposure diagnostic, and actual optimized-token accounting. These artifacts do not justify a 20M token budget or expected BPB by extrapolation.

## Reproduction

Run the validator against the exact downloaded Actions artifacts:

```bash
python tools/verify_learned_ladder_independent_v1.py \
  --contract configs/eval/learned_ladder_independent_verify_v1.json \
  --learn191-artifact learn191-real-3m.zip \
  --learn217-artifact learn217-terminal-10m.zip
```

The validator is fail-closed on artifact digest drift, missing members, source/model/data/tokenizer identity drift, foreign/pretrained or paid-compute claims, altered token accounting, reused process evidence, checkpoint payload corruption, missing evaluation non-mutation, missing recovery corruption rejection, missing retained roles, or attempted matched-budget scale ranking.
