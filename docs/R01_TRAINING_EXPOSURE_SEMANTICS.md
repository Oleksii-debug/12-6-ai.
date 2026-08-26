# R01 training exposure semantics

## Purpose

R01 plans 10x, 20x and 40x token-per-parameter exposure arms for the exact
20,613,440-parameter MODEL-341 candidate. Those numbers describe optimizer exposure, not a
claim that the project already owns the same number of unique training targets.

This contract closes that ambiguity before learned-20M work can be authorized.

## Four quantities that must never be conflated

1. Source bytes are acquisition capacity only.
2. Unique loss positions are distinct non-ignored causal targets in the immutable post-pack
   training corpus after all reservations, decontamination, quality/privacy, dedup, split and
   packing gates.
3. Total training exposures are all non-ignored causal targets consumed by optimizer updates.
4. Repeat exposures are optimizer-consumed positions seen again after their first unique use.

Replay can increase total exposure. It cannot increase unique data capacity.

## MODEL-341 planning arithmetic

For 20,613,440 parameters, the existing R01 arms imply:

- 10 tokens/parameter -> 206,134,400 total training exposures;
- 20 tokens/parameter -> 412,268,800 total training exposures;
- 40 tokens/parameter -> 824,537,600 total training exposures.

An illustrative 20,000,000-position unique corpus would therefore require about 10.30672,
20.61344 and 41.22688 effective passes respectively. The 20M figure is deliberately marked as
an illustrative planning floor, not a corpus authority and not a byte-to-token conversion.
Exact post-pack loss accounting remains mandatory.

## Data-constrained training boundary

Hoffmann et al. motivates joint model/data scaling, but its token/parameter relation is not a
rule that every planned optimizer exposure must be unique. Muennighoff et al. studies repeated
data in constrained regimes and reports little loss degradation through roughly four epochs in
the studied range, followed by diminishing value from further repetition.

The contract records four epochs only as a bounded research reference. It does not authorize
four epochs automatically and does not claim four is universally optimal. Any request above
the unique ledger must carry a separately preregistered maximum effective-epoch cap.

The executable assessment has four outcomes:

- `WITHIN_UNIQUE_LEDGER`;
- `BLOCKED_REPEAT_POLICY_REQUIRED`;
- `WITHIN_PREREGISTERED_REPEAT_CAP`;
- `BLOCKED_REPEAT_CAP_EXCEEDED`.

None of these statuses is training authorization. Corpus, tokenizer, checkpoint, evaluation,
training-recipe and material-compute gates remain separate.

## Why this matters for 20M -> 100M -> 1B

The project can use a smaller high-quality corpus for bounded pilots without pretending it is a
compute-optimal full campaign. As scale grows, the coordinator must decide explicitly whether
to acquire more unique data, use a measured repeat policy, or stop the campaign. This prevents
both failure modes: treating replay as new information, and incorrectly forbidding all
scientifically preregistered data-constrained training.

No model training, optimizer update, tokenizer fit, corpus mutation, final-test access, GPU/cloud
provisioning or paid compute is authorized by this package.
