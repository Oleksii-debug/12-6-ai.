# ~20M Readiness Controller — live authority reset

Machine decision: `BLOCK_LONG_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING`.

The primary ~20M model is no longer the main blocker. MODEL-341 mechanically qualifies the random-init 20,613,440-parameter candidate, including forward/backward/update, D05 save/load, static-KV, and 1024-token context mechanics.

The blocking path is the training corpus. DATA-301 remains terminal-blocked with no corpus identity, no shard identity, and zero balanced no-replay capacity. A real learned ~20M campaign therefore cannot start without fabricating data authority.

## Live changes beyond stale worker cutoffs

- English source diversity now has a terminal new source authority: bounded official CPython documentation, family `python.cpython.documentation`, terminal `ADMIT`, with 14 accepted chunks.
- Ukrainian diversity has a new independent bounded Wikisource authority with terminal rights and immutable snapshot evidence, but that source is not yet corpus-eligible because standard near-match evaluation decontamination is still required.
- NEXT100-066 correctly refuses decontamination because no exact candidate corpus identity and bound training-record inventory exist.
- The immutable selection-validation composite is non-empty for UA/EN; code selection-validation remains zero and must stay fail-closed until pristine, explicitly evaluation-reserved code objects exist.
- CHECKPOINT-346 is stale relative to the now-published primary MODEL-341 authority and should be requalified rather than treated as a permanent failure.

## Ordered next campaign

1. Compose a successor Research Corpus V1 intake from terminal source authorities without mutating the old DATA-300/301 contract.
2. Materialize and freeze an exact pre-decontamination candidate record inventory and cryptographic identity.
3. Run the standard exact/near-match evaluation decontamination against that exact candidate.
4. Run quality, privacy, cross-source dedup, cluster-safe split, deterministic sharding twice, and the full unique-loss ledger.
5. Requalify CHECKPOINT-346 and execute only the bounded LOCAL_FREE TRAIN-344 mechanics probe against MODEL-341.
6. Refresh LEARN-345 against terminal corpus/model/optimizer/recovery authorities.
7. Only when the campaign is data-ready should material training compute be separately authorized.

This ordering breaks the current circular failure mode: decontamination cannot run without a candidate identity, while a final corpus must not be declared terminal before decontamination. The successor needs a distinct immutable pre-decontamination candidate identity followed by a separate final corpus identity.

No model training was executed by this controller. No paid compute was used. The controller intentionally reports zero authorized unique optimized targets until a terminal successor Research Corpus V1 publishes a one-pass no-replay ledger.

Machine authority: `configs/control/20m_readiness_controller_v1.json`

Validator: `python tools/validate_20m_readiness_controller.py`
