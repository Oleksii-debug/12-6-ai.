# EVAL-135 reserved long-distance dependency diagnostics

`EVAL-135-LONG-DEPENDENCY-v1` is project-owned evaluation-only material for testing whether a 12-6 Base model actually uses distant context. It is not a generation benchmark and contains no instructions for the model to follow.

The generator materializes probes against the active inference backend. The dependency source is token index 0 and the scored target is at an exact token index of 32, 64, 128, 256, or 512. A matched 16-token condition is reserved as a short-context format control. A distance is never scored when it exceeds `backend.max_context_tokens`; this suite does not test extrapolation.

Each item has a correct one-token continuation and a balanced foil. Three conditions are scored: `full` retains the distant source; `truncated` keeps only a suffix shorter than the dependency distance, removing the source; `shuffled` preserves the full prefix length and all local tokens but replaces the distant source token with the foil. The principal measurements are correct-target NLL/bits, correct-vs-foil logit margin, pairwise accuracy with chance 0.5, and top-1 vocabulary accuracy. Positive dependency evidence requires a full-context advantage over both truncation and shuffled-source controls. Results are reported by family and exact token distance.

The 16-token short control is an interpretation gate. If a learned model cannot reliably resolve the same probe structure at short range, failure at a longer distance is not labelled an isolated context-capability failure. This separates probe-format/ordinary short-context quality from long-distance access.

The three v1 families are delayed symbol recurrence, structured key/value recurrence, and compact natural-language reference recurrence. Values and fillers are selected only from literals that encode to one token under the active tokenizer; fixed fragments may use multiple tokens. The generator pads with valid model tokens until the target-source distance is exact. The resulting materialized suite has its own SHA-256 identity in addition to the abstract suite identity.

Reservation is explicit: `data/reserved/eval135_long_dependency_v1.json` declares `source_purpose=evaluation_test` and `training_allowed=false`. The existing S0 contamination registry already forbids `evaluation_test` sources. No generated probe may be added to tokenizer training, corpus packaging, pretraining, fine-tuning, or augmentation.

For retained checkpoints, use `tools/run_long_dependency_probe.py` with an existing first-party `InferenceBackend` loader. The scorer only invokes next-token inference and never feeds probe sequences into a trainer. Unsupported distances are reported rather than silently shortened.

## Executed MODEL-17-era result

The original successful MODEL-17 workflow retained its evidence JSON but not the trained 128/256 checkpoints. For EVAL-135, the two tiny 95,568-parameter conditions were therefore reconstructed locally from the exact MODEL-17 architecture, seed, isolated one-token-overlap packing, optimizer configuration, S0 train split, and 32,768 optimized-token budget. This reconstruction reproduced the MODEL-17 train and held-out NLL/BPB values to approximately 1e-6, and the probes were never used for optimization. The machine report records both the original artifact identity and the reproduction deltas.

The learned 128-context reconstruction did not reliably solve the 16-token matched control (pairwise accuracy 0.53125), so its long-distance misses cannot be isolated as context failure. The learned 256-context reconstruction did resolve the short control above the suite's interpretation threshold (0.59375) but showed no distance at which full context beat both truncation and shuffled-source controls. At distance 256 its pairwise accuracy was 0.45833; truncation slightly improved target NLL, and replacing the distant source with the foil did not reduce the correct-vs-foil margin. This is no evidence of usable 256-token dependency learning in this training regime.

Combined with MODEL-17's original held-out result—128 native BPB 3.74856 versus 256 native BPB 4.76007, and a slightly negative within-256 gain from allowing history beyond 128—the EVAL-135 decision is to reject the MODEL-17-era 128→256 context increase for this tiny S0 regime. This is not a general context-length ceiling. No trained 512-context model was available, so 512 remains unevaluated rather than extrapolated.
