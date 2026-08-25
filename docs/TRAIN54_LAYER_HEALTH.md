# TRAIN-54 layer health diagnostics

TRAIN-54 extends the TRAIN-29 observability incumbent with temporary per-layer diagnostic windows. It does not change `TwelveSixDecoder.forward`, `TransformerBlock.forward`, D02 Trainer semantics, optimizer behavior, checkpoint state, or deterministic run identities.

## Window contract

At initialization and selected committed optimizer boundaries, the probe registers forward hooks only for the duration of one diagnostic forward/backward. It captures residual-stream RMS/variance before attention, after attention, and after the block; attention and MLP RMSNorm outputs; attention and MLP branch outputs; and pre-clip gradient norms for attention, MLP, and norm parameters. Hook handles are removed in `finally`, gradients and RNG state are restored, and no optimizer step is performed.

The machine report records the current `InitSpec` relationship explicitly: base weights use `Normal(0, 0.02)` and residual output projections use `0.02 / sqrt(2L)`. It also records whether the current global gradient clip threshold `1.0` would engage on the diagnostic gradient and the hypothetical clip factor. The diagnostic itself never clips.

## Controlled execution

The LOCAL_FREE matrix uses the exact fixed-control subset of the RESEARCH41 family: 95,568, 467,808, and 1,037,696 trainable parameters. Tokenizer, context, initialization, optimizer, deterministic cyclic byte trace, batch size, and sequence length remain fixed. Diagnostic windows are at optimizer steps 0, 4, 16, and 64.

Depth warnings are intentionally heuristic rather than theoretical critical thresholds. Residual endpoint ratios >=2.5 or <=0.4 and gradient endpoint ratios >=10 or <=0.1 are warning signals; large max/min spreads are also surfaced. Absence of these warnings is not a proof of universal stability.

## Truth boundary

The packaged S0 corpus is a tiny controlled compatibility fixture and is repeatedly cycled. Results are layer-health and optimization-mechanics evidence only, not representative-corpus quality evidence, architecture promotion, capability evidence, or paid-compute authorization.
