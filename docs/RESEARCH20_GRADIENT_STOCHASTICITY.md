# RESEARCH-20 gradient stochasticity diagnostics

This experiment reuses the RESEARCH41 fixed-control family rather than creating new stage geometries. The four measured models have 95,568, 267,912, 467,808, and 1,037,696 trainable parameters with the same byte tokenizer, vocabulary 256, model context 256, training sequence length 64, corpus, packing rule, AdamW recipe, seed, and fp32 CPU runtime.

## Estimator

At optimizer steps 0, 4, 16, and 48 the diagnostic evaluates eight deterministic microbatches from the same corpus. Probe gradients are obtained with `torch.autograd.grad`; the probe never calls `backward()`, `optimizer.step()`, scheduler operations, or Trainer mutation methods.

The report records global and per-block gradient norms. Its stochasticity statistic is the controlled trace ratio

`E[||g_i - g_bar||^2] / max(||g_bar||^2, epsilon)`.

This is a variance/noise proxy for this exact microbatch distribution. It is not named or interpreted as a universal gradient noise scale. The same observed gradients are grouped into empirical 1x, 2x, and 4x virtual batches so batch-size effects can be compared without changing model state.

Normal training remains authoritative for clipping and updates. The existing Trainer already returns the pre-clip global gradient norm. RESEARCH-20 therefore derives actual clip frequency from `grad_norm > gradient_clip_norm`, and measures the parameter update L2 divided by the pre-update weight L2 at each trained checkpoint.

## Non-mutation contract

Each probe fingerprints the complete checkpoint-safe Trainer state, model state, optimizer state, existing parameter `.grad` buffers, model train/eval mode, Trainer counters, Python RNG, and Torch RNG before evaluation. RNG state is restored after the probe. Any mismatch is a hard failure. Focused tests populate real AdamW state and non-empty sentinel gradients before probing to prove the diagnostic preserves both.

## Decision use

Batch-size decisions should use the measured proxy together with the 1x/2x/4x empirical reduction at each scale and checkpoint. Clipping decisions should use actual Trainer clip frequency, not only probe norms. Learning-rate transfer should be qualified using update/weight ratios under the identical learning rate; scale-dependent growth is a reason to test a lower LR, while shrinkage alone is not authority to increase LR.

No universal threshold is asserted for any of these quantities. The S0 project-authored fixture is intentionally small and recycled, so the result is controlled LOCAL_FREE optimization evidence, not representative large-corpus training evidence, a stage freeze, or paid-compute authorization.
