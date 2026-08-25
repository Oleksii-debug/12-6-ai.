# TRAIN-53 — batch-noise and effective-batch evidence

## Scope

TRAIN-53 estimates whether the incumbent fixed-control effective batch is noise-limited or already unnecessarily large. It extends TRAIN-29 by using `TrainingObserver` for bounded trajectory telemetry; it does not create another training logger or modify `Trainer` semantics.

The experiment fixes the established RESEARCH41 267,912-parameter member: byte vocabulary 256, context ceiling 256, `d_model=72`, four layers, six attention heads, `d_ff=192`, tied embeddings, random `InitSpec`, fp32 AdamW at `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, constant LR, gradient clipping at 1.0, seed 1337.

## Data truth boundary

Training and validation are loaded from the project-owned packaged S0 JSONL text records and passed through the existing deterministic packer and byte tokenizer. These are real text records rather than random token tensors. They are also small and repeatedly sampled. Therefore this experiment is controlled local evidence, not a representative external-corpus measurement.

Train and validation remain document-disjoint. No test set is optimized.

## Batch comparison

The base microbatch is four examples of length 64, giving 252 valid causal loss tokens for a full microbatch. Four accumulated effective batches are compared:

- accumulation 1: 252 loss tokens/update;
- accumulation 2: 504 loss tokens/update;
- accumulation 4: 1,008 loss tokens/update;
- accumulation 8: 2,016 loss tokens/update.

Every candidate consumes the same deterministic 64-microbatch trace, exactly 16,128 loss tokens total. Validation is measured at 0, 4,032, 8,064, and 16,128 optimized tokens. LR is deliberately not scaled with batch size, so the experiment answers the practical question "with the incumbent optimizer recipe and a fixed token budget, is a different effective batch better?" It does not answer a joint LR/batch retuning question.

Training wall time is measured through the incumbent observer around real `Trainer.train_microbatch()` transitions. The trace is pre-materialized, so these timings compare forward/backward/update work and optimizer-step overhead; they do not model production data-loader wait or GPU kernel efficiency.

## Gradient signal/variance estimator

On the accumulation-1 trajectory, diagnostics run at 0, 4,032, 8,064, and 16,128 optimized tokens. Each checkpoint draws 16 microbatches with replacement using an isolated deterministic PRNG. For microbatch mean gradients `g_i`:

- signal squared proxy: `||mean(g_i)||^2`;
- gradient second moment: `mean(||g_i||^2)`;
- variance proxy: unbiased `mean squared distance from mean`, i.e. trace of sample covariance;
- local noise-scale proxy: `trace(covariance) / ||mean(g_i)||^2` in units of the base microbatch;
- accumulated-gradient proxies: empirical grouped gradient covariance, relative deviation from the all-sample mean, cosine to the all-sample mean, and the `1/k` iid variance prediction for accumulation `k`.

The ratio is useful as a local signal/variance diagnostic. It is not called a theoretically exact critical batch size because the corpus is finite/recycled, document-level independence is not established, and gradients are only stationary at the local checkpoint by construction.

## Probe state preservation

Each diagnostic requires a checkpoint-safe Trainer boundary. Before gradient sampling it snapshots/fingerprints model state and full Trainer state, including optimizer/scheduler/scaler/counters, plus Python/Torch/CUDA RNG state and existing parameter gradients. The probe runs direct forward/backward passes only; it never calls optimizer or scheduler step. It then restores gradients, RNG and model mode and requires model and Trainer fingerprints to match exactly.

The unit contract additionally compares a training trajectory with a diagnostic inserted between two optimizer steps against an unprobed control and requires the next fp32 optimizer update to be bit-identical.

No duplicate model is retained. For the real 267,912-parameter probe, each diagnostic retains 16 float64 CPU gradient vectors during analysis. The evidence records exact bytes per vector and total retained gradient-sample bytes, and the vectors are released on return.

## Recommendation rule

At the final common token budget, candidates within 0.5% of the best held-out validation loss are considered quality-equivalent. Among them, the fastest measured training wall is preferred; if another acceptable candidate is within 5% of that wall, the smaller accumulation is chosen to avoid gratuitously large batches. The local gradient-noise proxy is reported as supporting evidence, not as an overriding theoretical optimum.

The resulting batch is only a provisional starting point for the 100K-1M campaign family. A representative-corpus and target-hardware recheck is required before treating it as a broad default, and batch sizes beyond the measured 2,016 loss tokens/update are not justified by this experiment.
