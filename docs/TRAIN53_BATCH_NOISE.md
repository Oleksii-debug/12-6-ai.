# TRAIN-53 — batch-noise and effective-batch evidence

## Scope

TRAIN-53 estimates whether the incumbent fixed-control effective batch is noise-limited or unnecessarily large. It extends TRAIN-29 `TrainingObserver`; it does not create another logger or change `Trainer` update semantics.

The fixed control is the established RESEARCH41 267,912-parameter member: byte vocabulary 256, context ceiling 256, `d_model=72`, four layers, six heads, `d_ff=192`, tied embeddings, fp32 AdamW at `3e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, zero weight decay, constant LR, clip 1.0, seed 1337.

## Real-corpus truth boundary

TRAIN-53 consumes DATA-21/22 through its existing `run_bounded_intake()` implementation. It does not use the packaged S0 fixture because those records are explicitly synthetic.

DATA-21/22 currently exposes three bounded real external objects from two sources whose experimental registry marks model training approved: one Ukrainian Verkhovna Rada open-data law object and two English Standard Ebooks Manual of Style objects pinned to git revision `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b`. The intake must accept all three objects or TRAIN-53 fails closed.

The split is fixed before measurement. The Standard Ebooks `9-metadata.rst` object is validation-only; the Rada object plus `8-typography.rst` are train-only. Packing remains document-isolated. No test data is optimized.

This is real-source evidence but not a canonical D03 corpus freeze. The Rada input is a bounded dated intake rather than a stored immutable snapshot, there are only three source objects, and validation is one English document. The report therefore marks corpus representativeness false and transfer to 100K–1M provisional.

## Batch comparison

A base microbatch is four 64-token examples, giving 252 valid causal loss tokens. Four accumulated effective batches are measured: 252, 504, 1,008 and 2,016 loss tokens/update via accumulation 1, 2, 4 and 8.

Every candidate consumes the exact same deterministic 256-microbatch trace and therefore the same 64,512 optimized loss-token budget. Validation is measured at 0, 16,128, 32,256 and 64,512 optimized tokens. LR is deliberately not batch-scaled, so this answers the practical fixed-recipe question rather than jointly retuning LR and batch size.

Wall time is measured by TRAIN-29 around real `Trainer.train_microbatch()` transitions. The data trace is pre-materialized, so CPU timing compares model compute and optimizer-step overhead; it is not a GPU efficiency claim.

## Gradient signal/variance estimator

On the accumulation-1 trajectory, diagnostics run at 0, 16,128, 32,256 and 64,512 optimized tokens. Each checkpoint uses 16 independently drawn with-replacement training microbatches from an isolated deterministic PRNG.

For microbatch gradients `g_i`, TRAIN-53 records `||mean(g_i)||^2`, the gradient second moment, unbiased trace of sample covariance, and `trace(covariance)/||mean(g_i)||^2`. The latter is a local noise-scale proxy in units of the 252-loss-token base microbatch; the report also converts it to a loss-token proxy. For accumulation 1/2/4/8 it records grouped-gradient covariance, relative deviation and cosine to the all-sample mean plus the iid `1/k` variance prediction.

The estimator is not labeled a theoretically exact critical batch size. The real bounded corpus is tiny, document-level iid independence is not established, and gradient stationarity is only local to a checkpoint.

## Optimizer-state preservation

Each probe runs only at a checkpoint-safe accumulation boundary. Before sampling it fingerprints the model and full Trainer state, including optimizer, scheduler, scaler and counters, and saves Python/Torch/CUDA RNG, train/eval mode and pre-existing parameter gradients. It performs direct forward/backward passes only and never calls optimizer or scheduler step. Everything is restored, then model and Trainer hashes must match exactly.

A focused unit test additionally inserts a probe between two real fp32 optimizer steps and requires the next parameter update to be bit-identical to an unprobed control.

No duplicate model is retained. For 267,912 parameters, one fp64 CPU gradient vector is 2,143,296 bytes; 16 samples retain 34,292,736 bytes (about 32.7 MiB) only during one diagnostic call, then release them.

## Recommendation rule

At the common final token budget, candidates within 0.5% of best held-out validation loss are treated as quality-equivalent. Among those, the fastest measured training wall time is preferred; if another acceptable candidate is within 5% of that wall time, the smaller accumulation is chosen to avoid gratuitous batching.

The selected batch is a provisional starting point for 100K–1M campaigns. If the largest measured batch wins, TRAIN-53 explicitly reports that the grid edge was reached rather than calling 2,016 tokens/update an optimum. No extrapolation above the measured grid is justified without another probe on a representative corpus and target hardware.
