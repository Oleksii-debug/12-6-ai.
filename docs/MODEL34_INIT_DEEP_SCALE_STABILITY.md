# MODEL-34 initialization and deep-scale stability audit

Status: engineering preflight. No canonical stage identity, checkpoint format, or paid-compute authority is changed by this work.

## Scope audited

The current decoder uses pre-RMSNorm, RoPE, SwiGLU and tied token/output embeddings. `InitSpec v1` initializes ordinary embedding/linear weights from `Normal(0, 0.02)` and reinitializes the two residual-output projections (`attn.out_proj` and `mlp.down_proj`) at:

`0.02 / sqrt(2 * n_layers)`

S1, S2 and S3 all carry the same InitSpec identity. The current Scale-04 S4 engineering candidate also carries that same identity. S0 remains untouched.

Checkpoint/run binding already fail-closes on `candidate.initspec_sha256`; the bound checkpoint training configuration persists `init_spec_sha256`, and the full run-manifest hash changes when InitSpec changes. MODEL-34 adds a regression test for this property rather than changing checkpoint format v1.

## Primary-source check

The current residual-output scaling is consistent with the long-lived Megatron/TransformerEngine family of scaled output initialization: ordinary weights receive a base normal sigma while transformer output projections use a sigma scaled by model depth. This is relevant to the current pre-normalized residual architecture.

Primary sources used for engineering context:

- OpenAI GPT-2 reference implementation: `conv1d` defaults to normal initialization with stddev 0.02 and token embeddings use stddev 0.02.
  https://github.com/openai/gpt-2/blob/master/src/model.py
- NVIDIA Megatron Core / TransformerEngine: transformer blocks distinguish base `init_method_normal` from `scaled_init_method_normal` for output layers.
  https://github.com/NVIDIA/Megatron-LM
  https://github.com/NVIDIA/TransformerEngine
- DeepNet / DeepNorm: Wang et al., 2022, arXiv:2203.00555. It changes residual/normalization semantics to stabilize extremely deep Transformers; it is not a drop-in initializer for this 3/4/6/13-layer pre-RMSNorm stack.
- Tensor Programs V / muTransfer: Yang et al., 2022, arXiv:2203.03466. It is a model parameterization and transfer protocol, not a single initializer constant. MODEL-34 does not introduce it without a contained base/target experiment.

The source review therefore supports testing the incumbent before replacement; it does not justify importing DeepNorm or muP semantics into S0/S1.

## Controlled local probe

A supporting CPU probe was executed against a locally reconstructed copy of the current model semantics because the execution container could not DNS-clone GitHub. This evidence is useful for direction, but it is not exact-head CI authority. The committed harness exists so the same experiment can be rerun from the actual repository and locked environment.

Data were deterministic random-token fixtures, held fixed across initialization candidates. Model-init seeds were varied. The optimizer matched current Trainer defaults: AdamW, LR 3e-4, betas 0.9/0.95, eps 1e-8, weight decay 0, clip norm 1.0, fp32. The probe is intentionally short and is not a model-quality benchmark.

### Incumbent InitSpec v1

| stage | params | width/layers | initial CE mean | uniform CE | logit std | final-block hidden RMS | pre-clip grad L2 | short-run clip fraction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 107,856 | 48 / 3 | 6.2492 | 6.2383 | 0.1421 | 0.0204 | 0.936 | 0.00 |
| S2 | 1,066,112 | 128 / 4 | 7.6550 | 7.6246 | 0.2290 | 0.0262 | 3.082 | 1.00 |
| S3 | 10,059,840 | 320 / 6 | 9.0688 | 9.0109 | 0.3575 | 0.0727 | 21.45 | 1.00 |
| S4 Scale-04 snapshot | 99,897,600 | 768 / 13 | 5.6800 | 5.5452 | 0.5459 | 0.2286 | 122.1 | 1.00 |

Seed dispersion remained small for loss/logit scale in this probe. The main scale-dependent signal was gradient norm: S4 was about 130x S1 before clipping.

### Counterfactual: remove residual scaling

Removing `1/sqrt(2L)` made the residual stream materially larger. Final-block hidden RMS rose from 0.0204 to 0.0234 (S1), 0.0262 to 0.0749 (S2), 0.0727 to 0.2735 (S3), and 0.2286 to 1.1918 (S4). At S4 the incumbent therefore reduced final-block residual-stream RMS by about 5.2x.

This is direct evidence against deleting the existing residual scaling.

The unscaled control sometimes had a lower global gradient norm. That is not evidence of better conditioning: the much larger residual stream is normalized before the head, so global gradient magnitude alone is not a monotonic stability score.

### Counterfactual: width-referenced base std

A contained control scales the base std by `sqrt(48 / d_model)` while retaining the incumbent residual-output depth scaling. It is representable with the existing InitSpec v1 `std` field and therefore receives its own InitSpec identity; it does not require a production schema change.

Observed initial metrics:

| stage | initial CE | logit std | final-block hidden RMS | pre-clip grad L2 |
| --- | ---: | ---: | ---: | ---: |
| S1 | 6.2492 | 0.1421 | 0.0204 | 0.936 |
| S2 | 7.6345 | 0.1383 | 0.0131 | 1.882 |
| S3 | 9.0283 | 0.1335 | 0.0103 | 11.00 |
| S4 | 5.5758 | 0.1724 | 0.0116 | 69.24 |

This control flattens initial activation/logit scale and moves initial CE closer to the uniform baseline. It still leaves strong growth in raw gradient norm, and the supporting probe is too short and synthetic to establish better learning dynamics. It is therefore an experimental control, not a selected future-stage InitSpec.

## Decision

1. Keep canonical `InitSpec v1 = Normal(0, 0.02) + residual 1/sqrt(2L)` unchanged, including S0 identity.
2. Do not promote the unscaled residual control.
3. Do not promote the width-referenced control yet. The signal is promising but insufficient to justify a production InitSpec v2.
4. Treat persistent clip saturation from S2 upward as a real qualification risk. Initialization and optimizer/clip policy are coupled here; TRAIN-33 should consume the exact-head reports rather than independently retuning from S0.
5. Do not introduce DeepNorm or muP from terminology alone. Either would require an explicit architecture/parameterization experiment and new identities.

## Exact-head qualification gate

Before paid 10M/100M training, run `tools/run_init_stability_matrix.py` in the locked repository environment and require:

- exact ModelSpec and InitSpec hashes in every report;
- all losses, activations and gradients finite for every seed;
- no canonical-init mutation or promotion authority in the report truth boundary;
- initial CE remains close to the vocabulary-uniform baseline; an excess above 0.25 nat is an investigation trigger;
- initial logit std below 1.0 and no gross layer-wise residual-stream explosion;
- seed coefficient of variation remains modest for loss/logit scale;
- clip fraction and raw gradient growth are reviewed jointly with optimizer experiments, not interpreted as an initializer-only pass/fail.

If the width-referenced control still materially reduces activation/logit drift on exact-head real-data qualification runs without worsening early loss progression or update behavior, then introduce a new versioned future-stage InitSpec field/schema. Until that evidence exists, no production initialization change is justified.

## Harness

`src/twelve_six/training/init_stability.py`
- loads exact stage configs;
- varies only InitSpec candidate and model-init seed;
- holds synthetic token data fixed;
- records token embedding, per-layer attention/MLP branch, per-block residual stream, final norm and logits;
- records shifted causal loss, global and per-block gradient norms, clipping factors and short-run loss trajectory;
- hashes each report and enforces a fail-closed non-authoritative truth boundary.

`tools/run_init_stability_matrix.py`
- runs the configured stage/candidate matrix;
- writes one hashed JSON report per stage/candidate plus an index.

`configs/experiments/s4_100m_scale04_init_probe.json`
- is an explicit snapshot of the current Scale-04 99,897,600-parameter engineering candidate;
- is probe-only and cannot promote or replace a canonical S4 config.

`tests/test_init_stability.py`
- proves candidate InitSpec identities differ;
- proves reports are hash-bound/fail-closed;
- proves canonical checkpoint binding persists and distinguishes InitSpec hashes.
