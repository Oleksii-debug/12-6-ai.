# R01 FLOPs/Vocabulary Accounting V1

Status: `CANDIDATE_PLANNING_ONLY` / `LOCAL_FREE`.

This package closes one planning ambiguity in the learned-20M -> 100M path: total parameter count is not an exact compute measure once context length, GQA geometry, and tokenizer vocabulary can change.

## Bound baseline

The accounting baseline is exact MODEL-341 candidate A:

- source branch: `model341/20m-candidate-a-20260826`
- source SHA: `e4ff486fd90802fc123bebf60eed4e59196a98df`
- ModelSpec identity: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- expected parameters: `20,613,440`
- geometry: vocab 256, d_model 320, 16 layers, 10 query heads, 2 KV heads, head_dim 32, SwiGLU d_ff 1080, two pre-RMSNorm vectors per block, final RMSNorm, tied embeddings, no projection/MLP/LM-head biases.

The exact parameter identity is reproduced as:

- token embedding: 81,920
- attention projection weights per block: 245,760
- SwiGLU weights per block: 1,036,800
- two RMSNorm vectors per block: 640
- block total: 1,283,200
- 16 blocks: 20,531,200
- final RMSNorm: 320
- total: 20,613,440

RoPE has no learned table in this accounting.

## Why vocabulary is a separate scaling axis

At fixed transformer geometry, changing a tied vocabulary from 256 to `V` adds `d_model * (V - 256)` learned embedding parameters. It also changes the output projection matmul even when the embedding weight is tied. Therefore a tokenizer comparison cannot truthfully say that two runs have the same model size and compute merely because hidden size/layers are unchanged.

Conversely, if total parameters are held approximately fixed, increasing vocabulary requires reducing another capacity term. V1 demonstrates that confound by adjusting only `d_ff` to the nearest configured multiple. This is a diagnostic surface, not a recommendation to tune only `d_ff` in the final architecture search.

## FLOP surface

The estimator deliberately does not call `6 * parameters` exact. It reports dominant matmul components per query token:

- Q/K/V/O projections;
- SwiGLU gate/up/down projections;
- vocabulary projection;
- QK and attention-weighted-V context work.

For a query attending to `C` KV positions, the attention-context term is approximately `4 * d_model * C` FLOPs. Dense matmul training is then exposed as a rough `3 * forward_dominant_matmul` estimate (forward plus two backward matmuls).

The estimate excludes elementwise activation, RMSNorm, softmax, RoPE, embedding lookup, optimizer updates, communication, memory traffic, padding/packing inefficiency, kernel launch overhead, recomputation/checkpointing overhead, and hardware utilization. It must therefore be calibrated later against measured throughput before material-compute authorization.

For causal full-sequence planning, `max_seq_len` should not automatically be substituted as every token's context. Sequence-average effective context is lower; packed-sequence boundaries must also be respected.

## Candidate-only geometry surfaces

V1 includes two arithmetic probes only:

- `R01-CANDIDATE-50M-A`: 50,009,472 parameters.
- `R01-CANDIDATE-100M-A`: 99,998,080 parameters.

They are not frozen ModelSpecs and carry no training authority. Their purpose is to exercise exact accounting around the intended scale points before a learned 20M campaign supplies empirical throughput/loss/memory evidence.

## Execution

Generate a deterministic JSON report with:

`PYTHONPATH=src python tools/report_r01_flops_vocab_accounting.py`

Optionally write it to a file with `--output <path>`.

Run focused tests with:

`PYTHONPATH=src pytest -q tests/test_r01_flops_vocab_accounting.py`

## Truth boundary

This package performs no tokenizer fit, corpus mutation, model training, optimizer update, GPU provisioning, final-test access, paid compute, learned-20M claim, 100M ModelSpec freeze, or stage promotion.

The next scientific use of this accounting is after terminal data/tokenizer/checkpoint/evaluation authorities exist: combine exact post-pack unique causal-loss positions with measured tokenizer efficiency and hardware throughput, then compute a bounded learned-20M run envelope. A 100M launch remains downstream of terminal learned-20M evidence and a new explicit compute authorization.
