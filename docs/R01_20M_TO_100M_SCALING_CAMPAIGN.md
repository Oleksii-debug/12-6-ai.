# R01 — Evidence-bound 20M to 100M scaling campaign

Status: `CANDIDATE_PLANNING_ONLY`  
Issue: #536  
Execution boundary: `LOCAL_FREE`; no long training or materially paid compute is authorized by this package.

## Why this package exists

The project now has an exact mechanical 20M authority, but a mechanically valid model is not the same thing as a learned model. The live 20M campaign is still gated by corpus materialization, evaluation decontamination, tokenizer identity, checkpoint-integrity retest, selection-validation evidence, and compute authorization.

This package prevents a common scaling failure mode: increasing parameter count before the data, tokenizer, metrics, optimizer evidence, checkpoint lineage, and cost envelope are ready. It defines what must be measured before the project freezes a ~100M ModelSpec.

## Exact incumbent control

`MODEL-341-20M-CANDIDATE-A` is bound exactly to:

- branch `model341/20m-candidate-a-20260826`;
- SHA `e4ff486fd90802fc123bebf60eed4e59196a98df`;
- ModelSpec SHA-256 `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`;
- 20,613,440 parameters;
- byte vocabulary 256;
- context 1024;
- D320, 16 layers, 10 query heads, 2 KV heads, head dimension 32, FFN 1080;
- pre-RMSNorm, RoPE, SwiGLU, tied embeddings;
- random initialization only.

This is the control geometry. It is not a learned-quality claim and is not a training authorization.

## Research findings

### Confirmed source facts

**Hoffmann et al., 2022 — Chinchilla**  
https://arxiv.org/abs/2203.15556

The study varied model size and training tokens under compute budgets and found that compute-optimal model size and data should grow together rather than scaling parameters while holding data fixed. Its empirical regime starts above the current 20M project stage, so it is used here as a scaling reference rather than a literal tiny-model recipe.

**Liu et al., 2024 — MobileLLM**  
https://arxiv.org/abs/2402.14905

For sub-billion models, architecture matters materially. The paper reports strong 125M/350M results from deep-and-thin networks together with embedding sharing and grouped-query attention. Those ingredients directionally agree with the incumbent 12-6 design.

**Ben Allal et al., 2025 — SmolLM2**  
https://arxiv.org/abs/2502.02737

SmolLM2 documents a data-centric small-model recipe with small-scale ablations and staged mixture refinement before committing to very large training exposure. This supports treating data mixture as an experimental axis, not as a fixed afterthought.

**Blake et al., 2024 — u-muP**  
https://arxiv.org/abs/2407.17465

u-muP combines maximal-update parameterization with unit scaling and is designed to make useful hyperparameters transfer more consistently across model widths/scales. This is evidence that hyperparameter transfer can be deliberately engineered. It is not evidence that the current standard-parameterized 20M recipe can be copied unchanged to 50M or 100M.

**DeepSeek LLM, 2024 — compute-aware scaling**  
https://arxiv.org/abs/2401.02954

The scaling analysis distinguishes parameter-count proxies from non-embedding FLOPs/token and shows that attention overhead can make simple parameter-based compute approximations materially inaccurate, especially at smaller scales. R01 therefore records parameter count but uses measured or exact non-embedding FLOPs/token as the primary scale-comparison axis.

**Meister, 2026 — TokEval**  
https://arxiv.org/abs/2608.18062

TokEval evaluates tokenizers with both intrinsic structure-sensitive metrics and controlled language-model pretraining, including bits-per-byte as a tokenizer-agnostic language-model metric. This strengthens the existing R01 rule that raw token-level perplexity must not be the primary comparison across different tokenizer identities.

### R01 inference from those facts

The incumbent D320/L16 + GQA + tied-embedding direction is a reasonable control for the present size, but it is not frozen for 100M. The next architectural decision should be based on matched data/tokenizer experiments rather than on a presumed larger-model template.

At <=100M there is no evidence-based reason to force tensor, pipeline, context, or expert parallelism into the model merely because those techniques will be needed later. Prefer the simplest execution topology that fits and measure throughput/memory before adding distributed complexity.

Parameter count remains an identity and deployment metric, but it is not sufficient as the sole compute scale variable. At every learned 20M/50M/100M point, record exact parameter count plus measured or exact non-embedding FLOPs/token, including attention cost, and report vocabulary projection compute separately.

### Experiment proposals, not accepted truths

The `[10, 20, 40]` tokens-per-parameter grid is a planned measurement grid for fitting local scaling behavior. It is not a statement that any one ratio is compute-optimal, sufficient for quality, or authorized for execution.

The AdamW beta2 candidates `[0.95, 0.98, 0.999]` are an ablation grid. They do not replace the current optimizer configuration without measured evidence.

The 20M/50M/100M size grid exists to measure whether loss and downstream contamination-safe validation improve enough to justify the next size. It is explicitly not a release ladder and does not freeze the ~100M architecture.

u-muP is an experiment proposal, not an adopted parameterization. A future 50M/100M recipe may use either scale-specific retuning under a matched data/tokenizer/metric contract or a validated transfer parameterization such as u-muP with a matched standard-parameterization control. Silent reuse of the 20M learning rate, betas, warmup, scheduler, batch-token geometry, clipping, precision, or initialization is forbidden.

## Cross-scale hyperparameter gate

Before R01-E30 can support freezing a 100M ModelSpec, the project must produce one of two evidence paths:

1. scale-specific retuning for the larger point under the same bound corpus/tokenizer/evaluation contract; or
2. a validated transfer-parameterization experiment, with u-muP currently the research candidate, compared against a matched standard-parameterization control.

Every path must record learning rate, optimizer betas, weight decay, warmup, scheduler, batch tokens, gradient clipping, precision, initialization/parameterization, seed, loss curve, and gradient-health evidence. The campaign validator fails closed if this gate is removed or if u-muP is marked adopted without evidence.

## Compute-accounting rule

For the 20M/50M/100M curve, parameter count is reported but `6 * parameters * tokens` is not the primary scientific scale axis. The campaign requires measured or exact non-embedding FLOPs/token, attention compute included, with vocabulary projection compute reported separately. This prevents width, context, GQA, or vocabulary changes from being misread as equal-compute comparisons merely because parameter counts look comparable.

The token-per-parameter grid remains a measurement grid. No value in `[10, 20, 40]` is declared compute-optimal in advance.

## The blocking order for a learned 20M run

1. Materialize one immutable Research Corpus V1 identity from terminal source authorities.
2. Decontaminate it against reserved selection/final evaluation identities and near-copy clusters.
3. Complete quality, privacy, global dedup, split, packing, and a unique post-pack causal-loss ledger.
4. Fit/reproduce the corpus-bound tokenizer candidates or explicitly retain the byte baseline with measured evidence.
5. Re-run checkpoint corruption/integrity evidence on the exact selected code head.
6. Bind selection-validation and stage metrics.
7. Run bounded local/free pilots first; reject NaN/Inf, unexplained spikes, resume mismatch, or unstable seed behavior.
8. Only after the above, C01 estimates the material run and the owner can explicitly authorize paid compute.

No missing gate may be replaced by a chat claim, parameter-count target, queued CI run, or stale report.

## Metrics that make the sweep scientifically comparable

Bits-per-byte is the primary cross-tokenizer language-model metric because token-level perplexity changes meaning when the tokenizer changes. Validation NLL/perplexity remain useful within one tokenizer identity.

Every candidate must also record the loss curve, gradient health, tokens/second, peak memory, non-embedding FLOPs/token, checkpoint/resume equivalence, deterministic rebuild or seed evidence, exact model/tokenizer/data/config identities, and contamination-safe evaluation provenance.

A faster or larger candidate is not promoted if its validation/evaluation evidence is worse or untrustworthy.

## 100M decision rule

Do not freeze a 100M ModelSpec now. First obtain terminal learned-20M evidence, then compare at least the planned 20M/50M/100M points under controlled tokenizer/data/metric contracts. Freeze the 100M ModelSpec only when the measured curve, cross-scale training-recipe evidence, and hardware profile justify it.

The long-term 1B and larger roadmap remains valid, but every jump must carry forward reproducible checkpoint lineage, dataset identity, tokenizer identity, evaluation separation, stage-specific optimization evidence, and cost evidence. Sparse MoE is a later large-scale research decision, not a reason to complicate the <=100M Base.

## Handoff by lane

- DATA: finish immutable Research Corpus V1 identity and post-pack unique-loss authority.
- TOK: execute reproducible corpus-bound tokenizer experiments and report BPB/fertility/roundtrip/throughput plus structure-sensitive intrinsic diagnostics.
- TRAIN: after gates, run bounded optimizer/token-budget pilots; do not silently copy 20M hyperparameters to larger scales.
- D06: compare contamination-safe metrics and fit the empirical scaling curve using BPB plus compute-normalized scale evidence.
- D05/AUDIT-A: close exact-head checkpoint integrity and resume evidence.
- C01: produce hardware/cost envelope and obtain explicit compute authorization before material spend.
- D01: use measured results and cross-scale recipe evidence to propose, then audit, the 100M ModelSpec.

## Truth boundary

This package changes no Base weights, no corpus bytes, no checkpoint implementation, and no evaluation records. It creates a durable scientific control contract so autonomous workers can move rapidly without converting guesses into canonical model decisions.
