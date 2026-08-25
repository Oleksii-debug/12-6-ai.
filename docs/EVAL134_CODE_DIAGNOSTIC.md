# EVAL-134 reserved code diagnostic v1

Status: `RESERVED_EVALUATION_ONLY`.

This package is a project-authored mechanistic code-modelling diagnostic for tiny raw 12-6 Base models. It is not an instruction-following benchmark and does not ask the model to generate applications. Passing it does not establish code-generation capability.

## Immutable suite

- suite: `eval134-code-diagnostic-v1`
- data: `eval/reserved/code_diag_v1/probes.jsonl`
- items: 32 contrastive probes, four per required phenomenon
- suite data SHA-256: `87e8085ef7bd9bb6b9755e5e88b2db040226bfd0bffd6696a3ca5f2afb0fe865`
- suite identity SHA-256: `df18192f6190cc5d8be9492103a15097daaaf31afdd1cd45b2f4c21af5721105`
- exact candidate hash root: `18bfda81d087b0f8db44793c0e224ec48f001058795f473ecf943bda3ddb9740`
- D03-normalized candidate hash root: `633662dcfb4d5d713567b37935e912fb50ccd7d893b650fe711bbfdd84c44c70`

The suite covers balanced delimiters, indentation-sensitive continuation, operator/type syntax, simple function-call structure, variable reuse, string/comment termination, JSON-like structure, and language-specific syntax. Language-specific corpus strata are Python and SQL; JSON is treated as a structural syntax stratum.

All probe text was authored for this repository. Identifiers use the reserved `qzv_` namespace, and distinguishing numeric literals use the 7xxx/8xxx range. The verifier requires that the namespace and every complete candidate continuation are absent from the current S0 and DATA-10 training inputs.

## Reservation and decontamination

`data/s0/contamination_registry.json` binds the suite identity, suite data hash, and exact/normalized candidate hash roots before any future corpus construction. Existing D03 code does not automatically consume the new `reserved_evaluation_suites` field. Therefore future pretraining composition must execute:

```text
PYTHONPATH=src:. python tools/eval134_code_suite.py verify --repo-root .
```

The verifier fails closed on suite drift, registry drift, exact candidate overlap, `qzv_` namespace overlap, missing phenomenon coverage, or language-stratum drift. This is a reservation gate, not a claim that older corpus builders already enforce a schema they do not know.

## Conditional-likelihood scoring

`src/twelve_six/code_diagnostic.py` scores an explicit prefix-token / continuation-token boundary. For each correct completion and distractor it reports:

- total conditional log-likelihood in nats;
- NLL per continuation token;
- NLL per UTF-8 source byte;
- bits per source byte;
- target token and byte counts;
- whether the completion is exactly one tokenizer token;
- whether independent prefix+completion tokenization equals joint tokenization.

The primary contrastive reports include raw likelihood accuracy/margin and source-byte-normalized accuracy/margin. This makes completion-length effects visible instead of treating a shorter tokenization as syntax knowledge.

Evaluation runs under `torch.no_grad()` and the EVAL-134 runner hashes model state and checks Trainer counters and model train/eval mode before and after scoring. Evaluation must not increment optimized tokens or mutate weights.

## Byte/BPE diagnostics

The learned comparison family uses the incumbent repeatable DATA-10 ByteLevel BPE experiment. Canonical `s0-byte-v1` is evaluated as a segmentation reference: one UTF-8 byte equals one token, so its token/source-byte ratio is exactly 1.0. The report records BPE tokens/source-byte for every candidate and how often BPE joint tokenization would cross the forced conditional boundary.

Model likelihood is not compared across incompatible byte-vocabulary and BPE-vocabulary weights. Instead, source-byte NLL/BPB supplies the tokenizer-length-normalized model metric, while byte tokenization supplies the exact segmentation baseline.

## Scaling execution

No successfully retained current-best ~100K/~500K/~1M learned artifacts are available on the live lineage: the current RESEARCH41 workflow stops before training because the aggregate dependency-lock profile is stale relative to `pyproject.toml`; LEARN-01 stopped at lint; LEARN-04 explicitly deferred to RESEARCH41. EVAL-134 therefore creates comparable learned states on the incumbent RESEARCH41 BPE control rather than mislabelling absent checkpoints.

The dedicated LOCAL_FREE workflow evaluates scratch random initialization and then trains/evaluates these exact geometries at a shared requested 65,536 optimized-token budget:

- 95,568 parameters;
- 467,808 parameters;
- 1,038,464 parameters.

Training reuses the incumbent project-authored DATA-10 UK/EN/code corpus, tokenizer, mixture, packing, Trainer, optimizer configuration, seed, ModelSpec family, and D05 checkpoint writer. It does not introduce a second training stack. Each learned state is retained as a D05 checkpoint inside the EVAL-134 evidence artifact.

The dedicated workflow intentionally does not call the currently stale aggregate `verify_locked_environment.py`. It installs the exact committed hash-locked toolchain/runtime files plus the exact tokenizer overlay, then runs the suite reservation verifier and focused tests before optimization. This bypass is narrowly scoped to the known stale metadata check and is not represented as canonical environment authority.

## Claim boundary

The output may support statements about relative conditional likelihood on this reserved mechanistic suite. It does not support claims of general programming ability, full code generation, instruction following, representative code-corpus quality, broad language proficiency, stage promotion, or intelligence. No foreign pretrained weights, instruction/SFT data, or paid compute are used.
