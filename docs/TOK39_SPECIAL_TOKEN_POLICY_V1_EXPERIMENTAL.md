# TOK-39 experimental Base special-token policy v1

Status: **EXPERIMENTAL_NOT_CANONICAL**. This policy does not alter S0 and has no promotion authority.

## Decision

The first Base special-token experiment is **EOS-only**.

| Role | Decision | ID | Rationale |
|---|---|---:|---|
| byte values | keep | 0..255 | preserve exact S0 byte semantics |
| EOS / end-of-document | add | 256 | explicit document boundary, generation stopping, enables incumbent cross-document packer |
| BOS | do not add | — | Base does not require it; current empty-context behavior remains fail-closed rather than using an untrained start token |
| PAD | do not add | — | incumbent batching is mask-defined: fill ID 0 is non-semantic, `attention_mask=0`, labels ignored |
| UNK | do not add | — | UTF-8 bytes cover every encodable input; unknown substitution would reduce fidelity |

No instruction, user, assistant, system, turn, role, or chat-template tokens are defined.

Contract identity:

- tokenizer version: `exp-byte-eos-v1`
- vocabulary size: `257`
- EOS surface: `<|end_of_document|>`
- config SHA-256: `9d26ab7c69e51f36192fbbe3313e13327a2a97ff1134dd24e79f2d1227dc59a0`
- vocabulary SHA-256: `2ae636644ebff60166fe69e1d83a15a1be45aada86a33977cb888c52cf5dc21d`
- canonical S0 config SHA-256 remains `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`
- canonical S0 vocabulary SHA-256 remains `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`

## Semantics

Text encoding without flags is byte-identical to S0. `encode(text, add_eos=True)` appends ID 256. `add_bos=True` is rejected. Decode strips EOS by default and preserves all byte content exactly.

EOS is both the end-of-sequence signal for generation and the explicit end-of-document marker used by the incumbent cross-document packer. These are compatible Base semantics; neither implies conversational turns.

EOS is a boundary **signal**, not an attention reset. With ordinary causal attention, tokens after EOS can still attend to tokens before EOS. Therefore `cross_document=True` is boundary-marked packing, not strict document-isolation. Experiments requiring hard independence must keep isolated packing or introduce a separately versioned attention-reset mechanism.

## Batching

No semantic PAD token is introduced. Fixed-shape training batches retain the incumbent contract: byte ID 0 may occupy masked fill positions, but those positions have `attention_mask=0` and ignored labels/loss. Code must never treat fill ID 0 as PAD semantics because ID 0 remains the real NUL byte.

Variable-length inference batching is permitted only where the runtime accepts explicit attention masks without requiring a semantic `pad_token_id`. A runtime that requires PAD must fail closed until a new token contract is trained/validated; EOS must not be aliased to PAD for convenience.

## Empty context

Empty-context generation/evaluation remains unsupported in v1. A BOS token would only be meaningful if included in the training objective. This contract therefore preserves the current zero-token rejection rather than inventing a BOS at inference time.

Evaluation must state whether first-token-of-document probability is excluded or, under EOS-packed evaluation, conditioned on a preceding EOS. Do not compare those two objectives as though they were identical.

## Evaluation accounting

EOS changes both token count and the objective. Report at least:

1. content-byte NLL/BPB excluding special-token targets, for comparability with S0;
2. EOS boundary NLL separately;
3. total packed-token NLL only as tokenizer-contract-specific evidence.

Raw token perplexity across 256- and 257-token contracts is not directly comparable without this decomposition.

## Transformers bridge

The current raw-Base Llama bridge correctly emits `bos_token_id=null`, `eos_token_id=null`, and `pad_token_id=null` for S0. The experimental helper may set only `eos_token_id=256`, and only after `ModelSpec.vocab_size` is 257. `bos_token_id` and `pad_token_id` remain null. `unk_token_id` is null in the token contract. No `chat_template` is allowed.

Existing S0 HF/runtime exports must continue to reject invented special-token semantics. An experimental 257-vocabulary checkpoint requires a new export identity; it must not be passed off as an S0 export.

## Checkpoint migration rules

1. Never relabel an existing `s0-byte-v1`, vocab-256 checkpoint as `exp-byte-eos-v1`.
2. Exact tokenizer version, config hash, vocabulary hash, and model vocabulary size must agree before load.
3. Vocab-256 and vocab-257 embedding/output tensors are shape-incompatible by design. Current S0 first-party loading should continue to fail closed.
4. No in-place S0-to-EOS checkpoint converter is authorized by TOK-39. The experimental campaign should train a 257-vocabulary model from an explicitly bound initialization/run config.
5. If a future warm-start converter is desired, it must create a new checkpoint identity, deterministically initialize the EOS row, record the parent checkpoint and conversion policy hash, and prove parity on IDs 0..255 before any use. It must never mutate the parent artifact.
6. Adding BOS, PAD, UNK, or any conversational token later requires a new contract version and new hashes. Existing IDs 0..256 are immutable within this contract.

## Controlled evidence

On the current S0 packaged training fixture at sequence length 128, isolated S0 packing used 21 sequences with 71.84% occupied positions. EOS-marked cross-document packing used 16 sequences with 94.97% occupied positions. The objective adds 19 boundary pairs on the ten-document training split: ten content-to-EOS targets and nine EOS-to-next-document-first-byte targets.

A directional three-seed tiny causal-LM probe on the same project-owned train/validation fixture found mean held-out content-byte NLL 1.95884 for isolated S0 versus 1.93296 for EOS packing. This is a microprobe, not qualification evidence; it shows no observed content-loss regression large enough to negate the packing result. Full details are in `evidence/tok39/special_token_eos_probe_20260825.json`.

## Promotion gate

TOK-39 leaves the contract experimental. Before canonical Base promotion, rerun the comparison with the selected production tokenizer/model size and report content BPB, EOS boundary loss, training throughput, generation stop behavior, and runtime/HF parity under a pinned environment.
