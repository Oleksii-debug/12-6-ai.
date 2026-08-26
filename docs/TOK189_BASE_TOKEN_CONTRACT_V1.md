# TOK-189 — Research Base Token Contract v1

## Decision

Contract: `base-byte-noeos-v1`

This is the research Base token contract for the next ~10M baseline campaign until a later version explicitly supersedes it.

- ordinary vocabulary: canonical `s0-byte-v1`, 256 raw UTF-8 byte IDs, where token ID equals byte value `0..255`
- normalization: none
- BOS: none
- EOS/EOD: none
- semantic PAD: none
- UNK: none
- instruction/system/chat tokens: none
- document boundary: isolate documents; do not create cross-document causal transitions
- tensor fill: ID `0` may be used only as non-semantic storage where `attention_mask=0`, labels are `-100`, and loss is masked
- empty generation context: reject; there is no BOS or other seed token
- generation termination: token/count/context or explicit caller stop controls only; no EOS termination semantics

Machine-readable authority: `configs/token_contracts/base_byte_noeos_v1.research.json`.

## Evidence cut

MILESTONE-150 retained `s0-byte-v1` as the accepted executed baseline for the 10M path. Its tokenizer config SHA-256 is `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`; its complete vocabulary SHA-256 is `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`.

RECOVER-173/TOK-115 successfully executed the EOS boundary experiment, but the retained scientific status is `PROVISIONAL_EOS_BOUNDARY_CONTRACT`. Candidate B (document-isolated EOS) measured about `3.1837692044` BPB versus about `3.1863011984` BPB for no-EOS, while 30 boundary-aware generation attempts produced zero EOS terminations. Under TOK-189's fail-closed rule, this is not sufficient evidence to force EOS into the Base contract.

At this contract cut, no published TOK-188 PR/commit/artifact was available in the repository. TOK-189 therefore does not invent a concurrent family decision. Byte remains the current executed 10M baseline authority. A later TOK-188 result that selects another deterministic family must create a new token-contract version rather than silently mutate v1.

## Stability requirements

`base-byte-noeos-v1` is valid only while all of these remain exact:

1. tokenizer version `s0-byte-v1`;
2. tokenizer config SHA-256 `b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1`;
3. vocabulary SHA-256 `905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571`;
4. model vocabulary size `256`;
5. packing version `s0-byte-pack-v1` for training resume;
6. document isolation and no EOS insertion;
7. no semantic special-token IDs.

The contract JSON is self-hashed. Runtime validation reconstructs all 256 vocabulary entries and verifies that entry `i` is byte `i`.

## Packing and padding

Without EOS/EOD there is no scientifically justified boundary token that could make adjacent documents one causal stream. Therefore cross-document packing is incompatible with this contract.

A short document may be physically extended to fixed tensor length with in-vocabulary fill ID `0`. This does not make byte zero a PAD token. Every fill position must have attention disabled and its target/loss disabled. Any path that attends to or optimizes a fill position violates the contract.

## Empty-context and generation semantics

An empty string encodes to zero tokens. Generation must reject this state before model execution because the contract has no BOS or equivalent seed token.

A non-empty prompt generates only ordinary IDs `0..255`. Generation can stop at the configured new-token limit, context limit, or an explicit caller-provided stop token/string. There is no model-level EOS stop condition in v1.

## Hugging Face / Transformers token mapping

Exports that describe this token contract must map:

- `vocab_size = 256`
- `bos_token_id = null`
- `eos_token_id = null`
- `pad_token_id = null`
- `unk_token_id = null`
- `decoder_start_token_id = null`
- `forced_bos_token_id = null`
- `forced_eos_token_id = null`
- `added_tokens = []`
- `special_tokens_map = {}`

This mapping is token-semantic authority only. It does not claim that `transformers.AutoModel` can instantiate the native 12-6 architecture; the existing HF-style export attestation remains conservative about architecture/runtime parity.

## Checkpoint compatibility

Before model, trainer, optimizer, scheduler, or RNG mutation, a checkpoint used under this contract must prove:

- tokenizer config hash equals the v1 byte hash;
- tokenizer vocabulary hash equals the v1 byte-vocabulary hash;
- checkpoint ModelSpec vocabulary size is 256;
- for training resume, the bound packing version is `s0-byte-pack-v1`;
- normal run/corpus/model bindings required by the checkpoint layer still match.

A 257-vocabulary EOS checkpoint, BPE/Unigram checkpoint, altered byte mapping, or different packing binding is incompatible and must fail before mutation.

## Migration rules

Legacy checkpoints are not rejected merely because they predate the TOK-189 document. A legacy no-EOS byte checkpoint is compatible only when its existing immutable identities prove the same tokenizer config hash, vocabulary hash, 256-entry ModelSpec vocabulary, and required training-resume packing binding. This is an identity-based compatibility rule, not a filename/version guess.

There is no silent migration from a 257-vocabulary EOS checkpoint or another tokenizer family. Such a change requires a new contract version, a compatible ModelSpec, explicit retokenization where needed, and a new checkpoint lineage.

There is no silent token-ID remapping and no silent addition of BOS/EOS/PAD/UNK/instruction/system/chat tokens.

If a future accepted TOK-188 family decision supersedes byte, create `base-<family>-...-v2` (or later), preserve v1 unchanged, and make cross-contract loading fail closed unless an explicit scientifically justified migration exists.

## Regression authority

`.github/workflows/tok189-base-token-contract-v1.yml` runs through the universal execution bootstrap with LOCAL_FREE CPU execution. It reuses the retained MILESTONE-150 ~500K geometry (`467,808` parameters) and DATA-25 truth model, trains to checkpoint step 500, resumes in a fresh process to step 1000, freshly verifies retained checkpoint/evaluation/generation behavior, exercises an intentionally wrong tokenizer identity and proves rejection before model/trainer/RNG mutation, and emits `tok189-evidence/final-report.json` plus all supporting artifacts.

The regression does not upgrade DATA-25 into representative external data. It is contract and learning-regression evidence only.
