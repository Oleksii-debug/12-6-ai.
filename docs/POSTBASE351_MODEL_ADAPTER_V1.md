# POSTBASE-351 Model Adapter V1

Worker: `POSTBASE-351-MODEL-ADAPTER-V1`
Recovery worker: `NEXT100-011-POSTBASE351-RECOVERY`

## Purpose

`PostBaseModelAdapter` is the single post-Base seam between immutable verified learned Base checkpoints and later deliberation/tool controllers. It does not add model weights, a second decoder, a trainer, checkpoint-writing behavior, chat semantics, tool execution, or a remote-model client.

## Base intake

The adapter delegates checkpoint intake exclusively to `load_first_party_backend()`. That maintained path first creates one D05 `VerifiedCheckpoint` snapshot, derives ModelSpec/tokenizer identity from that snapshot, loads weights from the same verified bytes, and restores no training RNG state.

POSTBASE-351 additionally requires positive checkpoint `step` and `tokens_seen` counters before it will call the checkpoint a learned Base input. This is a mechanical provenance gate; it is not a new scientific admission or capability claim.

The adapter never calls `save_checkpoint`, never creates an optimizer, never runs backward, and never writes to the supplied checkpoint directory. Generation executes under `torch.inference_mode()` through the accepted first-party generation path.

Direct construction is fail-closed: only the exact maintained `FirstPartyInferenceBackend` runtime class is accepted. External, network-backed, wrapper, and subclass backends are rejected. The normal supported entrypoint remains `PostBaseModelAdapter.from_checkpoint()`, which constructs that backend through the maintained first-party loader.

## ModelSpec compatibility

Compatibility is semantic rather than size-name based:

- ModelSpec schema v1;
- canonical `s0-byte-v1` vocabulary contract;
- a ModelSpec accepted by the maintained first-party decoder/checkpoint implementation.

There is deliberately no parameter-count allowlist. The exact current learned 10M geometry is covered: `10,000,640` parameters, ModelSpec `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`.

The recovered contract is also bound by test to the exact primary MODEL-341 candidate authority at commit `e4ff486fd90802fc123bebf60eed4e59196a98df`: `20,613,440` parameters, `D320/L16/10Q/2KV/F1080`, context `1024`, ModelSpec `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`. MODEL-341 is random-init mechanical qualification only; POSTBASE-351 does not claim that a learned 20M checkpoint exists.

## Controller port

Controllers submit `ControllerGenerationRequest` with one explicit controller kind:

- `deliberation`;
- `tool`.

Both receive ordinary first-party `GenerationResult`. POSTBASE-351 does not interpret the generated text, execute tools, create a tool schema, or decide deliberation policy. Those concerns remain above the adapter.

## Evidence firewall

Every response contains two different typed records:

1. `BaseCheckpointEvidence`, namespace `base`: checkpoint/model/tokenizer/data/run provenance and learned counters copied from the verified Base snapshot.
2. `PostBaseGenerationEvidence`, namespace `post_base`: adapter version, LOCAL_FREE runtime policy, controller kind, generation-config hash, prompt hash, token counts, generated-token hash, and stop reason.

The post-Base record intentionally contains no ModelSpec hash, dataset identity, tokenizer identity, training step, tokens-seen count, held-out score, or other Base evidence field. The Base record intentionally contains no prompt or controller-generation result fields. Controller behavior therefore cannot be promoted into Base evidence by flattening a shared record.

## Exact learned-10M recovery proof

The exact-head workflow consumes LEARN-217 artifact `9602650341` and first verifies the artifact ZIP SHA-256 `8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee`. It loads `scale141-evidence/retained/best` only through `load_first_party_backend()`, binds the exact learned-10M ModelSpec identity, exercises both controller ports under LOCAL_FREE CPU inference, and records Base and post-Base evidence separately.

Before inference the workflow hashes every file in the learned checkpoint tree into one deterministic tree digest. It repeats the same operation after both controller calls and requires byte identity with `cmp`. Any checkpoint-tree change fails the proof.

The proof is a post-Base adapter/integrity proof. Independent Base scientific admission remains governed by the separate Base verification authority; this adapter does not create or replace that authority.

## Truth boundary

D05 checkpoint verification establishes checkpoint integrity/compatibility for intake. It does not by itself establish scientific promotion, instruction following, reasoning quality, tool competence, alignment, or production readiness. POSTBASE-351 records post-Base controller executions separately and makes none of those claims.

`LOCAL_FREE` only. No external LLM, model hub, remote inference service, paid compute, SFT, RLHF, DPO, or Base checkpoint mutation is introduced by this adapter.
