# D09 Post-Training Infrastructure v1

Status: isolated infrastructure only. No canonical Base checkpoint, Base training loop, model weights, assistant behavior, refusal policy, ethics constitution, personality, or domain specialization is modified by this package.

## Purpose

This package establishes stable contracts that later SFT, preference optimization, reward-model, GRPO/PPO-family, verifier, process-supervision, synthetic-data and test-time-reasoning work can reuse without coupling S0 Base to a specific post-training framework.

## Package surface

- `twelve_six.posttraining.schemas`: content-addressed artifact refs, deterministic canonical JSON/fingerprints, SFT/preference/verifier record schemas, human/imported/synthetic provenance, and explicit provenance for synthetic generators.
- `twelve_six.posttraining.rollout`: framework-neutral sampling, rollout request/candidate and `RolloutProvider` protocol.
- `twelve_six.posttraining.verifiers`: versioned verifier protocol, fail-closed registry, result schema, and a minimal exact-match verifier used only for contract/harness smoke tests.
- `twelve_six.posttraining.experiment`: fail-closed execution boundary. Post-training output cannot target `base` or `base/*`; real behavioral training additionally requires an explicit owner authorization reference.

## Canonical Base boundary

A Base checkpoint may eventually be consumed as a read-only source checkpoint after an explicit post-training decision. The post-training result must be emitted to a separate lineage such as `posttraining/...`. The infrastructure deliberately rejects a post-training output target under `base/...` even when an authorization string is present.

`ExecutionMode.CONTRACT_ONLY` and `ExecutionMode.DRY_RUN` are infrastructure modes. `ExecutionMode.TRAIN` is blocked unless an explicit owner behavioral-training authorization reference is supplied. This is a schema guard, not a substitute for Coordinator/Auditor validation of that authorization.

External teacher/critic/synthetic generators are not implicitly approved. `SyntheticProvenance(external_generator=True)` requires an explicit `owner_policy_ref`, plus generator artifact identity, generation-config digest, prompt-template digest and seed.

## Framework integration boundary

The project reuse matrix names TRL for SFT/DPO/GRPO/reward modeling and verl for large-scale RL. Current TRL documentation exposes SFT, DPO, GRPO, reward-model and related trainers, including vLLM-backed online methods. Current verl documentation exposes FSDP/Megatron training with vLLM/SGLang/TGI rollout backends. Current vLLM exposes an offline `LLM` interface and sampling controls including max tokens, temperature, top-p and top-k.

D09 therefore keeps core project contracts independent of those frameworks and will add thin adapters only when the 12-6 model/checkpoint interfaces are stable enough. This avoids reimplementing an RL framework and avoids making S0 depend on heavy post-training runtimes.

## Tests in this package

The contract suite checks:

1. real behavioral training is denied without owner authorization;
2. post-training can never write into the canonical Base namespace;
3. contract-only experiments remain usable now without weight mutation;
4. external synthetic generators require owner-policy provenance;
5. fingerprints are deterministic;
6. duplicate verifier registration fails closed;
7. exact-match verifier behavior is deterministic;
8. rollout sampling parameters reject invalid ranges.

## Explicitly not implemented yet

- no SFT/DPO/GRPO/PPO/reward-model training execution;
- no behavioral datasets or instruction templates;
- no assistant persona, safety/refusal constitution or domain identity;
- no external teacher/critic calls;
- no vLLM/TRL/verl runtime adapter;
- no code-execution verifier or sandbox;
- no process-reward model;
- no Base weight save/update path;
- no paid compute.

Those remain future isolated packages gated by model maturity, exact checkpoint contracts, evaluation requirements and an explicit owner decision before behavioral weights are trained.
