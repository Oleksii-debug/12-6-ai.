# D09 Post-Training Boundary

Status: infrastructure-only. This document does not authorize behavioral post-training.

## Invariant

Early canonical checkpoints are 12-6 Base. D09 tooling may consume a Base checkpoint only as an immutable parent reference. Any experiment that writes post-training weights must produce a distinct `posttrain` lineage and must never overwrite, relabel, or silently replace the Base artifact.

The current Base lineage must not include behavioral alignment weights, instruction tuning, ethics/refusal policy, assistant personality, medical/engineering specialization, or another domain identity unless the owner later makes an explicit decision enabling that work.

## Stable contracts

`src/twelve_six/posttraining/contracts.py` defines framework-neutral checkpoint, dataset, candidate, verifier and experiment identities. `PostTrainingExperiment` fails closed if its requested output lineage is `base`. It also defaults `training_enabled=false`; any declaration with `training_enabled=true` requires an explicit `behavioral_training_authorization_id` referencing the owner's decision. This schema guard does not replace Coordinator/Auditor validation of that authorization.

Materially paid compute additionally requires an external compute-authorization reference; D09 does not grant that authorization itself.

`src/twelve_six/posttraining/provenance.py` defines deterministic manifest hashing. Synthetic rows must be explicitly marked and carry a generator identity. Parent hashes can record teacher/candidate ancestry without mixing generated data into an unlabelled corpus. An external synthetic generator/teacher additionally requires an explicit `owner_policy_ref`; merely naming an external model does not authorize its use.

`src/twelve_six/posttraining/verifiers.py` defines a minimal verifier protocol and deterministic exact-text/numeric baselines. These are plumbing tests, not claims of reasoning capability. Future math/code/logic verifiers should implement the same protocol or a versioned successor.

`src/twelve_six/posttraining/interfaces.py` is the integration boundary for mature external stacks. Implementations should wrap TRL, verl, vLLM-backed rollout infrastructure, or another reviewed framework instead of recreating full RL/distributed-training engines here.

## Dataset semantics

Each post-training record must have an immutable `record_id`, a declared `RecordKind`, an explicit train/validation/test split, payload fields, and provenance. Test/evaluation records are not authorized training inputs merely because a serializer can read them. D06 retains evaluation/stage-gate ownership; D03 retains source/data provenance ownership for pretraining corpora.

## Integration boundary

D10 may integrate this infrastructure as reusable code when audited, but current Base candidate composition must exclude any D09-produced behavioral weights. A future owner decision enabling post-training should name the parent checkpoint, algorithm/config, dataset manifest, verifier set, compute authorization state, output lineage, and promotion criteria explicitly.

## Current smoke config

`configs/posttraining/isolated_verifier_smoke.json` sets `training_enabled=false` and leaves both behavioral-training authorization and external-synthetic-generator permission unset. It exists only to exercise schema/integration assumptions. It is not a training authorization.
