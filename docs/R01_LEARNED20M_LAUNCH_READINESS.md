# R01 learned-20M launch readiness

Issue: #654. This hardening is stacked on clean PR #714.

This package converts the learned-20M critical path into one fail-closed integration gate. It does not implement DATA, tokenizer, checkpoint, evaluation, optimizer, audit or compute authorities; it consumes their terminal evidence only after those lanes produce it.

The evaluator deliberately exposes three different decisions:

1. `ready_for_local_free_pilot` means the exact model/code/data/tokenizer/loss-ledger/checkpoint/evaluation/training-recipe prerequisites are bound strongly enough for a bounded no-material-cost pilot.
2. `ready_for_compute_authorization_request` additionally requires terminal bounded-pilot evidence, a positive cost envelope and an independent audit. This is permission to ask for material compute, not permission to spend it.
3. `material_training_authorized` additionally requires both a durable explicit `COMPUTE_AUTHORIZED` authority whose maximum budget covers the estimated maximum cost and a separate durable `TRAINING_AUTHORIZED` authority bound to the exact training config and compute authorization.

No lower phase implies a higher phase. Financial authorization and permission to execute the exact learned-20M recipe are distinct authorities.

## Exact incumbent authorities

The gate binds the current MODEL-341 mechanics control exactly: branch `model341/20m-candidate-a-20260826`, SHA `e4ff486fd90802fc123bebf60eed4e59196a98df`, ModelSpec SHA-256 `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, and 20,613,440 random-init parameters.

It binds both the merged R01 campaign commit `a73ab38026cb7849f478cc13ad58b93534a76e2f` and the campaign config Git blob SHA-1 `c50154db609d41eceb2ffc97912360df567bcc04`. A copied campaign blob on unrelated ancestry therefore cannot silently become the launch-plan authority.

The exact training-code SHA is also insufficient by itself. It must carry a terminal `qualified_integration_head` authority backed by a successful workflow on that same exact head.

## Evidence contract

Terminal scientific authorities are typed, machine-addressable GitHub evidence references. Every reference must carry the exact repository, evidence kind, Git SHA, observed head SHA, evidence SHA-256, terminal state, and an explicit non-self-asserted marker. Gates backed by Actions additionally require a positive workflow run ID, `workflow_conclusion=success`, and `workflow_run_head_sha` equal to the cited Git SHA.

That typing matters: a successful corpus workflow cannot be reused as D05 checkpoint authority merely because it has a valid run ID and digest. A stale workflow whose success belongs to a previous head also fails closed.

Independent-audit evidence additionally requires distinct producer and verifier identities. The evaluator does not infer independence from the word `audit` or from a generic successful workflow.

The current checked-in packet is intentionally blocked. Null identities and zero unique loss positions are the truthful state until successor lanes populate exact terminal evidence.

A structural evidence reference is not a substitute for live GitHub reconciliation. The integration worker that fills a future launch packet must verify each referenced head/run/artifact against GitHub before publication. This module makes stale, untyped and self-asserted references structurally invalid; it does not fabricate network verification.

## Data-budget boundary

The gate does not convert source bytes, raw tokenizer tokens or replayed positions into training capacity. It consumes a separate terminal `data_budget_authority` and requires `data_budget_status=QUALIFIED` plus a positive exact post-pack unique causal-loss-position ledger.

The training recipe must also declare `requested_unique_loss_positions > 0`, and that request cannot exceed the exact ledger. This closes the seam where a valid corpus ledger could coexist with a recipe asking for more unique positions than actually exist.

The active R01 scaling-data-budget lane remains the owner of the numerical sufficiency policy.

## Compute and training boundary

The package executes no training and no paid compute. Cost estimation, financial compute authorization, and authorization to execute the exact learned-20M run are deliberately separate.

`COMPUTE_AUTHORIZED` must use scope `LEARNED_20M_MATERIAL_COMPUTE`, be explicitly owner-approved, and cover at least the estimated maximum cost. It still cannot make `material_training_authorized=true` by itself.

A separate `TRAINING_AUTHORIZED` authority must use scope `LEARNED_20M_MATERIAL_TRAINING`, be explicitly owner-approved, bind the exact training-config SHA-256, and reference the exact compute-authorization evidence SHA-256. This prevents a budget approval for one packet from becoming permission to run a different recipe. A general owner message such as `continue` satisfies neither authorization.

## Usage

From a repository checkout:

```bash
PYTHONPATH=src python tools/assess_r01_learned20m_launch_readiness.py
```

Exit code `0` is reserved for a packet that reaches `material_training_authorized=true`. A correctly blocked current packet exits `1` and prints exact blockers as JSON.
