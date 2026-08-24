# C01 / D13 Compute, Cost and Run-Queue Plan

Status: engineering control package only. It does not promote a model, grant audit authority, or authorize materially paid compute.

## Live authority cutoff — 2026-08-24

- Physical repository: `Oleksii-debug/12-6-ai.`; the trailing dot is part of the repository name.
- Canonical `main`: `f2e94c7212888cdb960bb66154d56d210e9b27ab` at this cutoff.
- Primary measured training evidence: PR #60 exact head `403831cf623120da18f0f4c62e830a352afcef91`.
- PR #60 exact-head CI `32745954583`: SUCCESS.
- PR #60 exact-head real-training workflow `32745954611`, job `97491395143`: SUCCESS.
- AUDIT-A and AUDIT-B durable verdicts remain `CHANGES_REQUIRED`.
- Materially paid compute is not authorized.

Older C01 PR #19 is historical control infrastructure. Its old 259-vs-256 vocabulary blocker is obsolete: canonical S0 raw-byte vocabulary is 256.

## Current measured S0 throughput

PR #60 executes the real 10,140-parameter D01 random-init Base with D03/D04 data/tokenizer and the D02 trainer on hosted CPU:

- 40 optimizer steps;
- 10,833 optimized tokens;
- wall time `0.166512882 s`;
- measured throughput `65,058.0296 optimized tokens/s`;
- train loss `5.557927956 -> 2.798575868`;
- validation loss `5.559258974 -> 2.790725003`;
- validation optimized tokens = 0;
- evidence SHA-256 `000de6556c08c94604e6c26a364ff11079a3800fb8d153fcdaf07523d1d32c50`.

The previous exact-green PR #60 head `84d767...` measured about 128,755 tok/s, but it is superseded. The large variance itself is evidence that tiny-S0 wall throughput is not a scaling predictor. No S1+ wall-clock estimate may linearly extrapolate this number; each hardware class must be benchmarked directly.

## Additional exact-green evidence available now

The queue records these as `EVIDENCE_AVAILABLE_NOT_COMPOSED`. Green evidence on another SHA is not launch authority until one exact integration candidate selectively composes it and reruns CI/evaluation.

- PR #61 `fa5e8365ae56c22da761e2183a4c71a0212a2aef`, CI `32745892911` SUCCESS: real uninterrupted versus checkpoint/destroy/restore/resume S0 trajectory, exact identity binding, D07 generation parity, repo-wide 197 tests PASS.
- PR #63 `9a1a6f8494b5d5dd9346afb644fc84de2e39d5e0`, CI `32745879308` SUCCESS: verified first-party S0 checkpoint loader/inference parity, repo-wide 171 tests PASS.
- PR #65 `cc1a7aa27b431326766b0172460b015b0e20e135`, CI `32746626964` SUCCESS and D06 evaluation run `32746627421` SUCCESS: real S0 evaluation 15/15 gates PASS, evaluation complete, save/load and exact resume verified, train loss `5.562200927 -> 2.899044275`, validation loss `5.554580927 -> 2.852916241`. The same report explicitly has `candidate.integrated=false`, `promotion_eligible=false`, and `promotion_authority_status=NOT_TESTED`.

## Exact run queue

`configs/runs/c01_s0_run_queue.v3.json` separates three phases:

1. `engineering_validation`: static/repo tests, measured training, deterministic repeat, checkpoint/reload/inference, resume contract, held-out eval, and future exact composed resume+D06 acceptance;
2. `scale_experiment`: optional profiling only on genuinely local/unmetered hardware;
3. `production_like_training`: S1+ material runs, all fail-closed until separately authorized.

Every run contains a candidate SHA when known, explicit artifact locations, cancellation criteria, failure criteria and retry policy. Candidate movement invalidates launchable references until ModelSpec, InitSpec, data, tokenizer, vocab, packing, training config, artifact hashes and CI IDs are rebound.

Current control state:

- `S0-E01-REAL-TRAIN-403831`: `COMPLETED_EVIDENCE`.
- S0 repeat/checkpoint/resume-contract/held-out/inference jobs on `403831...`: `READY_LOCAL_FREE`.
- `S0-E05-REAL-RESUME-COMPOSE`: `PREPARED_BLOCKED`; PR #61 evidence exists but is not in the primary candidate ancestry.
- `S0-E07-D06-COMPOSE`: `PREPARED_BLOCKED`; PR #65 has 15/15 real evaluation PASS but is not an integrated/promotion-authorized candidate.
- local GPU profile: `PREPARED_BLOCKED` until an actually local/unmetered device is established.
- `S1-PROD-TRAIN`: `PREPARED_NOT_LAUNCHED`.

## S1-S14 compute/storage planning

Machine-readable estimates live in `configs/runs/c01_stage_compute_plan_s1_s14.v1.json`.

Parameter references are planning references, not frozen stage authority:

- S1-S3 use D01 PR #24 candidate counts: 107,856; 1,066,112; 10,059,840.
- S4-S7 use D01 PR #37 non-frozen candidates: 100,384,512; 400,598,016; 999,106,560; 2,998,029,312.
- PR #67 now exposes alternative non-frozen S1-S4 architectures. If the integrator selects an alternative, D13 must regenerate the affected memory/FLOP rows before launch.
- S8-S14 use Drive engineering target scales because exact ModelSpecs are not frozen.

Planning formulas:

- BF16 model-only checkpoint = `2 * total_parameters` bytes.
- Conservative mixed-precision Adam state = `16 * total_parameters` bytes: BF16 weights 2B + gradients 2B + FP32 master weights 4B + FP32 moments 8B. Activations and temporary buffers are extra.
- Aggregate HBM lower bound reserves 30% headroom: `16*N / 0.70`.
- Training compute approximation = `6 * active_parameters * training_tokens` FLOPs.
- Inference approximation = `2 * active_parameters` FLOPs/token.
- S13 planning placeholder = 300B total / 75B active; S14 = 1T total / 50B active. Dense-total FLOPs remain separately recorded as upper bounds. Exact MoE topology is not frozen.

The plan includes cheap/balanced/fast token scenarios for every S1-S14 stage. State-only GPU counts for nominal 24GB/80GB/192GB devices are lower bounds, not runnable topologies. PR #74 is actively developing D12 DP/TP/PP/CP/EP and reshard contracts; its topology work must be integrated before later-stage device counts are converted into an executable distributed launch plan. No GPU/NCCL/multi-node benchmark exists yet.

## Cost boundary

No provider/SKU/region/reservation quote is bound and paid compute is explicitly unauthorized. Therefore this package does not invent a dollar figure. The eventual cost equation is `measured device-hours * current quoted rate`, after exact hardware topology, measured throughput, checkpoint/storage traffic, provider quote and explicit authorization are recorded.

## Validation

`tools/validate_c01_compute_plan.py` fails closed on stale primary SHA, external-evidence composition overclaim, paid launch readiness, missing artifact/failure/retry controls, throughput arithmetic drift, missing S1-S14 rows, and memory/FLOP formula drift.

`tests/test_c01_compute_plan.py` exercises both positive and negative paths. Exact-head GitHub Actions on PR #70 is the authoritative validation environment; queued or running CI is never called PASS.
