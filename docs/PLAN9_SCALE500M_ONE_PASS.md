# SCALE-500M one-pass preparation

This candidate describes one complete pass over the accepted Plan 2 local release. It binds the 494,204,160-parameter scratch profile and the accepted Plan 3 runtime. The physical stream has 71,256 blocks of width 32. The recipe uses two blocks per microbatch and 12 microbatches per optimizer update, yielding 35,628 microbatches and 2,969 complete updates. Its exposure limit is exactly 1,985,122 unique loss-bearing positions. The ordered stream uses no replay and the final test stays reserved. This is a first bounded campaign candidate. Its data volume and sequence length establish neither broad learned quality nor a production champion.

The recipe uses the native FP32 AdamW defaults for the learning rate, betas, epsilon and clipping. It uses cosine scheduling with 100 warmup updates and the canonical profile seed. The budget receipt was independently rebuilt from both accepted release materializations and checked seven source families and 22 notices.

From this exact Git revision, set PYTHONPATH to the repository and its src directory. On Windows PowerShell:

```powershell
$env:PYTHONPATH = 'src;.'
$release = '<verified local Plan 2 release>'
$recipe = 'configs/plan9/scale500m_one_pass_recipe_v1.json'
$lock = 'requirements/locks/windows-x86_64/runtime.lock.txt'
python -m tools.plan9_scale500m_launch_preparation_v1 --release $release --recipe $recipe --check-one-pass-budget
python -m tools.plan9_scale500m_launch_preparation_v1 --release $release --recipe $recipe --build-packet --environment-lock $lock --run-id scale500m-local-one-pass-candidate-v1
python -m tools.plan9_scale500m_launch_preparation_v1 --release $release --inspect-native-launch '<candidate packet JSON>' --packet-sha256 '<independently pinned packet SHA-256>' --checkpoint-root '<target checkpoint volume>' --device cpu
```

The packet command hashes the input lock bytes. Use the committed LF file, or a checkout with line-ending conversion disabled. Its SHA-256 is `a94337bafdfc74ad2779d3c05b2f315048c1c39200a3011825ac6ebdff7b628f`. An inspection denial returns exit code 2 and a structured receipt. A successful capacity screen still leaves full measurements and campaign acceptance pending.

For preparation on an external single CUDA device, the following commands bind
the committed Linux lock and report resource floors for one full FP32 AdamW replica.
These commands only read the repository and the verified data release.

```powershell
python -m tools.plan9_scale500m_launch_preparation_v1 --release $release --build-external-gpu-packet linux-x86_64 --run-id scale500m-external-gpu-general-v1
python -m tools.plan9_scale500m_launch_preparation_v1 --release $release --verify-external-gpu-packet '<saved packet JSON>' --packet-sha256 '<independently pinned packet SHA-256>'
```

The preparation order is 500M, 1B, then 2B. This general baseline measures the
interfaces against the already accepted corpus. The future cyber-focused campaign
requires a separately admitted data release and independent evaluation. The external
host still needs measured device capacity, runtime lock verification and checkpoint
recovery evidence.

The native runtime publishes a checkpoint at each committed optimizer boundary. The initial weights and all 2,969 checkpoints need at least 18,107,765,944,320 bytes under its current storage policy. An accepted host must be measured for training, checkpoint I/O and restart. The controller fails if it cannot commit a reserved optimizer step, encounters exhaustion, or exceeds the exposure ceiling. For the full campaign, separately pin the operator's stop criteria and execution authority before training. A paused checkpoint retirement prototype has been tested locally on tiny checkpoints. It does not change the active storage admission policy.

[Incumbent Plan 9 PR](https://github.com/Oleksii-debug/12-6-ai./pull/3237)
