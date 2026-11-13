# SCALE-500M external cybersecurity pilot

The pilot checks four optimizer updates, with a durable pause after update
two and continuation in a new process. It reads the first 96 ordered blocks
of the frozen cybersecurity candidate. The 2,696 active targets include
2,522 cybersecurity positions. This prefix is sized for runtime qualification.

Use a checkout of the exact source commit. The planning command runs on
the local machine with no CUDA requirement. It verifies the two physical
shards, source notices, tokenizer, recipe, runtime and checkpoint identity.

```bash
export PYTHONPATH=src:.
python -m tools.plan9_external_cyber_pilot_v1 plan --bundle "$LOCAL_FALLBACK" > pilot-packet.json
python -m tools.plan9_external_cyber_pilot_v1 stage-input \
  --bundle "$LOCAL_FALLBACK" --stage-dir "$LOCAL_STAGE" > pilot-stage-receipt.json
```

Staging copies two exact shards, 128 blocks in total, and pinned metadata
and notices. It writes the byte inventory to `pilot-input-manifest.json`.
Planning from the staged directory must reproduce the original packet.
The tokenizer is the frozen artifact in the checked-out source.
The source decision identifies the exact scope and unresolved external
processing facts. Use a host and transfer scope that satisfy those facts.

For a future external run, first qualify SCALE-500M with the
[external CUDA benchmark](PLAN3_PORTABLE_TRAINING_PACKET.md). Use the
same pinned Linux environment, selected GPU and checkpoint volume.
Save the benchmark's measured JSON receipt and hash the file. The pilot checks
the receipt against the exact source, model, runtime and installed lock. Both
commands use the same physical GPU and checkpoint filesystem. If the filesystem
does not provide an ID, the launcher compares its device number within one boot.
The four-update pilot has a 26,537,477,120-byte unpruned payload floor.
The launcher also asks for 1 GiB of space above that floor. The arithmetic
restore envelope requires 27,611,218,944 host bytes and 14,341,032,960 free CUDA
bytes. Workload and CUDA overhead still require measurement.

On the external host, use an existing checkpoint directory and the digest
from the locally verified packet. Prepare creates checkpoint zero. Pass
the resulting receipt digest to the first run. Pass the paused receipt
digest to the second run, in a fresh process. Use `status` to recover a
lost receipt. Use `--recover` only for an interrupted RUNNING or FAILED
receipt. The run reads the physical cursor bound to the last checkpoint.

```bash
pilot=(python -m tools.plan9_external_cyber_pilot_v1)
common=(--bundle "$STAGED_INPUT" --scratch "$CHECKPOINT_DIR")
execution=(--execute --expected-packet-sha256 "$PACKET_SHA"
  --install-report "$INSTALL_REPORT" --install-report-sha256 "$INSTALL_REPORT_SHA"
  --benchmark-receipt "$BENCHMARK_RECEIPT" --benchmark-receipt-sha256 "$BENCHMARK_RECEIPT_SHA")
"${pilot[@]}" prepare "${common[@]}" "${execution[@]}" > initial.json
"${pilot[@]}" run "${common[@]}" "${execution[@]}" \
  --expected-receipt-sha256 "$INITIAL_RECEIPT_SHA" > paused.json
"${pilot[@]}" run "${common[@]}" "${execution[@]}" \
  --expected-receipt-sha256 "$PAUSED_RECEIPT_SHA" > completed.json
"${pilot[@]}" verify-checkpoint "${common[@]}" \
  --expected-receipt-sha256 "$COMPLETED_RECEIPT_SHA" > checkpoint-proof.json
```

This path uses one visible CUDA GPU, selects `cuda:0`, and retains all five
checkpoints. The expected durable counts are 1,303 active positions at
the pause and 2,696 at completion. Measured host evidence and an actual
pilot receipt are still required. The full campaign has a separate
recipe, evaluation and checkpoint cost.

The command below performs the same four operations in separate processes
and writes the available receipts into an existing results directory. If
interrupted, rerun it against the same scratch directory and packet.

```bash
python -m tools.run_plan9_external_cyber_pilot_v1 --execute \
  --bundle "$STAGED_INPUT" --scratch "$CHECKPOINT_DIR" --output "$RESULTS_DIR" \
  --packet-sha256 "$PACKET_SHA" --install-report "$INSTALL_REPORT" \
  --benchmark-receipt "$BENCHMARK_RECEIPT" --benchmark-receipt-sha256 "$BENCHMARK_RECEIPT_SHA"
```
