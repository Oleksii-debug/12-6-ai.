# Plan 8 Section 7 — fixture fault campaigns

`configs/plan8/chaos/component-recovery-fixtures-v1.json` declares eight
versioned cases: process kill, network loss, disk corruption, checkpoint
interruption, stale state, provider failure, node failure and interrupted
publication. Its `source_sha` pins the original Q0 fixture candidate. The
manifest has exact expected recovery fingerprints and no external host grant.

`run_fault` writes an exclusive, fsynced one-shot claim before injection.
After a crash, call `recover_fault` with an independent observation; never
rerun an already claimed effect. A mismatch leaves a durable failure receipt.
`campaign_readback` checks claim and receipt bytes and returns passed,
missing and failed cases. Missing cases and forged receipts cannot become a
complete campaign. The process-kill test uses an actual child process; other
cases are controlled component fixtures. Q0 evidence does not assert real
network/provider outages, physical node failure or final product recovery.

Run `pytest -q tests/test_plan8_chaos_harness.py` against the exact commit.
Physical campaigns and higher evidence levels require separate host receipts.
