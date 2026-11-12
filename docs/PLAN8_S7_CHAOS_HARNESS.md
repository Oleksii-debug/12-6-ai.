# Plan 8 Section 7 — fixture fault campaigns

`configs/plan8/chaos/component-recovery-fixtures-v1.json` declares eight
versioned cases: process kill, network loss, disk corruption, checkpoint
interruption, stale state, provider failure, node failure and interrupted
publication. Its `source_sha` identifies the accepted component fixture
baseline. The manifest contains exact expected recovery fingerprints.
Qualification separately identifies the exact checkout used to run the campaign.

`run_fault` publishes a complete, fsynced one-shot claim before injection.
Injection and recovery share a process lock. The operating system releases
the lock when a worker exits. After a crash, `recover_fault` observes the
claimed case without running the injection again. A recovery mismatch is
recorded as a terminal failure for that campaign.

`campaign_readback` takes a stable snapshot under the same process lock and
validates claim, success and failure records. It reports claimed cases that
still need recovery separately from unstarted cases.
The tests kill actual local processes, close a real local socket and preserve
observed partial bytes during checkpoint and publication rollback. The
previous accepted bytes stay readable and an unstarted rollback fails.
The provider and node cases exercise local
component fixtures. External outages and whole-product recovery require
separate evidence.

Run `pytest -q tests/test_plan8_chaos_harness.py` against the exact commit.
Physical campaigns and higher evidence levels require separate host receipts.
