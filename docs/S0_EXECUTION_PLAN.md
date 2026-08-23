# S0 ~10K EXECUTION PLAN

Goal: prove the entire 12-6 factory end-to-end, not model quality.

Required output: random-init causal model near target parameter budget; exact parameter report; tiny deterministic corpus/byte or tiny tokenizer path; training loss decreases on learnable toy data; generation runs; checkpoint save/load; interrupted run resumes; run manifest/hashes; CI; independent audit.

Parallel lanes:
D01 architecture/config/count; D02 training/numerics; D03 provenance/tiny corpus; D04 tokenizer/packing; D05 checkpoint/export; D06 eval/gates; D07 generation/local test; D08 scale-ready trainer interfaces/simulation; D09 isolated future posttraining harness only; D10 CI/integration. R01 reviews design, C01 maintains run queue, AUDIT-A/B continuously audit committed evidence.

S0 is promoted only when the integrated exact SHA passes both audits.
