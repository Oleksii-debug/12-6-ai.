# 12-6 AI

From-scratch language-model research project with own ModelSpec, tokenizer/data pipeline, random-initialized canonical Base weights, pretraining, evaluation, later post-training/reasoning, and a scaling ladder from ~10K to 1T total parameters.

Operational model: AUTOPULSE. GitHub exact SHAs, PRs, CI and permanent lane issues are live truth. Google Drive stores canonical research/context/backups.

Current stage: S0 — build and audit the ~10K-parameter end-to-end training factory.

Important: infrastructure libraries are reused; foreign pretrained weights are not the canonical Base. Paid compute requires explicit authorization.
