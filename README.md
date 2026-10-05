# 12-6 AI

From-scratch language-model research project with own ModelSpec, tokenizer/data pipeline, random-initialized canonical Base weights, pretraining, evaluation, later post-training/reasoning, and a scaling ladder from ~10K to 1T total parameters.

Operational model: AUTOPULSE. GitHub exact SHAs, PRs, CI and permanent lane issues are live truth. Google Drive stores canonical research/context/backups.

Current stage: **PRE-LEARNED-20M terminal qualification**. The ~20.6M-parameter random-init ModelSpec and data/training/checkpoint/evaluation mechanics exist, but a lawful launch-authoritative corpus, tokenizer-fit authorization, positive post-pack unique-loss budget, real learned-20M optimizer updates, learned weights and Windows release are **not yet qualified**. Historical ~10K/3M/10M experiments are laboratory evidence, not a trained 20M product. See [scientific control #548](https://github.com/Oleksii-debug/12-6-ai./issues/548) for exact live heads, physical-evidence status and next admissible scale gate.

Important: infrastructure libraries are reused; foreign pretrained weights are not the canonical Base. Paid compute requires explicit authorization.

## Binding accessibility architecture

Accessibility is a product constraint, not a reason to interrupt the current model-research/training critical path.

The command-line/text execution path remains first-class and no graphical interface is required merely to satisfy this rule. If a standalone Windows graphical operator interface is introduced, its binding end-state is **WebView2 + semantic HTML + a correctly exposed Windows UI Automation host**, with keyboard-only and NVDA operation. Important status, training state, checkpoint identity, errors, evidence and controls must be real selectable/copyable text with semantic roles, names, state and deterministic focus.

This does **not** require rewriting the tokenizer, data, training, checkpoint, evaluation, inference or scaling core. Keep those components presentation-neutral so a Windows shell is an adapter over the same canonical runtime.

Visual layout/polish may come later. NVDA_VERIFIED requires physical keyboard-only acceptance on the exact packaged Windows candidate if or when such a GUI exists.
