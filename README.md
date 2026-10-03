# 12-6 AI

From-scratch language-model research project with own ModelSpec, tokenizer/data pipeline, random-initialized canonical Base weights, pretraining, evaluation, later post-training/reasoning, and a scaling ladder from ~10K to 1T total parameters.

Operational model: AUTOPULSE. GitHub exact SHAs, PRs, CI and permanent lane issues are live truth. Google Drive stores canonical research/context/backups.

Current stage: S0 — build and audit the ~10K-parameter end-to-end training factory.

Important: infrastructure libraries are reused; foreign pretrained weights are not the canonical Base. Paid compute requires explicit authorization.

## Binding accessibility architecture

Accessibility is a product constraint, not a reason to interrupt the current model-research/training critical path.

The command-line/text execution path remains first-class and no graphical interface is required merely to satisfy this rule. If a standalone Windows graphical operator interface is introduced, its binding end-state is **WebView2 + semantic HTML + a correctly exposed Windows UI Automation host**, with keyboard-only and NVDA operation. Important status, training state, checkpoint identity, errors, evidence and controls must be real selectable/copyable text with semantic roles, names, state and deterministic focus.

This does **not** require rewriting the tokenizer, data, training, checkpoint, evaluation, inference or scaling core. Keep those components presentation-neutral so a Windows shell is an adapter over the same canonical runtime.

Visual layout/polish may come later. NVDA_VERIFIED requires physical keyboard-only acceptance on the exact packaged Windows candidate if or when such a GUI exists.
