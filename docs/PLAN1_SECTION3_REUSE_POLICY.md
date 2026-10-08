# Plan 1 / Section 3: reuse and Base-lineage gate (partial implementation)

The canonical model, checkpoint, scheduler and evaluator stay in their incumbent modules. This new admission policy neither installs dependencies nor loads weights nor registers a second authority.

All 13 inventory entries are deliberately CANDIDATE_UNQUALIFIED. This is NOT an assertion of verified versions, license rights or security reviews. Only a separately authenticated exact-source, licensed, reviewed code adapter may be promoted; dataset rights and model weight lineage require independent evidence. Canonical Base ancestry must be rooted in independently trusted local random initialization; the lineage validator itself is NOT authentication of checkpoint bytes.

Still required for terminal DONE: qualify actually imported assets against real source/version/license/security hashes, wire independent Base genesis verification to the canonical checkpoint ingress/egress, qualify exact frozen SHA, integrate and read back. Until then do not update terminal status.
