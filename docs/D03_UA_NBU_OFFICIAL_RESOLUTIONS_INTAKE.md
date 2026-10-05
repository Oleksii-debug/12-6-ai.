# D03 — NBU official resolutions discovery intake

## Purpose

Open one collision-safe, high-yield Ukrainian source lane on the learned-20M data-capacity critical path: official resolutions of the Board of the National Bank of Ukraine.

The live NBU legislation catalogue exposes a large regulatory collection (the public catalogue reports thousands of documents and more than two thousand resolutions). Those counts are discovery signals only. They are **not** corpus capacity, immutable source authority, training bytes, tokenizer tokens, or optimized loss positions.

## Scope and rights boundary

Candidate family: `ua.nbu.official-resolutions`.

Only an exact page whose title begins with `Постанова Правління Національного банку України` and whose official PDF is served from the NBU `admin_uploads/law/` path can enter discovery evidence. The rights candidate is limited to the exact official resolution text under the project's existing Article 8(1)(3) official-act interpretation of Ukrainian Law No. 2811-IX. This package does not claim a site-wide license and does not make a legal conclusion.

News, consultations, explanatory text, images/video/design, third-party material, control copies, consolidated texts, and non-resolution attachments receive no authority from this package.

## Fail-closed discovery contract

The stdlib-only validator accepts only HTTPS `bank.gov.ua` canonical identities; restricts document paths to the frozen `Resolution_DDMMYYYY_NUMBER` shape; requires two independent catalogue discoveries and two independent page inspections; requires the exact page title and at least one same-origin official PDF path; strips cache-busting query/fragment components from identity; emits only body-free URL/hash discovery evidence; hard-bounds discovery at 120 documents; and rejects any attempt to create byte credit or training authority.

No dedicated workflow is added. Shared CI is sufficient for the network-free contract/adversarial tests, avoiding extra Actions fanout while the project has runner pressure.

## Required successor

Discovery alone is not an admitted source. Before any byte can count, a successor must pin the exact PDF bytes, materialize the exact official text, confirm document-level scope, run quality/privacy, and then feed survivors through the incumbent global exact/near/fragment/lineage dedup, reserved-evaluation decontamination, balance/family caps, cluster-safe split, deterministic tokenizer/packing two-clean-build proof, and positive exact unique causal-loss ledger.

Until those gates are terminal: canonical capacity credit = 0; training-authorized bytes = 0; tokenizer fit = unauthorized; optimizer updates = 0; final-test access = false; paid compute = false; learned-20M claim = false.
