# DATA-105 Source / Domain Balance

Worker: `DATA-105-DOMAIN-BALANCE`

Base: `data25/corpus-v01-20260825@8af17afa7baf3d75c2328caf8b08af2400a95e09`.

## Live-data truth boundary

No DATA-101 branch or pull request exists in the reconstructed live repository. The strongest
retained DATA-25 corpus identity available to this worker is
`422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`.
Its retained report records 43,238 training documents and 20,000,775 training byte tokens:
9,000,418 UK, 7,000,137 EN, and 4,000,220 code. It also records zero training-eligible external
sources. The corpus builder uses one project-authored source ID per stratum.

Therefore the current corpus is appropriate for source-balancing mechanics but not for claiming
that a real multi-source UK/EN/code corpus has been empirically balanced. Within each stratum the
current top-source share is 1.0 and effective source count is 1.0 because there is only one source,
not because a measured real source family has crowded out competitors.

## Layering contract

DATA-105 does not replace the incumbent modality scheduler. `MixturePlan` continues to select the
top-level 45/35/20 UK/EN/code stratum. `SourceBalancePlan` receives that selected stratum and chooses
only among sources belonging to it.

The source/domain taxonomy is derived from provenance metadata only. It never asks a model to label
content. The priority is explicit `source_family_id`, then provenance URL host (plus owner for
GitHub/GitLab), then a deterministic project-authored source-ID rule, then `source_id`.

The analysis reports token mass by language/modality, source, source family/domain,
document-length bucket, and real-external versus project-authored origin. Concentration is reported
using top-source share, Hill/Simpson effective source count, token entropy in bits, and normalized
entropy.

## Candidate policies

`raw_proportional` uses source token mass directly.

`bounded_source_cap` caps each within-stratum source mass at 35% of the stratum token mass before
renormalized sampling.

`tempered_source_sqrt` uses exact integer square-root weighting, equivalent to exponent 1/2 without
platform-dependent floating-point policy weights.

All document draws are explicitly with replacement at sampling time. No duplicate corpus documents
are materialized and no provenance records are copied to fake source diversity.

## Fixed experiment

The dedicated LOCAL_FREE workflow trains three fresh 267,912-parameter D72/L4/H6 MHA Base models
from identical random initialization. Byte vocabulary 256, AdamW 3e-4, betas 0.9/0.95, eps 1e-8,
weight decay 0, clipping 1.0, fp32, batch 4, 64 causal targets per sequence, 96 optimizer steps and
24,576 optimized tokens are fixed across policies.

Held-out BPB is evaluated in aggregate and per provenance-derived source family/domain. A candidate
may not be promoted if a minority domain (training share below 25%) regresses by more than 0.03 BPB
against raw proportional sampling.

## Current recommendation rule

The checked-in machine policy remains `raw_proportional` and balancing promotion is blocked while
the bound corpus has zero real external training sources or fewer than two sources in any UK/EN/code
stratum. The cap and tempering implementations remain executable and tested so a later real
multi-source manifest can be evaluated without changing the balancing semantics.

No foreign pretrained weights, instruction tuning, paid compute, or broad capability claim is part
of DATA-105.
