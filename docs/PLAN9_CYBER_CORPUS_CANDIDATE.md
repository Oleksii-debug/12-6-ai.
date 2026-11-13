# Cybersecurity corpus development candidate

The current development mixture uses 75% cybersecurity and 25% general
causal loss positions. The percentage is a preparation choice and can be
revised with a new version. CISA KEV and NIST SP 800-53 Rev. 5 supply the
cybersecurity material. The general subset uses the exact accepted Plan 2
data and inherits its local use scope. The derived bundle keeps all original
general rights files and the two cybersecurity source licenses.

The first receipt pins 22,833 packed blocks and 669,079 unique causal
positions, of which 501,812 are cybersecurity and 167,267 are general.
Two independent builds produced the same 387 files. Training documents
are excluded by exact normalized text overlap and protected general
evaluation hashes. CISA vendor groups and NIST control families stay in
one split each. The same-source holdout screen checks identifiers,
paragraphs and 8-grams and retains 172 validation and 94 test records.
This smaller dataset verifies the pipeline. A broader corpus, broader
behavioral evaluation, and admission to a future GPU campaign remain open.

Use local directories for the input data, licenses and output artifacts.
The repository receives the code and receipts only.

1. Download CISA `known_exploited_vulnerabilities.json` and `LICENSE` at
   [the frozen CISA commit](https://github.com/cisagov/kev-data/tree/6b8d7ba579672e4f84e2b1f255724aa97c8455b4).
   Name the local files `cisa-kev.json` and `cisa-license.txt`.
2. Download `nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json`
   and `LICENSE.md` at [the frozen NIST commit](https://github.com/usnistgov/oscal-content/tree/bc8a528770033611df899b3d52703fb3dc91a20d).
   Name the local files `nist-800-53.json` and `nist-license.txt`.
3. Download `enterprise-attack/enterprise-attack.json` and `LICENSE.txt`
   into a separate evaluation directory from [the frozen MITRE commit](https://github.com/mitre-attack/attack-stix-data/tree/f59b03ecf828579d5f9e93683ce25b7e2e5fd9d7).
   The 154 independently sourced development cases score extraction of
   mitigation identifiers for techniques across 15 primary tactics. They
   exercise knowledge retrieval. Some technique names appear in training.
   This suite stays outside the training bundle.

```powershell
$env:PYTHONPATH = 'src;.'
python -m tools.plan9_prepare_cyber_candidate_v1 --sources '<CISA and NIST>' --general-train '<S16 train.jsonl>' --general-release '<accepted S18 directory>' --out '<new corpus directory>'
python -m tools.plan9_cyber_holdout_audit_v1 --candidate '<new corpus directory>' --sources '<CISA and NIST>' --expected-candidate-receipt-sha256 aa49940f5419e5cda564c35baea3427b9c44c5037ed4b85807a4111f0c10563a --out '<new screened holdout>'
python -m tools.plan9_cyber_independent_eval_v1 build --source '<MITRE directory>' --corpus '<new corpus directory>' --out '<new evaluation directory>'
python -m tools.plan9_cyber_independent_eval_v1 score --cases '<evaluation directory>/cases.jsonl' --cases-sha256 5117a6523f3d257bd762de027e8f81bc63f1498457279b5c7cd76a3053d2dc23 --predictions '<candidate predictions JSONL>'
python -m tools.plan9_scale500m_launch_preparation_v1 --prepare-cyber-gpu-profiles '<new corpus directory>'
```

On Linux, set `PYTHONPATH=src:.` instead of the PowerShell assignment.
The profile check opens training data and rights notices and verifies every
packed shard. It uses the committed holdout and evaluation receipts without
opening evaluation payloads. The ordered profiles are SCALE-500M, SCALE-1B
and SCALE-2B. An example arithmetic schedule covers all 22,833 blocks with
three blocks per microbatch and three microbatches per update, for 2,537
updates. This schedule is not an approved recipe. Actual GPU capacity,
checkpoint throughput, data release, external transfer and campaign
authority are separate gates.

Predictions are JSONL objects with exactly `id` and `prediction` string
fields. Provide one prediction per case. The scorer extracts and
deduplicates mitigation IDs case-insensitively, then macro-averages set
precision, recall, F1 and exact match across cases. It also reports each
tactic. Scoring is offline and does not call a model.
