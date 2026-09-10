#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

MANIFEST = Path('configs/evaluation/eval303_selection_validation_composite_v1.json')
MEMBERSHIP = Path('data/evaluation/eval303/selection-validation/composite-membership.jsonl')
PROOF = Path('evidence/eval303/data300-exact-exclusion-proof-v1.json')

EXPECTED = {
    'selection_identity_sha256': '7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd',
    'membership_sha256': 'e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6',
    'proof_identity_sha256': 'ac9a0e2c3beab26c0d664b0006b11ec9fd155fa78be9f46d56ecb3ed336f2621',
    'proof_file_sha256': 'c80114fef670447efae54f8dcc70fcfb73ec9b01299ea15d8a21875456d049ef',
    'data300_contract_identity': '07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5',
    'data300_head_sha': '8ea7f830e50a23754d189dd4134f4afad76a7ee9',
    'data300_contract_blob_sha1': '39d4fa07ea17e66e042a3ccb1a55b8e5e1c5d7bf',
    'eval233_head_sha': 'b5512b4648cb09dd052b08884dc53f291e1ce935',
    'component_heads': {
        'ua': '029514654829cebc149cff6fc1fea2a8ba4fa566',
        'en': 'fb268061300127b62cc2a262664b30c614559dac',
        'code': '2cbe2f2d9c74984baa69e49e520e2280fc76421b',
    },
    'component_ids': {
        'ua': 'c32320a706a283049e35eb537eb20a1e7f5865b86c24397c8b73d1e3d2014164',
        'en': '727f229c091f86748a4eee9ea5aec72bb65347b68d6b687fabbf33166b0eca1e',
        'code': '9fd52e879c388f06f0b103afa02d68678388867c81cfb0f27ddbf0ca18867054',
    },
    'records': 10,
    'families': {
        'github:encode/httpx': 1,
        'github:psf/requests': 1,
        'kubernetes.website.docs': 4,
        'lang-uk.perestoroha-ocr': 4,
    },
}


class Eval303ValidationError(RuntimeError):
    """Fail-closed EVAL-303 immutable-authority validation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Eval303ValidationError(message)


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(obj: dict, field: str) -> str:
    clone = deepcopy(obj)
    clone.pop(field, None)
    return sha256_bytes((canonical(clone) + '\n').encode('utf-8'))


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(value, dict), f'{path} must contain a JSON object')
    return value


def load_records(path: Path) -> list[dict]:
    records = []
    raw = path.read_bytes()
    _require(not raw or raw.endswith(b'\n'), f'{path} must end with LF')
    for index, line in enumerate(raw.decode('utf-8').splitlines(), start=1):
        record = json.loads(line)
        _require(isinstance(record, dict), f'{path}:{index} must be a JSON object')
        _require(line == canonical(record), f'{path}:{index} is not canonical JSON')
        records.append(record)
    return records


def verify(repo_root: Path) -> dict:
    manifest_path = repo_root / MANIFEST
    membership_path = repo_root / MEMBERSHIP
    proof_path = repo_root / PROOF
    manifest = load_json(manifest_path)
    proof = load_json(proof_path)
    records = load_records(membership_path)

    _require(manifest.get('schema_version') == '12-6.eval303-selection-validation-composite.v1', 'manifest schema drift')
    _require(manifest.get('worker_id') == 'EVAL-303-SELECTION-VALIDATION-COMPOSITE', 'manifest worker drift')
    _require(manifest.get('purpose') == 'selection-validation', 'manifest purpose drift')
    _require(manifest.get('execution_profile') == 'LOCAL_FREE', 'execution profile drift')
    _require(manifest.get('selection_identity_sha256') == EXPECTED['selection_identity_sha256'], 'selection identity drift')
    _require(self_identity(manifest, 'selection_identity_sha256') == EXPECTED['selection_identity_sha256'], 'selection self-identity mismatch')

    membership_raw = membership_path.read_bytes()
    _require(sha256_bytes(membership_raw) == EXPECTED['membership_sha256'], 'membership file SHA-256 drift')
    composite = manifest.get('composite_membership', {})
    _require(composite.get('sha256') == EXPECTED['membership_sha256'], 'membership authority SHA-256 drift')
    _require(composite.get('bytes') == len(membership_raw), 'membership byte-count drift')
    _require(composite.get('documents') == EXPECTED['records'] == len(records), 'membership document-count drift')
    _require(composite.get('payload_bytes_copied_into_eval303') is False, 'EVAL-303 unexpectedly copied selection payload bytes')

    content_hashes = []
    strata = Counter()
    families = Counter()
    record_ids = set()
    for index, record in enumerate(records):
        record_id = record.get('record_id')
        _require(isinstance(record_id, str) and bool(record_id), f'record[{index}] record_id invalid')
        _require(record_id not in record_ids, f'duplicate record_id: {record_id}')
        record_ids.add(record_id)
        _require('text' not in record, f'record contains forbidden payload text: {record_id}')
        content_hash = record.get('content_sha256')
        _require(
            isinstance(content_hash, str)
            and len(content_hash) == 64
            and all(c in '0123456789abcdef' for c in content_hash),
            f'content SHA-256 invalid: {record_id}',
        )
        content_hashes.append(content_hash)
        strata[record.get('selection_stratum')] += 1
        families[record.get('source_family')] += 1
        _require(record.get('purpose') in {'selection-validation', 'selection_validation'}, f'purpose drift: {record_id}')
        _require(record.get('selection_eligible') is True, f'selection eligibility drift: {record_id}')
        _require(record.get('training_eligible') is False, f'training boundary weakened: {record_id}')
        _require(record.get('tokenizer_fit_eligible') is False, f'tokenizer-fit boundary weakened: {record_id}')
        _require(record.get('final_test_eligible') is False, f'final-test boundary weakened: {record_id}')
        _require(record.get('final_reporting_eligible') is False, f'final-reporting boundary weakened: {record_id}')
        _require(record.get('future_training_prohibited') is True, f'future-training prohibition missing: {record_id}')
    _require(len(content_hashes) == len(set(content_hashes)), 'selection content hashes are not unique')
    _require(dict(strata) == {'en': 2, 'ua': 8}, 'selection stratum vector drift')
    code_stratum = manifest.get('strata', {}).get('code', {})
    _require(code_stratum.get('documents') == 0, 'code stratum document-count drift')
    _require(code_stratum.get('selection_eligible') is False, 'code stratum unexpectedly selection eligible')
    _require(code_stratum.get('status') == 'BLOCKED_NO_ELIGIBLE_CODE_OBJECTS', 'code stratum fail-closed status drift')
    _require(dict(sorted(families.items())) == EXPECTED['families'], 'selection source-family vector drift')

    components = manifest.get('components', {})
    for stratum, head in EXPECTED['component_heads'].items():
        component = components.get(stratum, {})
        _require(component.get('head_sha') == head, f'{stratum} component head drift')
        _require(component.get('dedicated_workflow_conclusion') == 'success', f'{stratum} component is not terminal-success')
    _require(components.get('ua', {}).get('set_identity_sha256') == EXPECTED['component_ids']['ua'], 'UA component identity drift')
    _require(components.get('en', {}).get('authority_identity_sha256') == EXPECTED['component_ids']['en'], 'EN component identity drift')
    _require(components.get('code', {}).get('set_identity_sha256') == EXPECTED['component_ids']['code'], 'code component identity drift')
    code_component = components.get('code', {})
    _require(code_component.get('documents') == 0, 'code component document-count drift')
    rejected = code_component.get('rejected_candidates', [])
    _require(isinstance(rejected, list), 'code rejected_candidates must be a list')
    _require(all(not c.get('selection_admitted') for c in rejected if isinstance(c, dict)), 'code rejected candidate was admitted')
    _require(all(not c.get('evaluation_use_explicitly_authorized') for c in rejected if isinstance(c, dict)), 'code candidate unexpectedly has evaluation-use authorization')
    _require(all(not c.get('reserved_from_all_training') for c in rejected if isinstance(c, dict)), 'code candidate unexpectedly claims all-training reservation')

    _require(proof.get('schema_version') == '12-6.eval303-data300-exact-exclusion-proof.v1', 'DATA-300 proof schema drift')
    _require(proof.get('proof_identity_sha256') == EXPECTED['proof_identity_sha256'], 'DATA-300 proof identity drift')
    _require(self_identity(proof, 'proof_identity_sha256') == EXPECTED['proof_identity_sha256'], 'DATA-300 proof self-identity mismatch')
    _require(sha256_bytes(proof_path.read_bytes()) == EXPECTED['proof_file_sha256'], 'DATA-300 proof file SHA-256 drift')
    exclusion = manifest.get('data300_exclusion_proof', {})
    _require(exclusion.get('sha256') == EXPECTED['proof_file_sha256'], 'manifest DATA-300 proof file binding drift')
    _require(exclusion.get('proof_identity_sha256') == EXPECTED['proof_identity_sha256'], 'manifest DATA-300 proof identity binding drift')
    data300 = proof.get('data300', {})
    _require(data300.get('head_sha') == EXPECTED['data300_head_sha'], 'DATA-300 head drift')
    _require(data300.get('contract_identity_sha256') == EXPECTED['data300_contract_identity'], 'DATA-300 contract identity drift')
    _require(data300.get('contract_git_blob_sha1') == EXPECTED['data300_contract_blob_sha1'], 'DATA-300 contract blob drift')
    _require(sorted(proof.get('selection', {}).get('content_sha256', [])) == sorted(content_hashes), 'DATA-300 proof selection hashes drift')
    comparisons = proof.get('comparisons', {})
    _require(comparisons.get('selected_content_vs_training_raw_or_normalized_sha256_overlap') == [], 'selected content overlaps DATA-300 training bytes')
    _require(comparisons.get('selected_git_blob_vs_training_git_blob_overlap') == [], 'selected Git objects overlap DATA-300 training objects')
    verdict = proof.get('verdict', {})
    _require(verdict.get('exact_byte_overlap_count') == 0, 'DATA-300 exact-byte overlap is non-zero')
    _require(verdict.get('pinned_git_object_overlap_count') == 0, 'DATA-300 pinned-object overlap is non-zero')
    _require(verdict.get('status') == 'PASS_EXACT_DISTINCT_FROM_DATA300_TRAINING_PLAN', 'DATA-300 scoped verdict drift')
    _require(verdict.get('near_copy_or_dedup_cluster_scan_claimed') is False, 'historical proof falsely claims near-copy decontamination')
    _require(verdict.get('wave3_data300_g07_g08_still_required') is True, 'historical proof erased remaining gates')

    final_test = manifest.get('final_test_firewall', {})
    _require(final_test.get('head_sha') == EXPECTED['eval233_head_sha'], 'final-test authority head drift')
    _require(final_test.get('outcomes_read_by_eval303') is False, 'EVAL-303 read final-test outcomes')
    _require(final_test.get('final_test_payload_read_by_eval303') is False, 'EVAL-303 read final-test payload')
    _require(final_test.get('final_test_bytes_copied_into_composite') is False, 'EVAL-303 copied final-test bytes')
    _require(final_test.get('ua_component_exact_content_hash_overlap') == [], 'UA component overlaps final-test content hashes')
    _require(final_test.get('ua_component_source_family_overlap') == [], 'UA component overlaps final-test source families')
    _require(final_test.get('en_component_payload_read_for_construction') is False, 'EN component read final-test payload')
    _require(final_test.get('en_component_outcomes_read_for_construction') is False, 'EN component read final-test outcomes')

    usage = manifest.get('usage_contract', {})
    _require(usage.get('selection_only') is True, 'selection-only boundary drift')
    _require(usage.get('may_select_checkpoint') is True, 'checkpoint-selection contract drift')
    _require(usage.get('may_select_hyperparameters') is True, 'hyperparameter-selection contract drift')
    _require(usage.get('may_select_tokenizer_configuration') is True, 'tokenizer-configuration-selection contract drift')
    _require(usage.get('may_fit_tokenizer') is False, 'selection set may not fit tokenizer')
    _require(usage.get('may_update_model') is False, 'selection set may not update model')
    _require(usage.get('may_train') is False, 'selection set may not train')
    _require(usage.get('may_report_final_test') is False, 'selection set may not report final-test outcomes')
    _require(usage.get('code_aware_selection_available') is False, 'code-aware selection is unavailable with empty code stratum')

    return {
        'status': 'PASS',
        'selection_identity_sha256': manifest['selection_identity_sha256'],
        'documents': len(records),
        'strata': {'ua': 8, 'en': 2, 'code': 0},
        'membership_sha256': EXPECTED['membership_sha256'],
        'data300_exact_byte_overlap_count': 0,
        'final_test_outcomes_read': False,
        'code_selection_status': 'BLOCKED_NO_ELIGIBLE_CODE_OBJECTS',
    }


def materialize(repo_root: Path, output_dir: Path) -> None:
    verify(repo_root)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for rel in (MANIFEST, MEMBERSHIP, PROOF):
        src = repo_root / rel
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('verify', 'materialize'))
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.command == 'verify':
        print(json.dumps(verify(root), sort_keys=True, indent=2))
    else:
        if args.output_dir is None:
            parser.error('--output-dir is required for materialize')
        materialize(root, args.output_dir.resolve())
        print(json.dumps({'status': 'PASS', 'output_dir': str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
