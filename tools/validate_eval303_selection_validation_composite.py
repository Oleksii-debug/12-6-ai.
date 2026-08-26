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


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(obj: dict, field: str) -> str:
    clone = deepcopy(obj)
    clone.pop(field, None)
    return sha256_bytes((canonical(clone) + '\n').encode('utf-8'))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def load_records(path: Path) -> list[dict]:
    records = []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise AssertionError(f'{path} must end with LF')
    for index, line in enumerate(raw.decode('utf-8').splitlines(), start=1):
        record = json.loads(line)
        if line != canonical(record):
            raise AssertionError(f'{path}:{index} is not canonical JSON')
        records.append(record)
    return records


def verify(repo_root: Path) -> dict:
    manifest_path = repo_root / MANIFEST
    membership_path = repo_root / MEMBERSHIP
    proof_path = repo_root / PROOF
    manifest = load_json(manifest_path)
    proof = load_json(proof_path)
    records = load_records(membership_path)

    assert manifest['schema_version'] == '12-6.eval303-selection-validation-composite.v1'
    assert manifest['worker_id'] == 'EVAL-303-SELECTION-VALIDATION-COMPOSITE'
    assert manifest['purpose'] == 'selection-validation'
    assert manifest['execution_profile'] == 'LOCAL_FREE'
    assert manifest['selection_identity_sha256'] == EXPECTED['selection_identity_sha256']
    assert self_identity(manifest, 'selection_identity_sha256') == EXPECTED['selection_identity_sha256']

    assert sha256_bytes(membership_path.read_bytes()) == EXPECTED['membership_sha256']
    assert manifest['composite_membership']['sha256'] == EXPECTED['membership_sha256']
    assert manifest['composite_membership']['bytes'] == len(membership_path.read_bytes())
    assert manifest['composite_membership']['documents'] == EXPECTED['records'] == len(records)
    assert manifest['composite_membership']['payload_bytes_copied_into_eval303'] is False

    content_hashes = []
    strata = Counter()
    families = Counter()
    record_ids = set()
    for record in records:
        record_id = record['record_id']
        assert record_id not in record_ids
        record_ids.add(record_id)
        assert 'text' not in record
        content_hash = record['content_sha256']
        assert len(content_hash) == 64 and all(c in '0123456789abcdef' for c in content_hash)
        content_hashes.append(content_hash)
        strata[record['selection_stratum']] += 1
        families[record['source_family']] += 1
        assert record['purpose'] in {'selection-validation', 'selection_validation'}
        assert record['selection_eligible'] is True
        assert record['training_eligible'] is False
        assert record['tokenizer_fit_eligible'] is False
        assert record['final_test_eligible'] is False
        assert record['final_reporting_eligible'] is False
        assert record['future_training_prohibited'] is True
    assert len(content_hashes) == len(set(content_hashes))
    assert dict(strata) == {'en': 2, 'ua': 8}
    assert manifest['strata']['code']['documents'] == 0
    assert manifest['strata']['code']['selection_eligible'] is False
    assert manifest['strata']['code']['status'] == 'BLOCKED_NO_ELIGIBLE_CODE_OBJECTS'
    assert dict(sorted(families.items())) == EXPECTED['families']

    components = manifest['components']
    for stratum, head in EXPECTED['component_heads'].items():
        assert components[stratum]['head_sha'] == head
        assert components[stratum]['dedicated_workflow_conclusion'] == 'success'
    assert components['ua']['set_identity_sha256'] == EXPECTED['component_ids']['ua']
    assert components['en']['authority_identity_sha256'] == EXPECTED['component_ids']['en']
    assert components['code']['set_identity_sha256'] == EXPECTED['component_ids']['code']
    assert components['code']['documents'] == 0
    assert all(not c['selection_admitted'] for c in components['code']['rejected_candidates'])
    assert all(not c['evaluation_use_explicitly_authorized'] for c in components['code']['rejected_candidates'])
    assert all(not c['reserved_from_all_training'] for c in components['code']['rejected_candidates'])

    assert proof['schema_version'] == '12-6.eval303-data300-exact-exclusion-proof.v1'
    assert proof['proof_identity_sha256'] == EXPECTED['proof_identity_sha256']
    assert self_identity(proof, 'proof_identity_sha256') == EXPECTED['proof_identity_sha256']
    assert sha256_bytes(proof_path.read_bytes()) == EXPECTED['proof_file_sha256']
    assert manifest['data300_exclusion_proof']['sha256'] == EXPECTED['proof_file_sha256']
    assert manifest['data300_exclusion_proof']['proof_identity_sha256'] == EXPECTED['proof_identity_sha256']
    assert proof['data300']['head_sha'] == EXPECTED['data300_head_sha']
    assert proof['data300']['contract_identity_sha256'] == EXPECTED['data300_contract_identity']
    assert proof['data300']['contract_git_blob_sha1'] == EXPECTED['data300_contract_blob_sha1']
    assert sorted(proof['selection']['content_sha256']) == sorted(content_hashes)
    assert proof['comparisons']['selected_content_vs_training_raw_or_normalized_sha256_overlap'] == []
    assert proof['comparisons']['selected_git_blob_vs_training_git_blob_overlap'] == []
    assert proof['verdict']['exact_byte_overlap_count'] == 0
    assert proof['verdict']['pinned_git_object_overlap_count'] == 0
    assert proof['verdict']['status'] == 'PASS_EXACT_DISTINCT_FROM_DATA300_TRAINING_PLAN'
    assert proof['verdict']['near_copy_or_dedup_cluster_scan_claimed'] is False
    assert proof['verdict']['wave3_data300_g07_g08_still_required'] is True

    final_test = manifest['final_test_firewall']
    assert final_test['head_sha'] == EXPECTED['eval233_head_sha']
    assert final_test['outcomes_read_by_eval303'] is False
    assert final_test['final_test_payload_read_by_eval303'] is False
    assert final_test['final_test_bytes_copied_into_composite'] is False
    assert final_test['ua_component_exact_content_hash_overlap'] == []
    assert final_test['ua_component_source_family_overlap'] == []
    assert final_test['en_component_payload_read_for_construction'] is False
    assert final_test['en_component_outcomes_read_for_construction'] is False

    usage = manifest['usage_contract']
    assert usage['selection_only'] is True
    assert usage['may_select_checkpoint'] is True
    assert usage['may_select_hyperparameters'] is True
    assert usage['may_select_tokenizer_configuration'] is True
    assert usage['may_fit_tokenizer'] is False
    assert usage['may_update_model'] is False
    assert usage['may_train'] is False
    assert usage['may_report_final_test'] is False
    assert usage['code_aware_selection_available'] is False

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
