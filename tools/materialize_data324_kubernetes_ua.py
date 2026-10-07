from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

CONFIG_PATH = Path('configs/data/data324_kubernetes_ua_recovery_v1.json')
USER_AGENT = '12-6-ai-DATA324/1.0'


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f'blob {len(payload)}\0'.encode('ascii')
    return hashlib.sha1(header + payload).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        + '\n'
    ).encode('utf-8')


def fetch_bounded(url: str, max_bytes: int) -> bytes:
    req = Request(
        url,
        headers={'User-Agent': USER_AGENT, 'Accept-Encoding': 'identity'},
    )
    with urlopen(req, timeout=30) as response:
        length = response.headers.get('Content-Length')
        if length is not None and int(length) > max_bytes:
            raise RuntimeError(f'oversized response before read: {url}')
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f'oversized response: {url}')
    return payload


def normalize_markdown_uk(raw: bytes) -> str:
    text = raw.decode('utf-8', errors='strict').replace('\r\n', '\n').replace('\r', '\n')
    text = unicodedata.normalize('NFKC', text)
    if text.startswith('---\n'):
        end = text.find('\n---\n', 4)
        if end < 0:
            raise RuntimeError('unterminated YAML frontmatter')
        text = text[end + 5:]
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\{\{<.*?>\}\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\{\{%.*?%\}\}', ' ', text, flags=re.DOTALL)
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'[`*_#>|]+', ' ', text)
    lines = [' '.join(line.split()) for line in text.split('\n')]
    normalized = '\n'.join(line for line in lines if line).strip() + '\n'
    return normalized


def language_evidence(text: str) -> dict[str, object]:
    alpha = [ch for ch in text if ch.isalpha()]
    cyrillic = [ch for ch in alpha if '\u0400' <= ch <= '\u04ff']
    latin = [ch for ch in alpha if ('A' <= ch <= 'Z') or ('a' <= ch <= 'z')]
    ukrainian_specific = [ch for ch in alpha if ch in 'ІіЇїЄєҐґ']
    if not alpha:
        raise RuntimeError('normalized text has no alphabetic content')
    cyrillic_share = len(cyrillic) / len(alpha)
    latin_share = len(latin) / len(alpha)
    decision = (
        'PASS'
        if cyrillic_share >= 0.70 and len(ukrainian_specific) >= 20
        else 'FAIL'
    )
    evidence = {
        'alphabetic_chars': len(alpha),
        'cyrillic_chars': len(cyrillic),
        'cyrillic_share_of_alpha': cyrillic_share,
        'latin_chars': len(latin),
        'latin_share_of_alpha': latin_share,
        'ukrainian_specific_chars': len(ukrainian_specific),
        'ukrainian_specific_set_present': sorted(set(ukrainian_specific)),
        'decision': decision,
        'rule': (
            'cyrillic_share>=0.70 AND count(ІіЇїЄєҐґ)>=20 '
            'after English-comment stripping'
        ),
    }
    if evidence['decision'] != 'PASS':
        raise RuntimeError(f'UA language gate failed: {evidence}')
    return evidence


def privacy_evidence(text: str) -> dict[str, object]:
    email_hits = re.findall(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        text,
    )
    phone_hits = re.findall(r'(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)', text)
    return {
        'email_like_hits': len(email_hits),
        'phone_like_hits': len(phone_hits),
        'decision': 'PASS' if not email_hits and not phone_hits else 'REVIEW',
        'note': (
            'bounded technical documentation heuristic only; '
            'no claim of universal PII absence'
        ),
    }


def resolve(root: Path, relative: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--config', default=str(CONFIG_PATH))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = json.loads(config_path.read_text(encoding='utf-8'))

    if config['local_free_only'] is not True:
        raise RuntimeError('LOCAL_FREE gate is not asserted')
    upstream = config['upstream']
    revision = upstream['revision']
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise RuntimeError('upstream revision must be exact 40-hex Git commit')
    if len(config['file_set']) != 1:
        raise RuntimeError('bounded recovery V1 requires exactly one source file')

    bounds = config['acquisition_bounds']
    item = config['file_set'][0]
    raw_url = (
        f"https://raw.githubusercontent.com/{upstream['repo']}/"
        f"{revision}/{item['path']}"
    )
    license_url = (
        f"https://raw.githubusercontent.com/{upstream['repo']}/"
        f"{revision}/{upstream['license_path']}"
    )

    raw = fetch_bounded(raw_url, bounds['max_source_bytes'])
    license_raw = fetch_bounded(license_url, bounds['max_license_bytes'])
    if len(raw) + len(license_raw) > bounds['max_total_download_bytes']:
        raise RuntimeError('combined acquisition exceeds total bound')
    if git_blob_sha1(raw) != item['expected_git_blob_sha1']:
        raise RuntimeError('source blob identity mismatch')
    if git_blob_sha1(license_raw) != upstream['expected_license_git_blob_sha1']:
        raise RuntimeError('license blob identity mismatch')

    normalized = normalize_markdown_uk(raw)
    normalized_bytes = normalized.encode('utf-8')
    if len(normalized_bytes) > bounds['max_normalized_bytes']:
        raise RuntimeError('normalized output exceeds bound')

    raw_hash = sha256(raw)
    normalized_hash = sha256(normalized_bytes)
    incumbents = config['dedup']['incumbent_text_hashes']
    exact_raw_collision = raw_hash in set(incumbents['raw_sha256'])
    exact_normalized_collision = normalized_hash in set(
        incumbents['normalized_sha256']
    )
    if exact_raw_collision or exact_normalized_collision:
        raise RuntimeError('candidate duplicates an incumbent training text object')

    lang = language_evidence(normalized)
    privacy = privacy_evidence(normalized)

    raw_rel = config['outputs']['raw_snapshot']
    norm_rel = config['outputs']['normalized_snapshot']
    lic_rel = config['outputs']['license_evidence']
    attr_rel = config['outputs']['attribution']
    manifest_rel = config['outputs']['manifest']
    report_rel = config['outputs']['report']

    resolve(root, raw_rel).write_bytes(raw)
    resolve(root, norm_rel).write_bytes(normalized_bytes)
    resolve(root, lic_rel).write_bytes(license_raw)

    attribution = (
        'Kubernetes Ukrainian documentation snapshot\n'
        'Creator/attribution: Kubernetes Authors\n'
        f"Source repository: https://github.com/{upstream['repo']}\n"
        f"Exact revision: {revision}\n"
        f"Source path: {item['path']}\n"
        'License: Creative Commons Attribution 4.0 International (CC BY 4.0)\n'
        'License URI: https://creativecommons.org/licenses/by/4.0/\n'
        'Project modification: normalized derivative removes YAML frontmatter, '
        'HTML comments, Hugo shortcodes and Markdown punctuation; applies Unicode '
        'NFKC and deterministic whitespace normalization.\n'
        'No endorsement by Kubernetes, CNCF, or contributors is implied.\n'
    )
    resolve(root, attr_rel).write_text(
        attribution,
        encoding='utf-8',
        newline='\n',
    )

    manifest_core = {
        'schema_version': '12-6.data324-kubernetes-ua-snapshot-manifest.v1',
        'worker_id': config['worker_id'],
        'verdict': 'ADMIT',
        'source_family': config['family_identity']['source_family'],
        'canonical_upstream': config['family_identity']['canonical_upstream'],
        'family_lineage_rule': config['family_identity']['lineage_rule'],
        'upstream_revision': revision,
        'license_id': config['license']['id'],
        'rights': config['license']['rights'],
        'evaluation': 'NOT_SEPARATELY_ADMITTED',
        'objects': [
            {
                'source_id': item['source_id'],
                'language': 'uk',
                'path': item['path'],
                'source_url': raw_url,
                'git_blob_sha1': item['expected_git_blob_sha1'],
                'raw_sha256': raw_hash,
                'raw_bytes': len(raw),
                'normalized_sha256': normalized_hash,
                'normalized_utf8_bytes': len(normalized_bytes),
                'raw_snapshot_path': raw_rel,
                'normalized_snapshot_path': norm_rel,
            }
        ],
        'license_evidence': {
            'path': upstream['license_path'],
            'url': license_url,
            'git_blob_sha1': upstream['expected_license_git_blob_sha1'],
            'sha256': sha256(license_raw),
            'bytes': len(license_raw),
            'materialized_path': lic_rel,
        },
        'language_evidence': lang,
        'privacy_evidence': privacy,
        'dedup': {
            'exact_raw_collision_with_current_training_texts': False,
            'exact_normalized_collision_with_current_training_texts': False,
            'dedup_key': normalized_hash,
            'family_collapse_required_for_any_kubernetes_website_translation_or_sibling': True,
        },
        'normalization': config['normalization'],
        'attribution_path': attr_rel,
        'local_free_only': True,
        'training_executed': False,
    }
    manifest = {
        **manifest_core,
        'manifest_identity_sha256': sha256(canonical_bytes(manifest_core)),
    }
    resolve(root, manifest_rel).write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
        encoding='utf-8',
        newline='\n',
    )

    report_core = {
        'schema_version': '12-6.data324-kubernetes-ua-recovery-report.v1',
        'worker_id': config['worker_id'],
        'verdict': 'ADMIT',
        'recovered_from': config['recovered_from'],
        'prior_failure_classification': (
            'ENVIRONMENT_BOOTSTRAP_FAILURE_NOT_RIGHTS_REJECTION'
        ),
        'snapshot_manifest_identity_sha256': manifest['manifest_identity_sha256'],
        'raw_sha256': raw_hash,
        'normalized_sha256': normalized_hash,
        'license_sha256': sha256(license_raw),
        'language_decision': lang['decision'],
        'privacy_decision': privacy['decision'],
        'dedup_decision': 'PASS',
        'rights_decision': (
            'PASS_WITH_CC_BY_4_0_ATTRIBUTION_AND_MODIFICATION_NOTICE'
        ),
        'evaluation_decision': 'NOT_ADMITTED_REQUIRE_SEPARATE_AUTHORITY',
        'materialized_snapshot': True,
        'local_free_only': True,
        'training_executed': False,
    }
    report = {
        **report_core,
        'report_identity_sha256': sha256(canonical_bytes(report_core)),
    }
    resolve(root, report_rel).write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
