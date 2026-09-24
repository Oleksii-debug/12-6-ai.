"""Bounded zero-credit intake for one immutable Common Pile Foodista shard."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from twelve_six.common_pile_rights import load_and_validate as load_rights
CONFIG = ROOT / 'configs/data/d03_common_pile_foodista_bounded_v1.json'
SCHEMA = '12-6.d03-common-pile-foodista-bounded.v1'
REPORT_SCHEMA = '12-6.d03-common-pile-foodista-bounded-report.v1'
CLAIM_ISSUE = 1168
SOURCE_REVISION = '04d1b7a6562c6d6459426d2a3b88184b8a98f7b6'
SOURCE_FILE = 'v0/documents/00000_foodista.jsonl.gz'
SOURCE_SHA256 = '286b801bc826f161efad892af5529b201f91f42c5a932e3962d74d24037fe087'
SOURCE_COMPRESSED_BYTES, SOURCE_MAX_BYTES = (6760466, 7000000)
SOURCE_URL = f'https://huggingface.co/datasets/common-pile/foodista/resolve/{SOURCE_REVISION}/{SOURCE_FILE}'
SOURCE_LABEL = 'foodista'
EXPECTED_LICENSE = 'Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/'
METADATA_HOSTS = ('www.foodista.com', 'foodista.com')
FIELDS = {'id', 'text', 'source', 'added', 'created', 'metadata'}
META_FIELDS = {'authors', 'license', 'provenance', 'url'}
PROVENANCE_RE = re.compile('^foodista-dolma-[0-9]{4}\\.json\\.gz:[1-9][0-9]*$')
MAX_LINE = 5 * 1024 * 1024
EMAIL_RE = re.compile('\\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}\\b', re.I)
PHONE_RE = re.compile('(?<!\\w)(?:\\+?\\d[\\d ().-]{7,}\\d)(?!\\w)')
IPV4_RE = re.compile('(?<!\\d)(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1?\\d?\\d)){3}(?!\\d)')
CONTROL_RE = re.compile('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]')
SECRET_RE = re.compile('-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----|\\bbearer\\s+[A-Za-z0-9._~+/=-]{12,}|\\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)\\s*[:=]\\s*[^\\s]{6,}', re.I)
ZERO_KEYS = ('training_authorized_bytes', 'canonical_capacity_credited', 'family_credit_added', 'unique_causal_loss_positions_authorized', 'optimizer_updates')
FALSE_KEYS = ('training_eligible', 'evaluation_eligible', 'tokenizer_fit_authorized', 'model_training_executed', 'final_test_accessed', 'paid_compute_used', 'research_corpus_v1_released')
DOWNSTREAM = ['global_cross_source_dedup', 'reserved_evaluation_decontamination', 'final_source_rights_review', 'post_composition_quality_privacy', 'balance_and_family_caps', 'cluster_safe_split', 'deterministic_packing_two_clean_builds', 'positive_unique_loss_ledger']

class FoodistaIntakeError(RuntimeError):
    pass

def req(ok: bool, msg: str) -> None:
    if not ok:
        raise FoodistaIntakeError(msg)

def zero(value: Any, name: str) -> None:
    req(type(value) is int and value == 0, f'{name} must remain integer zero')

def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()

def _validate_parent_rights(cfg: Mapping[str, Any]) -> str:
    path = (ROOT / cfg['parent_rights']['registry_path']).resolve()
    req(path.is_relative_to(ROOT.resolve()), 'rights registry escaped repo root')
    rights = load_rights(path)
    rows = rights.get('sources')
    req(isinstance(rows, list), 'validated rights registry lost source rows')
    matches = [r for r in rows if isinstance(r, Mapping) and r.get('key') == 'foodista']
    req(len(matches) == 1, 'Foodista rights row is not unique')
    row = matches[0]
    expected = {'hf_dataset': 'common-pile/foodista', 'audited_collector_path': 'sources/food', 'collector_path_status': 'PRESENT_AT_AUDITED_CODE_COMMIT', 'rights_basis_class': 'SITEWIDE_OPEN_LICENSE', 'license_or_status_signals': ['CC-BY'], 'upstream_rights_claim': 'Upstream paper states all Foodista content is licensed under CC BY.', 'provenance_summary': 'Foodista HTML collected with a custom text extraction pipeline.', 'project_review_status': 'REVIEW_REQUIRED', 'canonical_training_authorized': False, 'evaluation_role': 'TRAINING_CANDIDATE_ONLY', 'final_test_excluded': True}
    for key, value in expected.items():
        req(row.get(key) == value, f'rights {key} drift')
    zero(row.get('credited_bytes'), 'rights credited_bytes')
    zero(row.get('authorized_loss_positions'), 'rights authorized_loss_positions')
    identity = rights.get('registry_identity_sha256')
    req(isinstance(identity, str) and re.fullmatch('[0-9a-f]{64}', identity) is not None, 'validated rights registry identity missing')
    return identity

def load_config(path: Path=CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoodistaIntakeError(f'cannot load config: {path}') from exc
    req(isinstance(cfg, dict), 'config root must be an object')
    req(set(cfg) == {'schema_version', 'execution_profile', 'claim_issue', 'parent_rights', 'collector_authority', 'source', 'selection', 'quality', 'privacy', 'downstream_required', 'claim_boundary'}, 'config keys must be exact')
    req(cfg['schema_version'] == SCHEMA and cfg['execution_profile'] == 'LOCAL_FREE' and (cfg['claim_issue'] == CLAIM_ISSUE), 'config identity drift')
    p = cfg['parent_rights']
    req(set(p) == {'merged_pr', 'registry_path', 'source_key', 'hf_dataset', 'required_collector_path', 'required_rights_basis_class', 'required_rights_signal', 'required_upstream_rights_claim', 'required_provenance_summary', 'required_project_review_status'}, 'parent rights keys drift')
    req((p['merged_pr'], p['registry_path'], p['source_key'], p['hf_dataset'], p['required_collector_path'], p['required_rights_basis_class'], p['required_rights_signal'], p['required_project_review_status']) == (769, 'configs/data/common_pile_source_rights_v1.json', 'foodista', 'common-pile/foodista', 'sources/food', 'SITEWIDE_OPEN_LICENSE', 'CC-BY', 'REVIEW_REQUIRED'), 'parent rights binding drift')
    req(p['required_upstream_rights_claim'] == 'Upstream paper states all Foodista content is licensed under CC BY.' and p['required_provenance_summary'] == 'Foodista HTML collected with a custom text extraction pipeline.', 'parent rights evidence drift')
    c = cfg['collector_authority']
    req(c == {'repository': 'https://github.com/r-three/common-pile', 'revision': '9457f04a14cb2355ab00023420369d46ffd4a395', 'readme_path': 'sources/food/README.md', 'readme_git_blob_sha1': '155d861c68194b68a18df1fcfa3c32e9105c645d', 'supported_fact': 'The audited Common Pile Foodista collector README states the Foodista shared recipe site is licensed under CC-BY-3.0.', 'legal_conclusion_claimed': False}, 'collector authority drift')
    s = cfg['source']
    req(set(s) == {'revision', 'file', 'url', 'sha256', 'compressed_bytes', 'max_compressed_bytes', 'compression', 'expected_source_label', 'expected_row_fields', 'expected_metadata_fields', 'expected_metadata_license', 'required_metadata_hosts', 'required_provenance_regex'}, 'source contract keys drift')
    req((s['revision'], s['file'], s['url'], s['sha256'], s['compressed_bytes'], s['max_compressed_bytes'], s['compression'], s['expected_source_label']) == (SOURCE_REVISION, SOURCE_FILE, SOURCE_URL, SOURCE_SHA256, SOURCE_COMPRESSED_BYTES, SOURCE_MAX_BYTES, 'gzip', SOURCE_LABEL), 'immutable source contract drift')
    req(s['expected_row_fields'] == ['id', 'text', 'source', 'added', 'created', 'metadata'] and s['expected_metadata_fields'] == ['authors', 'license', 'provenance', 'url'] and (s['expected_metadata_license'] == EXPECTED_LICENSE) and (s['required_metadata_hosts'] == list(METADATA_HOSTS)) and (s['required_provenance_regex'] == PROVENANCE_RE.pattern), 'source row contract drift')
    req(cfg['selection'] == {'max_scanned_records': 8192, 'max_retained_records': 4096, 'max_retained_normalized_utf8_bytes': 4800000, 'order': 'exact_shard_file_order_after_fail_closed_filters'}, 'selection contract drift')
    req(cfg['quality'] == {'min_chars': 200, 'max_chars': 50000, 'min_alpha_fraction': 0.25, 'min_latin_alpha_ratio': 0.75, 'max_cyrillic_alpha_ratio': 0.05}, 'quality contract drift')
    req(cfg['privacy'] == {'reject_email': True, 'reject_phone': True, 'reject_ipv4': True, 'reject_secret_markers': True, 'reject_control_characters': True, 'metadata_authors_emitted': False, 'metadata_url_emitted': False, 'metadata_provenance_emitted': False, 'universal_pii_absence_claimed': False}, 'privacy contract drift')
    req(cfg['downstream_required'] == DOWNSTREAM, 'downstream gates drift')
    b = cfg['claim_boundary']
    req(set(b) == set(ZERO_KEYS + FALSE_KEYS), 'boundary keys drift')
    for key in ZERO_KEYS:
        zero(b.get(key), key)
    for key in FALSE_KEYS:
        req(b.get(key) is False, f'{key} must remain false')
    _validate_parent_rights(cfg)
    return cfg

def download_exact(cfg: Mapping[str, Any], output: Path) -> None:
    req(cfg['source']['url'] == SOURCE_URL, 'download URL drift')
    request = urllib.request.Request(SOURCE_URL, headers={'User-Agent': '12-6-ai-D03-common-pile-foodista/1', 'Accept': 'application/gzip,application/octet-stream;q=0.9,*/*;q=0.1'})
    partial, digest, total = (output.with_suffix(output.suffix + '.partial'), hashlib.sha256(), 0)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as target:
            while (chunk := response.read(1024 * 1024)):
                total += len(chunk)
                req(total <= SOURCE_MAX_BYTES, 'download exceeded compressed safety cap')
                digest.update(chunk)
                target.write(chunk)
        req(total == SOURCE_COMPRESSED_BYTES, 'source compressed size mismatch')
        req(digest.hexdigest() == SOURCE_SHA256, 'source SHA-256 mismatch')
        partial.replace(output)
    except OSError as exc:
        raise FoodistaIntakeError('exact source download failed') from exc
    finally:
        if partial.exists():
            partial.unlink()

def verify_exact_source(path: Path) -> int:
    digest, size = (hashlib.sha256(), 0)
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                size += len(chunk)
                req(size <= SOURCE_MAX_BYTES, 'source exceeds compressed safety cap')
                digest.update(chunk)
    except OSError as exc:
        raise FoodistaIntakeError(f'cannot read source shard: {path}') from exc
    req(size == SOURCE_COMPRESSED_BYTES, 'source compressed size mismatch')
    req(digest.hexdigest() == SOURCE_SHA256, 'source SHA-256 mismatch')
    return size

def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).replace('\r\n', '\n').replace('\r', '\n')
    return '\n'.join((' '.join(line.split()) for line in text.split('\n') if line.split())).strip()

def _validate_structural_row(row: Mapping[str, Any]) -> None:
    req(set(row) == FIELDS, 'source row field drift')
    req(type(row.get('id')) is int and row['id'] >= 0, 'source record id invalid')
    for key in ('text', 'source', 'added', 'created'):
        req(isinstance(row.get(key), str), f'source {key} is not a string')
    meta = row.get('metadata')
    req(isinstance(meta, Mapping), 'metadata field is not an object')
    req(set(meta) == META_FIELDS, 'metadata field drift')

def _metadata_matches(meta: Mapping[str, Any]) -> bool:
    if meta.get('license') != EXPECTED_LICENSE or not isinstance(meta.get('authors'), list) or (not all((isinstance(x, str) for x in meta['authors']))):
        return False
    provenance = meta.get('provenance')
    if not isinstance(provenance, str) or PROVENANCE_RE.fullmatch(provenance) is None:
        return False
    raw_url = meta.get('url')
    if not isinstance(raw_url, str):
        return False
    try:
        parsed, port = (urlsplit(raw_url), urlsplit(raw_url).port)
    except ValueError:
        return False
    return parsed.scheme == 'https' and parsed.hostname in METADATA_HOSTS and (parsed.username is None) and (parsed.password is None) and (port in (None, 443)) and parsed.path.startswith('/') and (not parsed.fragment)

def assess_row(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str, str]:
    _validate_structural_row(row)
    if row['source'] != SOURCE_LABEL:
        return (False, 'source_label_mismatch', '')
    if not _metadata_matches(row['metadata']):
        return (False, 'metadata_rights_or_origin_mismatch', '')
    text = row['text']
    for pattern, reason in ((CONTROL_RE, 'control_character'), (EMAIL_RE, 'email'), (IPV4_RE, 'ipv4'), (PHONE_RE, 'phone'), (SECRET_RE, 'secret_marker')):
        if pattern.search(text):
            return (False, reason, '')
    normalized, q = (normalize(text), cfg['quality'])
    if len(normalized) < q['min_chars']:
        return (False, 'too_short', '')
    if len(normalized) > q['max_chars']:
        return (False, 'too_long', '')
    alpha = [c for c in normalized if c.isalpha()]
    if len(alpha) / max(1, len(normalized)) < q['min_alpha_fraction']:
        return (False, 'low_alpha_fraction', '')
    latin = sum(('LATIN' in unicodedata.name(c, '') for c in alpha)) / max(1, len(alpha))
    cyr = sum(('CYRILLIC' in unicodedata.name(c, '') for c in alpha)) / max(1, len(alpha))
    if latin < q['min_latin_alpha_ratio']:
        return (False, 'low_latin_alpha_ratio', '')
    if cyr > q['max_cyrillic_alpha_ratio']:
        return (False, 'high_cyrillic_alpha_ratio', '')
    return (True, 'accepted', normalized)

def _iter_gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with gzip.open(path, 'rb') as handle:
            while (raw := handle.readline(MAX_LINE + 1)):
                req(len(raw) <= MAX_LINE, 'source JSONL line exceeds safety cap')
                try:
                    row = json.loads(raw.decode('utf-8', errors='strict'))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise FoodistaIntakeError('source JSONL is malformed') from exc
                req(isinstance(row, dict), 'source JSONL row must be an object')
                yield row
    except (OSError, EOFError) as exc:
        raise FoodistaIntakeError('cannot stream exact gzip shard') from exc

def select_rows(rows: Iterable[Mapping[str, Any]], cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Counter[str], int]:
    s, accepted, reasons, seen_text, seen_ids, retained, scanned = (cfg['selection'], [], Counter(), set(), set(), 0, 0)
    for row in rows:
        if scanned >= s['max_scanned_records']:
            break
        scanned += 1
        rid = row.get('id')
        if type(rid) is int and rid in seen_ids:
            raise FoodistaIntakeError('duplicate source record id')
        if type(rid) is int:
            seen_ids.add(rid)
        ok, reason, text = assess_row(row, cfg)
        if not ok:
            reasons[reason] += 1
            continue
        encoded = text.encode()
        sha = hashlib.sha256(encoded).hexdigest()
        if sha in seen_text:
            reasons['exact_normalized_duplicate'] += 1
            continue
        seen_text.add(sha)
        if len(accepted) >= s['max_retained_records']:
            reasons['retained_record_cap_reached'] += 1
            continue
        if retained + len(encoded) > s['max_retained_normalized_utf8_bytes']:
            reasons['retained_byte_cap_reached'] += 1
            continue
        accepted.append({'record_id': rid, 'source_key': 'foodista', 'source_label': SOURCE_LABEL, 'normalized_sha256': sha, 'normalized_bytes': len(encoded), 'training_eligible': False, 'evaluation_eligible': False, 'text': text})
        retained += len(encoded)
        reasons['accepted'] += 1
    req(sum(reasons.values()) == scanned, 'selection accounting mismatch')
    return (accepted, reasons, scanned)

def candidate_payload(accepted: Iterable[Mapping[str, Any]]) -> bytes:
    return b''.join((canonical(dict(row)) + b'\n' for row in accepted))

def build_report(cfg: Mapping[str, Any], *, rights_registry_identity: str, accepted: list[Mapping[str, Any]], reasons: Counter[str], scanned: int, candidate: bytes, observed_source_bytes: int) -> dict[str, Any]:
    report = {'schema_version': REPORT_SCHEMA, 'status': 'COMMON_PILE_FOODISTA_CANDIDATE_ONLY_ZERO_CREDIT', 'claim_issue': CLAIM_ISSUE, 'source_revision': SOURCE_REVISION, 'source_file': SOURCE_FILE, 'source_sha256': SOURCE_SHA256, 'observed_source_bytes': observed_source_bytes, 'collector_revision': '9457f04a14cb2355ab00023420369d46ffd4a395', 'collector_readme_git_blob_sha1': '155d861c68194b68a18df1fcfa3c32e9105c645d', 'rights_registry_identity_sha256': rights_registry_identity, 'scanned_records': scanned, 'retained_records': len(accepted), 'retained_normalized_bytes': sum((int(r['normalized_bytes']) for r in accepted)), 'selection_reasons': dict(sorted(reasons.items())), 'candidate_payload_sha256': hashlib.sha256(candidate).hexdigest(), 'candidate_payload_bytes': len(candidate), 'durable_report_contains_source_text': False, 'metadata_authors_emitted': False, 'metadata_url_emitted': False, 'metadata_provenance_emitted': False, 'universal_pii_absence_claimed': False, 'downstream_required': list(cfg['downstream_required'])}
    report.update(cfg['claim_boundary'])
    return report

def run(shard: Path, output_jsonl: Path, report_path: Path, *, download: bool) -> dict[str, Any]:
    cfg = load_config()
    if download:
        download_exact(cfg, shard)
    observed, rights = (verify_exact_source(shard), _validate_parent_rights(cfg))
    accepted, reasons, scanned = select_rows(_iter_gzip_jsonl(shard), cfg)
    req(bool(accepted), 'bounded source intake retained zero records')
    payload = candidate_payload(accepted)
    report = build_report(cfg, rights_registry_identity=rights, accepted=accepted, reasons=reasons, scanned=scanned, candidate=payload, observed_source_bytes=observed)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_bytes(payload)
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=Path, required=True)
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    try:
        report = run(args.shard, args.output_jsonl, args.report, download=args.download)
    except FoodistaIntakeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
