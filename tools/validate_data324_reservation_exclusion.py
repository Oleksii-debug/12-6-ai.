from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/data/data324_kubernetes_ua_recovery_v1.json',
    )
    parser.add_argument(
        '--report',
        default='reports/data324/kubernetes-ua-recovery-v1.json',
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))

    raw_sha256 = report.get('raw_sha256')
    if not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
        raise RuntimeError('materialized report lacks bound raw SHA-256')

    reservations = config.get('reservations')
    if not isinstance(reservations, list) or not reservations:
        raise RuntimeError('no evaluation reservation authorities are bound')

    checked: list[dict[str, object]] = []
    for reservation in reservations:
        authority = reservation.get('authority')
        head_sha = reservation.get('head_sha')
        run_id = reservation.get('workflow_run_id')
        reserved = reservation.get('reserved_source_raw_sha256')
        if not isinstance(authority, str) or not authority:
            raise RuntimeError('reservation authority is not bound')
        if not isinstance(head_sha, str) or len(head_sha) != 40:
            raise RuntimeError(f'{authority}: reservation head SHA is not exact')
        if not isinstance(run_id, int) or run_id <= 0:
            raise RuntimeError(f'{authority}: reservation run is not bound')
        if not isinstance(reserved, list) or not reserved:
            raise RuntimeError(f'{authority}: reserved raw identities are absent')
        reserved_set = {str(value) for value in reserved}
        if any(len(value) != 64 for value in reserved_set):
            raise RuntimeError(f'{authority}: malformed reserved SHA-256')
        if raw_sha256 in reserved_set:
            raise RuntimeError(
                f'{authority}: acquired training source is evaluation-reserved: '
                f'{raw_sha256}'
            )
        checked.append(
            {
                'authority': authority,
                'head_sha': head_sha,
                'workflow_run_id': run_id,
                'reserved_source_objects_checked': len(reserved_set),
                'collision': False,
            }
        )

    print(
        json.dumps(
            {
                'decision': 'PASS',
                'training_source_raw_sha256': raw_sha256,
                'reservation_authorities': checked,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
