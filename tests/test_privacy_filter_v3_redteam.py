from __future__ import annotations

import hashlib
import json

from twelve_six.data.privacy_filter_v3 import (
    DETECTOR_ACTIONS,
    assert_hash_safe_evidence,
    build_manifest,
    detect,
    hash_safe_scan,
    policy_manifest,
)


def S(*parts: str) -> str:
    return "".join(parts)


# Synthetic-only fixtures. Secret-looking forms are assembled from fragments so
# repository scanners do not confuse fixtures with live credentials.
POSITIVES = [
    ("api-github", "api_token", S("gh", "p_", "A1b2" * 9)),
    ("api-gitlab", "api_token", S("gl", "pat-", "Ab12_" * 5)),
    ("api-hf", "api_token", S("h", "f_", "Ab12" * 6)),
    ("api-npm", "api_token", S("np", "m_", "Ab12" * 6)),
    ("api-openai", "api_token", S("s", "k-proj-", "Ab12_" * 6)),
    ("cloud-aws-id", "cloud_credential", S("AK", "IA", "A1B2" * 4)),
    ("cloud-aws-secret", "cloud_credential", "aws_secret_access_key=" + S("Ab1/", "Cd2+", "Ef3/", "Gh4+", "Ij5/", "Kl6+", "Mn7/", "Op8+", "Qr9/", "St0+")),
    ("cloud-gcp", "cloud_credential", S("AI", "za", "A" * 35)),
    ("cloud-azure", "cloud_credential", "AccountKey=" + S("QUJD" * 12)),
    ("auth-basic", "authorization_header", "Authorization: Basic " + S("QWxpY2U6", "UzNjcjN0IQ==")),
    ("auth-bearer", "authorization_header", "Authorization: Bearer " + S("eyJ", "A1b2" * 7, ".", "B2c3" * 7, ".", "C3d4" * 7)),
    ("auth-token", "authorization_header", "Proxy-Authorization: Token " + S("Ab12" * 8)),
    ("cred-url-http", "credential_url", "https://alice:" + S("S3c", "ret!") + "@internal.example/path"),
    ("cred-url-git", "credential_url", "git+https://svc:" + S("T0k", "en-987") + "@git.example/repo"),
    ("path-linux", "private_filesystem_path", "/home/alice/projects/private/config.yaml"),
    ("path-macos", "private_filesystem_path", "/Users/bob/Documents/notes.txt"),
    ("path-root", "private_filesystem_path", "/root/.ssh/id_ed25519"),
    ("path-windows", "private_filesystem_path", r"C:\Users\carol\AppData\Roaming\cloud\credentials.json"),
    ("email-1", "email", "owner+prod@corp.example.org"),
    ("email-2", "email", "Інфо: test.person@sub.example.com"),
    ("phone-e164", "phone", "+421 905 123 456"),
    ("phone-ua", "phone", "+380 (67) 123-45-67"),
    ("id-ssn", "sensitive_id", "SSN 123-45-6789"),
    ("id-passport", "sensitive_id", "passport number: AB123456"),
    ("id-rnokpp", "sensitive_id", "РНОКПП: 1234567890"),
    ("ssh-private", "ssh_material", S("-----BEGIN OPEN", "SSH PRIVATE KEY-----\n", "fixture-only")),
    ("ssh-public", "ssh_material", "ssh-ed25519 " + S("QUJD" * 18) + " alice@host"),
    ("db-credentialed", "database_url", "postgresql://svc:" + S("Db", "Pass9!") + "@db.corp.local:5432/app"),
    ("db-remote", "database_url", "mongodb://cluster.internal:27017/app"),
    ("env-token", "environment_secret_assignment", "SERVICE_TOKEN=" + S("Ab12" * 8)),
    ("env-password", "environment_secret_assignment", "export DB_PASSWORD=" + S("Str0ng", "!Pass99")),
    ("env-database", "environment_secret_assignment", "DATABASE_URL=postgresql://svc:" + S("P4ss", "word!") + "@db.internal/app"),
    ("generic-secret", "environment_secret_assignment", "client_secret=" + S("AbCd", "1234!fixture")),
]

NEGATIVES = [
    ("plain-doc", "This documentation explains authentication headers without values."),
    ("placeholder-token", "SERVICE_TOKEN=${SERVICE_TOKEN}"),
    ("placeholder-password", "password=changeme"),
    ("example-api", "api_key=<redacted>"),
    ("date", "Release date 2026-08-26."),
    ("ipv4", "Server 192.168.10.25 is an example address."),
    ("build-id", "Build number 1234-56-78."),
    ("invalid-ssn", "Test 000-12-3456."),
    ("generic-linux-home", "/home/user/project/readme.md"),
    ("generic-macos-home", "/Users/example/Documents/file.txt"),
    ("generic-windows-home", r"C:\Users\user\project\README.md"),
    ("url-no-creds", "https://example.com/docs/auth"),
    ("url-placeholder-creds", "https://user:<password>@example.com/docs"),
    ("db-local", "postgresql://localhost/app"),
    ("db-example", "mongodb://db.example.com/sample"),
    ("env-debug", "DEBUG=1"),
    ("env-port", "PORT=8080"),
    ("env-placeholder", "API_KEY=your_api_key"),
    ("header-placeholder", "Authorization: Bearer <token>"),
    ("header-short", "Authorization: Token demo"),
    ("code-email-ish", "value = 'user at example dot org'"),
    ("short-id", "id=42"),
    ("uuid", "request_id=550e8400-e29b-41d4-a716-446655440000"),
    ("ssh-command", "ssh -i ~/.ssh/id_ed25519 host.example"),
    ("public-path", "/usr/share/doc/package/README"),
]


def _metrics() -> dict[str, object]:
    tp = fn = tn = fp = 0
    per_class = {name: {"positive": 0, "tp": 0, "fn": 0} for name in DETECTOR_ACTIONS}
    for _, expected, text in POSITIVES:
        per_class[expected]["positive"] += 1
        ids = {f.detector_id for f in detect(text)}
        if expected in ids:
            tp += 1
            per_class[expected]["tp"] += 1
        else:
            fn += 1
            per_class[expected]["fn"] += 1
    for _, text in NEGATIVES:
        if detect(text):
            fp += 1
        else:
            tn += 1
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp, "per_class": per_class}


def test_synthetic_adversarial_metrics_gate() -> None:
    metrics = _metrics()
    assert {k: metrics[k] for k in ("TP", "FN", "TN", "FP")} == {
        "TP": 33,
        "FN": 0,
        "TN": 25,
        "FP": 0,
    }
    assert all(row["positive"] >= 2 for row in metrics["per_class"].values())


def test_hash_safe_evidence_never_contains_match_values_or_hashes() -> None:
    raw = ("Authorization: Basic " + S("QWxpY2U6", "UzNjcjN0IQ==")).encode()
    result = hash_safe_scan(raw)
    evidence = result.evidence()
    assert evidence["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "QWxpY2U6" not in json.dumps(evidence)
    assert_hash_safe_evidence(evidence)


def test_manifest_policy_binding_and_zero_value_retention() -> None:
    rows = [hash_safe_scan("owner@example.org"), hash_safe_scan("DEBUG=1")]
    inventory_sha = hashlib.sha256(b"inventory").hexdigest()
    manifest = build_manifest(rows, inventory_sha256=inventory_sha)
    assert manifest["privacy_policy_sha256"] == policy_manifest()["policy_sha256"]
    assert manifest["records_total"] == 2
    assert manifest["detector_counts"]["email"] == 1
    assert manifest["public_evidence_policy"]["matched_value_hashes_retained"] is False
    assert_hash_safe_evidence(manifest)
