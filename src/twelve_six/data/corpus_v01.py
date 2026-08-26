"""Deterministic UK/EN/code corpus V0.1 assembly."""
from __future__ import annotations

import argparse, hashlib, json, shutil, unicodedata
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Mapping

CONFIG_SCHEMA = "12-6.corpus-build-config.v1"
MANIFEST_SCHEMA = "12-6.corpus-manifest.v1"
EXTERNAL_SCHEMA = "12-6.external-source-registry.v1"
RESERVED_SCHEMA = "12-6.reserved-fingerprints.v1"
STRATA = ("uk", "en", "code")

class CorpusBuildError(ValueError): pass

def cjson(v: Any) -> bytes:
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()

def sha(v: bytes) -> str: return hashlib.sha256(v).hexdigest()
def load(path: Path) -> dict[str, Any]:
    v = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(v, dict): raise CorpusBuildError(f"{path}: JSON object required")
    return v

def registry_identity(r: Mapping[str, Any], schema: str, key: str) -> str:
    if r.get("schema_version") != schema or not isinstance(r.get(key), list):
        raise CorpusBuildError("registry schema/content invalid")
    expected = sha(cjson({"schema_version": schema, key: r[key]}))
    if r.get("registry_identity_sha256") != expected: raise CorpusBuildError("registry identity mismatch")
    return expected

def eligible_external(r: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    registry_identity(r, EXTERNAL_SCHEMA, "sources")
    if not r["sources"]: return ()
    from twelve_six.data.external_sources import validate_external_source_registry
    out = []
    for source in validate_external_source_registry(r):
        source.assert_training_eligible()
        out.append(source.to_dict())
    return tuple(out)

def reserved_hashes(r: Mapping[str, Any]) -> frozenset[str]:
    registry_identity(r, RESERVED_SCHEMA, "sets")
    out: set[str] = set()
    for item in r["sets"]:
        for digest in item.get("normalized_sha256", []):
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise CorpusBuildError("reserved fingerprint invalid")
            out.add(digest)
    return frozenset(out)

def norm(text: str, code: bool) -> str:
    if "\ufffd" in text or any(0xD800 <= ord(c) <= 0xDFFF for c in text): raise CorpusBuildError("invalid Unicode")
    text.encode("utf-8", "strict")
    x = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if code: x = x.strip("\n")
    else: x = "\n".join(" ".join(line.split()) for line in x.split("\n") if line.strip()).strip()
    if not x: raise CorpusBuildError("empty normalized record")
    return x

def quality(s: str, text: str) -> bool:
    if len(text) < 80: return False
    if s == "code": return "\n" in text and "def " in text and "return " in text
    letters = [c for c in text if c.isalpha()]
    script = "CYRILLIC" if s == "uk" else "LATIN"
    ratio = sum(script in unicodedata.name(c, "") for c in letters) / max(1, len(letters))
    return len(letters) >= 40 and ratio >= (0.85 if s == "uk" else 0.95) and (s != "uk" or any(c in "іїєґІЇЄҐ" for c in text))

def split_for(record_id: str, salt: str, bp: int) -> str:
    if not 1 <= bp <= 5000: raise CorpusBuildError("validation_basis_points invalid")
    bucket = int.from_bytes(hashlib.sha256(f"{salt}\0{record_id}".encode()).digest()[:8], "big") % 10000
    return "validation" if bucket < bp else "train"

def authored_text(s: str, n: int) -> str:
    uk_t = ("дані","мовна модель","алгоритм","мережа","пам'ять","текст","навчання","перевірка","корпус","токенізація","відтворюваність","якість","метрика","контекст","помилка","версія")
    uk_a = ("пояснює","порівнює","перевіряє","вимірює","зберігає","відокремлює","нормалізує","узгоджує")
    en_t = ("dataset","language model","algorithm","network","memory","text","training","validation","corpus","tokenization","reproducibility","quality","metric","context","error","version")
    en_a = ("explains","compares","checks","measures","stores","separates","normalizes","aligns")
    a, b, d = n % 16, (n // 16) % 8, (n // 1024) % 11 + 1
    if s == "uk": return f"Приклад {n}: цей український фрагмент {uk_a[b]} поняття «{uk_t[a]}» з явним походженням. Спочатку система формулює мету, потім описує вхідні дані, а після цього перевіряє результат. Якщо крок {d} змінює представлення, попередня версія не підмінюється і лишається доступною для аудиту. Запис містить повні речення, українські літери і технічну лексику та прямо позначений як проєктний."
    if s == "en": return f"Example {n}: this English passage {en_a[b]} the concept of {en_t[a]} with visible provenance. The system states the goal first, describes the input next, and then checks the resulting artifact. If step {d} changes a representation, the previous version is not silently replaced and remains auditable. This complete technical passage is explicitly project-authored data."
    if s == "code":
        scale, bias = n % 17 + 2, n % 23 - 11
        return f'def record_{n:07d}(values: list[int]) -> list[int]:\n    """Deterministic project-authored example {n}."""\n    result: list[int] = []\n    for index, value in enumerate(values):\n        adjusted = value * {scale} + {bias} + index\n        if adjusted % 2 == 0:\n            result.append(adjusted)\n        else:\n            result.append(adjusted - 1)\n    return result\n'
    raise CorpusBuildError("unknown stratum")

def authored(config: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    spec = config.get("project_authored")
    if not isinstance(spec, Mapping) or spec.get("enabled") is not True: return
    version, limit = str(spec.get("source_version", "0.1.0")), int(spec.get("max_candidates_per_stratum", 200000))
    for n in range(limit):
        for s in STRATA:
            source_id = f"project-authored:{s}:corpus-v01"; raw = authored_text(s, n)
            rid = sha(f"{source_id}\0{version}\0{n}\0{raw}".encode())[:24]
            yield {"record_id": f"{source_id}:{rid}", "source_id": source_id, "source_version": version, "stratum": s, "external": False, "project_authored": True, "raw_text": raw}

def external_records(sources: tuple[dict[str, Any], ...]) -> Iterable[dict[str, Any]]:
    if sources: raise CorpusBuildError("eligible external sources exist but no materialized source adapter output is configured for corpus V0.1")
    return ()

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[str, int]:
    payload = b"".join(cjson(r) for r in rows); path.write_bytes(payload); return sha(payload), len(payload)

def build_corpus(config_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path); config = load(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA: raise CorpusBuildError("unsupported config schema")
    root = config_path.resolve().parents[2]
    er, rr = load(root / config["external_registry"]), load(root / config["reserved_registry"])
    eid, rid = registry_identity(er, EXTERNAL_SCHEMA, "sources"), registry_identity(rr, RESERVED_SCHEMA, "sets")
    sources, reserved = eligible_external(er), reserved_hashes(rr)
    target = {k: int(v) for k, v in config["target_train_byte_tokens"].items()}
    if set(target) != set(STRATA) or any(v <= 0 for v in target.values()): raise CorpusBuildError("target budget invalid")
    out = Path(output_dir) if output_dir else root / config["output_dir"]
    if out.exists(): shutil.rmtree(out)
    shard_dir = out / "shards"; shard_dir.mkdir(parents=True)
    seen: set[str] = set(); accepted = {s: [] for s in STRATA}; train_tokens = Counter(); counts = Counter()
    for rec in chain(external_records(sources), authored(config)):
        s = rec["stratum"]; counts["candidate_documents"] += 1
        try: text = norm(rec["raw_text"], s == "code")
        except CorpusBuildError: counts["quality_rejected_documents"] += 1; continue
        if not quality(s, text): counts["quality_rejected_documents"] += 1; continue
        fp = sha(text.encode())
        if fp in seen: counts["dedup_rejected_documents"] += 1; continue
        seen.add(fp)
        if fp in reserved: counts["reserved_eval_rejected_documents"] += 1; continue
        split = split_for(rec["record_id"], config["split_salt"], int(config["validation_basis_points"]))
        row = {"record_id": rec["record_id"], "source_id": rec["source_id"], "source_version": rec["source_version"], "stratum": s, "modality": "code" if s == "code" else "natural", "split": split, "external": bool(rec["external"]), "project_authored": bool(rec["project_authored"]), "content_sha256": fp, "byte_tokens": len(text.encode()), "text": text}
        accepted[s].append(row); counts["accepted_documents"] += 1
        if split == "train": train_tokens[s] += row["byte_tokens"]
        if all(train_tokens[s] >= target[s] for s in STRATA): break
    else: raise CorpusBuildError("candidate generator exhausted before target budget")
    rows: list[dict[str, Any]] = []; kept = Counter()
    for s in STRATA:
        for row in accepted[s]:
            if row["split"] == "validation": rows.append(row)
            elif kept[s] < target[s]: rows.append(row); kept[s] += row["byte_tokens"]
    th = {r["content_sha256"] for r in rows if r["split"] == "train"}; vh = {r["content_sha256"] for r in rows if r["split"] == "validation"}
    if th & vh: raise CorpusBuildError("validation leaked into training")
    rows.sort(key=lambda r: (r["split"], r["stratum"], r["record_id"]))
    shards, current, current_bytes = [], [], 0; target_shard = int(config["shard_target_bytes"])
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        size = len(cjson(row))
        if current and current_bytes + size > target_shard: groups.append(current); current, current_bytes = [], 0
        current.append(row); current_bytes += size
    if current: groups.append(current)
    aggs = {k: defaultdict(Counter) for k in ("split","stratum","modality","source","split_stratum")}
    for i, group in enumerate(groups):
        path = shard_dir / f"part-{i:05d}.jsonl"; h, size = write_jsonl(path, group)
        shards.append({"path": f"shards/{path.name}", "sha256": h, "size_bytes": size, "documents": len(group), "byte_tokens": sum(r["byte_tokens"] for r in group)})
        for r in group:
            keys = (("split",r["split"]),("stratum",r["stratum"]),("modality",r["modality"]),("source",r["source_id"]),("split_stratum",f'{r["split"]}:{r["stratum"]}'))
            for kind, key in keys:
                a = aggs[kind][key]; a["documents"] += 1; a["bytes"] += r["byte_tokens"]; a["byte_tokens"] += r["byte_tokens"]
    core = {"schema_version": MANIFEST_SCHEMA, "corpus_version": config["corpus_version"], "builder_sha256": sha(Path(__file__).read_bytes()), "config_sha256": sha(cjson(config)), "external_registry_identity_sha256": eid, "reserved_registry_identity_sha256": rid, "external_training_eligible_sources": len(sources), "truth_boundary": {"contains_external_training_data": any(r["external"] for r in rows), "contains_project_authored_data": any(r["project_authored"] for r in rows), "external_source_diversity_representative": any(r["external"] for r in rows), "claim": "Representative across intended UK/EN/code modalities for local small-model mechanics; not evidence of real-world external corpus representativeness when external_training_eligible_sources is zero."}, "pipeline": ["source_registry","extraction","normalization","quality_filtering","exact_dedup","reserved_eval_removal","stable_train_validation_split","physical_shards","manifest"], "tokenizer_accounting": "canonical byte tokenizer; byte_tokens equals UTF-8 bytes of normalized text", "mixture_target_percent": config["mixture_target_percent"], "target_train_byte_tokens": target, "counters": dict(sorted(counts.items())), "by_split": {k:dict(v) for k,v in sorted(aggs["split"].items())}, "by_stratum": {k:dict(v) for k,v in sorted(aggs["stratum"].items())}, "by_modality": {k:dict(v) for k,v in sorted(aggs["modality"].items())}, "by_source": {k:dict(v) for k,v in sorted(aggs["source"].items())}, "by_split_stratum": {k:dict(v) for k,v in sorted(aggs["split_stratum"].items())}, "shards": shards, "train_validation_content_overlap": 0}
    manifest = {**core, "corpus_identity_sha256": sha(cjson(core))}; (out / "manifest.json").write_bytes(cjson(manifest)); return manifest

def verify_rebuild(config_path: str | Path, first_output: str | Path, second_output: str | Path) -> dict[str, Any]:
    a, b = build_corpus(config_path, first_output), build_corpus(config_path, second_output)
    if a["corpus_identity_sha256"] != b["corpus_identity_sha256"]: raise CorpusBuildError("corpus identity changed on rebuild")
    if [(x["path"],x["sha256"]) for x in a["shards"]] != [(x["path"],x["sha256"]) for x in b["shards"]]: raise CorpusBuildError("shard hashes changed on rebuild")
    return a

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("config", type=Path); p.add_argument("--output-dir", type=Path); p.add_argument("--verify-rebuild", action="store_true"); a = p.parse_args()
    m = verify_rebuild(a.config, a.output_dir / "rebuild-a", a.output_dir / "rebuild-b") if a.verify_rebuild else build_corpus(a.config, a.output_dir)
    print(json.dumps(m, ensure_ascii=False, sort_keys=True, indent=2))

if __name__ == "__main__": main()
