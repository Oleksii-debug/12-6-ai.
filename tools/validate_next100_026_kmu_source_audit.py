from __future__ import annotations
import hashlib, json, re, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/"configs/data/next100_026_kmu_source_audit_v1.json"
def h(b): return hashlib.sha256(b).hexdigest()
def norm(s):
    s=unicodedata.normalize("NFKC",s)
    return "\n".join(x for x in (" ".join(line.split()) for line in s.splitlines()) if x).strip()+"\n"
def sh(s,k=5):
    w=re.findall(r"[\w’'-]+",s.lower(),flags=re.UNICODE)
    return {" ".join(w[i:i+k]) for i in range(max(0,len(w)-k+1))}
def main():
    m=json.loads(M.read_text(encoding="utf-8"))
    assert m["worker_id"]=="NEXT100-026-DATA-UA-CABINET-MINISTRY"
    assert m["verdict"]=="ADMIT" and m["local_free_only"] and not m["training_executed"]
    assert m["source_family"]["family_id"]=="ua.kmu.portal.secretariat-news"
    assert "ua.rada.open-data.laws-texts" in m["source_family"]["independent_from"]
    assert m["rights"]["license_id"]=="CC-BY-4.0"
    assert m["rights"]["training"]=="ALLOWED_PRETRAINING"
    assert m["rights"]["redistribution"]=="ALLOWED_WITH_ATTRIBUTION"
    assert m["purpose"]=={"evaluation":"NOT_SEPARATELY_ADMITTED","final_test":"PROHIBITED"}
    assert m["selection"]["exclude_ministry_syndication"] and m["selection"]["exclude_normative_acts"]
    incumbent={"154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83","94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a","72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50"}
    seen_r=set(); seen_n=set(); texts=[]
    for r in m["records"]:
        raw=(ROOT/r["raw_path"]).read_bytes(); n=norm(raw.decode()).encode()
        assert h(raw)==r["raw_sha256"] and len(raw)==r["raw_bytes"]
        assert h(n)==r["normalized_sha256"] and len(n)==r["normalized_bytes"]
        assert r["raw_sha256"] not in seen_r and r["normalized_sha256"] not in seen_n
        assert r["normalized_sha256"] not in incumbent
        seen_r.add(r["raw_sha256"]); seen_n.add(r["normalized_sha256"])
        assert r["email_count"]==r["phone_count"]==0 and not r["sensitive_personal_data_detected"]
        assert r["quality"]=="PASS" and r["word_count"]>=60
        assert r["language"]=="uk" and r["cyrillic_letter_ratio"]>=0.99 and r["ukrainian_specific_letter_count"]>=20
        texts.append(n.decode())
    mx=0.0
    for i,a in enumerate(texts):
        A=sh(a)
        for b in texts[i+1:]:
            B=sh(b); mx=max(mx,len(A&B)/len(A|B))
    assert mx<0.10
    assert len(m["records"])==m["selection"]["record_count"]==6
    assert sum(r["raw_bytes"] for r in m["records"])==m["aggregate"]["raw_bytes"]
    assert sum(r["normalized_bytes"] for r in m["records"])==m["aggregate"]["normalized_bytes"]
    check=dict(m); ident=check.pop("manifest_identity_sha256")
    assert h(json.dumps(check,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())==ident
    print(json.dumps({"status":"PASS","records":6,"normalized_bytes":m["aggregate"]["normalized_bytes"],"max_pairwise_5word_shingle_jaccard":mx,"manifest_identity_sha256":ident},sort_keys=True))
if __name__=="__main__": main()
