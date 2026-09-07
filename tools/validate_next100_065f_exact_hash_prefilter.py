"""Validate hash-only exact duplicate prefilter over terminal V7 + #818 artifacts."""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'evidence/next100_065f/exact_hash_prefilter_v1.json'
EXPECTED_ID=None
class PrefilterError(ValueError): pass
def req(c:bool,m:str)->None:
    if not c: raise PrefilterError(m)
def identity(d:dict[str,Any])->str:
    x=deepcopy(d); x.pop('evidence_identity_sha256',None)
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate(d:dict[str,Any])->None:
    req(d.get('schema_version')=='12-6.next100-065f-exact-hash-prefilter.v1','schema drift')
    req(identity(d)==d.get('evidence_identity_sha256'),'identity drift')
    h=d['historical_v7']; n=d['new_bundle']; r=d['result']; b=d['claim_boundary']
    req(h['head_sha']=='d3333ec1b4a508df232a5aefccd6686adda745fb','V7 head drift')
    req(h['workflow_run_id']==33045763964 and h['workflow_conclusion']=='success','V7 run drift')
    req(h['artifact_zip_sha256']=='cca6921a2093d4e033976b23b0af180e9dc1945b624b82e218780f8d20bafd18','V7 artifact drift')
    req(h['report_sha256']=='80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267','V7 report drift')
    req(h['dedup_report_sha256']=='c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84','V7 dedup drift')
    req(len(h['raw_sha256'])==35 and h['source_count']==35,'historical count drift')
    req(n['execution_head_sha']=='a045c602bfcead862f7852924fb78a5d78c992d6','bundle head drift')
    req(n['workflow_run_id']==34155446113 and n['workflow_conclusion']=='success','bundle run drift')
    req(n['artifact_zip_sha256']=='9febb6e4e900df63c58b1cb0ed2a7003df56d6e9bd593e4b4adbaa54496010ac','bundle artifact drift')
    req(n['report_identity_sha256']=='80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985','bundle report drift')
    req(len(n['raw_sha256'])==229 and n['file_count']==229 and n['family_count']==6,'bundle count drift')
    req(n['eligible_utf8_bytes']==3_880_009,'bundle byte drift')
    old=h['raw_sha256']; new=n['raw_sha256']
    req(all(re.fullmatch(r'[0-9a-f]{64}',x) for x in old+new),'invalid sha')
    intersections=sorted(set(old)&set(new)); dup=sorted(x for x,c in Counter(new).items() if c>1)
    req(len(intersections)==r['historical_vs_new_raw_exact_sha256_intersections'],'intersection evidence drift')
    req(len(dup)==r['new_internal_exact_duplicate_sha256_groups'],'internal duplicate evidence drift')
    req(len(intersections)==0 and len(dup)==0,'raw exact duplicates present')
    req(r['raw_exact_prefilter']=='PASS_NO_RAW_BYTE_EXACT_DUPLICATES','verdict drift')
    for k in ('global_dedup_terminal','normalized_exact_match_proven','near_copy_match_proven','fragment_match_proven','code_skeleton_match_proven','post_dedup_capacity_claimed','tokenizer_fit_authorized','model_training_executed','final_test_accessed','paid_compute_used'):
        req(b[k] is False,f'boundary promoted: {k}')
    req(b['authorized_training_exposure']==0,'training exposure promoted')
def main()->None:
    d=json.loads(PATH.read_text()); validate(d); print('NEXT100-065F EXACT-HASH PREFILTER PASS 35x229 raw intersections=0 new_internal=0')
if __name__=='__main__': main()
