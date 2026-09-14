#!/usr/bin/env python3
"""Fail-closed independent authority checks for SWARM-2065 physical clean successor."""
from __future__ import annotations
import hashlib,json,os,subprocess
from collections.abc import Mapping,Sequence
from pathlib import Path
from typing import Any

BASE="0a594f91ceb61586fdf5d7062e4eae2a53ed90f7"
V7_HEAD="d3333ec1b4a508df232a5aefccd6686adda745fb"; V7_TREE="f6bb58379e9e249583480c246b844b673be38b4c"
P623_HEAD="70d6ccc87396d129d00771bbf0b6b29bc673bfc4"; P623_TREE="e10c66fe59d88997a7de40de4775257da896e28a"
BLOCK_ID="ua.verba.nomis1864.bounded24"; BLOCK_FAM="ua.verba.public-domain.nomis1864"
BLOCK_SHA="1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"; BLOCK_BYTES=1659
QUAR="e9f29dd9f710fac057550e5cd671b7f412720e1ceb36568565a79909f11cf5b6"
V7_REPORT="80997ac88b9d604afaf652807cda2a2d9fd0f6cb75754460ae6f4aa7af6e0267"
V7_V3="c33e0d06a469473aac191e9b5bf7baec23322cd3e9200f0caab2633c921afd84"
BULK_REPORT="80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
SRC_SCHEMA="12-6.d03-nomis-free-clean-successor-report.v1"; SURV_SCHEMA="12-6.next100-065f-post-dedup-survivors.v1"
EV_SCHEMA="12-6.d03-nomis-free-data526-successor-evidence.v1"; INV_SCHEMA="12-6.data526-record-inventory.v1"
CLUSTER=sorted(["data-bulk-code1:pallets/werkzeug:src/werkzeug/__init__.py","data-bulk-code1:pallets/werkzeug:src/werkzeug/routing/__init__.py","data-bulk-code1:pallets/werkzeug:src/werkzeug/wrappers/__init__.py"])
SELECTED="data-bulk-code1:pallets/werkzeug:src/werkzeug/routing/__init__.py"
BASE_BLOBS={
"tools/run_next100_065f_global_dedup_v8.py":"3a4df4b3cf6381893a58641dde479d9fd7fbe46d",
"tools/derive_next100_065f_v8_survivors.py":"ae91d60e5d62466c69394abb1c4b27d2e49f40e3",
"src/twelve_six/data/external_llm_provenance_quarantine_v1.py":"5615c2e4732d14cbd5dc418bb6277e52afd579e6",
"configs/data/d03_external_llm_provenance_quarantine_v1.json":"c746f130d3100f90b233c80ffed6f877fa6eabcf",
"tools/materialize_data_bulk_code1_permissive_python_bundle.py":"eda12f74cc8fef35f7de007a930f2bea254059e4",
"configs/data/data_bulk_code1_permissive_python_bundle_v1.json":"05d2f5e6a83d4a8cf8159f422bd9a66a9dd3f393",
"evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json":"b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0",
"configs/data/next100_065f_global_dedup_v8.json":"51d72c0448c4197807780e38b3cbbb94be36ba96",
"tools/materialize_data526_record_inventory_v1.py":"9b0c5b9df66df468e7b6a4a317c2cd82412ee659"}
BEHAVIOR=("tools/run_d03_nomis_free_clean_successor_v1.py","tools/materialize_d03_nomis_free_data526_successor_v1.py","tools/run_d03_nomis_free_v7_data_only_v1.py","tools/verify_d03_nomis_free_execution_authority_v1.py",".github/workflows/d03-nomis-free-clean-successor-v1.yml")
P623={"tools/materialize_data526_records_from_v7.py":"190d8d5d7727230aefb04cc1464ba42e023b32eb","configs/data/data526_record_materialization_v5.json":"ccc6ed614955739439e4ea09423a0ecbe2f9d3b9"}

class AuthorityError(RuntimeError): pass
def req(x:bool,m:str)->None:
    if not x: raise AuthorityError(m)
def canon(x:Any)->bytes:return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def sha(x:bytes)->str:return hashlib.sha256(x).hexdigest()
def blob(x:bytes)->str:return hashlib.sha1(f"blob {len(x)}\0".encode()+x).hexdigest()
def _read_json(p:Path)->dict[str,Any]:
    try:x=json.loads(p.read_text())
    except Exception as e: raise AuthorityError(f"cannot read JSON {p}: {e}") from e
    req(isinstance(x,dict),f"JSON root invalid: {p}"); return x
def git(root:Path,*args:str,text=True):
    env=dict(os.environ);env["GIT_NO_REPLACE_OBJECTS"]="1"
    return subprocess.run(["git","-C",str(root),*args],check=True,capture_output=True,text=text,env=env).stdout
def gtext(root:Path,*args:str)->str:return str(git(root,*args)).strip()
def gbytes(root:Path,*args:str)->bytes:return bytes(git(root,*args,text=False))
def no_replace(root:Path)->None:req(gtext(root,"for-each-ref","--format=%(refname)","refs/replace")=="",f"replace refs present: {root}")
def path_at_head(root:Path,head:str,rel:str)->None:
    try:w=(root/rel).read_bytes();c=gbytes(root,"show",f"{head}:{rel}")
    except Exception as e: raise AuthorityError(f"cannot bind {rel}: {e}") from e
    req(w==c,f"worktree drift: {rel}")

def verify_product_checkout(root:Path,expected:str)->dict[str,Any]:
    req(len(expected)==40 and all(c in "0123456789abcdef" for c in expected),"invalid expected Product head")
    no_replace(root);head=gtext(root,"rev-parse","HEAD");req(head==expected,"Product HEAD drift")
    for rel in BEHAVIOR:path_at_head(root,head,rel)
    for rel,h in BASE_BLOBS.items():
        path_at_head(root,head,rel);req(blob((root/rel).read_bytes())==h,f"base-main binding drift: {rel}")
    return {"product_head_sha":head,"base_main_sha":BASE,"base_main_bound_blob_count":len(BASE_BLOBS)}
def _checkout(root:Path,head:str,tree:str,label:str,blobs:Mapping[str,str]|None=None)->dict[str,str]:
    no_replace(root);req(gtext(root,"rev-parse","HEAD")==head,f"{label} HEAD drift");req(gtext(root,"rev-parse","HEAD^{tree}")==tree,f"{label} tree drift")
    req(gtext(root,"status","--porcelain=v1","--untracked-files=all")=="",f"{label} worktree dirty")
    for rel,h in (blobs or {}).items():path_at_head(root,head,rel);req(blob((root/rel).read_bytes())==h,f"{label} blob drift: {rel}")
    return {"head_sha":head,"tree_sha":tree}
def verify_v7_checkout(root:Path):return _checkout(root,V7_HEAD,V7_TREE,"terminal V7")
def verify_pr623_checkout(root:Path):return _checkout(root,P623_HEAD,P623_TREE,"PR623",P623)
def selfhash(x:Mapping[str,Any],key:str)->str:
    y=dict(x);y.pop(key,None);return sha(canon(y))
def zero(b:Mapping[str,Any],label:str,keys:Sequence[str],optimizer:bool)->None:
    req(b.get("authorized_training_exposure")==0,f"{label} exposure drift")
    if optimizer:req(b.get("optimizer_updates")==0,f"{label} optimizer drift")
    for k in keys:req(k in b and b[k] is False,f"{label} positive/missing {k}")
def srcrows(d:Mapping[str,Any])->dict[str,Mapping[str,Any]]:
    rows=d.get("sources");req(isinstance(rows,list),"source rows missing");out={}
    for r in rows:
        req(isinstance(r,Mapping) and isinstance(r.get("source_id"),str),"invalid source row");sid=str(r["source_id"]);req(sid not in out,"duplicate source id");out[sid]=r
    return out
def clusters(d:Mapping[str,Any])->list[list[str]]:
    t=d.get("terminal_candidates");req(isinstance(t,Mapping) and isinstance(t.get("duplicate_clusters"),list),"clusters missing")
    out=[]
    for c in t["duplicate_clusters"]:
        req(isinstance(c,Sequence) and not isinstance(c,(str,bytes)),"invalid cluster");v=sorted(map(str,c));req(len(v)>=2 and len(v)==len(set(v)),"invalid cluster members");out.append(v)
    return sorted(out)

def verify_source_authority(*,current_root:Path,expected_product_head:str,report:Mapping[str,Any],survivor:Mapping[str,Any])->dict[str,Any]:
    p=verify_product_checkout(current_root,expected_product_head);req(report.get("schema_version")==SRC_SCHEMA,"source schema drift");req(report.get("base_main_sha")==BASE,"base drift");req(report.get("report_sha256")==selfhash(report,"report_sha256"),"source self hash drift");req(report.get("raw_text_emitted") is False,"raw text leak")
    b=report.get("truth_boundary");req(isinstance(b,Mapping),"source boundary missing");zero(b,"source",("tokenizer_fit_authorized","model_training_executed","learned_weights_created","final_test_payload_read","paid_compute_used","foreign_pretrained_weights"),True)
    inc=report.get("incumbent_bindings");req(isinstance(inc,Mapping),"incumbent bindings missing")
    expected={"v8_tool_git_blob_sha1":BASE_BLOBS["tools/run_next100_065f_global_dedup_v8.py"],"survivor_tool_git_blob_sha1":BASE_BLOBS["tools/derive_next100_065f_v8_survivors.py"],"quarantine_module_git_blob_sha1":BASE_BLOBS["src/twelve_six/data/external_llm_provenance_quarantine_v1.py"],"quarantine_config_git_blob_sha1":BASE_BLOBS["configs/data/d03_external_llm_provenance_quarantine_v1.json"],"historical_v7_head_sha":V7_HEAD,"historical_v7_report_sha256":V7_REPORT,"historical_v7_nested_v3_sha256":V7_V3,"bulk_terminal_report_identity_sha256":BULK_REPORT};req(dict(inc)==expected,"incumbent binding drift")
    r=report.get("deauthorization");req(isinstance(r,Mapping),"deauth missing");req(r.get("quarantine_identity_sha256")==QUAR and r.get("blocked_source_id")==BLOCK_ID and r.get("blocked_family")==BLOCK_FAM and r.get("blocked_payload_sha256")==BLOCK_SHA and r.get("blocked_payload_bytes")==BLOCK_BYTES,"deauth identity drift");req(r.get("removed_before_new_global_dedup") is True and r.get("pre_source_object_count")==35 and r.get("post_source_object_count")==34,"deauth transition drift")
    hist=report.get("clean_historical");req(isinstance(hist,Mapping),"clean historical missing");hv=hist.get("source_vector");req(isinstance(hv,Mapping),"historical vector missing");req(hv.get("source_object_count")==34 and hv.get("source_capacity_bytes_before_global_dedup")==2213956 and hv.get("source_family_counts")=={"uk":3,"en":5,"code":6},"historical vector drift");hd=hist.get("dedup_v3");req(isinstance(hd,Mapping),"historical v3 missing");hr=srcrows(hd);req(len(hr)==34 and BLOCK_ID not in hr,"historical source cut drift");req(all(x.get("source_family")!=BLOCK_FAM and x.get("verified_raw_sha256")!=BLOCK_SHA for x in hr.values()),"historical prohibited root survived")
    v=report.get("source_vector");req(isinstance(v,Mapping),"vector missing");req(v.get("source_object_count")==263 and v.get("source_family_counts")=={"uk":3,"en":5,"code":12} and v.get("source_capacity_bytes_before_global_dedup")==6093965 and v.get("conservative_unique_capacity_bytes_after_global_dedup")==6093662 and v.get("duplicate_discount_bytes")==303 and v.get("duplicate_cluster_count")==1,"composed vector drift")
    d=report.get("dedup_v3");req(isinstance(d,Mapping),"composed v3 missing");rows=srcrows(d);req(len(rows)==263 and BLOCK_ID not in rows,"composed source cut drift");req(all(x.get("source_family")!=BLOCK_FAM and x.get("verified_raw_sha256")!=BLOCK_SHA for x in rows.values()),"composed prohibited root survived");req(clusters(d)==[CLUSTER],"exact Werkzeug cluster drift")
    req(survivor.get("schema_version")==SURV_SCHEMA and survivor.get("v8_report_sha256")==report.get("report_sha256"),"survivor source binding drift");req(survivor.get("survivor_authority_sha256")==selfhash(survivor,"survivor_authority_sha256"),"survivor self hash drift");req(survivor.get("post_dedup_survivor_source_object_count")==261 and survivor.get("post_dedup_declared_capacity_bytes")==6093662,"survivor aggregate drift")
    sb=survivor.get("truth_boundary");req(isinstance(sb,Mapping),"survivor boundary missing");zero(sb,"survivor",("tokenizer_fit_authorized","model_training_executed","final_test_payload_read","paid_compute_used"),False)
    sc=survivor.get("duplicate_clusters");req(isinstance(sc,list) and len(sc)==1 and isinstance(sc[0],Mapping),"survivor cluster missing");c=sc[0];req(c.get("member_source_ids")==CLUSTER and c.get("selected_source_id")==SELECTED,"survivor exact cluster drift");cap=max(int(rows[x]["declared_capacity_bytes"]) for x in CLUSTER);tied=sorted(x for x in CLUSTER if int(rows[x]["declared_capacity_bytes"])==cap);req(tied[0]==SELECTED and c.get("selected_declared_capacity_bytes")==cap,"independent survivor oracle drift")
    sr=survivor.get("survivors");req(isinstance(sr,list),"survivor rows missing");sids=[str(x.get("source_id")) for x in sr if isinstance(x,Mapping)];expected_ids=set(rows)-(set(CLUSTER)-{SELECTED});req(len(sids)==261 and set(sids)==expected_ids,"exact survivor set drift")
    return {**p,"source_report_sha256":report["report_sha256"],"survivor_authority_sha256":survivor["survivor_authority_sha256"],"duplicate_cluster_selected_source_id":SELECTED}

def read_jsonl(p:Path)->tuple[list[dict[str,Any]],bytes]:
    raw=p.read_bytes();out=[]
    for i,line in enumerate(raw.splitlines(),1):
        if not line.strip():continue
        try:x=json.loads(line.decode())
        except Exception as e: raise AuthorityError(f"invalid JSONL {p}:{i}: {e}") from e
        req(isinstance(x,dict),"record not object");out.append(x)
    req(raw==b"".join(canon(x)+b"\n" for x in out),f"noncanonical JSONL: {p}");return out,raw
def inventory(records:list[dict[str,Any]])->dict[str,Any]:
    items=[];seen=set()
    for r in records:
        req(set(r)=={"record_id","source_id","family","modality","normalized_payload"},"record schema drift");rid=r["record_id"];req(all(isinstance(r[k],str) and r[k] for k in r),"record text field invalid");req(rid not in seen,"duplicate record id");seen.add(rid);raw=r["normalized_payload"].encode();items.append({"record_id":rid,"source_id":r["source_id"],"family":r["family"],"modality":r["modality"],"payload_sha256":sha(raw),"payload_bytes":len(raw)})
    items.sort(key=lambda x:x["record_id"]);proj=[{"record_id":x["record_id"],"payload_sha256":x["payload_sha256"],"payload_bytes":x["payload_bytes"]} for x in items]
    return {"schema_version":INV_SCHEMA,"record_count":len(items),"total_payload_bytes":sum(x["payload_bytes"] for x in items),"record_inventory_digest_sha256":sha(canon(items)),"payload_inventory_digest_sha256":sha(canon(proj)),"records":items}
def reject(records:Sequence[Mapping[str,Any]])->None:
    for r in records:
        req(r.get("source_id")!=BLOCK_ID and r.get("record_id")!=BLOCK_ID and r.get("family")!=BLOCK_FAM,"blocked identity survived DATA526");p=r.get("normalized_payload");req(isinstance(p,str) and sha(p.encode())!=BLOCK_SHA,"blocked payload survived DATA526")

def verify_data526_authority(*,current_root:Path,expected_product_head:str,source_report:Mapping[str,Any],survivor:Mapping[str,Any],historical_records_path:Path,records_path:Path,inventory:Mapping[str,Any],evidence:Mapping[str,Any],pr623_root:Path|None=None)->dict[str,Any]:
    s=verify_source_authority(current_root=current_root,expected_product_head=expected_product_head,report=source_report,survivor=survivor)
    if pr623_root is not None:verify_pr623_checkout(pr623_root)
    hist,hraw=read_jsonl(historical_records_path);recs,rraw=read_jsonl(records_path);hi=globals()["inventory"](hist);fi=globals()["inventory"](recs);req(hi["record_count"]==47 and hi["total_payload_bytes"]==2213956,"historical DATA526 aggregate drift");req(fi["record_count"]==274 and fi["total_payload_bytes"]==6093662,"DATA526 aggregate drift");req(dict(inventory)==fi,"retained inventory not physical rehash");reject(hist);reject(recs)
    sr=survivor.get("survivors");req(isinstance(sr,list),"survivor rows missing");surv_ids={str(x["source_id"]) for x in sr if isinstance(x,Mapping)};final_ids={str(x["source_id"]) for x in recs};hist_ids={str(x["source_id"]) for x in hist};req(final_ids==surv_ids,"DATA526 source set != survivor set")
    hd=source_report["clean_historical"]["dedup_v3"];hrows=srcrows(hd);req(hist_ids==set(hrows),"physical historical set drift");agg={}
    for r in hist:
        a=agg.setdefault(r["source_id"],{"family":r["family"],"modality":r["modality"],"bytes":0});req(a["family"]==r["family"] and a["modality"]==r["modality"],"historical split metadata drift");a["bytes"]+=len(r["normalized_payload"].encode())
    for sid,a in agg.items():x=hrows[sid];req(a["family"]==x.get("source_family") and a["modality"]==x.get("modality") and a["bytes"]==x.get("declared_capacity_bytes"),f"historical physical binding drift: {sid}")
    req(SELECTED in final_ids and all(x==SELECTED or x not in final_ids for x in CLUSTER),"physical Werkzeug survivor drift")
    req(evidence.get("schema_version")==EV_SCHEMA and evidence.get("execution_head_sha")==expected_product_head,"evidence execution binding drift");req(evidence.get("source_report_sha256")==source_report.get("report_sha256") and evidence.get("survivor_authority_sha256")==survivor.get("survivor_authority_sha256"),"evidence upstream binding drift")
    hm=evidence.get("historical_materializer");req(isinstance(hm,Mapping) and hm.get("head_sha")==P623_HEAD and hm.get("tool_git_blob_sha1")==P623["tools/materialize_data526_records_from_v7.py"] and hm.get("config_git_blob_sha1")==P623["configs/data/data526_record_materialization_v5.json"] and hm.get("transform_contract_reused_without_modification") is True,"PR623 evidence binding drift")
    eh=evidence.get("clean_historical");ef=evidence.get("clean_data526");req(isinstance(eh,Mapping) and isinstance(ef,Mapping),"clean evidence sections missing");req(eh.get("record_count")==47 and eh.get("total_payload_bytes")==2213956 and eh.get("record_payload_jsonl_sha256")==sha(hraw) and eh.get("record_inventory_digest_sha256")==hi["record_inventory_digest_sha256"] and eh.get("payload_inventory_digest_sha256")==hi["payload_inventory_digest_sha256"],"historical evidence rehash drift");req(ef.get("record_count")==274 and ef.get("source_object_count")==261 and ef.get("total_payload_bytes")==6093662 and ef.get("record_payload_jsonl_sha256")==sha(rraw) and ef.get("record_inventory_digest_sha256")==fi["record_inventory_digest_sha256"] and ef.get("payload_inventory_digest_sha256")==fi["payload_inventory_digest_sha256"],"DATA526 evidence rehash drift")
    req(evidence.get("raw_text_emitted_to_durable_evidence") is False,"durable raw text leak");b=evidence.get("truth_boundary");req(isinstance(b,Mapping) and b.get("clean_data526_record_graph_materialized") is True,"physical DATA526 truth flag missing");zero(b,"DATA526",("tokenizer_fit_authorized","model_training_executed","learned_weights_created","final_test_payload_read","paid_compute_used","foreign_pretrained_weights"),True);req(evidence.get("evidence_identity_sha256")==selfhash(evidence,"evidence_identity_sha256"),"evidence self hash drift")
    return {**s,"data526_evidence_identity_sha256":evidence["evidence_identity_sha256"],"record_count":274,"payload_bytes":6093662}
