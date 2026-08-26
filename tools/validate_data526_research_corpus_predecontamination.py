#!/usr/bin/env python3
"""Fail-closed gate for DATA-526 Research Corpus V1 pre-decontamination intake."""
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any
DEFAULT_PATH=Path('configs/data/data526_research_corpus_v1_predecontamination_intake_v1.json')
SCHEMA='12-6.data526-research-corpus-v1-predecontamination-intake.v1'; WORKER='DATA526-RESEARCH-CORPUS-V1-PREDECONTAMINATION-INTAKE'
EXPECTED_DEPENDENCY={'canonical_issue':530,'canonical_pr':538,'branch':'next100-063/canonical-source-registry-convergence-20260826','head_sha':'958ebec0f7c9cb00238c7df70566cefd6b504d92','base_sha':'b0523ccbc4b957615aac849d476cfa851be87578','candidate_registry_identity_sha256':'77fb69c558df8c59fdae00583c955c62ad088cda98fd16b335eedb26fb2d7526'}
EXPECTED_DATA300={'head_sha':'8ea7f830e50a23754d189dd4134f4afad76a7ee9','contract_identity_sha256':'07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5'}
EXPECTED_DATA301={'head_sha':'8820ba1b255f6bb95c7db0531fd846078a1aae01','evidence_identity_sha256':'939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81'}
EXPECTED_DATA287={'head_sha':'b0523ccbc4b957615aac849d476cfa851be87578','registry_identity_sha256':'917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c'}
class ValidationError(RuntimeError): pass
def _require(c:bool,m:str)->None:
    if not c: raise ValidationError(m)
def _empty_manifest(m:dict[str,Any])->None:
    _require(m.get('materialized') is False,'blocked intake cannot be materialized')
    for key in ('source_object_ids','family_ids','content_hashes','rights_purpose_decisions','normalization_identities'): _require(m.get(key)==[],f'blocked intake must not populate {key}')
    _require(m.get('composite_identity_sha256') is None,'blocked intake cannot claim composite identity')
def validate(data:dict[str,Any])->dict[str,Any]:
    _require(data.get('schema_version')==SCHEMA,'schema mismatch'); _require(data.get('worker_id')==WORKER,'worker mismatch'); _require(data.get('issue')==526,'issue binding mismatch'); _require(data.get('execution_profile')=='LOCAL_FREE','execution profile must remain LOCAL_FREE'); _require(data.get('status')=='BLOCKED_WAIT_SOURCE_CONVERGENCE','nonterminal dependency requires blocked status')
    for key in ('model_training_executed','tokenizer_fit_executed','paid_compute_used','final_test_payload_read','corpus_or_shards_promoted'): _require(data.get(key) is False,f'unsafe boundary: {key}')
    dep=data.get('source_convergence_dependency'); _require(isinstance(dep,dict),'source convergence dependency missing')
    for key,expected in EXPECTED_DEPENDENCY.items(): _require(dep.get(key)==expected,f'canonical source convergence binding changed: {key}')
    _require(dep.get('candidate_decision')=='CONVERGED_TERMINAL_SOURCE_VECTOR_PRE_GLOBAL_DEDUP_NOT_CORPUS_FREEZE','candidate truth boundary changed'); _require(dep.get('candidate_normalized_bytes')==565743,'candidate bytes drift'); _require(dep.get('candidate_independent_family_count')==13,'candidate family count drift'); _require(dep.get('candidate_source_authority_count')==14,'candidate authority count drift')
    _require(dep.get('candidate_by_stratum')=={'uk':{'normalized_bytes':100856,'family_count':4},'en':{'normalized_bytes':168544,'family_count':4},'code':{'normalized_bytes':296343,'family_count':5}},'candidate stratum vector drift')
    _require(dep.get('terminal_required') is True,'terminal dependency requirement missing'); _require(dep.get('terminal_at_cutoff') is False,'blocked cutoff cannot claim terminal source convergence'); _require(dep.get('pr_state_at_cutoff')=='OPEN_DRAFT','canonical successor was not recorded as open draft'); _require(dep.get('exact_head_terminal_verification_claimed') is False,'exact-head terminal verification cannot be invented'); _require(dep.get('consumed_as_authority') is False,'nonterminal successor cannot be consumed as authority')
    ci=dep.get('observed_generic_ci'); _require(isinstance(ci,dict),'generic CI observation missing'); _require(ci.get('run_id')==33005811593 and ci.get('name')=='CI','generic CI binding changed'); _require(ci.get('status_at_cutoff')=='queued' and ci.get('conclusion_at_cutoff') is None,'recorded cutoff must remain historical queued evidence'); _require(ci.get('accepted_as_terminal_authority') is False,'generic queued CI cannot authorize source convergence')
    lineage=data.get('incumbent_lineage_reference'); _require(isinstance(lineage,dict),'incumbent lineage missing')
    for name,expected in (('data300',EXPECTED_DATA300),('data301',EXPECTED_DATA301),('data287',EXPECTED_DATA287)):
        row=lineage.get(name); _require(isinstance(row,dict),f'{name} lineage missing')
        for key,value in expected.items(): _require(row.get(key)==value,f'{name} binding changed: {key}')
    d301=lineage['data301']; _require(d301.get('terminal_state')=='TERMINAL_BLOCKED','DATA-301 historical state drift'); _require(d301.get('corpus_identity') is None and d301.get('shard_identity') is None,'DATA-301 cannot gain corpus identity here')
    manifest=data.get('predecontamination_candidate_manifest'); _require(isinstance(manifest,dict),'candidate manifest missing'); _empty_manifest(manifest)
    contract=data.get('fail_closed_contract'); _require(isinstance(contract,dict) and contract,'fail-closed contract missing')
    for key,value in contract.items(): _require(value is False,f'fail-closed boundary weakened: {key}')
    req=data.get('activation_requirements'); _require(isinstance(req,list) and len(req)>=5,'activation requirements incomplete'); joined='\n'.join(str(x) for x in req)
    for term in ('terminal','registry identity','source object id','content hash','rights-purpose','RETEST','composite identity'): _require(term.lower() in joined.lower(),f'activation contract missing {term}')
    handoff=data.get('handoff'); _require(isinstance(handoff,dict),'handoff missing'); _require(handoff.get('current_next_action')=='WAIT_CANONICAL_NEXT100_063_TERMINAL_THEN_MATERIALIZE_PREDECONTAMINATION_MANIFEST','next action changed'); _require(handoff.get('after_activation')=='RESERVED_EVALUATION_EXACT_NEAR_MATCH_DECONTAMINATION','decontamination handoff changed')
    for key in ('training_authorized_now','tokenizer_fit_authorized_now','decontamination_authorized_now'): _require(handoff.get(key) is False,f'premature authorization: {key}')
    boundary=data.get('claim_boundary'); _require(isinstance(boundary,dict) and boundary,'claim boundary missing')
    for key,value in boundary.items(): _require(value is False,f'premature claim: {key}')
    return {'status':'PASS_BLOCKED','project_state':'BLOCKED_WAIT_SOURCE_CONVERGENCE','dependency_pr':538,'dependency_head':EXPECTED_DEPENDENCY['head_sha'],'candidate_registry_identity':EXPECTED_DEPENDENCY['candidate_registry_identity_sha256'],'candidate_manifest_materialized':False,'next_action':handoff['current_next_action']}
def main(argv:list[str]|None=None)->int:
    args=sys.argv[1:] if argv is None else argv; path=Path(args[0]) if args else DEFAULT_PATH
    try: report=validate(json.loads(path.read_text(encoding='utf-8')))
    except (OSError,json.JSONDecodeError,ValidationError,KeyError,TypeError,ValueError) as exc: print(f'DATA526 FAIL: {exc}',file=sys.stderr); return 1
    print(json.dumps(report,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
