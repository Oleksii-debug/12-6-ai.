#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
EXPECTED='aea821a40b6c175ee744d422115dd1bb419ef26d7fcf3133e9ff81c6922d0fae'
PARENT_HEAD='33017f1e344534841b31df5a6e0bfaf5b7cb2bcc'; PARENT_ID='448dd61ed3e0d78d0bca9e202529a79c02811fd67beebe4833373d0c2ab0c0a7'
SEAL='1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b'
def canon(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def req(c,m):
 if not c: raise SystemExit('NEXT100-108 validation failed: '+m)
def validate(p:Path):
 d=json.loads(p.read_text()); ident=d.pop('authority_identity_sha256',None); req(ident==EXPECTED,'authority drift'); req(hashlib.sha256(canon(d)).hexdigest()==ident,'self hash')
 req(d['decision']=='SOURCE_REGISTRY_SUCCESSOR_DELTA_PRE_GLOBAL_DEDUP_NOT_CORPUS_FREEZE','decision')
 par=d['parent_registry']; req(par['head_sha']==PARENT_HEAD and par['registry_identity_sha256']==PARENT_ID,'parent')
 req(par['candidate_normalized_bytes']==303374 and par['candidate_independent_family_count']==11,'parent vector')
 add=d['additive_source']; req(add['family']=='en.project-gutenberg.public-domain-books','family'); req(add['normalized_bytes']==1672110 and add['record_count']==3,'PG bytes'); req(add['seal_authority_identity_sha256']==SEAL,'seal'); req(add['dedicated_workflow_run']==32998859164 and add['dedicated_workflow_conclusion']=='success','source run'); req(add['terminal_decision']=='ADMIT','source not admit'); req(add['evaluation']=='NOT_AUTHORIZED','evaluation broadened')
 inv=d['successor_pre_global_dedup_inventory']; by=inv['by_stratum']; req(by=={'uk':{'normalized_bytes':100856,'family_count':4},'en':{'normalized_bytes':1822753,'family_count':4},'code':{'normalized_bytes':51875,'family_count':4}},'stratum vector')
 req(inv['candidate_normalized_bytes']==sum(x['normalized_bytes'] for x in by.values())==1975484,'total arithmetic'); req(inv['candidate_independent_family_count']==12,'family arithmetic'); req(inv['target_gap_normalized_bytes']==20000000-1975484,'target gap'); req(inv['target_stratum_gaps']=={'uk':8899144,'en':5177247,'code':3948125},'stratum gaps')
 mix=inv['target_mix']; ceilings={k:int(by[k]['normalized_bytes']/mix[k]) for k in ('uk','en','code')}; req(ceilings==inv['stratum_only_ceiling_by_stratum'],'ceiling arithmetic'); req(min(ceilings.values())==inv['stratum_only_no_replay_ceiling_normalized_bytes']==224124,'balanced ceiling'); req(inv['stratum_only_no_replay_ceiling_limiter']=='uk','limiter drift')
 req(inv['gutenberg_candidate_share_global']>0.84 and inv['gutenberg_candidate_share_en']>0.91,'PG share arithmetic'); req(inv['gutenberg_family_requires_downselection_under_25pct_global_60pct_within_stratum_caps'] is True,'family cap warning lost')
 pri=d['acquisition_priority']; req(pri['immediate_balanced_capacity']=='UK_FIRST_CODE_SECOND_EN_NONLIMITING_AT_CURRENT_SCALE','priority drift')
 gates=d['downstream_gates']; req(gates['global_cross_source_exact_near_fragment_lineage_dedup']=='REQUIRED_NEXT','dedup skipped'); req(gates['long_training']=='BLOCKED' and gates['paid_compute']=='NOT_AUTHORIZED','training/compute promoted')
 c=d['claim_boundary']; req(c['corpus_identity'] is None and c['shard_identity'] is None and c['authorized_unique_loss_positions']==0,'corpus/loss fabricated'); req(c['model_training_executed'] is False and c['optimizer_updates']==0 and c['learned_20m'] is False and c['learned_100m'] is False,'model claim')
 return {'status':'PASS','authority_identity_sha256':ident,'candidate_normalized_bytes':1975484,'balanced_source_ceiling':224124,'limiter':'uk','next_priority':['uk','code']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('path',nargs='?',default='configs/data/next100_108_gutenberg_registry_delta_v1.json'); print(json.dumps(validate(Path(ap.parse_args().path)),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
