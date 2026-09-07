from __future__ import annotations
import json
from copy import deepcopy
from tools.validate_next100_065f_exact_hash_prefilter import PATH, PrefilterError, validate
import pytest
def load(): return json.loads(PATH.read_text())
def test_exact_evidence_passes(): validate(load())
def test_no_raw_exact_overlap_and_no_new_internal_dups():
 d=load(); assert d['result']['historical_vs_new_raw_exact_sha256_intersections']==0; assert d['result']['new_internal_exact_duplicate_sha256_groups']==0
def test_remains_nonterminal_for_full_dedup():
 d=load(); b=d['claim_boundary']; assert b['global_dedup_terminal'] is False; assert b['near_copy_match_proven'] is False; assert b['authorized_training_exposure']==0
@pytest.mark.parametrize('where,key,val',[('historical_v7','workflow_conclusion','failure'),('new_bundle','artifact_zip_sha256','0'*64),('claim_boundary','global_dedup_terminal',True),('claim_boundary','authorized_training_exposure',1)])
def test_rejects_drift(where,key,val):
 d=load(); d=deepcopy(d); d[where][key]=val
 with pytest.raises(PrefilterError): validate(d)
def test_rejects_fabricated_zero_overlap():
 d=load(); d=deepcopy(d); d['new_bundle']['raw_sha256'][0]=d['historical_v7']['raw_sha256'][0]; d.pop('evidence_identity_sha256');
 from tools.validate_next100_065f_exact_hash_prefilter import identity
 d['evidence_identity_sha256']=identity(d)
 with pytest.raises(PrefilterError, match='intersection'): validate(d)
