"""Plan2 S12 frozen BPE local fixture, positive/negative/compatibility/restart."""
from __future__ import annotations
import copy
from pathlib import Path
import pytest
from tools import plan2_tokenizer_fit_freeze_v1 as fit
from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.tokenization.base import require_tokenizer_identity,TokenizerCompatibilityError
ROOT=Path(__file__).resolve().parents[1]


def test_identity_from_fixture_has_fixed_specials_and_reproducible_frozen_vocab():
    a=fit.freeze_fixture(ROOT)
    b=fit.freeze_fixture(ROOT)
    assert a==b and a['manifest_sha256']==fit.sha({k:v for k,v in a.items() if k!='manifest_sha256'})
    model=fit.verify_frozen(a)
    assert model.identity.to_dict()==a['tokenizer_identity']
    assert model.special_tokens=={'pad':256,'bos':257,'eos':258,'unk':259}
    assert a['train_record_count']>0 and len(a['fit_merges'])>0
    assert model.vocab_size==260+len(a['fit_merges'])
    assert model.vocab_size<=a['target_vocab_size_future']
    assert not any(a[k] for k in ('training_corpus_authorized','physical_tokenizer_fit_authorized','production_release_authorized','checkpoint_weight_reuse_authorized','paid_compute_used'))


@pytest.mark.parametrize('sample',['','Українська мова','Hello, world!','def f(x):\n return x+1','🙂\r\n\x00','e\u0301 != é','<bos> 𐍈'])
def test_roundtrip_and_bos_eos_special_id_semantics(sample):
    m=fit.verify_frozen(fit.freeze_fixture(ROOT))
    token_ids=m.encode(sample,add_bos=True,add_eos=True)
    assert token_ids[0]==m.bos_id and token_ids[-1]==m.eos_id
    assert m.decode(token_ids)==sample
    assert m.encode(sample)==m.encode(sample)
    with pytest.raises(fit.FitDenied):
        m.decode(token_ids,skip_special_tokens=False)


@pytest.mark.parametrize('bad',[[-1],[True],[99999],['260']])
def test_forged_token_ids_fail_closed(bad):
    m=fit.verify_frozen(fit.freeze_fixture(ROOT))
    with pytest.raises(fit.FitDenied):
        m.decode(bad)


@pytest.mark.parametrize('bad',[[[260,1]],[[256,1]],[[0,260]],[[0,0],[0,0]],[['1',2]]])
def test_future_special_or_duplicate_merge_fails(bad):
    with pytest.raises(fit.FitDenied):
        fit.FrozenBPE(bad)


def test_fit_order_invariant_and_fails_on_duplicate_or_invalid_training():
    records=[{'record_id':str(i),'text':f'Україна та text text {i}'} for i in range(5)]
    assert fit._fit(records,30)==fit._fit(list(reversed(records)),30)
    with pytest.raises(fit.FitDenied):
        fit._fit(records+[records[0]],30)
    with pytest.raises(fit.FitDenied):
        fit._fit([{'record_id':'x','text':''}],30)
    with pytest.raises(fit.FitDenied):
        fit._fit([{'record_id':'x','text':'\ud800'}],30)


def test_manifest_mutation_and_false_permission_fail_closed():
    original=fit.freeze_fixture(ROOT)
    for key,val in [('fit_merges',[]),('manifest_sha256','0'*64),('production_release_authorized',True),('tokenizer_identity',{})]:
        forged=copy.deepcopy(original)
        forged[key]=val
        with pytest.raises(fit.FitDenied):
            fit.verify_frozen(forged)


def test_checkpoint_identity_never_silently_matches_incumbent():
    m=fit.verify_frozen(fit.freeze_fixture(ROOT))
    baseline=ByteTokenizer().identity
    with pytest.raises(TokenizerCompatibilityError):
        require_tokenizer_identity(m,expected_version=baseline.version,
            expected_config_sha256=baseline.config_sha256,
            expected_vocab_size=baseline.vocab_size,
            expected_vocab_sha256=baseline.vocab_sha256)
    ident=m.identity
    require_tokenizer_identity(m,expected_version=ident.version,
        expected_config_sha256=ident.config_sha256,
        expected_vocab_size=ident.vocab_size,
        expected_vocab_sha256=ident.vocab_sha256)


def test_restart_clean_rebuild_and_corrupt_artifact(tmp_path):
    one=fit.stage(ROOT,tmp_path/'one')
    again=fit.stage(ROOT,tmp_path/'one')
    clean=fit.stage(ROOT,tmp_path/'two')
    assert one==again==clean
    p=tmp_path/'one'/'frozen-tokenizer-fixture.json'
    q=tmp_path/'two'/'frozen-tokenizer-fixture.json'
    assert p.read_bytes()==q.read_bytes()
    p.write_text('{}',encoding='utf-8')
    with pytest.raises(fit.FitDenied):
        fit.stage(ROOT,tmp_path/'one')


def test_symlink_destination_and_tampered_policy_rejected(tmp_path):
    (tmp_path/'actual').mkdir()
    (tmp_path/'alias').symlink_to(tmp_path/'actual',target_is_directory=True)
    with pytest.raises(fit.FitDenied):
        fit.stage(ROOT,tmp_path/'alias'/'child')
    policy=ROOT/fit.POLICY_PATH
    assert fit.git_blob(policy.read_bytes())==fit.POLICY_BLOB
    replacement=tmp_path/'other'
    (replacement/'configs'/'data').mkdir(parents=True)
    (replacement/fit.POLICY_PATH).write_text('{"purpose":"PRODUCTION"}')
    with pytest.raises(fit.FitDenied):
        fit._policy(replacement)

def test_frozen_special_ids_and_merge_table_reject_in_place_mutation():
    model = fit.verify_frozen(fit.freeze_fixture(ROOT))
    with pytest.raises(TypeError):
        model.special_tokens["pad"] = 999
    with pytest.raises(TypeError):
        model.merges[0][0] = 999
    with pytest.raises(TypeError):
        model.tokens[0] = b"forged"
