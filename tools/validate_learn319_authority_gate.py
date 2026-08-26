from __future__ import annotations
import hashlib, json, pathlib, sys

DEFAULT = pathlib.Path('evidence/learn319/authority-gate.json')

def canonical_hash(obj: dict) -> str:
    clone = dict(obj)
    clone.pop('identity_sha256', None)
    raw = json.dumps(clone, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def validate(path: pathlib.Path) -> None:
    obj = json.loads(path.read_text(encoding='utf-8'))
    assert obj['worker_id'] == 'LEARN-319-EXTERNAL-REAL-3M-SHORT'
    assert obj['identity_sha256'] == canonical_hash(obj)
    assert obj['decision'] == 'BLOCKED_NO_TERMINAL_FROZEN_CORPUS'
    truth = obj['truth']
    assert truth['training_executed'] is False
    assert truth['optimizer_updates'] == 0
    assert truth['learned_checkpoint_emitted'] is False
    assert truth['stage_promotion_authorized'] is False
    auth = obj['authority_cutoff']
    assert auth['data300_contract_head'] == '8ea7f830e50a23754d189dd4134f4afad76a7ee9'
    assert auth['data300_corpus_state'] == 'NOT_BUILT_NOT_FROZEN_NOT_TERMINAL'
    assert auth['data301_vs_data300'] == 'identical'
    assert auth['data301_ahead_by'] == 0
    assert auth['data301_terminal_release_found'] is False
    cand = obj['observed_candidate']
    assert cand['full_five_source_unique_loss_ledger_present'] is False
    assert cand['terminal_nonempty_selection_validation_present'] is False
    prereg = obj['preregistered_successor_contract']
    assert prereg['model']['parameters'] == 3213120
    assert prereg['tokenizer'] == {'id':'s0-byte-v1','special_tokens':0,'type':'canonical_byte','vocab_size':256}
    budget = prereg['token_budget']
    assert budget['requested_short_anchor_optimized_targets'] == 131938
    assert budget['rule'] == 'min(131938, terminal_one_pass_unique_train_optimized_targets)'
    assert budget['replay_allowed'] is False
    assert budget['sampling_with_replacement_allowed'] is False
    assert prereg['selection']['best_and_final_retained_separately'] is True
    assert prereg['selection']['final_test_influences_selection'] is False
    print(json.dumps({'validation':'PASS','runnable':False,'identity_sha256':obj['identity_sha256']}, sort_keys=True))

if __name__ == '__main__':
    validate(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
