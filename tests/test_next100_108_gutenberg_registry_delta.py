from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/'tools/validate_next100_108_gutenberg_registry_delta.py'; C=ROOT/'configs/data/next100_108_gutenberg_registry_delta_v1.json'
s=importlib.util.spec_from_file_location('v',V); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
class T(unittest.TestCase):
 def load(self): return json.loads(C.read_text())
 def chk(self,d):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.json'; p.write_text(json.dumps(d)); return m.validate(p)
 def test_pass(self): self.assertEqual(m.validate(C)['balanced_source_ceiling'],224124)
 def test_reject_total_inflation(self):
  d=self.load(); d['successor_pre_global_dedup_inventory']['candidate_normalized_bytes']+=1
  with self.assertRaises(SystemExit): self.chk(d)
 def test_reject_pg_family_multi_credit(self):
  d=self.load(); d['successor_pre_global_dedup_inventory']['candidate_independent_family_count']+=2
  with self.assertRaises(SystemExit): self.chk(d)
 def test_reject_fake_balanced_ceiling(self):
  d=self.load(); d['successor_pre_global_dedup_inventory']['stratum_only_no_replay_ceiling_normalized_bytes']=1000000
  with self.assertRaises(SystemExit): self.chk(d)
 def test_reject_training_authorization(self):
  d=self.load(); d['downstream_gates']['long_training']='AUTHORIZED'
  with self.assertRaises(SystemExit): self.chk(d)
if __name__=='__main__': unittest.main()
