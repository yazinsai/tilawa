import unittest
from sparse_mining_core import attach_timing,failures,freeze_arms,persistent,runs,word_target,validated_weights
class Sparse(unittest.TestCase):
 def test_repeat_needs_two_emissions(self):
  t=[dict(token=2,word=0,start=1,end=2),dict(token=2,word=1,start=4,end=5)]
  self.assertEqual(len(failures(t,runs([0,2,0,0,0]))['missing']),1)
 def test_word_boundary_survives_duration_collapse(self):
  t=word_target([{'phones':'aa'},{'phones':'a'}],lambda s:list(s))
  self.assertEqual([(r['token'],r['word']) for r in t],[('a',0),('a',1)])
 def test_far_match_not_borrowed(self):
  t=[dict(token=2,word=0,start=1,end=2)]
  self.assertEqual(failures(t,[dict(token=2,start=30,end=31)])['missing'],[0])
 def test_timing_and_persistence(self):
  t=attach_timing([dict(token=2,word=0)],runs([0,2,0]))
  self.assertTrue(t[0]['short'])
  self.assertEqual(persistent(t,[{'missing':[0]},{'missing':[0]},{'missing':[]}],{2}),[0])
 def test_cap_weights_and_random_reproducible(self):
  keys=[str(i) for i in range(1000)]; c=keys[:300]
  a=freeze_arms(keys,c);self.assertEqual(a['n'],20);self.assertEqual(len(a['random']),20)
  self.assertEqual(a,freeze_arms(keys,c));self.assertEqual(a['weight'],3.)
 def test_unverified_manifest_refused(self):
  import hashlib
  keys=[str(i) for i in range(100)]
  d={'poolSha256':hashlib.sha256('\n'.join(keys).encode()).hexdigest(),
     'policy':{'ctcWeight':1.,'kdWeight':2.,'updates':200,'weight':3.},'selected':['1'],'random':['2']}
  with self.assertRaisesRegex(ValueError,'Unverified'):validated_weights(keys,d,'selected')
 def test_lexical_review_keeps_wrong_attempt(self):
  from finalize_sparse_manifest import normalized
  self.assertNotEqual(normalized('وجعلوا جعلوا'),normalized('جعلوا'))
  self.assertEqual(normalized('ٱلْإِنسَانَ'),normalized('الإنسان'))
 def test_duration_emissions_keep_boundaries(self):
  t=attach_timing([dict(token=2,word=0)],runs([0,2,0,2,0]))
  self.assertEqual(len(t),1);self.assertEqual(len(t[0]['emissions']),2)
  self.assertFalse(t[0]['short'])
  self.assertIsNone(attach_timing([dict(token=2,word=0),dict(token=2,word=1)],runs([0,2,0,2,0])))
 def test_uthmani_review_preserves_word_count(self):
  from finalize_sparse_manifest import lexical_match
  self.assertTrue(lexical_match([{'text':'ٱلسَّمَـٰوَٰتِ'}],'السماوات'))
  self.assertFalse(lexical_match([{'text':'جَعَلُوا'},{'text':'جَعَلُوا'}],'جعلوا'))
if __name__=='__main__':unittest.main()
