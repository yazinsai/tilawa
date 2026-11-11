"""Mathematical and sampling tests; no practice inputs."""
import unittest
import numpy as np
from student_pairwise_phone import architecture, coupled_logits, mine_pairs
from student_phone_occupancy import occupancies, sequence_score

class PhoneTournaments(unittest.TestCase):
    def test_parameter_parity(self):
        for vocab in (3, 29, 47, 56, 65):
            for pairs in (1, 5, 12, 40):
                widths, biases, hidden, count = architecture(pairs, vocab)
                self.assertEqual(sum(34*w for w in widths)+biases, count)
                self.assertEqual((33+vocab)*hidden+vocab, count)
                self.assertGreaterEqual(min(widths), 4)

    def test_sparse_cycle_reconciliation(self):
        control = np.log([[.2,.1,.2,.3,.2]])
        pairs = [(1,2),(2,3),(1,3)]
        actual = coupled_logits(control,np.array([[2.,-1.,0.]]),pairs)
        np.testing.assert_allclose(np.exp(actual).sum(1),1.,atol=1e-12)
        np.testing.assert_array_equal(actual[:,[0,4]],control[:,[0,4]])
        np.testing.assert_allclose(np.exp(actual[:,1:4]).sum(1),.6)
        # Least-squares residual is orthogonal to the incidence columns.
        scores = actual[0,1:4]
        incidence = np.array([[1,-1,0],[0,1,-1],[1,0,-1]])
        np.testing.assert_allclose(incidence.T@(incidence@scores-[2,-1,0]),0,atol=1e-12)

    def test_confusions_come_from_decodes_with_independent_speakers(self):
        rows=[]
        for speaker in ('a','b'):
            rows.append(dict(speaker=speaker,targets=np.eye(4)[[1,2,3]],logProbs=np.log(np.array([
                [.01,.04,.9,.05],[.01,.9,.04,.05],[.01,.05,.05,.89]]))))
        pairs=mine_pairs(rows,'بتث')
        self.assertIn((1,2),pairs)
        with self.assertRaises(ValueError):
            mine_pairs(rows[:1],'بتث')

    def test_soft_occupancies_match_enumerated_ctc_paths(self):
        import itertools
        probabilities=np.array([[.2,.5,.3],[.3,.3,.4],[.4,.4,.2]])
        weighted=[]
        for path in itertools.product(range(3),repeat=3):
            collapsed=[p for i,p in enumerate(path) if p and (not i or p!=path[i-1])]
            if collapsed==[1,2]:
                weighted.append((path,np.prod([probabilities[t,p] for t,p in enumerate(path)])))
        wanted=np.zeros_like(probabilities)
        for path,weight in weighted:
            for t,p in enumerate(path):
                wanted[t,p]+=weight
        wanted/=sum(w for _,w in weighted)
        actual,_=occupancies(np.log(probabilities),[1,2])
        np.testing.assert_allclose(actual,wanted,atol=1e-7)
        self.assertTrue(np.any((actual>0)&(actual<1)))

    def test_mining_does_not_invent_unobserved_confusions(self):
        rows = [dict(speaker=s, targets=np.eye(4)[[1,2,3]],
                     logProbs=np.log(np.eye(4)[[1,2,3]] + 1e-9)) for s in ('a','b')]
        with self.assertRaises(ValueError):
            mine_pairs(rows, 'بتث')

    def test_complete_ctc_acceptance(self):
        neutral=np.full((8,3),-1.)
        self.assertAlmostEqual(sequence_score(neutral,[1]),-8.)
        self.assertAlmostEqual(sequence_score(neutral,[1,1,2]),-8.)
        self.assertEqual(sequence_score(neutral[:1],[1,1]),-np.inf)

if __name__=='__main__':
    unittest.main()
