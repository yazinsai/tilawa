"""Small numerical and provenance checks; no dataset or practice examples."""
import unittest
import numpy as np
import torch
from student_session_readout import ridge_update, occupancies, prepare, speaker_folds

class ReadoutTests(unittest.TestCase):
    def test_outer_query_loss_differentiates_through_initialization_and_regularizer(self):
        w=torch.zeros((2,3),dtype=torch.float64,requires_grad=True)
        rho=torch.ones(2,dtype=torch.float64,requires_grad=True)
        x=torch.tensor([[1.,1.],[2.,1.]],dtype=torch.float64)
        y=torch.tensor([[.1,.8,.1],[.2,.7,.1]],dtype=torch.float64)
        adapted=ridge_update(w,rho.exp(),x,y,torch.full((2,),.5,dtype=torch.float64))
        q=torch.tensor([[3.,1.]],dtype=torch.float64)
        loss=-(q@adapted).log_softmax(-1)[0,1]
        loss.backward()
        self.assertGreater(float(w.grad.abs().sum()),0)
        self.assertGreater(float(rho.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(rho.grad).all())

    def test_ridge_matches_sdk_closed_form(self):
        x=torch.ones((2,2),dtype=torch.float64)
        y=torch.tensor([[.1,.8,.1],[.2,.7,.1]],dtype=torch.float64)
        w=ridge_update(torch.zeros((2,3),dtype=torch.float64),torch.ones(2,dtype=torch.float64),x,y,torch.full((2,),.5,dtype=torch.float64))
        torch.testing.assert_close(w[0],y.log().mean(0)/3)
        torch.testing.assert_close(w[1],w[0])

    def test_repeated_phone_occupancies_require_complete_paths(self):
        lp=np.log([[.1,.8,.1],[.8,.1,.1],[.1,.8,.1]])
        y,_=occupancies(lp,[1,1])
        np.testing.assert_allclose(y.sum(1),1)
        self.assertEqual(occupancies(lp[:2],[1,1]),None)

    def test_episodes_use_available_earlier_support_and_complete_later_words(self):
        lp=np.log(np.tile([.1,.8,.1],(30,1)))
        row=dict(speaker='train-speaker',features=np.arange(30)[:,None],
                 logProbs=lp,pronunciation=[1,1,1],accepted=[
                     dict(fromFrame=a,toFrame=a+2,availableFrame=a+4,
                          logProbs=lp[a:a+2],pronunciation=[1]) for a in (0,12,24)])
        record=prepare([row],np.array([0.]),np.array([1.]))[0]
        self.assertEqual(len(record['episodes']),2)
        sx,sy,weights,qx,qy=record['episodes'][0]
        self.assertEqual(sx[:,0].tolist(),[0.,1.])
        self.assertEqual(qx[:,0].tolist(),[12.,13.,24.,25.])
        self.assertAlmostEqual(float(weights.sum()),1.)
        # Future query supervision cannot alter the same accepted-prefix solve.
        row['pronunciation']=[2,2,2]
        changed=prepare([row],np.array([0.]),np.array([1.]))[0]['episodes'][0]
        for before,after in zip((sx,sy,weights),changed[:3]):
            torch.testing.assert_close(before,after)
        self.assertFalse(torch.equal(qy,changed[4]))

    def test_speakers_are_indivisible_partitions(self):
        rows=[dict(speaker=s) for s in ('a','b','c','a','d')]
        folds=speaker_folds(rows)
        self.assertEqual(set(folds),{'a','b','c','d'})
        with self.assertRaises(ValueError):speaker_folds(rows[:2])

if __name__=='__main__':unittest.main()
