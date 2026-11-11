"""Small synthetic numerical checks; no practice recordings or model weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from student_preactivation_export import activation_sites
from student_preactivation_residual import boundary, design, observations, predictor_fit, topology, update


class ResidualTests(unittest.TestCase):
    def test_identifies_complete_noninvertible_activation_formula(self):
        def node(op, inputs, output):
            return SimpleNamespace(op_type=op,input=inputs,output=[output],name='encoder/'+output,attribute=[])
        nodes = [node('Sub',['x','shift'],'shifted'),node('Exp',['shifted'],'exp'),
            node('Add',['exp','one'],'plus'),node('Log',['plus'],'log'),
            node('Mul',['x','beta'],'linear'),node('Sub',['log','linear'],'difference'),
            node('Sub',['difference','offset'],'activated')]
        constants = [SimpleNamespace(name=k,data=np.array(v)) for k,v in
            [('shift',4.),('one',1.),('beta',.08),('offset',.035)]]
        graph = SimpleNamespace(graph=SimpleNamespace(node=nodes,initializer=constants))
        with patch.dict('sys.modules',{'onnx':SimpleNamespace(numpy_helper=SimpleNamespace(to_array=lambda t:t.data))}):
            sites = activation_sites(graph)
            self.assertEqual([(s['pre'],s['post'],s['kind']) for s in sites],[('x','activated','Swoosh')])
            constants[2].data = np.array(0.)
            # beta zero is monotone softplus, not the proposed information loss.
            self.assertEqual(activation_sites(graph),[])

    def test_excludes_injective_primitives_and_head_activations(self):
        nodes = [SimpleNamespace(op_type=op,input=['x'],output=[str(i)],name=name,attribute=[])
            for i,(op,name) in enumerate([('Sigmoid','encoder/sigmoid'),('Tanh','encoder/tanh'),
                ('Relu','ctc_output/relu'),('Relu','encoder/relu')])]
        graph = SimpleNamespace(graph=SimpleNamespace(node=nodes,initializer=[]))
        with patch.dict('sys.modules',{'onnx':SimpleNamespace(numpy_helper=SimpleNamespace(to_array=lambda t:t.data))}):
            self.assertEqual([s['post'] for s in activation_sites(graph)],['3'])

    def test_complete_boundary_marginal_matches_exhaustive_paths(self):
        values = np.log(np.array([[.2,.5,.3],[.5,.3,.2],[.3,.4,.3],[.6,.1,.3]]))
        for ids in ([1], [1,2], [1,1]):
            labels, edges = topology(ids)
            paths = [(s, values[0, labels[s]], 0.) for s in (0,1)]
            for t in range(1,len(values)):
                paths = [(v, likelihood+values[t,labels[v]]-np.log(len(edges[s])),
                    prior-np.log(len(edges[s]))) for s,likelihood,prior in paths for v in edges[s]]
            complete = [(a,b) for s,a,b in paths if s >= len(labels)-2]
            expected = np.logaddexp.reduce([a for a,b in complete])-np.logaddexp.reduce([b for a,b in complete])
            score, occupancy = boundary(values,ids,True)
            self.assertAlmostEqual(score,expected,places=12)
            np.testing.assert_allclose(occupancy.sum(1),1,atol=1e-12)
        self.assertEqual(boundary(values[:1],[1,1]),-np.inf)

    def test_ridge_removes_predictable_pre_information(self):
        rng = np.random.default_rng(7)
        post = rng.normal(size=(400,2))
        pre = np.column_stack([3*post[:,1],np.zeros(400)])
        site = dict(width=2,preProjection=[[1,0]]*8,postProjection=[[1,0]]*16,
            nonlinearProjection=[[0,0]]*64)
        rows = [dict(pairs=[dict(pre=pre,post=post)],log_probs=np.zeros((200,3)))]
        predictor = predictor_fit(rows,site,0,1e-6)
        result = observations(rows,site,0,predictor)[0]
        self.assertLess(np.abs(result['residual'][:,8:]).max(),1e-4)
        np.testing.assert_allclose(result['residual'][:,:8],result['control'][:,:8])

    def test_ambiguous_activation_branch_survives_conditioning(self):
        # ReLU maps every negative preactivation to the SAME post tensor.
        pre = np.array([[-1.,0],[-3.,0]]*80)
        post = np.zeros_like(pre)
        site = dict(width=2,preProjection=[[1,0]]*8,postProjection=[[1,0]]*16,
            nonlinearProjection=[[0,0]]*64)
        rows = [dict(pairs=[dict(pre=pre,post=post)],log_probs=np.zeros((160,3)))]
        predictor = predictor_fit(rows,site,0,1.)
        predicted = design(post,site,predictor) @ np.array(predictor['coefficients']).T
        np.testing.assert_allclose(predicted,-2.,atol=1e-12)
        result = observations(rows,site,0,predictor)[0]
        np.testing.assert_allclose(result['residual'][0,8:],1.)
        np.testing.assert_allclose(result['residual'][1,8:],-1.)
        np.testing.assert_allclose(result['control'],0.)

    def test_regularized_sparse_phone_covariance_is_positive(self):
        count=np.array([2.,0.]); sums=np.zeros((2,16))
        cross=np.array([np.eye(16)*4,np.zeros((16,16))])
        fitted=update(count,sums,cross,16)
        for e in fitted:
            precision=np.array(e['precision'])
            self.assertTrue(np.isfinite(precision).all())
            self.assertTrue((precision.diagonal()>0).all())


if __name__ == '__main__':
    unittest.main()
