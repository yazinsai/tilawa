"""Algebra/topology checks; run on the CPU executor with numpy and ONNX."""
import unittest
import numpy as np
from student_deconvolved_phone import boundary, topology, noise_covariance, latent_posterior, fit
from student_deconvolved_phone_export import projection


class DeconvolvedPhoneTests(unittest.TestCase):
    def test_noise_is_separated_from_between_occurrence_variance(self):
        rng=np.random.default_rng(9)
        latent=rng.normal(size=(20000,2))*[3,1]
        a=latent+rng.normal(size=latent.shape)*[.2,2]
        b=latent+rng.normal(size=latent.shape)*[.2,2]
        noise=noise_covariance([dict(view0=a,view1=b)])
        np.testing.assert_allclose(np.diag(noise),[.04,4],rtol=.04)
        self.assertLess(noise[0,0],np.var(a[:,0])/100)

    def test_latent_posterior_matches_gaussian_conditioning(self):
        mean=np.array([1.,2.]);cov=np.array([[2.,.4],[.4,1.]])
        noise=np.array([[3.,.1],[.1,2.]])
        average=np.array([[2.,-1.]])
        expected,uncertainty=latent_posterior(average,mean,cov,noise)
        exact=np.linalg.inv(np.linalg.inv(cov)+2*np.linalg.inv(noise))
        np.testing.assert_allclose(uncertainty,exact,atol=1e-12)
        np.testing.assert_allclose(expected,(exact@(np.linalg.solve(cov,mean)+2*np.linalg.solve(noise,average[0])))[None,:])

    def test_constrained_fit_recovers_latent_spread_instead_of_subtracting(self):
        rng=np.random.default_rng(2); rows=[]
        for _ in range(4):
            y=rng.normal(size=(500,2))*[.3,.7]+[1.,2.]
            a=y+rng.normal(size=y.shape)*[1.,.5]
            b=y+rng.normal(size=y.shape)*[1.,.5]
            # One complete phone, blank effectively absent. Identical data/initialization.
            rows.append(dict(view0=a,view1=b,log_probs=np.tile([-1000.,0.],(len(a),1)),ids=[1]))
        noise=noise_covariance(rows)
        ordinary=fit(rows,2,1.,noise,False)
        latent=fit(rows,2,1.,noise,True)
        def covariance(arm):
            l=np.linalg.inv(arm['phones'][1]['precision']);return l@l.T
        self.assertLess(covariance(latent)[0,0],covariance(ordinary)[0,0]/2)
        self.assertGreater(np.linalg.eigvalsh(covariance(latent)).min(),0)
        np.testing.assert_allclose(latent['phones'][1]['mean'],[1,2],atol=.08)

    def test_replicates_require_identical_frames(self):
        with self.assertRaises(ValueError):
            noise_covariance([dict(view0=np.zeros((5,2)),view1=np.zeros((4,2)))])

    def test_quantized_head_uses_effective_codes_and_per_channel_scales(self):
        from onnx import helper, numpy_helper, TensorProto
        q = np.array([[5, 2], [7, -3], [1, 4]], dtype=np.int8)
        scale = np.array([.2, .4], dtype=np.float32)
        zero = np.array([1, -1], dtype=np.int8)
        initializers = [numpy_helper.from_array(v, k) for k,v in [('weight',q),('wscale',scale),('wzero',zero)]]
        nodes = [helper.make_node('DynamicQuantizeLinear',['h'],['hq','hs','hz']),
            helper.make_node('MatMulInteger',['hq','weight','hz','wzero'],['integer'],name='/ctc_output/MatMul'),
            helper.make_node('Cast',['integer'],['float'],to=TensorProto.FLOAT),
            helper.make_node('Mul',['hs','wscale'],['scale']),
            helper.make_node('Mul',['float','scale'],['logits']),
            helper.make_node('LogSoftmax',['logits'],['log_probs'],axis=-1)]
        graph = helper.make_model(helper.make_graph(nodes,'test',[],[],initializer=initializers))
        feature, logits, weight, extra = projection(graph)
        self.assertEqual(feature, 'deconvolved_effective_features')
        self.assertEqual(logits, 'logits')
        np.testing.assert_allclose(weight,(q.astype(float)-zero)*scale)
        self.assertEqual(extra[0].op_type,'DequantizeLinear')

    def test_boundary_mass_sums_to_one_and_accepts_only_complete_labels(self):
        values = np.zeros((4, 3))
        for ids in ([1], [1, 2], [1, 1]):
            score, occupancy = boundary(values, ids, True)
            self.assertAlmostEqual(score, 0)
            np.testing.assert_allclose(occupancy.sum(1), 1)
        self.assertEqual(boundary(values[:1], [1, 2]), -np.inf)

    def test_soft_occupancies_match_exhaustive_complete_paths(self):
        values = np.array([[.1,.3,.2],[.5,.4,.1],[.2,.8,.4],[.4,.2,.7]])
        for ids in ([1], [1,2], [1,1]):
            labels, edges = topology(ids)
            paths = []
            def visit(path, prior):
                if len(path) == len(values):
                    if path[-1] >= len(labels)-2:
                        mass = prior*np.exp(sum(values[t,labels[s]] for t,s in enumerate(path)))
                        paths.append((path,prior,mass))
                    return
                for target in edges[path[-1]]:
                    visit(path+[target],prior/len(edges[path[-1]]))
            visit([0],1.)
            visit([1],1.)
            total = sum(p[2] for p in paths)
            expected = np.zeros_like(values)
            for path, _, mass in paths:
                for t, s in enumerate(path):
                    expected[t,labels[s]] += mass/total
            score, occupancy = boundary(values,ids,True)
            self.assertAlmostEqual(score,np.log(total/sum(p[1] for p in paths)))
            np.testing.assert_allclose(occupancy,expected,atol=1e-12)


if __name__ == '__main__':
    unittest.main()
