import itertools
import math
import unittest
import numpy as np
from student_fisher import boundary, frame_gradients, normalized_projection
from student_fisher_export import projection

class FisherTests(unittest.TestCase):
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
        self.assertEqual(feature, 'fisher_effective_features')
        self.assertEqual(logits, 'logits')
        np.testing.assert_allclose(weight,(q.astype(float)-zero)*scale)
        self.assertEqual(extra[0].op_type,'DequantizeLinear')

    def test_gradient_matches_finite_difference_likelihood(self):
        mixture=[dict(weight=.4,mean=[-1.,.2],variance=[.6,1.2]),dict(weight=.6,mean=[1.,-.2],variance=[.8,.7])]
        model={'mixture':mixture}
        x=np.array([[.3,-.1],[1.2,.5]])
        gradients=frame_gradients(model,x).mean(0).reshape(2,4)
        def likelihood():
            from scipy.special import logsumexp
            return float(np.mean(logsumexp(np.column_stack([math.log(g['weight'])-.5*(np.log(2*np.pi*np.array(g['variance']))+(x-g['mean'])**2/g['variance']).sum(1) for g in mixture]),axis=1)))
        eps=1e-5
        for k,g in enumerate(mixture):
            for d in range(2):
                original=g['mean'][d]
                g['mean'][d]=original+eps
                plus=likelihood()
                g['mean'][d]=original-eps
                minus=likelihood()
                g['mean'][d]=original
                expected=(plus-minus)/(2*eps)*math.sqrt(g['variance'][d]/g['weight'])
                self.assertAlmostEqual(gradients[k,d],expected,places=7)
                original=g['variance'][d]
                g['variance'][d]=original+eps
                plus=likelihood()
                g['variance'][d]=original-eps
                minus=likelihood()
                g['variance'][d]=original
                expected=(plus-minus)/(2*eps)*original*math.sqrt(2/g['weight'])
                self.assertAlmostEqual(gradients[k,2+d],expected,places=7)

    def test_complete_boundary_sum_matches_enumeration(self):
        values=np.zeros((4,3,3))
        for end in range(4):
            for n in range(1,min(end+1,3)+1):
                values[end,n-1]=np.array([-.3,.2,-.1])*n
        labels=[0,1,0,1,0]
        paths=[]
        priors=[]
        for durations in itertools.product(range(4),repeat=len(labels)):
            if sum(durations)!=4 or any(p and n==0 for p,n in zip(labels,durations)):
                continue
            end=0
            score=0.
            prior=0.
            for p,n in zip(labels,durations):
                end+=n
                prior-=math.log(3+(p==0))
                if n:
                    score+=values[end-1,n-1,p]
            paths.append(score+prior)
            priors.append(prior)
        from scipy.special import logsumexp
        score,w=boundary(values,[1,1],True)
        self.assertAlmostEqual(score,logsumexp(paths)-logsumexp(priors),places=10)
        # Two repeated phone occurrences each consume one segment.
        self.assertAlmostEqual(w[:,:,1].sum(),2.,places=10)
        self.assertAlmostEqual((w*np.arange(1,4)[None,:,None]).sum(),4.,places=10)
        self.assertEqual(boundary(values,[1,2,1,2,1]),-np.inf)

    def test_power_normalization_is_finite_for_zero(self):
        self.assertTrue(np.isfinite(normalized_projection(np.zeros((1,3)),np.eye(3))).all())

if __name__=='__main__':
    unittest.main()
