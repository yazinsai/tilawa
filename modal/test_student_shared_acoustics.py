"""Algebra/topology checks; run on the CPU executor with numpy and ONNX."""
import unittest
import numpy as np
from student_shared_acoustics import boundary, topology, shared_projection
from student_shared_acoustics_export import projection


class SharedAcousticsTests(unittest.TestCase):
    def test_shared_projection_rejects_high_variance_independent_noise(self):
        rng=np.random.default_rng(9)
        rows=[]
        for _ in range(4):
            speech=rng.normal(size=(1000,1))
            a=np.concatenate([speech+.1*rng.normal(size=speech.shape),10*rng.normal(size=speech.shape)],axis=1)
            b=np.concatenate([speech+.1*rng.normal(size=speech.shape),10*rng.normal(size=speech.shape)],axis=1)
            rows.append(dict(view0=a,view1=b))
        result=shared_projection(rows,dimensions=1)
        self.assertGreater(abs(result['sharedProjection'][0][0]),20*abs(result['sharedProjection'][0][1]))
        self.assertGreater(abs(result['controlProjection'][0][1]),20*abs(result['controlProjection'][0][0]))
        self.assertGreater(result['correlations'][0],.4)
        self.assertEqual(len(result['sharedProjection']),len(result['controlProjection']))

    def test_replicates_require_identical_frames(self):
        with self.assertRaises(ValueError):
            shared_projection([dict(view0=np.zeros((5,2)),view1=np.zeros((4,2)))],1)

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
        self.assertEqual(feature, 'shared_effective_features')
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
