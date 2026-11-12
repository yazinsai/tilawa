import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from student_attention_dispersion import boundary, topology, update, matched_features, train


class MarginalTests(unittest.TestCase):
    def test_exhaustive_ctc_paths_and_normalization(self):
        values = np.log(np.array([[.3,.6,.1],[.4,.2,.4],[.2,.3,.5],[.5,.1,.4]]))
        for ids in ([1],[1,2],[1,1]):
            labels, edges = topology(ids)
            paths = [[0],[1]]
            for _ in range(1,len(values)):
                paths = [p+[n] for p in paths for n in edges[p[-1]]]
            paths = [p for p in paths if p[-1] >= len(labels)-2]
            weights = [np.prod([1/len(edges[s]) for s in p[:-1]]) for p in paths]
            likelihood = sum(w*np.exp(sum(values[t,labels[s]] for t,s in enumerate(p))) for p,w in zip(paths,weights))
            expected = np.log(likelihood/sum(weights)) if paths else -np.inf
            self.assertAlmostEqual(boundary(values,ids),expected)

    def test_fractional_occupancy_sums_to_one(self):
        _, w = boundary(np.log(np.full((5,3),1/3)),[1,2],True)
        np.testing.assert_allclose(w.sum(1),1)

    def test_conditioning_does_not_reward_extra_paths(self):
        values = np.full((7,3),-.75)
        for ids in ([1],[1,2],[1,1]):
            self.assertAlmostEqual(boundary(values,ids),-.75*7)

    def test_covariance_regularization_remains_finite(self):
        phones = update(np.array([2.,1.]),np.zeros((2,16)),np.zeros((2,16,16)),4)
        for e in phones:
            self.assertTrue(np.isfinite(e['precision']).all())
            self.assertTrue(np.isfinite(e['logDet']))


class AttentionVarianceTests(unittest.TestCase):
    def test_opposite_values_do_not_lose_dispersion(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper,TensorProto
        from student_attention_export import add_variance
        node = helper.make_node('MatMul',['weights','values'],['mean'])
        graph = helper.make_graph([node],'attention',[
            helper.make_tensor_value_info('weights',TensorProto.FLOAT,[1,1,2]),
            helper.make_tensor_value_info('values',TensorProto.FLOAT,[1,2,1]),
        ],[helper.make_tensor_value_info('mean',TensorProto.FLOAT,[1,1,1])])
        model = helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
        model.ir_version = 10
        add_variance(model,node,'variance')
        model.graph.output.append(helper.make_tensor_value_info('variance',TensorProto.FLOAT,[1,1,1]))
        session = ort.InferenceSession(model.SerializeToString(),providers=['CPUExecutionProvider'])
        weights = np.array([[[.5,.5]]],dtype=np.float32)
        a = session.run(None,dict(weights=weights,values=np.array([[[-1.],[1.]]],dtype=np.float32)))
        b = session.run(None,dict(weights=weights,values=np.zeros((1,2,1),dtype=np.float32)))
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_array_equal(a[1],np.ones((1,1,1)))
        np.testing.assert_array_equal(b[1],np.zeros((1,1,1)))

    def test_unscoped_value_matmuls_exclude_nonlinear_attention(self):
        from onnx import helper
        from student_attention_export import value_attention_nodes
        nodes = [
            helper.make_node('Mul',['before','encoder.encoders.1.out_combiner.bypass_scale'],['scale1']),
            helper.make_node('Add',['before','scale1'],['stage1']),
            helper.make_node('Softmax',['logits'],['weights']),
            helper.make_node('Concat',['cached_val1_0','current'],['values'],axis=0),
            helper.make_node('MatMul',['weights','values'],['mean']),
            helper.make_node('Concat',['cached_nonlin_attn_0','other'],['nonlinear'],axis=0),
            helper.make_node('MatMul',['weights','nonlinear'],['nonlinear_mean']),
            helper.make_node('Mul',['mean','encoder.encoders.2.out_combiner.bypass_scale'],['scale2']),
            helper.make_node('Add',['mean','scale2'],['stage2']),
        ]
        graph = helper.make_model(helper.make_graph(nodes,'unscoped',[],[]))
        selected = value_attention_nodes(graph,dict(states=[dict(name='cached_val1_0'),dict(name='cached_nonlin_attn_0')]))
        self.assertEqual([n.output[0] for n in selected],['mean'])

    def test_masked_cached_values_do_not_contribute(self):
        weights = np.array([[[0.,.25,.75]]])
        values = np.array([[[1000.],[-2.],[2.]]])
        mean = weights @ values
        variance = weights @ (values**2) - mean**2
        np.testing.assert_array_equal(mean,[[[1.]]])
        np.testing.assert_array_equal(variance,[[[3.]]])

    def test_arms_have_equal_dimension_and_invalid_statistics_abstain(self):
        row = dict(log_probs=np.zeros((2,3)),candidates=[dict(mean=[[0.,0.],[1.,1.]],variance=[[1.,1.],[1.,1.]])])
        result = matched_features([row],0,np.ones((16,4)),np.ones((16,2)))[0]
        self.assertEqual(result['dispersion'].shape,result['control'].shape)
        row['candidates'][0]['variance'][0][0] = -1.
        with self.assertRaises(ValueError):
            matched_features([row],0,np.ones((16,4)),np.ones((16,2)))


class TrainingEligibilityTests(unittest.TestCase):
    def rejected(self, row, excluded, message):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'metadata.json').write_text(json.dumps(dict(tokens='بت',heads=[])))
            (root/'excluded.json').write_text(json.dumps(excluded))
            (root/'input.jsonl').write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,message):
                train(SimpleNamespace(input=str(root/'input.jsonl'),metadata=str(root/'metadata.json'),
                    excluded_speakers=str(root/'excluded.json'),out=str(root/'out')))

    def test_scripted_mistakes_and_nontraining_splits_are_rejected(self):
        base = dict(source='help-train',kind='none',split='train',excluded=False,speaker='training-speaker')
        for fields in (dict(kind='wrong_word'),dict(split='held-out'),dict(source='practice'),dict(excluded=True)):
            self.rejected(dict(base,**fields),[],'Only explicitly eligible')

    def test_excluded_and_missing_speakers_are_rejected_before_fitting(self):
        base = dict(source='help-train',kind='none',split='train',excluded=False,speaker='excluded-speaker')
        self.rejected(base,['excluded-speaker'],'release-held-out speaker')
        self.rejected(dict(base,speaker=''),[],'Missing')


if __name__ == '__main__':
    unittest.main()
