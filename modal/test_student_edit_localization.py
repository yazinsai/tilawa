"""Small exhaustive checks of the localization head's boundary marginalization."""
import itertools
import importlib.util
import unittest

import numpy as np

from student_edit_localization import (ctc_occupancy, partition, weak_targets, word_features,
                                      preservation_point, validate_training_labels)


class LocalizationTests(unittest.TestCase):
    def test_boundary_marginals_equal_enumerated_ctc_paths(self):
        probabilities = np.asarray([[.5,.3,.2],[.2,.4,.4],[.3,.2,.5],[.6,.3,.1]])
        for labels in [[1], [1,1], [1,2]]:
            states = [0]
            for label in labels:
                states.extend([label,0])
            weighted = np.zeros((4,len(labels)))
            total = 0
            for path in itertools.product(range(len(states)), repeat=4):
                if path[0] not in [0,1] or path[-1] not in [len(states)-2,len(states)-1]:
                    continue
                if any(b-a not in [0,1,2] or (b-a==2 and (states[b]==0 or states[b]==states[a]))
                       for a,b in zip(path,path[1:])):
                    continue
                mass = np.prod([probabilities[t,states[s]] for t,s in enumerate(path)])
                total += mass
                for t,s in enumerate(path):
                    if s%2:
                        weighted[t,(s-1)//2] += mass
            np.testing.assert_allclose(ctc_occupancy(np.log(probabilities),labels),weighted/total,atol=1e-12)

    def test_no_veto_features_from_an_impossible_reading(self):
        self.assertIsNone(ctc_occupancy(np.log([[.5,.5]]),[1,1]))
        self.assertIsNone(word_features(np.ones((1,2)),np.log([[.5,.5]]),["bb"],"b"))

    def test_pooling_preserves_multiple_word_inputs(self):
        encoder = np.asarray([[1.,2.],[2.,1.],[3.,0.],[0.,3.]])
        logp = np.log([[.5,.3,.2],[.2,.4,.4],[.3,.2,.5],[.6,.3,.1]])
        features,spans = word_features(encoder,logp,["b","c"],"bc")
        self.assertEqual(features.shape,(2,12))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(spans),2)
        self.assertTrue(all(0<=a<b<=4 for a,b in spans))

    def test_repeat_is_an_explicit_control_class(self):
        label = {"events":[{"errorClass":"wrong_word","location":{"surah":1,"ayah":1,"word":1}}],
                 "non_errors":[{"kind":"repeat","location":{"surah":1,"ayah":1,"word":2}}]}
        self.assertEqual(weak_targets(label,[(1,1,1),(1,1,2)]),[(0,1),(1,4)])

    def test_speaker_partition_is_stable(self):
        self.assertEqual(partition("speaker"),partition("speaker"))
        self.assertEqual({partition(str(i)) for i in range(50)},{"fit","validation","calibration"})

    def test_calibration_preserves_every_substitution_including_ties(self):
        values = [.4, .7, .2, .2]
        threshold = preservation_point(values)
        self.assertTrue(all(p >= threshold for p in values))
        self.assertEqual(threshold,.2)
        for invalid in [[], [np.nan], [0.0], [-1.0]]:
            with self.assertRaises(ValueError):
                preservation_point(invalid)

    def test_release_and_unspecified_speakers_rejected_before_audio_reads(self):
        controls = [{"split":"train","speaker":"a","kind":"none"},
                    {"split":"train","speaker":"b","kind":"repeat","non_errors":[{"kind":"repeat"}]}]
        validate_training_labels(controls)
        for split in ["eval",None]:
            with self.assertRaises(ValueError):
                validate_training_labels(controls+[{"speaker":"held-out","split":split}])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch runs in supervisor compute")
    def test_distilled_inference_can_preserve_several_changes(self):
        import torch
        from student_edit_localization import make_head
        head = make_head(3,True,True)
        with torch.no_grad():
            head.output.weight.zero_()
            head.output.bias.fill_(10)
            logits,_ = head(torch.zeros(4,3))
        self.assertTrue(torch.all(logits[:,0].sigmoid() > .99))
        self.assertGreater(float(logits[:,0].sigmoid().sum()),3.9)


if __name__ == "__main__":
    unittest.main()
