"""Training-only checks: validation speakers must never select fitted parameters."""
import copy
import unittest

import numpy as np

from interruption_fit import partition, train


class TrainingPartitions(unittest.TestCase):
    def examples(self):
        out = []
        for speaker in range(15):
            for target in (0, 1):
                for i in range(12):
                    x = [0.0] * 13
                    x[0] = (2*target-1) + (i-6)/20
                    out.append({'speaker': f'training-{speaker}', 'split': 'train', 'construction': 'interruption-v1',
                                'features': x, 'label': target})
        return out

    def test_validation_data_cannot_change_the_model(self):
        examples = self.examples()
        assignments = partition([e['speaker'] for e in examples])
        a, _ = train(examples)
        changed = copy.deepcopy(examples)
        for e in changed:
            if assignments[e['speaker']] == 0:
                e['features'] = [50.0] * 13
                e['label'] = 1-e['label']
        b, _ = train(changed)
        self.assertEqual(a, b)
        self.assertEqual(set(assignments.values()), {0, 1, 2, 3, 4})

    def test_rejects_nontraining_examples(self):
        examples = self.examples()
        examples[0]['split'] = 'eval'
        with self.assertRaises(ValueError):
            train(examples)

    def test_model_is_finite_and_separates_training_controls(self):
        model, report = train(self.examples())
        self.assertTrue(np.isfinite(model['weights']).all())
        self.assertGreaterEqual(report['validation']['interruptionRecall'], 0.9)
        self.assertLessEqual(report['validation']['falseInterruptionRate'], 0.1)


if __name__ == '__main__':
    unittest.main()
