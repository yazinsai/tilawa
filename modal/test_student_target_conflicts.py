import unittest

from student_target_conflicts import (CompleteTargets, contradictory_families,
                                      pronunciation_equivalences, supervision_weights)


class FullTargets(unittest.TestCase):
    def test_existing_representation_equivalences(self):
        targets = CompleteTargets()
        self.assertTrue(targets.compatible('بَااانںڇ', 'بَانن'))
        self.assertFalse(targets.compatible('بَن', 'بَم'))
        self.assertFalse(targets.compatible('بَت', 'بَط'))  # acoustic soft cost is not lawful identity

    def test_direct_same_word_forms_but_no_transitive_phone_closure(self):
        targets = CompleteTargets([('بَن', 'بَں'), ('ءَبَ', 'بَ'), ('دَ', 'ذَ'), ('ذَ', 'زَ')])
        self.assertTrue(targets.compatible('تَءَبَمَ', 'تَبَمَ'))
        self.assertFalse(targets.compatible('تَدَمَ', 'تَزَمَ'))

    def test_complete_content_not_a_shared_prefix_or_free_edge(self):
        targets = CompleteTargets()
        self.assertFalse(targets.compatible('بَن', 'بَنتَ'))
        self.assertFalse(targets.compatible('بَن', 'ءَبَن'))
        self.assertFalse(targets.compatible('بَن', 'بِن'))

    def test_reference_forms_supply_only_the_existing_same_word_alternatives(self):
        reference = {'ayahs': [dict(p='للبَ', o=[0], iso=['لبَ'])]}
        pairs = pronunciation_equivalences(reference)
        self.assertTrue(CompleteTargets(pairs).compatible('للبَ', 'لبَ'))
        self.assertFalse(CompleteTargets(pairs).compatible('للبَ', 'لتَ'))

    def test_family_requires_a_real_content_disagreement_and_excludes_every_copy(self):
        records = [dict(key=k, target_text=t, timing=timing) for k, t, timing in
                   [('a', 'بَت', [0, 1]), ('b', 'بَط', [2, 3]),
                    ('c', 'بَت', [0, 2]), ('d', 'بَت', [1, 3])]]
        families = dict(a='f', b='f', c='f', d='different-performance')
        conflicts = contradictory_families(records, families, CompleteTargets())
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]['recordings'], ['a', 'b', 'c'])
        views = [dict(recording=k, original_weight=1) for k in families for _ in range(3)]
        original, filtered = supervision_weights(views, families, conflicts)
        self.assertEqual(original, [1/12]*12)
        self.assertEqual(filtered[:9], [0]*9)
        self.assertEqual(filtered[9:], [1/3]*3)

    def test_annotations_alone_are_not_conflicting_phone_targets(self):
        records = [dict(key='a', target_text='بَاااں', spelling='first', timing=[1, 3]),
                   dict(key='b', target_text='بَان', spelling='second', timing=[2, 4])]
        self.assertEqual(contradictory_families(records, dict(a='f', b='f'), CompleteTargets()), [])

    def test_no_conflicts_give_exactly_identical_weights(self):
        views = [dict(recording='a', original_weight=2), dict(recording='b', original_weight=3)]
        a, b = supervision_weights(views, dict(a='f', b='g'), [])
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            supervision_weights(views, dict(a='f', b='f'), [dict(family='f')])


if __name__ == '__main__':
    unittest.main()
