"""Performed-recording identity tests with synthetic audio."""
import unittest

from student_source_families import CopyPolicy, partition_aliases, source_families, verify_reencoding, waveform_hash


class Identity(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.policy = CopyPolicy(sample_rate=1000, max_delay_seconds=.01)
        # A fixed generic multi-band waveform, independent of benchmark audio.
        rng = np.random.default_rng(831)
        self.wave = rng.normal(size=2000).astype('float32')

    def test_exact_copies_and_gain_delay_reencodings_share_a_family(self):
        shifted = self.np.concatenate([self.np.zeros(4), self.wave*.7])
        groups, certificates = source_families([('a', self.wave), ('b', self.wave.copy()), ('c', shifted)], self.policy)
        self.assertEqual(len(set(groups.values())), 1)
        self.assertTrue(any(c['kind'] == 'exact-decoded' for c in certificates))
        self.assertTrue(any(c['kind'] == 'verified-reencoding' for c in certificates))

    def test_separate_performances_do_not_group(self):
        # The same energy envelope and nominal pitch are insufficient identity.
        t = self.np.arange(2000)/1000
        a = self.np.sin(2*self.np.pi*91*t)
        b = self.np.sin(2*self.np.pi*91*t + .25*self.np.sin(2*self.np.pi*3*t))
        self.assertIsNone(verify_reencoding(a, b, self.policy))
        groups, _ = source_families([('same-reader-same-text-take-1', a), ('same-reader-same-text-take-2', b)], self.policy)
        self.assertEqual(len(set(groups.values())), 2)

    def test_a_locally_different_word_rejects_full_recording_identity(self):
        # Most of the waveform agrees but one short block has different speech.
        other = self.wave.copy()
        other[800:850] *= -1
        self.assertIsNone(verify_reencoding(self.wave, other, self.policy))

    def test_registration_cannot_hide_missing_onset_or_ending_speech(self):
        policy = CopyPolicy(sample_rate=1000, max_delay_seconds=.01, block_seconds=.02)
        for shortened in (self.wave[4:], self.wave[:-4]):
            self.assertIsNone(verify_reencoding(self.wave, shortened, policy))
        shifted_and_truncated = self.np.concatenate([self.np.zeros(4), self.wave[:-4]*.7])
        self.assertIsNone(verify_reencoding(self.wave, shifted_and_truncated, policy))

    def test_short_consonant_change_cannot_hide_in_the_global_correlation(self):
        policy = CopyPolicy(sample_rate=1000, block_seconds=.02)
        other = self.wave.copy()
        other[800:804] *= -1
        self.assertIsNone(verify_reencoding(self.wave, other, policy))

    def test_silence_never_proves_a_shared_performance(self):
        x = self.np.zeros(2000)
        self.assertIsNone(verify_reencoding(x, x, self.policy))
        groups, _ = source_families([('a', x), ('b', x)], self.policy)
        self.assertEqual(len(set(groups.values())), 2)

    def test_sample_rate_is_part_of_exact_identity(self):
        self.assertNotEqual(waveform_hash(self.wave, 1000), waveform_hash(self.wave, 2000))

    def test_complete_link_does_not_chain_and_partition_audit_still_checks_every_pair(self):
        noise = self.np.random.default_rng(42).normal(size=len(self.wave))
        rows = [('a', self.wave), ('b', self.wave+.07*noise), ('c', self.wave-.07*noise)]
        self.assertIsNotNone(verify_reencoding(rows[0][1], rows[1][1], self.policy))
        self.assertIsNotNone(verify_reencoding(rows[0][1], rows[2][1], self.policy))
        self.assertIsNone(verify_reencoding(rows[1][1], rows[2][1], self.policy))
        families, _ = source_families(rows, self.policy)
        self.assertEqual(families['a'], families['b'])
        self.assertNotEqual(families['a'], families['c'])
        aliases = partition_aliases(rows, {'a': 'train', 'b': 'train', 'c': 'validation'}, self.policy)
        self.assertEqual([(a['left'], a['right']) for a in aliases], [('a', 'c')])
        self.assertEqual(partition_aliases(rows, {k: 'train' for k, _ in rows}, self.policy), [])


if __name__ == '__main__':
    unittest.main()
