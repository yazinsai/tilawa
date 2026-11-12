"""Low-compute pairing invariants; no labelled benchmark fixtures."""
import unittest
from student_duration_negatives import match_words, random_pairs, partition, vowel_sequence


def word(pronunciation,frames=12,speaker='speaker',take='take',capture='phone'):
    return dict(ids=pronunciation,x=[[0]*32 for _ in range(frames)],speaker=speaker,
        take=take,capture=capture,descriptor=[0]*9)


class AudioSwaps(unittest.TestCase):
    def test_actual_different_audio_with_matched_vowels_and_duration(self):
        tokens='بتنَُ'
        words=[word([1,4,1]),word([2,4,2]),word([3,4,3],40),word([3,5,3]),word([3,4,3],speaker='other')]
        self.assertEqual(match_words(words,tokens),[(0,1)])

    def test_random_control_retains_exact_positives_and_bidirectional_balance(self):
        words=[word([i+1,8]) for i in range(6)]
        matched=[(0,1),(2,3),(4,5)]
        pairs=random_pairs(words,matched)
        self.assertEqual(sorted(i for p in pairs for i in p),list(range(6)))
        self.assertTrue(all(words[a]['ids']!=words[b]['ids'] for a,b in pairs))
        self.assertEqual(pairs,random_pairs(words,matched))

    def test_identity_audio_and_capture_changes_cannot_be_hard_negatives(self):
        words=[word([1,3]),word([1,3]),word([2,3],capture='laptop')]
        self.assertEqual(match_words(words,'بتَ'),[])
        # Different held-phone counts must not let duration alone solve the task.
        self.assertEqual(match_words([word([1,3]),word([1,1,3])],'بتَ'),[])

    def test_speaker_assignment_and_held_vowels_are_stable(self):
        self.assertEqual(partition('same-speaker'),partition('same-speaker'))
        self.assertEqual(vowel_sequence([1,2,2,3],'بَُ'),('a','u'))


if __name__=='__main__':unittest.main()
