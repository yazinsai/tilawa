import copy
import hashlib
import tempfile
import unittest
import wave
from pathlib import Path
from session_build import build, chains


class SessionBuildTests(unittest.TestCase):
    def test_never_mixes_speakers_or_pads_gaps(self):
        order = {'1:1': 0, '1:2': 1, '1:3': 2}
        clips = [{'id': 'a', 'speaker': 'a', 'split': 'eval', 'ayahs': ['1:1']},
                 {'id': 'b', 'speaker': 'b', 'split': 'eval', 'ayahs': ['1:2']},
                 {'id': 'c', 'speaker': 'a', 'split': 'eval', 'ayahs': ['1:3']}]
        self.assertEqual([len(r) for r in chains(clips, order)], [1, 1, 1])

    def test_continuous_audio_and_timings_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); audio = p / 'source.wav'
            with wave.open(str(audio), 'wb') as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\1\0' * 16000)
            clips = [{'id': str(i), 'speaker': 'one', 'split': 'eval', 'ayahs': [f'1:{i}'], 'audio': str(audio),
                      'durationS': 1, 'timing': 'reviewed', 'words': [{'at': {'surah': 1, 'ayah': i, 'word': 1}, 'startS': 0, 'endS': 1}],
                      'events': [], 'nonErrors': []} for i in [1, 2]]
            a = build({'version': 1, 'clips': clips}, p / 'a', min_seconds=1)
            b = build({'version': 1, 'clips': list(reversed(clips))}, p / 'b', min_seconds=1)
            sa, sb = a['sessions'][0]['session'], b['sessions'][0]['session']
            self.assertEqual(sa['words'], sb['words'])
            self.assertEqual(hashlib.sha256(Path(sa['audio']).read_bytes()).hexdigest(), hashlib.sha256(Path(sb['audio']).read_bytes()).hexdigest())
            self.assertGreater(sa['words'][1]['startS'], 1.3)
            with wave.open(sa['audio']) as w:
                self.assertEqual(w.getnframes() / 16000, sa['durationS'])
            bad = copy.deepcopy(clips); bad[0]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, 'held-out'):
                build({'version': 1, 'clips': bad}, p / 'bad', min_seconds=1)


if __name__ == '__main__':
    unittest.main()
