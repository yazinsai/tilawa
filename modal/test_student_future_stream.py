"""Replay checks without loading a large acoustic model or external weights."""
import unittest

import numpy as np

from student_future_export import stream


def manifest():
    return dict(window=45, shift=32, fbank=dict(mels=80, padValue=-23.),
                states=[dict(name="position", shape=[1], dtype="int64"),
                        dict(name="history", shape=[1, 1], dtype="float32")])


class RecordingRuntime:
    def __init__(self, invalid=None):
        self.calls = []
        self.invalid = invalid

    def run(self, names, feed):
        self.calls.append({k: v.copy() for k, v in feed.items()})
        position = int(feed["position"][0])
        np.testing.assert_array_equal(feed["history"], [[position]])
        rows = np.arange(position, position + 8, dtype=np.float32)
        lp = np.stack([rows, -rows], axis=-1)[None]
        if self.invalid == "nonfinite":
            lp[0, 0, 0] = np.nan
        result = [lp, np.array([position + 8], dtype=np.int64),
                  np.array([[position + 8]], dtype=np.float32)]
        if names[-1] == "native":
            feature = np.repeat(rows[None, :, None], 3, axis=2)
            result.append(feature[:, :-1] if self.invalid == "clock" else feature)
        return result


class FrozenStreamTest(unittest.TestCase):
    def test_drain_keeps_native_rows_and_state_across_every_tail_phase(self):
        # Exercise full and partial windows at all four feature phases of the
        # 40 ms clock. A padded row must never become a supervised frame.
        for length in range(9, 130):
            with self.subTest(length=length):
                features = np.repeat(np.arange(length, dtype=np.float32)[:, None], 80, axis=1)
                original = features.copy()
                runtime = RecordingRuntime()
                lp, native = stream(runtime, manifest(), features, "native")
                expected_rows = ((length - 7) // 2 + 1) // 2
                np.testing.assert_array_equal(lp[:, 0], np.arange(expected_rows))
                np.testing.assert_array_equal(native[:, 0], np.arange(expected_rows))
                np.testing.assert_array_equal(features, original)
                self.assertEqual(len(runtime.calls), (expected_rows + 7) // 8)
                for i, call in enumerate(runtime.calls):
                    view = features[i * 32:i * 32 + 45]
                    np.testing.assert_array_equal(call["x"][0, :len(view)], view)
                    np.testing.assert_array_equal(call["x"][0, len(view):],
                                                  np.full((45 - len(view), 80), -23.))
                    self.assertEqual(call["position"].dtype, np.dtype("int64"))
                    self.assertEqual(call["history"].dtype, np.dtype("float32"))

    def test_primary_only_replay_and_invalid_evidence(self):
        features = np.zeros((45, 80), dtype=np.float32)
        lp, native = stream(RecordingRuntime(), manifest(), features)
        self.assertEqual(lp.shape, (10, 2))
        self.assertIsNone(native)
        for bad in (features[:, :79], features[:8], np.full_like(features, np.nan)):
            with self.assertRaises(ValueError):
                stream(RecordingRuntime(), manifest(), bad)
        for invalid in ("clock", "nonfinite"):
            with self.assertRaises(ValueError):
                stream(RecordingRuntime(invalid), manifest(), features, "native")


if __name__ == "__main__":
    unittest.main()
