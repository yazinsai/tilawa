import copy
import io
import json
import unittest
import wave

from live_review_core import issue_markdown, validate


class LiveReviewTest(unittest.TestCase):
    def setUp(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)
        self.audio = out.getvalue()
        self.report = {"schema": 1, "id": "12345678-1234-1234-1234-123456789abc", "durationMs": 1000,
                       "build": {"revision": "main-sha"}, "model": {"modelVersion": "test"},
                       "events": [], "evidence": [],
                       "flags": [{"timeMs": 500, "component": "tracker", "note": "Wrong ayah", "position": None}]}

    def test_valid_and_issue_evidence(self):
        self.assertIs(validate(self.report, self.audio), self.report)
        issue = issue_markdown(self.report, self.report["flags"][0], 1)
        for text in ["Status: needs-triage", "main-sha", "Wrong ayah", "../audio.wav", "../report.json"]:
            self.assertIn(text, issue)

    def test_reject_invalid_reports(self):
        for update in [{"id": "../../escape"}, {"durationMs": 180001}, {"durationMs": float("nan")},
                       {"durationMs": 300}, {"flags": []}, {"evidence": None}, {"build": None}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate({**self.report, **update}, self.audio)

    def test_reject_invalid_flags(self):
        for update in [{"timeMs": -1}, {"timeMs": 1001}, {"timeMs": float("inf")},
                       {"component": "../other"}, {"note": " "}, {"note": "a" * 2001}]:
            report = copy.deepcopy(self.report)
            report["flags"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate(report, self.audio)

    def test_reject_invalid_audio(self):
        for audio in [b"bad", self.audio[:44], self.audio[:-1], b"X" + self.audio[1:]]:
            with self.assertRaises(ValueError):
                validate(self.report, audio)


if __name__ == "__main__":
    unittest.main()
