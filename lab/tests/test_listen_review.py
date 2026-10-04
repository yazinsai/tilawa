"""Queue construction, Wilson intervals, and the listen page. No audio, no Modal."""

from __future__ import annotations

import importlib.util
import json
import threading
import unittest
import urllib.request
import wave
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "listen_review", Path(__file__).resolve().parents[1] / "scripts" / "listen_review.py"
)
lr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lr)


def _slip(clip: str, surah: int, ayah: int, word: int, kind: str, span=(0.4, 0.8)) -> dict:
    return {
        "id": clip,
        "surah": surah,
        "ayah": ayah,
        "word_index": word,
        "kind": kind,
        "span_s": list(span),
        "evidence": {"v3": {"word_end": word + 1}},
    }


class ListenReviewTest(unittest.TestCase):
    def test_wilson_centre_and_empty(self) -> None:
        self.assertIsNone(lr.wilson(0, 0))
        lo, hi = lr.wilson(50, 100)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)
        self.assertGreater(lo, 0.39)
        self.assertLess(hi, 0.61)
        self.assertEqual(lr.wilson(0, 10)[0], 0.0)
        self.assertEqual(lr.wilson(10, 10)[1], 1.0)

    def test_held_out_matches_eval_split(self) -> None:
        import hashlib

        for clip in ("tlog_1_2_3", "abc", "tlog_99_1_1"):
            n = int(hashlib.sha1(clip.encode()).hexdigest()[:8], 16)
            expect = "tune" if n % 2 == 0 else "held"
            self.assertEqual(lr.tlog_half(clip), expect)

    def test_stratified_sample_is_stable_and_covers_rare_kinds(self) -> None:
        rows = []
        for i in range(10):
            rows.append(_slip(f"tlog_s_{i}", 2, 1, 1, "substituted"))
        for i in range(5):
            rows.append(_slip(f"tlog_o_{i}", 2, 2, 1, "omitted"))
        for i in range(3):
            rows.append(_slip(f"tlog_r_{i}", 2, 3, 1, "repeated"))
        for i in range(2):
            rows.append(_slip(f"tlog_z_{i}", 2, 4, 1, "restarted"))
        a = lr.stratified_sample(rows, 10, 0)
        b = lr.stratified_sample(rows, 10, 0)
        self.assertEqual([lr.slip_key(row) for row in a], [lr.slip_key(row) for row in b])
        self.assertEqual(len(a), 10)
        kinds = {row["kind"] for row in a}
        self.assertIn("restarted", kinds)
        self.assertIn("repeated", kinds)

    def test_queue_a_is_held_new_flags_only_and_shuffled_blind(self) -> None:
        held_id = next(f"tlog_h_{i}" for i in range(100) if lr.tlog_half(f"tlog_h_{i}") == "held")
        tune_id = next(f"tlog_t_{i}" for i in range(100) if lr.tlog_half(f"tlog_t_{i}") == "tune")
        rows = [
            _slip(held_id, 1, 1, 2, "omitted"),
            _slip(tune_id, 1, 2, 1, "omitted"),
            _slip(held_id, 1, 1, 4, "substituted"),
        ]
        flags = {
            "slips": [
                {"id": held_id, "surah": 1, "ayah": 1, "word_index": 2, "scored": True,
                 "shipped_old": True, "shipped_new": True, "a0w_old": False, "a0w_new": True},
                {"id": held_id, "surah": 1, "ayah": 1, "word_index": 4, "scored": True,
                 "shipped_old": False, "shipped_new": False, "a0w_old": False, "a0w_new": False},
                {"id": tune_id, "surah": 1, "ayah": 2, "word_index": 1, "scored": True,
                 "shipped_old": False, "shipped_new": True, "a0w_old": False, "a0w_new": True},
            ]
        }
        help_id = "helpclip"
        help_rows = [_slip(help_id, 1, 3, 0, "substituted")]
        queues = lr.build_queues(rows, flags, help_rows, {help_id: "helpclip.wav"}, seed=0)
        self.assertEqual(len(queues["A"]), 1)
        self.assertEqual(queues["A"][0]["word_index"], 2)
        self.assertEqual(queues["C"][0]["audio"]["name"], "helpclip.wav")
        self.assertEqual(len(queues["B"]), 3)
        again = lr.build_queues(rows, flags, help_rows, {help_id: "helpclip.wav"}, seed=0)
        self.assertEqual([item["tag_id"] for item in queues["B"]], [item["tag_id"] for item in again["B"]])

    def test_precision_gate_and_recall(self) -> None:
        item = {
            "queue": "A",
            "tag_id": "A|c|1|1|2",
            "id": "c",
            "surah": 1,
            "ayah": 1,
            "word_index": 2,
            "locator_kind": "omitted",
            "flags": {"shipped_old": True, "shipped_new": True, "a0w_old": False, "a0w_new": True},
            "scored": True,
        }
        miss = {
            **item,
            "tag_id": "A|c|1|1|3",
            "word_index": 3,
            "flags": {"shipped_old": False, "shipped_new": True, "a0w_old": False, "a0w_new": True},
        }
        tags = {
            item["tag_id"]: {"verdict": "real", "kind": "skipped"},
            miss["tag_id"]: {"verdict": "not_slip"},
        }
        # Two new flags, one false: precision 0.5. Shipped old only flagged the real one: precision 1.
        report = lr.summarize_queues({"A": [item, miss], "B": [], "C": []}, tags)
        self.assertEqual(report["queues"]["A"]["gate_shipped"], "fail")
        self.assertEqual(report["queues"]["A"]["precision_new"]["k"], 1)
        self.assertEqual(report["queues"]["A"]["precision_new"]["n"], 2)
        self.assertFalse(report["pass"])
        good = lr.summarize_queues({"A": [item], "B": [dict(item, queue="B", tag_id="B|c|1|1|2")], "C": []}, {
            item["tag_id"]: {"verdict": "real", "kind": "skipped"},
            "B|c|1|1|2": {"verdict": "real", "kind": "skipped"},
        })
        self.assertEqual(good["queues"]["A"]["gate_shipped"], "pass")
        self.assertEqual(good["queues"]["B"]["recall"]["shipped_new"]["k"], 1)
        self.assertEqual(good["queues"]["B"]["validity"]["p"], 1.0)

    def test_page_tag_and_summarize(self) -> None:
        root = Path("/tmp/tilawa-listen-fixture")
        if root.exists():
            for path in sorted(root.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
        cache = lr.Cache(root)
        held_id = next(f"tlog_h_{i}" for i in range(200) if lr.tlog_half(f"tlog_h_{i}") == "held")
        rows = [
            _slip(held_id, 1, 1, 1, "omitted", span=(0.2, 0.6)),
            _slip(held_id, 1, 2, 0, "repeated", span=(0.1, 0.3)),
        ]
        (cache.meta / "tlog_candidates.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (cache.meta / "manifest.json").write_text(json.dumps({"samples": []}), encoding="utf-8")
        (cache.meta / "help_located.jsonl").write_text("\n", encoding="utf-8")
        slips = []
        for row in rows:
            slips.append({
                "id": row["id"], "surah": row["surah"], "ayah": row["ayah"], "word_index": row["word_index"],
                "scored": True, "shipped_old": False, "shipped_new": True, "a0w_old": False, "a0w_new": False,
            })
        (cache.meta / "rule_flags.json").write_text(json.dumps({"slips": slips}), encoding="utf-8")
        saved = lr.build_from_cache(cache, seed=0, rebuild=True)
        self.assertGreaterEqual(len(saved["queues"]["A"]), 1)
        # Pre-seed audio so the page does not touch the volume.
        for item in saved["queues"]["A"] + saved["queues"]["B"]:
            dest = cache.audio_path(item)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(dest), "w") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
        app = lr.App(cache, saved)
        server = lr.ThreadingHTTPServer(("127.0.0.1", 0), lr.make_handler(app))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
            self.assertIn("Play window", page)
            self.assertIn("tajweed-only", page)
            item = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/item?queue=A&i=0"
            ).read().decode("utf-8"))
            blob = json.dumps(item)
            for banned in ("shipped", "a0w", "flags", "locator", '"id"', "tlog_h_"):
                self.assertNotIn(banned, blob)
            self.assertGreater(len(item["words"]), 0)
            start, end = item["highlight"]
            self.assertGreater(end, start)
            self.assertLess(start, len(item["words"]))
            audio = urllib.request.urlopen(item["audio"] if item["audio"].startswith("http") else f"http://127.0.0.1:{port}{item['audio']}")
            self.assertEqual(audio.status, 200)
            self.assertGreater(len(audio.read()), 1000)
            body = json.dumps({
                "queue": "A", "index": 0, "verdict": "real", "kind": "skipped", "notes": "heard a skip",
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/tag", data=body, headers={"content-type": "application/json"}
            )
            saved_tag = json.loads(urllib.request.urlopen(req).read().decode())
            self.assertEqual(saved_tag["tag"]["verdict"], "real")
            self.assertEqual(saved_tag["tag"]["kind"], "skipped")
            report = lr.summarize_queues(saved["queues"], cache.load_tags())
            text = lr.format_report(report)
            self.assertIn("new-flag precision", text)
            self.assertIn("1/1", text)
            self.assertIn("incomplete", text)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
