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


def _wipe(root: Path) -> None:
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()


class ListenModesTest(unittest.TestCase):
    def test_tlog_mined_keeps_tags_json_and_hides_the_clip(self) -> None:
        root = Path("/tmp/tilawa-listen-mined")
        _wipe(root)
        cache = lr.Cache(root)
        self.assertEqual(cache.tags_path.name, "tags.json")
        self.assertEqual(cache.queue_path.name, "queue.json")
        self.assertEqual(cache.mined_queue_path.name, "queue_mined.json")
        kept = {
            "verdict": "real",
            "kind": "skipped",
            "notes": "mid-listen",
            "updated": "2020-01-01T00:00:00Z",
        }
        cache.save_tags({"tlog_mined|already|1|1|0": kept})
        clip = "secret_tlog_clip"
        item = {
            "queue": "tlog_mined",
            "tag_id": "tlog_mined|secret_tlog_clip|1|1|2",
            "id": clip,
            "surah": 1,
            "ayah": 1,
            "word_index": 2,
            "highlight": [2, 3],
            "span": [0.2, 0.5],
            "role": "control",
            "mined_kind": "vowel",
            "stratum": "sister_ayah",
            "qc_pass": True,
            "audio": {"source": "tlog", "name": "secret_tlog_clip.flac"},
        }
        (cache.meta / "tlog_mined_queue.json").write_text(
            json.dumps({"items": [item, dict(item, tag_id="tlog_mined|secret_tlog_clip|1|1|3", word_index=3)]}) + "\n",
            encoding="utf-8",
        )
        corpus = Path("/tmp/tilawa-self-repair/zipformer_quran.json")
        if corpus.is_file():
            (cache.meta / "zipformer_quran.json").write_bytes(corpus.read_bytes())
        saved = lr.build_mined(cache, seed=0, rebuild=False)
        ids = [row["tag_id"] for row in saved["queues"]["tlog_mined"]]
        self.assertEqual(ids, [item["tag_id"], "tlog_mined|secret_tlog_clip|1|1|3"])
        self.assertEqual(saved["focus"], "tlog_mined")
        disk = json.loads(cache.mined_queue_path.read_text(encoding="utf-8"))
        self.assertEqual(disk["focus"], "tlog_mined")
        self.assertEqual([row["tag_id"] for row in disk["queues"]["tlog_mined"]], ids)
        again = lr.build_mined(cache, seed=0, rebuild=False)
        self.assertEqual([row["tag_id"] for row in again["queues"]["tlog_mined"]], ids)
        audio = cache.audio_path(saved["queues"]["tlog_mined"][0])
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"fLaC" + b"\x00" * 64)

        app = lr.App(cache, saved, lr.mined_mode())
        view = app.item("tlog_mined", 0)
        blob = json.dumps(view)
        for banned in (clip, "control", "vowel", "sister_ayah", "mined_kind", '"id"', "locator"):
            self.assertNotIn(banned, blob)
        app.set_tag("tlog_mined", 0, "not_slip", None, "")
        tags = json.loads(cache.tags_path.read_text(encoding="utf-8"))
        self.assertEqual(tags["tlog_mined|already|1|1|0"], kept)
        self.assertEqual(tags[item["tag_id"]]["verdict"], "not_slip")
        self.assertFalse((root / "self_repair_tags.json").exists())
        self.assertFalse((root / "clean_guard_tags.json").exists())

        label = lr.verified_slip_label(
            item, {"verdict": "real", "kind": "wrong_word"}
        )
        self.assertEqual(label["label_kind"], "vowel")
        self.assertTrue(label["control"])
        self.assertFalse(label["in_scoreboard"])
        self.assertTrue(label["provisional"])
        self.assertFalse(label["train_ready"])
        report = lr.summarize_queues(saved["queues"], cache.load_tags())
        text = lr.format_report(report)
        self.assertIn("control slips", text)
        self.assertNotIn(clip, text)

        page = lr._page_for(lr.mined_mode())
        self.assertIn('tlog_mined: "M"', page)
        self.assertIn('const ORDER = ["tlog_mined"]', page)
        self.assertIn("const label = LABEL[name] || name", page)

    def test_clean_guard_tags_stay_off_the_shared_file(self) -> None:
        root = Path("/tmp/tilawa-listen-guard")
        _wipe(root)
        cache = lr.Cache(root, tags_name="clean_guard_tags.json", queue_name="clean_guard_queue.json")
        (root / "tags.json").write_text("{}\n", encoding="utf-8")
        item = {
            "queue": "clean_guard",
            "tag_id": "clean_guard|hideme|2|3|1",
            "id": "hideme",
            "surah": 2,
            "ayah": 3,
            "word_index": 1,
            "highlight": [1, 2],
            "span": [0.1, 0.2],
            "role": "edit",
            "pool": "help",
            "split": "dev",
            "edit_class": "vowel",
            "audio": {"source": "help", "name": "hideme.wav"},
        }
        saved = {
            "queues": {"clean_guard": [item]},
            "words": {"2:3": ["word"]},
            "surah_names": {},
        }
        audio = cache.audio_path(item)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFFxxxxWAVEfmt ")
        app = lr.App(cache, saved, lr.guard_mode())
        view = app.item("clean_guard", 0)
        blob = json.dumps(view)
        self.assertNotIn("hideme", blob)
        self.assertNotIn("edit_class", blob)
        self.assertNotIn('"id"', blob)
        app.set_tag("clean_guard", 0, "real_lapse", None, "")
        self.assertEqual((root / "tags.json").read_text(encoding="utf-8"), "{}\n")
        tags = json.loads((root / "clean_guard_tags.json").read_text(encoding="utf-8"))
        self.assertEqual(tags[item["tag_id"]]["verdict"], "real_lapse")
        self.assertIsNone(tags[item["tag_id"]]["kind"])
        report = lr.summarize_clean_guard(saved["queues"]["clean_guard"], tags)
        self.assertEqual(report["decision"], "mislabelled_guard")
        self.assertEqual(lr.help_lapse_decision({"n": 0, "p": None}), "incomplete")
        self.assertEqual(lr.help_lapse_decision({"n": 4, "p": 0.2}), "model_habits")
        self.assertEqual(lr.help_lapse_decision({"n": 4, "p": 0.4}), "inconclusive")
        page = lr.guard_mode().page
        self.assertIn("real_lapse", page)
        self.assertIn('const ORDER = ["clean_guard"]', page)
        self.assertNotIn('const ORDER = ["A"', page)
        self.assertNotIn("tajweed-only", page)

    def test_self_repair_queue_is_blind_and_uses_its_own_tags(self) -> None:
        root = Path("/tmp/tilawa-listen-repair")
        _wipe(root)
        root.mkdir(parents=True, exist_ok=True)
        shared = {
            "tlog_mined|kept|1|1|0": {
                "verdict": "real", "kind": "skipped", "notes": "yazin", "updated": "2020-01-01T00:00:00Z",
            }
        }
        (root / "tags.json").write_text(json.dumps(shared) + "\n", encoding="utf-8")
        (root / "queue_mined.json").write_text("{}\n", encoding="utf-8")
        candidates = []
        for i in range(3):
            candidates.append({
                "id": f"help_{i}",
                "surah": 1,
                "ayah": 1,
                "word_index": i,
                "kind": "letter",
                "span_s": [0.1, 0.4],
                "edit_tokens": 1,
                "unseen": True,
                "source_set": "help-clean",
                "audio": {"source": "help", "name": f"help_{i}.wav"},
            })
        # Same clip, higher edit cost, replaces the first help_0 row.
        candidates.append({
            **candidates[0], "word_index": 4, "edit_tokens": 9, "kind": "skip",
        })
        candidates.append({
            "id": "tlog_late",
            "surah": 1,
            "ayah": 2,
            "word_index": 1,
            "kind": "substitution",
            "span_s": [0.2, 0.3],
            "edit_tokens": 8,
            "unseen": True,
            "source_set": "tlog",
            "audio": {"source": "tlog", "name": "tlog_late.flac"},
        })
        candidates.append({
            "id": "acted_seen",
            "surah": 1,
            "ayah": 3,
            "word_index": 0,
            "kind": "vowel",
            "unseen": False,
            "source_set": "help-clean",
            "audio": {"source": "help", "name": "acted.wav"},
        })
        controls = [
            {
                "id": f"ctrl_{i}",
                "surah": 2,
                "ayah": 1,
                "word_index": 0,
                "kind": "vowel",
                "span_s": [0.0, 0.2],
                "source_set": "help-clean",
                "audio": {"source": "help", "name": f"ctrl_{i}.wav"},
            }
            for i in range(4)
        ]
        controls.append({**controls[0], "id": "help_0"})
        # 3 help clips fill the candidate slots, so the later tlog row stays out.
        items = lr.select_self_repair_queue(candidates, controls, n=5, n_controls=2, seed=0)
        self.assertLessEqual(len(items), 5)
        roles = [row["role"] for row in items]
        self.assertLessEqual(roles.count("control"), roles.count("candidate"))
        self.assertEqual(roles.count("candidate"), 3)
        self.assertEqual(roles.count("control"), 2)
        ids = [row["id"] for row in items]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("acted_seen", ids)
        self.assertNotIn("tlog_late", ids)
        self.assertIn("help_0", ids)
        chosen = next(row for row in items if row["id"] == "help_0")
        self.assertEqual(chosen["word_index"], 4)
        self.assertEqual(chosen["mined_kind"], "skip")

        cache = lr.Cache(root, tags_name="self_repair_tags.json", queue_name="queue_self_repair.json")
        saved = {
            "queues": {"self_repair": items},
            "words": {"1:1": ["ا", "ب", "ت", "ث", "ج"], "1:2": ["ا"], "2:1": ["ا"]},
            "surah_names": {},
        }
        audio = cache.audio_path(chosen)
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFFxxxxWAVEfmt ")
        app = lr.App(cache, saved, lr.repair_mode())
        view = app.item("self_repair", items.index(chosen))
        blob = json.dumps(view)
        for banned in ("help_0", "candidate", "control", "skip", "mined_kind", '"id"', "locator", "letter"):
            self.assertNotIn(banned, blob)
        app.set_tag("self_repair", items.index(chosen), "real", "skipped", "")
        self.assertEqual(json.loads((root / "tags.json").read_text(encoding="utf-8")), shared)
        self.assertEqual((root / "queue_mined.json").read_text(encoding="utf-8"), "{}\n")
        tags = json.loads((root / "self_repair_tags.json").read_text(encoding="utf-8"))
        self.assertEqual(tags[chosen["tag_id"]]["kind"], "skipped")
        label = lr.natural_label(chosen, tags[chosen["tag_id"]])
        self.assertEqual(label["kind"], "skip_word")
        self.assertEqual(label["word"], 5)
        self.assertEqual(label["elicitation"], "natural")
        self.assertNotIn("id", label)
        report = lr.summarize_self_repair(items, tags)
        text = lr.format_self_repair(report)
        self.assertNotIn("help_0", text)
        self.assertIn("natural labels 1", text)
        page = lr.repair_mode().page
        self.assertIn('const ORDER = ["self_repair"]', page)
        self.assertIn("tajweed-only", page)
        self.assertNotIn("help_0", page)


if __name__ == "__main__":
    unittest.main()
