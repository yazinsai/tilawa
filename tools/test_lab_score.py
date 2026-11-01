"""Spawned scoring (autolab ticket 03); Modal, the runner module and the volume are faked."""

import io
import json
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lab

TS = {"id": "synthetic", "manifest": "m.json", "exclude": None, "sessions": {"practice": ["p1", "p2"]}}
REPORT = {"testSet": "synthetic", "sessions": {"excluded": []}}


def tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        data = b"export {}\n"
        info = tarfile.TarInfo("packages/core/x.ts")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Errors:
    class Error(Exception):
        pass

    class TimeoutError(Error):
        pass

    class FunctionTimeoutError(TimeoutError):
        pass

    class OutputExpiredError(TimeoutError):
        pass

    class NotFoundError(Error):
        pass


def summary(ok=True):
    shards = [{"shard": i, "code": 0 if ok or i else 1, "seconds": 1.0, "tail": "" if ok or i else "boom: session p1"} for i in range(2)]
    return {"run": "main-abc1234-T", "shards": shards, "merge": {"shard": 0, "code": 0, "seconds": 1.0, "tail": ""} if ok else None, "ok": ok}


class SpawnedScore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.call = MagicMock(object_id="fc-1")
        self.fn = MagicMock()
        self.fn.spawn.return_value = self.call
        self.modal = SimpleNamespace(exception=Errors, Function=MagicMock(), FunctionCall=MagicMock())
        self.modal.Function.from_name.return_value = self.fn
        self.modal.FunctionCall.from_id.return_value = self.call
        runner = SimpleNamespace(source=lambda wt: (tarball(), "abc1234"), run_id=lambda name, sha: f"{name}-{sha}-T")
        for p in (patch.dict(sys.modules, {"modal": self.modal}), patch.object(lab, "runner", return_value=runner),
                  patch.object(lab, "RUNS", self.tmp), patch.object(lab, "POLL", 0),
                  patch.object(lab, "vol_get", return_value=REPORT)):
            p.start()
            self.addCleanup(p.stop)

    def score(self):
        with redirect_stdout(io.StringIO()):
            return lab.score(None, Path("/wt"), name="main", listener="l", ts=TS, splits=("practice",))

    def run_files(self):
        return list(self.tmp.glob("*.run"))

    def test_fresh_start_spawns_once_and_writes_the_json_run_file(self):
        self.call.get.side_effect = [TimeoutError(), summary()]
        self.assertEqual(self.score(), REPORT)
        self.modal.Function.from_name.assert_called_once_with("tilawa-lab-run", "fanout")
        self.assertEqual(self.fn.spawn.call_count, 1)
        _, cmd, n, run, merge = self.fn.spawn.call_args.args
        self.assertEqual((n, run), (2, "main-abc1234-T"))
        self.assertIn("--split practice", cmd)
        [f] = self.run_files()
        rec = json.loads(f.read_text())
        self.assertEqual((rec["run"], rec["callId"]), ("main-abc1234-T", "fc-1"))
        self.assertIn("waiting on call fc-1", f.with_suffix(".log").read_text())
        lab.vol_get.assert_called_once_with("/runs/main-abc1234-T/score.json")

    def test_existing_call_id_resumes_without_spawning(self):
        self.call.get.side_effect = [summary()]
        self.score()
        self.fn.spawn.reset_mock()
        self.call.get.side_effect = [TimeoutError(), TimeoutError(), summary()]
        self.assertEqual(self.score(), REPORT)
        self.fn.spawn.assert_not_called()
        self.modal.FunctionCall.from_id.assert_called_with("fc-1")
        self.assertIn("resuming call fc-1", self.run_files()[0].with_suffix(".log").read_text())

    def test_expired_call_starts_again(self):
        self.call.get.side_effect = [summary()]
        self.score()
        self.call.get.side_effect = [Errors.OutputExpiredError(), summary()]
        self.score()
        self.assertEqual(self.fn.spawn.call_count, 2)

    def test_plain_text_run_file_is_read_and_replaced(self):
        self.assertEqual(lab.read_run(Path(self.tmp / "old.run")), {})
        (self.tmp / "old.run").write_text("main-abc1234-T\n")
        self.assertEqual(lab.read_run(self.tmp / "old.run"), {"run": "main-abc1234-T"})

    def test_failed_shard_raises_stop_with_the_log_tail_and_forgets_the_call(self):
        self.call.get.side_effect = [summary(ok=False)]
        with self.assertRaises(lab.Stop) as stop:
            self.score()
        self.assertIn("shard 0: exit 1", str(stop.exception))
        self.assertIn("boom: session p1", str(stop.exception))
        self.assertEqual(self.run_files(), [])
        lab.vol_get.assert_not_called()

    def test_without_the_modal_package_scoring_borrows_a_uv_python(self):
        done = SimpleNamespace(returncode=0, stdout='{"run": "main-abc1234-T"}\n')
        with patch.dict(sys.modules, {"modal": None}), patch.object(lab.subprocess, "run", return_value=done) as run:
            self.assertEqual(self.score(), REPORT)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["uv", "run"])
        self.assertEqual(json.loads(argv[-1])[:2], ["/wt", "main"])
        self.fn.spawn.assert_not_called()
        failed = SimpleNamespace(returncode=0, stdout='{"error": "scoring wt failed"}\n')
        with patch.dict(sys.modules, {"modal": None}), patch.object(lab.subprocess, "run", return_value=failed), \
                self.assertRaises(lab.Stop) as stop:
            self.score()
        self.assertEqual(str(stop.exception), "scoring wt failed")

    def test_call_that_raised_raises_stop(self):
        self.call.get.side_effect = [RuntimeError("container died")]
        with self.assertRaises(lab.Stop) as stop:
            self.score()
        self.assertIn("RuntimeError: container died", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
