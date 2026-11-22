"""Practice-only fold orchestration; all scores and git/remote calls here are synthetic."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lab
import lab_ledger as L


def totals(missed):
    return {"sessions": 1, "recitedMinutes": 2,
            "overall": {"wrongCalls": missed, "missed": missed, "falseWarnings": 0, "mistakes": 20, "per5Min": missed * 2.5},
            "attribution": {"listener": 0, "tracker": 0, "checker": missed,
                            "missed": {"listener": 0, "tracker": 0, "checker": missed},
                            "falseWarnings": dict.fromkeys(L.COMPONENTS, 0)},
            "components": {"listener": {"edits": 0, "phonemes": 50, "per": 0},
                           "tracker": {"falseJumps": 0, "onPlaceShare": 1},
                           "checker": {"wrongCalls": missed, "missed": missed, "falseWarnings": 0}}}


class PracticeOnlyFold(unittest.TestCase):
    def fold(self, refresh, check=True, **opts):
        branch_listener = opts.pop("branch_listener", L.DEFAULT_LISTENER)
        self.seen_listeners = []
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / Path(lab.SUMMARY).parent).mkdir(parents=True)
            base = {"sha": "aaaa111", "head": "aaaa111", "at": "t0", "testSet": "synthetic",
                    "listener": L.DEFAULT_LISTENER, "run": "base", "practice": L.score(totals(10))}
            card = L.new_card("synthetic", "checker", "synthetic practice improvement", -5, -5,
                              test_set="synthetic", author="test")
            ledger = {"version": 1, "kind": "lab-ledger", "main": "main", "testSet": "synthetic",
                      "testSets": {"synthetic": {"id": "synthetic", "sessions": {"practice": ["p1"]}}},
                      "mainLine": [base], "experiments": {}}
            L.add_experiment(ledger, card, branch="exp/synthetic", worktree=str(wt))
            L.record_result(ledger, "synthetic", {**base, "sha": "bbbb222", "listener": branch_listener, "practice": L.score(totals(5))}, vs=base)
            args = SimpleNamespace(**{"id": "synthetic", "main": "main", "yes": True, "practice_only": True, **opts})
            requested = []

            def score(*_args, splits, **_kwargs):
                self.seen_listeners.append(_kwargs.get("listener"))
                requested.append(splits)
                self.assertEqual(splits, ("practice",))
                return {"kind": "lab-score", "sha": "bbbb222", "run": "rebased", "testSet": "synthetic",
                        "listener": L.DEFAULT_LISTENER, "sessions": {"excluded": []},
                        "splits": {"practice": totals(5)}}

            def score_main(*_args, splits, **_kwargs):
                requested.append(splits)
                self.assertEqual(splits, ("practice",))
                return {**base, "sha": "cccc333"}

            calls = []

            def git(*cmd, **_kwargs):
                calls.append(cmd)
                return "dddd444" if cmd == ("rev-parse", "HEAD") else ""

            stdout = io.StringIO()
            with patch.object(lab, "load", side_effect=lambda _a: copy.deepcopy(ledger)), \
                    patch.object(lab, "update", side_effect=lambda _a, fn: fn(ledger)), \
                    patch.object(lab, "score", side_effect=score), \
                    patch.object(lab, "score_main", side_effect=score_main) as refresh_score, \
                    patch.object(lab, "main_head", return_value="cccc333" if refresh else "aaaa111"), \
                    patch.object(lab, "git", side_effect=git), \
                    patch.object(lab.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                    patch.object(lab, "main_worktree", return_value=wt), \
                    patch.object(lab, "remove_worktree"), redirect_stdout(stdout):
                out = lab.cmd_fold(args)
            if not check:
                return out, ledger, calls

            self.assertEqual(refresh_score.call_count, int(refresh))
            self.assertEqual(requested, [("practice",)] * (1 + int(refresh)))
            self.assertEqual(ledger["experiments"]["synthetic"]["state"], "folded")
            accepted = ledger["mainLine"][-1]
            self.assertEqual(accepted["practice"]["wrongCalls"], 5)
            self.assertNotIn("exam", accepted)
            self.assertEqual(accepted["measurementScope"], ["practice"])
            self.assertEqual(accepted["examStatus"], "unmeasured_for_this_head")
            self.assertEqual(accepted["head"], "dddd444")
            self.assertIn(("merge", "-q", "--ff-only", "dddd444"), calls)
            self.assertIn("exam not measured", stdout.getvalue())

    def test_fold_scores_and_records_only_practice(self):
        self.fold(False)

    def test_stale_main_refresh_also_scores_only_practice(self):
        self.fold(True)

    def test_fold_rescores_with_the_listener_the_branch_was_scored_with(self):
        self.fold(False, check=False, yes=True, branch_listener="exp-x-abc")
        self.assertEqual(self.seen_listeners, ["exp-x-abc"])

    def test_no_merge_records_and_stops_before_merging(self):
        out, ledger, calls = self.fold(False, check=False, yes=False, no_merge=True)
        self.assertEqual(out["state"], "waiting")
        self.assertEqual((out["run"], out["base"], out["branchRun"]), ("rebased", "base", "base"))
        self.assertEqual(out["compare"]["overall"]["change"], -5)
        self.assertIsNone(out["examTripwire"])
        self.assertEqual(ledger["experiments"]["synthetic"]["state"], "open")
        self.assertEqual(ledger["experiments"]["synthetic"]["results"][-1]["kind"], "rebased")
        self.assertFalse(any(c[0] == "merge" for c in calls))

    def gated(self, verdict):
        res = {"verdict": verdict, "tuning": False, "checks": [{"name": "gain", "ok": verdict == "pass", "blocking": True, "why": "overall -5"}]}
        with patch.object(lab, "vol_get", return_value={}), patch.object(lab.G, "gate", return_value=res) as gate:
            out = self.fold(False, check=False, yes=False, gate=True, json=True)
        self.assertEqual(gate.call_count, 1)
        return out

    def test_gate_pass_folds_without_asking(self):
        out, ledger, calls = self.gated("pass")
        self.assertEqual(out["state"], "folded")
        self.assertEqual(ledger["experiments"]["synthetic"]["state"], "folded")
        self.assertIn(("merge", "-q", "--ff-only", "dddd444"), calls)

    def test_gate_fail_discards_and_never_merges(self):
        out, ledger, calls = self.gated("fail")
        self.assertEqual(out["state"], "gate-failed")
        self.assertEqual(ledger["experiments"]["synthetic"]["state"], "discarded")
        self.assertIn("autolab gate: overall -5", ledger["experiments"]["synthetic"]["decision"]["note"])
        self.assertFalse(any(c[0] == "merge" for c in calls))

    def test_baseline_practice_only_scores_no_exam(self):
        led = {"version": 1, "kind": "lab-ledger", "main": "main", "testSet": "synthetic", "mainLine": [], "experiments": {},
               "testSets": {"synthetic": {"id": "synthetic", "manifest": "m.json", "sessions": {"practice": ["p1"]}}}}
        res = {"sha": "aaaa111", "run": "base", "testSet": "synthetic", "listener": L.DEFAULT_LISTENER, "practice": L.score(totals(10))}
        args = SimpleNamespace(manifest="m.json", exclude="", note="", practice_only=True, main="main")
        with patch.object(lab, "load", return_value=led), patch.object(lab, "update", side_effect=lambda _a, fn: fn(led)), \
                patch.object(lab, "score_main", return_value=res) as sm, redirect_stdout(io.StringIO()):
            out = lab.cmd_baseline(args)
        self.assertEqual(sm.call_args.kwargs["splits"], ("practice",))
        self.assertNotIn("exam", led["mainLine"][-1])
        self.assertEqual(out["practice"]["wrongCalls"], 10)

    def test_json_mode_never_prompts(self):
        with self.assertRaises(lab.Stop):
            self.fold(False, check=False, yes=False, json=True)

    def test_folded_result_names_the_new_head(self):
        out, _, _ = self.fold(False, check=False)
        self.assertEqual((out["state"], out["head"]), ("folded", "dddd444"))

    def test_json_flag_prints_one_object_and_sends_prose_to_stderr(self):
        led = {"version": 1, "kind": "lab-ledger", "main": "main", "testSet": None, "testSets": {}, "mainLine": [], "experiments": {}}
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(lab, "load", return_value=led), patch.object(lab.sys, "argv", ["lab.py", "--json", "status"]), \
                patch.object(lab.L, "format_status", return_value=["prose"]), redirect_stdout(stdout), redirect_stderr(stderr):
            lab.main()
        self.assertEqual(json.loads(stdout.getvalue()), {"testSet": None, "rows": []})
        self.assertEqual(stderr.getvalue().strip(), "prose")

    def test_json_errors_are_objects_with_a_nonzero_exit(self):
        stdout = io.StringIO()
        with patch.object(lab, "load", side_effect=lab.Stop("boom")), patch.object(lab.sys, "argv", ["lab.py", "--json", "status"]), \
                redirect_stdout(stdout), self.assertRaises(SystemExit) as exit:
            lab.main()
        self.assertEqual(exit.exception.code, 1)
        self.assertEqual(json.loads(stdout.getvalue()), {"error": "boom"})

    def test_main_refresh_forwards_the_requested_split(self):
        report = {"kind": "lab-score", "sha": "aaaa111", "run": "refresh", "testSet": "synthetic",
                  "listener": L.DEFAULT_LISTENER, "sessions": {"excluded": []}, "splits": {"practice": totals(10)}}
        with patch.object(lab, "main_checkout") as checkout, patch.object(lab, "score", return_value=report) as score, \
                patch.object(lab, "load", return_value=L.empty()):
            checkout.return_value.__enter__.return_value = (Path("/unused"), "aaaa111")
            result = lab.score_main(SimpleNamespace(), {}, splits=("practice",))
        self.assertEqual(score.call_args.kwargs["splits"], ("practice",))
        self.assertNotIn("exam", result)

    def test_summary_does_not_inspect_exam_and_preserves_committed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            (wt / Path(lab.SUMMARY).parent).mkdir(parents=True)
            prior = {"kind": "main", "at": "t0", "sha": "aaaa111", "testSet": "synthetic",
                     "experiment": None, "note": "historical opaque row", "practice": {}, "exam": {"opaque": "retained"}}
            (wt / lab.SUMMARY).write_text(json.dumps({"rows": [prior]}))
            led = {"main": "main", "testSet": "synthetic", "experiments": {}, "mainLine": [
                {"at": "t0", "sha": "aaaa111", "testSet": "synthetic", "practice": L.score(totals(10)), "exam": object()},
                {"at": "t1", "sha": "bbbb222", "testSet": "synthetic", "practice": L.score(totals(5))},
            ]}
            result = lab.fold_summary(led, wt, True)
        self.assertEqual(result["rows"][0], prior)
        self.assertIsNone(result["rows"][1]["exam"])


if __name__ == "__main__":
    unittest.main()
