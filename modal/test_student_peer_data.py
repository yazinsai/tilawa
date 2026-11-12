import unittest

from student_peer_data import admitted_help, deployment_candidates, partition, recording_orders, select_deployment


class TrainingDataTest(unittest.TestCase):
    def test_mistake_scripts_and_mixed_split_speakers_are_excluded(self):
        def row(speaker, kind="none", split="train", **kw):
            return dict(speaker=speaker, kind=kind, split=split, **kw)
        data = [row("mixed"), row("mixed", split="eval"), row("mistake", kind="substitution"),
                row("repeat", non_errors=[{"kind": "repeat"}]), row("clean"),
                row("extra", scored=False), row("excluded", excluded="review"),
                row("script", events=[{"errorClass": "added_word"}]), row(None)]
        self.assertEqual([r["speaker"] for r in admitted_help(data)], ["clean"])

    def test_all_recordings_of_a_speaker_share_a_stable_group(self):
        speakers = ["a", "b", "c", "a", "d", "e", "f"]
        groups = partition(speakers)
        self.assertEqual(groups, partition(list(reversed(speakers))))
        self.assertEqual(set(groups.values()), {0, 1, 2})
        for fold in range(3):
            train = {s for s in speakers if groups[s] != fold}
            held = {s for s in speakers if groups[s] == fold}
            self.assertFalse(train & held)

    def test_unidentified_or_insufficient_speakers_fail(self):
        for speakers in (["a", "a"], ["a", "b", None], ["a", "b", ""]):
            with self.assertRaises(ValueError):
                partition(speakers)

    def test_different_orders_have_identical_total_exposure(self):
        exposures = [[0] * 31 for _ in range(3)]
        for epoch in range(18):
            orders = recording_orders(31, epoch)
            self.assertEqual(orders, recording_orders(31, epoch))
            self.assertEqual(len({tuple(o) for o in orders}), 3)
            for peer, order in enumerate(orders):
                self.assertEqual(sorted(order), list(range(31)))
                for i in order:
                    exposures[peer][i] += 1
        self.assertTrue(all(n == 18 for counts in exposures for n in counts))
        self.assertNotEqual(recording_orders(31, 0), recording_orders(31, 1))

    def test_deployment_selection_weights_speakers_across_unequal_folds(self):
        folds = [[dict(speakers=1, per=p, kl=0.) for p in (0., .4, .8)],
                 [dict(speakers=9, per=p, kl=0.) for p in (.6, .4, .8)]]
        candidates = deployment_candidates(1., folds)
        self.assertAlmostEqual(candidates[0]["score"], .54)
        self.assertEqual(select_deployment(candidates)["head"], 1)
        # Additional training recordings affect a speaker's own mean metric;
        # the selection weight remains its speaker count, never frame count.
        preserved = deployment_candidates(.1, [[dict(speakers=2, per=.2, kl=k) for k in (2., 0., 1.)]])
        self.assertEqual(select_deployment(preserved)["head"], 1)

    def test_selection_ties_are_stable_and_invalid_metrics_fail(self):
        folds = [[dict(speakers=1, per=.2, kl=.3)] * 3]
        candidates = deployment_candidates(1., folds) + deployment_candidates(.1, folds)
        for order in (candidates, list(reversed(candidates))):
            chosen = select_deployment(order)
            self.assertEqual((chosen["strength"], chosen["head"]), (.1, 0))
        for values in ([], [[dict(speakers=0, per=.1, kl=0.)] * 3],
                       [[dict(speakers=1, per=float("nan"), kl=0.)] * 3]):
            with self.assertRaises(ValueError):
                deployment_candidates(1., values)


if __name__ == "__main__":
    unittest.main()
