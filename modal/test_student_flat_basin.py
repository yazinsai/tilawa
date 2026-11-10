"""Training data isolation and matched-arm planning; no GPU or lab inputs."""

import unittest

from student_flat_basin import batch_records, check_geometry, help_role, reserve_groups


class TrainingPartitionsTest(unittest.TestCase):
    def test_speaker_reservation_is_order_independent(self):
        groups = [f"speaker-{i}" for i in range(10)]
        reserved = reserve_groups(groups)
        self.assertEqual(len(reserved), 2)
        self.assertEqual(reserved, reserve_groups(list(reversed(groups)) + groups))
        train = [g for g in groups for _ in range(3) if g not in reserved]
        validation = [g for g in groups for _ in range(3) if g in reserved]
        self.assertFalse(set(train) & set(validation))

    def test_no_ungrouped_validation(self):
        for groups in ([], ["one", "one"]):
            with self.assertRaises(ValueError):
                reserve_groups(groups)

    def test_only_explicit_clean_training_takes_have_targets(self):
        clean = dict(split="train", kind="none", events=[], non_errors=[], unsupported=[])
        self.assertEqual(help_role(clean), "none")
        for patch in (dict(split="eval"), dict(split=None), dict(excluded="review"),
                      dict(scored=False), dict(events=[{"errorClass": "wrong_word"}]),
                      dict(non_errors=[{"kind": "repeat"}]), dict(unsupported=[{"rule": "other"}])):
            self.assertIsNone(help_role({**clean, **patch}))

    def test_mistake_scripts_are_preservation_only(self):
        for kind in ("substitution", "skip_word", "skip_ayah", "vowel", "tajweed", "repeat"):
            label = dict(split="train", kind=kind, events=[{"expected": "x", "observed": "y"}])
            self.assertEqual(help_role(label), "preserve")
            self.assertIsNone(help_role({**label, "split": "eval"}))

    def test_control_and_sam_share_resumable_batch_plan(self):
        data = [dict(key=f"{role}/{i}", role=role) for role in ("original", "none", "preserve") for i in range(10)]
        uninterrupted = [batch_records(data, i) for i in range(7)]
        resumed = [batch_records(data, i) for i in range(4, 7)]
        self.assertEqual(uninterrupted[4:], resumed)
        self.assertEqual([r["role"] for r in uninterrupted[0]], ["original"] * 4 + ["none", "preserve"])

    def test_export_rejects_inventory_or_timing_changes(self):
        source = dict(tokens="abc", frameMs=40, window=45, states=[dict(name="cache", shape=[1, 8])])
        check_geometry(dict(source, sha256="new-weights"), source)
        for patch in (dict(tokens="acb"), dict(frameMs=20), dict(window=46), dict(states=[])):
            with self.assertRaises(ValueError):
                check_geometry({**source, **patch}, source)


if __name__ == "__main__":
    unittest.main()
