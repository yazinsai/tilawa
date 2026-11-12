"""Dependency-free checks of phonetic routing and probability conservation."""
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS
from student_hierarchical_tree import matched_widths, phonetic_tree


class PhoneticTreeTest(unittest.TestCase):
    def test_inventory_paths_and_generic_families(self):
        nodes, paths = phonetic_tree(TOKENS)
        self.assertEqual(len(paths), 43)
        self.assertEqual(paths[0], [(0, 0)])
        self.assertEqual(set(nodes[0].children[1]), set(range(1, 43)))
        for phone, path in enumerate(paths):
            parent = set(range(43))
            for index, child in path:
                self.assertEqual(set(x for group in nodes[index].children for x in group), parent)
                parent = set(nodes[index].children[child])
            self.assertEqual(parent, {phone})
        families = {n.name: {TOKENS[i-1] for c in n.children for i in c} for n in nodes[1:]}
        self.assertEqual(families["nasals"], set("منں۾"))
        self.assertEqual(families["pauses"], set("ؙۜ"))
        self.assertEqual(families["vowels"], set("اَٲۦِ۪ۥُ"))

    def test_arbitrary_conditional_probabilities_form_one_normalized_inventory(self):
        nodes, paths = phonetic_tree(TOKENS)
        rng = random.Random(41)
        for _ in range(30):
            conditionals = []
            for node in nodes:
                scores = [rng.uniform(-10, 10) for c in node.children]
                mass = [math.exp(x - max(scores)) for x in scores]
                conditionals.append([p / sum(mass) for p in mass])
            leaves = [math.prod(conditionals[i][j] for i, j in path) for path in paths]
            self.assertAlmostEqual(sum(leaves), 1., places=13)
            for node, probabilities in zip(nodes, conditionals):
                parent = sum(leaves[p] for child in node.children for p in child)
                for child, conditional in zip(node.children, probabilities):
                    self.assertAlmostEqual(sum(leaves[p] for p in child), parent * conditional, places=13)

    def test_routing_tracks_phone_identity_when_inventory_order_changes(self):
        original, paths = phonetic_tree(TOKENS)
        reversed_nodes, reversed_paths = phonetic_tree(TOKENS[::-1])
        self.assertEqual([n.name for n in original], [n.name for n in reversed_nodes])
        for phone in TOKENS:
            self.assertEqual(paths[TOKENS.index(phone) + 1], reversed_paths[TOKENS[::-1].index(phone) + 1])
        for invalid in (TOKENS[:-1], TOKENS + "x", TOKENS + TOKENS[0]):
            with self.assertRaises(ValueError):
                phonetic_tree(invalid)

    def test_small_exact_parameter_budget_uses_all_nodes(self):
        nodes, _ = phonetic_tree(TOKENS)
        hidden, flat, count = matched_widths(256, TOKENS)
        self.assertEqual((hidden, flat, count), (25, 326, 97800))
        self.assertEqual(sum(256 * hidden + hidden + hidden * len(n.children) for n in nodes), count)
        self.assertEqual(256 * flat + flat + flat * 43, count)
        self.assertLess(count * 4, 1024 * 1024)
        with self.assertRaises(ValueError):
            matched_widths(0, TOKENS)


if __name__ == "__main__":
    unittest.main()
