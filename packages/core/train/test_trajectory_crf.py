import itertools
import unittest
import numpy as np
from trajectory_crf import Graph, Layer, partition, compile_graph, conditional_loss


class ConditionalTrajectoryTests(unittest.TestCase):
    def graph(self):
        rng = np.random.default_rng(17)
        layers = []
        for _ in range(3):
            cost = rng.uniform(.1, 2, (3, 4))
            cost[0, 1:] = np.inf
            parent = np.array([[0, -1, -1], [1, 0, 0], [2, 1, 0]])
            layers.append(Layer(cost, rng.normal(size=(3, 4, 2)), parent))
        return Graph(np.array([0., -np.inf, -np.inf]), layers, np.zeros(3))

    def enumerate(self, graph, weights):
        paths = [(0, 0., np.zeros(len(weights)))]
        for layer in graph.layers:
            next_paths = []
            for src, old_score, old_x in paths:
                for dest, kind in itertools.product(range(len(layer.cost)), range(3)):
                    if layer.parent[dest, kind] != src or not np.isfinite(layer.cost[dest, kind]):
                        continue
                    score = old_score - layer.cost[dest, kind] - layer.x[dest, kind] @ weights
                    x = old_x + layer.x[dest, kind]
                    next_paths.append((dest, score, x))
                    for j in range(dest + 1, len(layer.cost)):
                        if not np.isfinite(layer.cost[j, 3]):
                            break
                        score -= layer.cost[j, 3] + layer.x[j, 3] @ weights
                        x = x + layer.x[j, 3]
                        next_paths.append((j, score, x))
            paths = next_paths
        paths = [(s, score, x) for s, score, x in paths if np.isfinite(graph.terminal[s])]
        scores = np.array([p[1] + graph.terminal[p[0]] for p in paths])
        shift = scores.max()
        probs = np.exp(scores - shift); probs /= probs.sum()
        return shift + np.log(np.exp(scores - shift).sum()), -sum(p * path[2] for p, path in zip(probs, paths)), scores.max()

    def test_full_partition_gradient_and_map_equal_exhaustive_paths(self):
        graph, weights = self.graph(), np.array([.2, -.3])
        z, grad, best = partition(graph, weights)
        ez, eg, eb = self.enumerate(graph, weights)
        self.assertAlmostEqual(z, ez, places=12)
        self.assertAlmostEqual(best, eb, places=12)
        np.testing.assert_allclose(grad, eg, atol=1e-12)
        for i in range(len(weights)):
            delta = np.eye(len(weights))[i] * 1e-5
            finite = (partition(graph, weights + delta)[0] - partition(graph, weights - delta)[0]) / 2e-5
            self.assertAlmostEqual(grad[i], finite, places=8)

    def row(self, route):
        arc = lambda cost: {"cost": cost, "x": [cost]}
        return {"initial": {"lo": 0, "n": 4, "at": 0}, "coordinates": route,
                "frames": [{"lo": 0, "source": 0, "arcs": [
                    [arc(1), None, None, None], [arc(1), arc(0), None, arc(1)],
                    [arc(1), arc(0), None, arc(1)], [arc(1), arc(0), arc(2), arc(1)]]}]}

    def test_omission_route_uses_existing_jump_not_fabricated_deletions(self):
        row = self.row([0, 3])
        pair = compile_graph(row, 1), compile_graph(row, 1, True)
        self.assertEqual(partition(pair[1], np.zeros(1))[0], -2)
        loss, g, _ = conditional_loss(pair, np.zeros(1))
        self.assertGreater(loss, 0)
        self.assertTrue(np.isfinite(g).all())
        row["frames"][0]["arcs"][3][2] = None
        with self.assertRaisesRegex(ValueError, "not reachable"):
            conditional_loss((compile_graph(row, 1), compile_graph(row, 1, True)), np.zeros(1))

    def test_reference_only_closure_and_complete_endpoint(self):
        row = self.row([0, 1, 2, 3])
        gold = compile_graph(row, 1, True)
        z, g, best = partition(gold, np.zeros(1))
        self.assertGreater(z, -2)  # both consuming and deletion paths contribute
        self.assertAlmostEqual(best, -2)
        gold.terminal[:] = -np.inf
        self.assertEqual(partition(gold, np.zeros(1))[0], -np.inf)

    def test_recenter_projects_coordinates_without_new_arcs(self):
        row = self.row([0, 1, 2, 3])
        next_frame = {"lo": 1, "source": 1, "arcs": row["frames"][0]["arcs"][1:]}
        row["frames"].append(next_frame)
        full = compile_graph(row, 1)
        self.assertEqual(full.layers[1].parent[0, 0], 1)
        self.assertEqual(full.layers[1].parent[0, 1], 0)
        self.assertEqual(full.layers[1].cost[0, 3], np.inf)
        z, g, _ = partition(full, np.array([.1]))
        self.assertTrue(np.isfinite(z))
        self.assertTrue(np.isfinite(g).all())

    def test_repeat_occurrences_remain_distinct_without_duplicate_partition_mass(self):
        arc = lambda price: {"cost": price, "x": [price]}
        row = {"initial": {"lo": 0, "n": 3, "at": 0}, "coordinates": [0, 1, 2, 1, 2], "frames": []}
        for src in [0, 1, 2, 1]:
            row["frames"].append({"lo": 0, "source": src, "arcs": [
                [arc(1), None, None, None], [arc(1), arc(0), arc(2) if src != 0 else None, arc(1)],
                [arc(1), arc(0), arc(2) if src != 1 else None, arc(1)]]})
        pair = compile_graph(row, 1), compile_graph(row, 1, True)
        loss, gradient, _ = conditional_loss(pair, np.array([.2]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(gradient).all())
        z, g, best = partition(pair[1], np.array([.2]))
        ez, eg, eb = self.enumerate(pair[1], np.array([.2]))
        self.assertAlmostEqual(z, ez, places=12)
        self.assertAlmostEqual(best, eb, places=12)
        np.testing.assert_allclose(g, eg, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
