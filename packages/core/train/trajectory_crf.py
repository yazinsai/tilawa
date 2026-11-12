"""Exact conditional partitions and feature expectations on the exported tracker DAG.

All consuming arcs read the preceding observation layer. Deletions read the current
layer's preceding reference cell, so their closure is acyclic and consumes no token.
The numerator intersects that graph with ordered transcript occurrences, including
repeats; it does not independently label or normalize observations or word matches.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp


@dataclass
class Layer:
    cost: np.ndarray
    x: np.ndarray
    parent: np.ndarray  # destination x 3 consuming arc kinds


@dataclass
class Graph:
    initial: np.ndarray
    layers: list
    terminal: np.ndarray


def compile_graph(row, dimension, transcript=False):
    layers = []
    init = row["initial"]
    route = np.array(row["coordinates"], dtype=int)
    previous = route if transcript else np.arange(init["lo"], init["lo"] + init["n"])
    initial = np.full(len(previous), -np.inf)
    if transcript:
        initial[0] = 0
    else:
        initial[init["at"] - init["lo"]] = 0
    for frame in row["frames"]:
        coords = route if transcript else np.arange(frame["lo"], frame["lo"] + len(frame["arcs"]))
        cost = np.full((len(coords), 4), np.inf)
        x = np.zeros((len(coords), 4, dimension))
        parent = np.full((len(coords), 3), -1, dtype=int)
        for j, at in enumerate(coords):
            cell = at - frame["lo"]
            if cell < 0 or cell >= len(frame["arcs"]):
                continue
            for kind, arc in enumerate(frame["arcs"][cell]):
                if arc is None:
                    continue
                if kind == 3:
                    if j == 0 or coords[j - 1] != at - 1:
                        continue
                elif transcript:
                    src = j if kind == 0 else j - 1
                    if src < 0:
                        continue
                    expected = at if kind == 0 else at - 1 if kind == 1 else frame["source"]
                    if previous[src] != expected:
                        continue
                    parent[j, kind] = src
                else:
                    expected = at if kind == 0 else at - 1 if kind == 1 else frame["source"]
                    src = expected - previous[0]
                    if src < 0 or src >= len(previous):
                        continue
                    parent[j, kind] = src
                cost[j, kind] = arc["cost"]
                x[j, kind] = arc["x"]
        layers.append(Layer(cost, x, parent))
        previous = coords
    terminal = np.zeros(len(previous))
    if transcript:
        terminal[:-1] = -np.inf
    return Graph(initial, layers, terminal)


def closure(seed, delete, reverse=False, maximum=False):
    """Linear-time log-sum (or max) closure, split at absent deletion arcs."""
    result = np.empty_like(seed)
    boundaries = np.r_[0, np.flatnonzero(~np.isfinite(delete[1:])) + 1, len(seed)]
    accumulate = np.maximum.accumulate if maximum else np.logaddexp.accumulate
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        prefix = np.r_[0., np.cumsum(delete[a + 1:b])]
        if reverse:
            result[a:b] = -prefix + accumulate((seed[a:b] + prefix)[::-1])[::-1]
        else:
            result[a:b] = prefix + accumulate(seed[a:b] - prefix)
    return result


def partition(graph, weights, gradient=True):
    scores = [-layer.cost - layer.x @ weights for layer in graph.layers]
    alpha, maps = [graph.initial], graph.initial
    for layer, score in zip(graph.layers, scores):
        parents = np.maximum(layer.parent, 0)
        seeds = alpha[-1][parents] + score[:, :3]
        seeds[layer.parent < 0] = -np.inf
        alpha.append(closure(logsumexp(seeds, axis=1), score[:, 3]))
        best = maps[parents] + score[:, :3]
        best[layer.parent < 0] = -np.inf
        maps = closure(np.max(best, axis=1), score[:, 3], maximum=True)
    z = logsumexp(alpha[-1] + graph.terminal)
    viterbi = np.max(maps + graph.terminal)
    if not gradient or not np.isfinite(z):
        return z, None, viterbi
    expectation = np.zeros_like(weights)
    beta = graph.terminal
    for t in range(len(graph.layers) - 1, -1, -1):
        layer, score = graph.layers[t], scores[t]
        beta = closure(beta, score[:, 3], reverse=True)
        parents = np.maximum(layer.parent, 0)
        direct = alpha[t][parents] + score[:, :3] + beta[:, None] - z
        direct[layer.parent < 0] = -np.inf
        mass = np.exp(direct)
        expectation += np.einsum("ik,ikd->d", mass, layer.x[:, :3])
        deleted = np.exp(alpha[t + 1][:-1] + score[1:, 3] + beta[1:] - z)
        expectation += deleted @ layer.x[1:, 3]
        previous = np.full(len(alpha[t]), -np.inf)
        for kind in range(3):
            valid = layer.parent[:, kind] >= 0
            np.logaddexp.at(previous, layer.parent[valid, kind], (score[:, kind] + beta)[valid])
        beta = previous
    # Scores are negative costs: derivative of log Z is negative expected features.
    return z, -expectation, viterbi


def conditional_loss(pair, weights, gradient=True):
    all_z, all_g, all_v = partition(pair[0], weights, gradient)
    gold_z, gold_g, gold_v = partition(pair[1], weights, gradient)
    if not np.isfinite(gold_z):
        raise ValueError("Transcript route is not reachable in the unchanged candidate graph")
    if gold_z > all_z + 1e-7:
        raise ValueError("Numerator is not a subset of the candidate graph")
    return all_z - gold_z, all_g - gold_g if gradient else None, all_v - gold_v
