"""Inventory-only phonetic tree and exact parameter matching; no acoustic data.

The manner groups agree with core's generic phonetic vocabulary. Inventory
extensions (ghunna, vowel colouring, length/release and pause marks) remain
ordinary leaves. Affricate ج and emphatic ض are grouped with oral closures;
this is a coarse acoustic partition, not a claim of one universal realization.
"""
from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class Node:
    name: str
    children: tuple[tuple[int, ...], ...]


def phonetic_tree(tokens: str) -> tuple[list[Node], list[list[tuple[int, int]]]]:
    """Return child leaf sets and root-to-leaf (node, child) paths in ID order."""
    if len(set(tokens)) != len(tokens):
        raise ValueError("Phone inventory contains duplicates")
    spec = ("root", [0, ("units", [
        ("obstruents", [("closures", list("ءبتجدضطقك")),
                         ("frication", list("ثحخذزسشصظعغفه"))]),
        ("sonorants", [("nasals", list("منں۾")), ("liquids", list("رل")),
                       ("glides", list("وي"))]),
        ("vowels", [("low", list("اَٲ")), ("front", list("ۦِ۪")),
                    ("back", list("ۥُ"))]),
        ("length-release", list("ـڇ")),
        ("pauses", list("ؙۜ")),
    ])])
    ids = {phone: i + 1 for i, phone in enumerate(tokens)}
    nodes, paths = [], [[] for _ in range(len(tokens) + 1)]
    seen = set()

    def visit(branch, path):
        if isinstance(branch, (str, int)):
            if branch != 0 and branch not in ids:
                raise ValueError("Phonetic tree and phone inventory differ")
            leaf = 0 if branch == 0 else ids[branch]
            if leaf in seen:
                raise ValueError("Phonetic tree repeats a leaf")
            seen.add(leaf)
            paths[leaf] = path
            return (leaf,)
        name, children = branch
        index = len(nodes)
        nodes.append(None)
        descendants = tuple(visit(child, path + [(index, i)]) for i, child in enumerate(children))
        if len(descendants) < 2 or any(not child for child in descendants):
            raise ValueError("Each internal node needs at least two nonempty branches")
        nodes[index] = Node(name, descendants)
        return tuple(leaf for child in descendants for leaf in child)

    visit(spec, [])
    if seen != set(range(len(tokens) + 1)):
        raise ValueError("Phonetic tree does not cover the unchanged inventory")
    return nodes, paths


def matched_widths(dim: int, tokens: str) -> tuple[int, int, int]:
    """Uniform node bottleneck vs a flat MLP, with every parameter active.

    Biased D->H and bias-free H->arity at every node. The flat control has
    biased D->F and bias-free F->V. Choose the smallest exact uniform H>=8,
    using inventory alone, with a one-megabyte fp32 trainable-head bound.
    """
    if dim < 1:
        raise ValueError("Encoder width must be positive")
    nodes, _ = phonetic_tree(tokens)
    node_unit = sum(dim + 1 + len(n.children) for n in nodes)
    flat_unit = dim + 1 + len(tokens) + 1
    step = flat_unit // gcd(node_unit, flat_unit)
    hidden = ((8 + step - 1) // step) * step
    count = hidden * node_unit
    if count * 4 > 1024 * 1024:
        raise ValueError("No small exact-parameter flat control for this encoder")
    return hidden, count // flat_unit, count
