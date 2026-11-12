"""Unchanged-weight norm-neutral attention export for the streaming student.

Only the content query/key dot products change. Raw cached keys, positional
scores and value retrieval retain their original graph paths. Calibration is
supplied by the caller and must come from training recordings, not evaluation.
"""

from copy import deepcopy


def attention_sites(model):
    """Find the pinned Zipformer's content attention, rejecting ambiguous graphs."""
    import onnx

    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    sites = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or not node.name.endswith("/self_attn_weights/MatMul"):
            continue
        following = consumers.get(node.output[0], [])
        if len(following) != 1 or following[0].op_type != "Add":
            raise ValueError(f"Unrecognized positional attention at {node.name}")
        sites.append((node, following[0]))
    if not sites:
        raise ValueError("No streaming Zipformer content attention found")
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: v.type.tensor_type.shape for v in [
        *inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output,
    ]}
    for node, _ in sites:
        q, k = [shapes.get(name) for name in node.input]
        if q is None or k is None or len(q.dim) != 4 or len(k.dim) != 4:
            raise ValueError(f"Attention query/key must be rank four at {node.name}")
        if q.dim[0].dim_value <= 0 or q.dim[0].dim_value != k.dim[0].dim_value:
            raise ValueError(f"Unknown head count at {node.name}")
    return [(n.name, n.input[0], n.input[1], n.output[0], add.output[0],
             shapes[n.input[0]].dim[0].dim_value) for n, add in sites]


def recheck_graph(model, scales, *, temperature=False):
    """Export norm-neutral Q/K or its calibrated temperature-only control.

    ``scales`` maps every content MatMul name to one positive scale per head.
    Norm-neutral scales apply to cosine content scores before the unchanged
    positional addition. Control scales apply to full unmasked logits; the
    padding mask is never scaled. Neither arm writes normalized keys to cache.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    sites = attention_sites(model)
    if set(scales) != {s[0] for s in sites}:
        raise ValueError("Calibration must cover every attention site exactly")
    out = deepcopy(model)
    replacements = {}
    for i, (name, q, k, content, combined, heads) in enumerate(sites):
        scale = np.asarray(scales[name], dtype=np.float32)
        if scale.shape != (heads,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"Invalid head scales at {name}")
        prefix = f"norm_neutral_{i}"
        out.graph.initializer.append(numpy_helper.from_array(scale.reshape(heads, 1, 1, 1), prefix + "_scale"))
        if temperature:
            replacements[combined] = (prefix + "_scaled", [helper.make_node(
                "Mul", [combined, prefix + "_scale"], [prefix + "_scaled"], name=prefix + "_temperature")])
            continue
        extra = []
        normalized = []
        out.graph.initializer.append(numpy_helper.from_array(np.array(np.finfo(np.float32).tiny, np.float32), prefix + "_eps"))
        for source, suffix, axis in [(q, "q", -1), (k, "k", -2)]:
            base = prefix + "_" + suffix
            out.graph.initializer.append(numpy_helper.from_array(np.array([axis], np.int64), base + "_axis"))
            extra.extend([
                helper.make_node("Mul", [source, source], [base + "_square"], name=base + "_square"),
                helper.make_node("ReduceSum", [base + "_square", base + "_axis"], [base + "_ss"], keepdims=1, name=base + "_sum"),
                helper.make_node("Max", [base + "_ss", prefix + "_eps"], [base + "_safe"], name=base + "_safe"),
                helper.make_node("Sqrt", [base + "_safe"], [base + "_norm"], name=base + "_norm"),
                helper.make_node("Div", [source, base + "_norm"], [base + "_unit"], name=base + "_unit"),
            ])
            normalized.append(base + "_unit")
        extra.extend([
            helper.make_node("MatMul", normalized, [prefix + "_cosine"], name=prefix + "_cosine"),
            helper.make_node("Mul", [prefix + "_cosine", prefix + "_scale"], [prefix + "_scaled"], name=prefix + "_scaled"),
        ])
        replacements[content] = (prefix + "_scaled", extra)
    # Insert operations after the original producer, replacing only its consumers.
    nodes = []
    for node in out.graph.node:
        for j, source in enumerate(node.input):
            if source in replacements:
                node.input[j] = replacements[source][0]
        nodes.append(node)
        for produced in node.output:
            if produced in replacements:
                nodes.extend(replacements[produced][1])
    del out.graph.node[:]
    out.graph.node.extend(nodes)
    onnx.checker.check_model(out)
    return out


def spread_scale(unit, positional, original):
    """Positive cosine scale matching original per-query centered logit spread.

    Arrays contain valid keys only, one training query per row. Positional
    scores stay fixed, so their covariance with content is included explicitly.
    """
    import numpy as np

    def center(x):
        x = np.asarray(x, dtype=np.float64)
        return x - x.mean(axis=-1, keepdims=True)

    u, p, o = center(unit), center(positional), center(original)
    a = np.mean(u * u)
    b = 2 * np.mean(u * p)
    c = np.mean(p * p) - np.mean(o * o)
    if a <= 0:
        raise ValueError("Training queries have no content spread")
    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError("Original spread is unattainable with fixed position scores")
    scale = (-b + np.sqrt(disc)) / (2 * a)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("No positive spread-matching scale")
    return float(scale)


def entropy_temperature(original, neutral):
    """Fixed control temperature matching norm-neutral training entropy."""
    import numpy as np

    original, neutral = np.asarray(original), np.asarray(neutral)

    def entropy(x):
        z = np.asarray(x, dtype=np.float64)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        return float(np.mean(-np.sum(p * (z - np.log(e.sum(axis=-1, keepdims=True))), axis=-1)))

    target = entropy(neutral)
    lo, hi = 0.0, 1.0
    while entropy(original * hi) > target:
        hi *= 2
        if hi > 2 ** 20:
            raise ValueError("Cannot match training entropy")
    for _ in range(60):
        mid = (lo + hi) / 2
        if entropy(original * mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
