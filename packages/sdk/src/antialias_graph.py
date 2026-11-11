"""Geometry-fixed, centered temporal antialiasing for an exported streaming ear.

This is a separate inference graph: original weights and output/state geometry
are retained. Run on CPU beside the SDK; it does not train or quantize weights.
"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def kernel(stride):
    """A unit-DC binomial low-pass, radius stride - 1, with zero phase."""
    if stride < 2:
        raise ValueError("A temporal stride must exceed one")
    degree = 2 * (stride - 1)
    return [math.comb(degree, k) / 2**degree for k in range(degree + 1)]


def centered_filter(source, rank, axis, stride, name):
    """No channel mixing, no decimation and no coordinate or length change.

    Edge extension is explicit, so even chunk-edge constant inputs retain DC.
    The symmetric filter is centered on each original sample; its causal FIR
    delay (stride - 1 input frames) is compensated by the left extension.
    """
    nodes, constants = [], []

    def const(label, data, dtype):
        key = name + "/" + label
        constants.append(numpy_helper.from_array(np.asarray(data, dtype=dtype), key))
        return key

    weights = kernel(stride)
    radius = stride - 1
    pads = [0] * (2 * rank)
    pads[axis] = pads[rank + axis] = radius
    padded = name + "/padded"
    nodes.append(helper.make_node("Pad", [source, const("pads", pads, np.int64)], [padded],
                                  name=padded, mode="edge"))
    terms = []
    for k, weight in enumerate(weights):
        sliced, term = name + f"/slice{k}", name + f"/term{k}"
        end = k - 2 * radius if k < 2 * radius else np.iinfo(np.int64).max
        nodes.append(helper.make_node("Slice", [padded, const(f"start{k}", [k], np.int64),
            const(f"end{k}", [end], np.int64), const(f"axis{k}", [axis], np.int64)],
            [sliced], name=sliced))
        nodes.append(helper.make_node("Mul", [sliced, const(f"weight{k}", weight, np.float32)],
                                      [term], name=term))
        terms.append(term)
    output = name + "/filtered"
    nodes.append(helper.make_node("Sum", terms, [output], name=output))
    return nodes, constants, output


def geometry(model):
    """Locate acoustic striding in the exported Zipformer graph, not masks.

    Conv2dSubsampling uses N,C,time,frequency. SimpleDownsample splits T,B,C
    into T/stride,stride,B,C, then reduces the stride dimension with the
    shipped learned weights. Shape inference verifies those splits explicitly.
    """
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        shapes[v.name] = [d.dim_value if d.HasField("dim_value") else None
                          for d in v.type.tensor_type.shape.dim]
    sites = []
    for n in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        shape = shapes.get(n.input[0], []) if n.input else []
        if n.op_type == "Conv" and len(n.input) > 1 and "encoder_embed" in n.input[1]:
            strides = attrs.get("strides", [1, 1])
            if strides[0] > 1:
                if len(shape) != 4:
                    raise ValueError(f"Unknown acoustic Conv layout: {n.name}, {shape}")
                sites.append((n, 4, 2, strides[0]))
        elif n.op_type == "Reshape" and "downsample" in n.name.lower():
            out = shapes.get(n.output[0], [])
            if len(shape) == 3 and len(out) == 4:
                if not (out[1] and out[1] > 1 and shape[0] and out[0]
                        and shape[0] == out[0] * out[1]
                        and all(a is None or b is None or a == b for a, b in zip(shape[1:], out[2:]))):
                    raise ValueError(f"Unknown acoustic downsample split: {n.name}, {shape} -> {out}")
                sites.append((n, 3, 0, out[1]))
    if not sites:
        raise ValueError("No supported temporal stride operations found")
    return sites


def build(source, destination):
    model = onnx.load(source)
    sites = geometry(model)
    selected = {n.name: (rank, axis, stride) for n, rank, axis, stride in sites}
    nodes, report = [], []
    for n in model.graph.node:
        if n.name in selected:
            rank, axis, stride = selected[n.name]
            added, constants, output = centered_filter(n.input[0], rank, axis, stride, n.name + "/antialias")
            report.append(dict(node=n.name, input=n.input[0], axis=axis, stride=stride,
                               kernel=kernel(stride), inputDelay=0))
            nodes.extend(added)
            model.graph.initializer.extend(constants)
            n.input[0] = output
        nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    return dict(recipe="centered-binomial-strides-v1", sourceSha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest(), sites=report)


def bundle(source, destination):
    """Copy the complete primary bundle unchanged, including existing auxiliary ears."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("The recheck must be a separate bundle")
    manifest = json.loads((source / "manifest.json").read_text())
    digest = hashlib.sha256((source / manifest["file"]).read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Source listener checksum mismatch")
    shutil.copytree(source, destination, dirs_exist_ok=True)
    report = build(source / manifest["file"], destination / "antialias.onnx")
    (destination / "antialias.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    print(json.dumps((bundle if args.bundle else build)(args.source, args.destination), indent=2))
