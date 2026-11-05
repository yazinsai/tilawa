"""Attach the frozen CTC word-separator projection to the shipped encoder.

Inputs are the shipped export and the existing phones-plus-| export. Only the
head's ancestors outside the encoder are copied; matching feature producers
bind it to the shipped graph. The original phone graph and its caches are kept.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def attach(main_dir: Path, separator_dir: Path, out_dir: Path) -> dict:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    main = json.loads((main_dir / "manifest.json").read_text())
    side = json.loads((separator_dir / "manifest.json").read_text())
    for field in ("sampleRate", "frameMs", "window", "shift", "inventory"):
        if main[field] != side[field]:
            raise ValueError(f"Incompatible separator frame clock: {field}")
    separator_id = side["tokens"].index("|") + 1
    base = onnx.load(str(main_dir / main["file"]))
    auxiliary = onnx.load(str(separator_dir / side["file"]))
    base_inputs = {v.name for v in base.graph.input}
    available = base_inputs | {v for n in base.graph.node for v in n.output}
    base_producers = {v: n for n in base.graph.node for v in n.output}
    producers = {v: i for i, n in enumerate(auxiliary.graph.node) for v in n.output}
    initializers = {v.name: v for v in auxiliary.graph.initializer}
    selected: set[int] = set()
    parameters: set[str] = set()
    shared: set[str] = set()

    def same_producer(value: str) -> bool:
        if value not in base_producers or value not in producers:
            return False
        left, right = base_producers[value], auxiliary.graph.node[producers[value]]
        return (left.op_type == right.op_type and left.domain == right.domain and
                list(left.input) == list(right.input) and list(left.output) == list(right.output) and
                [a.SerializeToString() for a in left.attribute] == [a.SerializeToString() for a in right.attribute])

    def visit(value: str):
        if not value:
            return
        if value in base_inputs or (value in available and same_producer(value)):
            shared.add(value)
            return
        if value in initializers:
            parameters.add(value)
            return
        if value not in producers:
            raise ValueError(f"No source for separator tensor {value}")
        i = producers[value]
        if i in selected:
            return
        node = auxiliary.graph.node[i]
        selected.add(i)
        for source in node.input:
            visit(source)

    visit("log_probs")
    if sum(len(initializers[n].raw_data) for n in parameters) > 1024 * 1024:
        raise ValueError("Export would duplicate encoder weights instead of attaching a small head")
    rename = lambda value: value if not value or value in shared else "separator/" + value
    graph = copy.deepcopy(base.graph)
    for i in sorted(selected):
        node = copy.deepcopy(auxiliary.graph.node[i])
        node.name = "separator/" + node.name
        for values in (node.input, node.output):
            for j, value in enumerate(values):
                values[j] = rename(value)
        graph.node.append(node)
    for name in sorted(parameters):
        parameter = copy.deepcopy(initializers[name])
        parameter.name = rename(name)
        graph.initializer.append(parameter)
    graph.initializer.append(numpy_helper.from_array(np.array(separator_id, dtype=np.int64), "separator/token_id"))
    graph.node.extend([
        helper.make_node("ArgMax", [rename("log_probs")], ["separator/greedy"], axis=2, keepdims=0),
        helper.make_node("Equal", ["separator/greedy", "separator/token_id"], ["separator/is_boundary"]),
        helper.make_node("Cast", ["separator/is_boundary"], ["word_separator"], to=TensorProto.INT64),
    ])
    graph.output.append(helper.make_tensor_value_info("word_separator", TensorProto.INT64, [1, None]))
    result = copy.deepcopy(base)
    result.graph.CopyFrom(graph)
    onnx.checker.check_model(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"
    onnx.save(result, str(path))
    data = path.read_bytes()
    manifest = {**main, "id": main["id"] + "+word-separator", "file": path.name,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps({"headNodes": len(selected), "sharedFeatures": sorted(shared),
                      "addedBytes": len(data) - (main_dir / main["file"]).stat().st_size}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path)
    parser.add_argument("separator", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    print(json.dumps(attach(args.main, args.separator, args.out), ensure_ascii=False))
