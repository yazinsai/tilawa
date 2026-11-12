"""Graft the acoustic edit/CTC heads onto the exact shipped encoder graph.

The primary graph produces source phones, timed edit distributions and CTC
support jointly. The SDK executes the monotone lattice. No second acoustic
model or expected text is needed. All existing verification assets are copied
and checked, and the encoder caches/frontend/alphabet are inherited unchanged.
"""
import copy
import json
import shutil
from pathlib import Path

from student_edit_native import digest, insertion_assets, load_source


def export_head(head, path, width, frames, direct=False):
    import torch
    from torch import nn

    class Export(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head.eval()

        def forward(self, features, source_log_probs, edit_source_previous, edit_valid_frames):
            ids = source_log_probs.argmax(-1)
            previous = torch.cat((edit_source_previous[:, None], ids[:, :-1]), 1)
            valid = torch.arange(frames, device=features.device)[None] < edit_valid_frames[:, None]
            boundary = (ids != previous) & (ids != 0) & valid
            order = (boundary.to(torch.int64).cumsum(1) - 1).clamp(min=0)
            membership = torch.nn.functional.one_hot(order, frames + 1) * boundary[..., None]
            source = (membership * ids[..., None]).sum(1)
            clock = torch.arange(frames, device=features.device)[None, :, None]
            starts = (membership * clock).sum(1) + (membership.sum(1) == 0) * frames
            actions, phones, support, primary = self.head(features, source, starts, direct=direct)
            return actions, phones, support, primary, ids.gather(1, (edit_valid_frames - 1)[:, None]).squeeze(1)

    features = torch.randn(1, frames, width)
    source = torch.randn(1, frames, head.embedding.num_embeddings).log_softmax(-1)
    with torch.no_grad():
        torch.onnx.export(Export(), (features, source, torch.zeros(1, dtype=torch.int64),
            torch.full((1,), frames, dtype=torch.int64)),
            str(path), opset_version=17, dynamo=False,
            input_names=["features", "source_log_probs", "edit_source_previous", "edit_valid_frames"],
            output_names=["edit_actions", "edit_phones", "support_log_probs", "direct_log_probs",
                          "new_edit_source_previous"])


def graft(source, head, out, direct=False):
    import onnx
    from onnx import helper, TensorProto
    m, graph, feature, width = load_source(source)
    if m["shift"] % 4 or m["shift"] < 4:
        raise ValueError("Native shift is not an integer number of 40 ms frames")
    auxiliary = insertion_assets(source)
    frames = m["shift"] // 4
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export must use a new empty bundle directory")
    if Path(source).resolve() == out.resolve() or Path(source).resolve() in out.resolve().parents:
        raise ValueError("Cannot modify or nest inside the installed bundle")
    out.mkdir(parents=True, exist_ok=True)
    part = out / "head.onnx"
    export_head(head.cpu(), part, width, frames, direct)
    subgraph = onnx.load(str(part))
    terminal = next(n for n in graph.graph.node if "log_probs" in n.output)
    terminal.output[list(terminal.output).index("log_probs")] = "source_log_probs"
    for value in graph.graph.output:
        if value.name == "log_probs":
            value.name = "source_log_probs"
    names = {"features": feature, "source_log_probs": "source_log_probs",
        "edit_source_previous": "edit_source_previous", "edit_valid_frames": "edit_valid_frames",
        "new_edit_source_previous": "new_edit_source_previous",
        "edit_actions": "edit_actions", "edit_phones": "edit_phones", "support_log_probs": "support_log_probs",
        "direct_log_probs": "direct_log_probs"}
    def name(old):
        return names.get(old, "acoustic_edit/" + old) if old else old
    for value in subgraph.graph.initializer:
        value.name = name(value.name)
    for node in subgraph.graph.node:
        node.name = "acoustic_edit/" + node.name
        for i, old in enumerate(node.input):
            node.input[i] = name(old)
        for i, old in enumerate(node.output):
            node.output[i] = name(old)
    graph.graph.node.extend(subgraph.graph.node)
    graph.graph.initializer.extend(subgraph.graph.initializer)
    graph.graph.input.append(helper.make_tensor_value_info("edit_source_previous", TensorProto.INT64, [1]))
    graph.graph.input.append(helper.make_tensor_value_info("edit_valid_frames", TensorProto.INT64, [1]))
    for value in subgraph.graph.output:
        value.name = name(value.name)
        graph.graph.output.append(value)
    primary = "direct_log_probs" if direct else "support_log_probs"
    graph.graph.node.append(helper.make_node("Identity", [primary], ["log_probs"], name="acoustic_edit/primary"))
    graph.graph.output.append(helper.make_tensor_value_info("log_probs", TensorProto.FLOAT,
        [1, frames, head.embedding.num_embeddings]))
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(graph)
    path = out / "model.onnx"
    onnx.save(graph, str(path))
    part.unlink()
    for file, checksum in auxiliary.items():
        target = out / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / file, target)
        if digest(target) != checksum:
            raise ValueError("Unchanged verification asset failed copy integrity")
    if insertion_assets(out) != auxiliary:
        raise ValueError("Incomplete insertion verification bundle")
    new = copy.deepcopy(m)
    new.update(id=m["id"] + ("-direct-ctc-control" if direct else "-acoustic-edit"),
               file=path.name, bytes=path.stat().st_size, sha256=digest(path),
               format="onnx-int8-encoder-fp32-edit")
    new["states"].append({"name": "edit_source_previous", "shape": [1], "dtype": "int64"})
    new["editFrameCountInput"] = "edit_valid_frames"
    if direct:
        new.pop("editDecoder", None)
        new["supportOutput"] = "support_log_probs"
    else:
        new["editDecoder"] = {"kind": "acoustic-edit", "actions": "edit_actions",
                              "phones": "edit_phones", "source": "source_log_probs"}
    (out / "manifest.json").write_text(json.dumps(new, ensure_ascii=False, indent=1))
    return path
