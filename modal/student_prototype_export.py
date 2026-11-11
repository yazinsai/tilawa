"""Export the training-selected prototype listener with the existing SDK ABI.

Selection is frozen in selection.json before this command. Export validation
uses training clips; practice windows must never choose a model or checkpoint.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import student_export as E
import student_train as T
from student_prototypes import PrototypeHead


def streaming_torch(model, feats, chunk=16, left=128):
    s = E.wrap(model.cpu(), chunk, left)
    states = s.init_states()
    frames = feats.shape[0]
    need = ((frames - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (chunk // 2)))
    total = chunk * 2 + E.PAD + (windows - 1) * chunk * 2
    feats = torch.cat([feats, torch.full((max(0, total - frames), 80), E.LOG_EPS)])
    result = []
    with torch.no_grad():
        for start in range(0, total - chunk * 2 - E.PAD + 1, chunk * 2):
            lp, states = s(feats[None, start:start + chunk * 2 + E.PAD], states)
            result.append(lp[0])
    return torch.cat(result).numpy()


def export_checkpoint(ckpt, out, validation):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = T.build_model(cfg["size"], cfg["phone_head"], cfg["prototypes"])
    model.load_state_dict(ck["model"])
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    fp = out / "float.onnx"
    states = E.export_onnx(copy.deepcopy(model), str(fp), 16, 128)
    quantize_dynamic(str(fp), str(out / "model.onnx"), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
    checks = []
    for name, features in validation:
        a = streaming_torch(copy.deepcopy(model), features)
        b = E.stream_onnx(str(fp), features, states, 16)
        c = E.stream_onnx(str(out / "model.onnx"), features, states, 16)
        if a.shape != b.shape or b.shape != c.shape:
            raise RuntimeError("export changed the frame clock or alphabet")
        error = float(np.abs(np.exp(a) - np.exp(b)).max())
        if error > .001:
            raise RuntimeError(f"fp32 streaming export parity failed: {error}")
        checks.append(dict(clip=name, frames=len(a), fp32_probability_error=error,
                           int8_probability_error=float(np.abs(np.exp(b) - np.exp(c)).max()),
                           int8_argmax_agreement=float((b.argmax(-1) == c.argmax(-1)).mean())))
    m = E.manifest(out.name, states, 16, 128, str(out / "model.onnx"), ck)
    m["format"] = "onnx-int8"
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    (out / "parity.json").write_text(json.dumps(checks, indent=2))
    assert hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest() == m["sha256"]
    fp.unlink()
    return m


def main(args):
    from student_prototype_train import packed, training_records

    E.onnx_deps()
    torch.set_num_threads(8)
    root = Path(args.run)
    selection = json.loads((root / "selection.json").read_text())
    rows, _ = training_records(Path(args.data), args.clips, args.seed)
    ds = T.ClipDataset(packed(rows[:8]), T.DEFAULTS, False, None)
    validation = [(r["key"], ds[i]["feats"]) for i, r in enumerate(rows[:8])]
    chosen = selection["selected"]
    m = export_checkpoint(chosen["path"], Path(args.out), validation)
    count = chosen["arm"].rsplit("-", 1)[1]
    control = next(r for r in selection["arms"] if r["arm"] == f"conventional-{count}")
    export_checkpoint(control["path"], Path(args.out).with_name("conventional-control"), validation)
    print(json.dumps(dict(bundle=str(args.out), sha256=m["sha256"], selected=chosen, control=control)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="/lab/data")
    ap.add_argument("--clips", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
