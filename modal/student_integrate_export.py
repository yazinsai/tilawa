"""Export a selected CIF/CTC checkpoint and verify its complete streaming bundle.

The CIF graph exports raw mass and affine phone logits alongside the independent CTC
head. No text or phone-count input exists. This command runs directly in the train image.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import student_export as E
import student_train as T
from student_integrate import attach_heads, decode_events
from student_integrate_assets import copy_auxiliaries


def verify(model, path, specs, chunk, left):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wrapper = E.wrap(model, chunk, left)
    torch_states = wrapper.init_states()
    onnx_states = {s["name"]: t.numpy().copy() for s, t in zip(specs, torch_states)}
    names = [o.name for o in session.get_outputs()]
    rng = np.random.default_rng(71)
    worst = dict(log_probs=0.0, phone_mass=0.0, phone_scores=0.0)
    torch_mass, onnx_mass, torch_scores, onnx_scores = [], [], [], []
    for i in range(24):
        x = (rng.normal(-3, 2, (1, chunk * 2 + E.PAD, 80)).astype(np.float32)
             if i % 4 else np.full((1, chunk * 2 + E.PAD, 80), E.LOG_EPS, np.float32))
        with torch.no_grad():
            expected = wrapper(torch.from_numpy(x), torch_states)
        torch_states = expected[-1]
        actual = dict(zip(names, session.run(None, dict(x=x, **onnx_states))))
        if any(not np.isfinite(value).all() for value in actual.values()):
            raise RuntimeError("Export produced a nonfinite acoustic output or recurrent state")
        onnx_states = {s["name"]: actual["new_" + s["name"]] for s in specs}
        for name, tensor in zip(["log_probs", *(["phone_mass", "phone_scores"] if wrapper.event_heads else [])], expected[:-1]):
            diff = float(np.abs(tensor.numpy() - actual[name]).max())
            worst[name] = max(worst[name], diff)
        if wrapper.event_heads:
            torch_mass.append(expected[1][0])
            onnx_mass.append(torch.from_numpy(actual["phone_mass"][0]))
            torch_scores.append(expected[2][0])
            onnx_scores.append(torch.from_numpy(actual["phone_scores"][0]))
    if max(worst.values()) > 1e-3:
        raise RuntimeError(f"Streaming ONNX parity failed: {worst}")
    events_equal = None
    if wrapper.event_heads:
        a = decode_events(torch.cat(torch_mass), torch.cat(torch_scores))
        b = decode_events(torch.cat(onnx_mass), torch.cat(onnx_scores))
        events_equal = [e["id"] for e in a] == [e["id"] for e in b]
        if not events_equal:
            raise RuntimeError("Export changed the raw-mass event phone sequence")
    return dict(windows=24, worst_absolute_difference=worst, event_tokens_equal=events_equal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-bundle", required=True, help="Shipped model bundle whose installed auxiliaries must be preserved")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--left", type=int, default=128)
    args = ap.parse_args()
    torch.set_num_threads(8)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = attach_heads(T.build_model(checkpoint["cfg"]["size"]))
    model.load_state_dict(checkpoint["model"])
    arm = checkpoint["cfg"]["arm"]
    if arm == "cif":
        model.export_events = True
    elif arm == "ctc":
        # The matched control's identity head was trained with framewise CTC, not pooling.
        model.ctc_output = torch.nn.Sequential(torch.nn.Identity(), model.phone_output)
    else:
        raise ValueError(arm)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Export requires a fresh bundle directory")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    states = E.export_onnx(model, str(path), args.chunk, args.left)
    parity = verify(model, path, states, args.chunk, args.left)
    parity["preserved_auxiliaries"] = copy_auxiliaries(args.base_bundle, out)
    manifest = E.manifest(out.name, states, args.chunk, args.left, str(path), None)
    if arm == "cif":
        manifest["decoder"] = dict(kind="cif", mass="phone_mass", scores="phone_scores")
    manifest["trainStep"] = checkpoint["step"]
    (out / "parity.json").write_text(json.dumps(parity, indent=1))
    # Written last, only after every graph output and recurrent state passed verification.
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(dict(arm=arm, manifest=manifest, parity=parity)), flush=True)


if __name__ == "__main__":
    main()
