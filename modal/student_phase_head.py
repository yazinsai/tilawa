"""Frozen exported-encoder head refit with a matched two-origin CTC control.

Run cache/train/export on the authorized Modal worker, never on practice audio.
No training checkpoint is needed: expose the final encoder activation in the
shipped streaming graph, preserving its quantized encoder and recurrent states.

    python modal/student_phase_head.py cache --model MODEL --help-train HELP --out CACHE
    python modal/student_phase_head.py train --cache CACHE --out FIT
    python modal/student_phase_head.py export --model MODEL --fit FIT --out BUNDLE --arm consistent

All runs use CPU. Cache provenance and the speaker split are frozen before the
two arms train. Neither checkpoint selection nor this module reads practice labels.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student_phase_consistency import (authorized_none, origins, output_frames,
                                       paired_loss, speaker_partition)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def readout(graph, vocabulary):
    """Find the last linear readout, including ORT's dynamic-int8 representation."""
    import numpy as np
    from onnx import numpy_helper

    producers = {o: n for n in graph.node for o in n.output}
    arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    output = graph.output[0].name
    softmax = producers[output]
    if softmax.op_type != "LogSoftmax":
        raise ValueError("primary output must end in the shipped LogSoftmax")
    add = producers[softmax.input[0]]
    if add.op_type != "Add":
        raise ValueError("expected a final linear readout bias")
    biases = [i for i in add.input if i in arrays and arrays[i].shape == (vocabulary,)]
    if len(biases) != 1:
        raise ValueError("ambiguous final head bias")
    bias = arrays[biases[0]].copy()
    stack = [i for i in add.input if i != biases[0]]
    seen, matches = set(), []
    while stack:
        name = stack.pop()
        if name in seen or name not in producers:
            continue
        seen.add(name)
        node = producers[name]
        if node.op_type in ("MatMulInteger", "MatMul") and node.input[1] in arrays:
            weight = arrays[node.input[1]]
            if weight.ndim == 2 and weight.shape[1] == vocabulary:
                matches.append(node)
                continue  # Do not traverse into the frozen encoder.
        stack.extend(node.input)
    if len(matches) != 1:
        raise ValueError("cannot identify a unique exported linear phone head")
    linear = matches[0]
    weight = arrays[linear.input[1]].copy()
    feature = linear.input[0]
    if linear.op_type == "MatMulInteger":
        quantizer = producers[feature]
        if quantizer.op_type != "DynamicQuantizeLinear":
            raise ValueError("unsupported head activation quantization")
        feature = quantizer.input[0]
        zero = arrays[linear.input[3]] if len(linear.input) > 3 else 0
        scale_name = linear.input[1].removesuffix("_quantized") + "_scale"
        if scale_name not in arrays:
            raise ValueError("missing head weight quantization scale")
        weight = (weight.astype(np.float32) - zero) * arrays[scale_name]
    return feature, weight.astype(np.float32).T.copy(), bias.astype(np.float32), softmax


class Encoder:
    def __init__(self, directory, threads=2):
        import numpy as np
        import onnx
        import onnxruntime as ort

        self.manifest = json.loads((Path(directory) / "manifest.json").read_text())
        self.path = Path(directory) / self.manifest["file"]
        if digest(self.path) != self.manifest["sha256"]:
            raise ValueError("source model checksum mismatch")
        self.graph = onnx.load(self.path)
        feature, self.weight, self.bias, _ = readout(self.graph.graph, len(self.manifest["tokens"]) + 1)
        self.feature = feature
        self.graph.graph.output.append(onnx.helper.make_tensor_value_info(
            feature, onnx.TensorProto.FLOAT, [1, None, self.weight.shape[1]]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(self.graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
        self.states = self.manifest["states"]
        self.np = np
        origins(self.manifest)

    def stream(self, features, phase):
        """Original feature rows at identical 40 ms acoustic times in either view."""
        np, m = self.np, self.manifest
        if phase not in origins(m):
            raise ValueError("origin is not one of the frozen legal grids")
        count = output_frames(len(features))
        if count == 0:
            raise ValueError("recording is too short")
        f = np.concatenate((np.full((phase, 80), m["fbank"]["padValue"], np.float32), features))
        need = output_frames(len(f))
        windows = (need + m["chunk"] // 2 - 1) // (m["chunk"] // 2)
        total = m["window"] + (windows - 1) * m["shift"]
        f = np.concatenate((f, np.full((max(0, total - len(f)), 80), m["fbank"]["padValue"], np.float32)))
        state = {s["name"]: np.zeros(s["shape"], np.int64 if s["dtype"] == "int64" else np.float32)
                 for s in self.states}
        lp, hidden = [], []
        for start in range(0, total - m["window"] + 1, m["shift"]):
            result = self.session.run(None, {"x": f[None, start:start + m["window"]], **state})
            lp.append(result[0][0])
            hidden.append(result[-1][0])
            state = {s["name"]: value for s, value in zip(self.states, result[1:-1])}
        offset = phase // 4
        return dict(features=np.concatenate(hidden)[offset:offset + count],
                    log_probs=np.concatenate(lp)[offset:offset + count],
                    times_ms=np.arange(count, dtype=np.int64) * m["frameMs"], phase=phase)


def fbank(path):
    import subprocess
    import numpy as np
    import torch
    import torchaudio

    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                             "-f", "f32le", "pipe:1"], capture_output=True, check=True)
    wave = torch.from_numpy(np.frombuffer(result.stdout, np.float32).copy())
    return torchaudio.compliance.kaldi.fbank(wave[None], num_mel_bins=80, sample_frequency=16000,
                                            dither=0., energy_floor=1e-10, high_freq=-400).numpy()


def training_rows(help_root, references, corpus_index=None):
    """Only supplied train-speaker none takes; do not infer transcripts of errors."""
    from student import phoneme_ids

    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in references["ayahs"]}
    labels_path = help_root / "labels.jsonl"
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    # A speaker with any eval/held-out record is forbidden, even if another take
    # incorrectly has a train tag. The supplied volume already excludes release speakers.
    forbidden = {r["speaker"] for r in labels if r.get("split") != "train"}
    corpus = []
    if corpus_index:
        corpus_index = Path(corpus_index)
        corpus = [json.loads(line) for line in corpus_index.read_text().splitlines() if line]
        forbidden |= {r["speaker"] for r in corpus if r.get("split") != "train"}
    rows = []
    for label in labels:
        if not authorized_none(label) or label["speaker"] in forbidden:
            continue
        path = (help_root / label["wav"]).resolve()
        if not path.is_relative_to(help_root.resolve()):
            raise ValueError("training recording escapes the authorized root")
        # Reference runs encode held sounds, not separate CTC emissions. Use the
        # same run collapse as student.spliced_target, independently per ayah so
        # a real same-phone repetition across a pause remains two occurrences.
        phones = "".join("".join(c for i, c in enumerate(ayahs[a]) if i == 0 or c != ayahs[a][i - 1])
                         for a in label["ayahs"])
        rows.append(dict(id=label["id"], speaker=label["speaker"], path=path,
                         target=phoneme_ids(phones)))
    for row in corpus:
        # The caller supplies a training-only catalog of performed transcripts,
        # not canonical prompts or a mined practice-set catalog.
        if (row.get("split") != "train" or row["speaker"] in forbidden
                or row.get("excluded") or row.get("authorization") != "transcribed-training"
                or not row.get("transcript")):
            continue
        path = (corpus_index.parent / row["audio"]).resolve()
        if not path.is_relative_to(corpus_index.parent.resolve()):
            raise ValueError("corpus recording escapes the authorized catalog root")
        rows.append(dict(id=row["id"], speaker=row["speaker"], path=path,
                         target=phoneme_ids(row["transcript"])))
    if not rows:
        raise ValueError("no authorized none-labelled training recordings")
    return rows


def cache(args):
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    encoder = Encoder(args.model, args.threads)
    started = time.time()
    rows = training_rows(Path(args.help_train), json.loads(Path(args.reference).read_text()), args.corpus_index)
    split = speaker_partition([r["speaker"] for r in rows], args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    index = []
    # Full original recordings retain speech and real pause frames. No new
    # augmentation is introduced and both arms use exactly the same cached views.
    from student_phase_consistency import occurrence_weights
    for i, row in enumerate(rows):
        if time.time() >= started + min(args.hours, 24) * 3600:
            raise TimeoutError("one-day cache/refit budget exhausted")
        features = fbank(row["path"])
        views = [encoder.stream(features, phase) for phase in origins(encoder.manifest)]
        for view in views:
            occurrence_weights(torch.from_numpy(view["log_probs"]), row["target"])
        np.testing.assert_array_equal(views[0]["times_ms"], views[1]["times_ms"])
        record = dict(target=row["target"], views=views)
        name = f"record-{i:05d}.pt"
        torch.save(record, out / name)
        index.append(dict(id=row["id"], speaker=row["speaker"], partition=split[row["speaker"]],
                          audio_sha256=digest(row["path"]), file=name, cache_sha256=digest(out / name),
                          frames=len(views[0]["features"])))
        print(json.dumps(dict(cached=i + 1, total=len(rows), frames=index[-1]["frames"])), flush=True)
    torch.save(dict(weight=encoder.weight, bias=encoder.bias), out / "initial.pt")
    provenance = dict(schema=1, source_sha256=encoder.manifest["sha256"], seed=args.seed, started_at=started,
                      origins=list(origins(encoder.manifest)), frame_ms=40, speakers=split,
                      labels_sha256=digest(Path(args.help_train) / "labels.jsonl"),
                      corpus_index_sha256=digest(args.corpus_index) if args.corpus_index else None,
                      source="help-train/train/none + authorized performed transcripts", augmentation="none",
                      transcript_policy="canonical within-ayah run collapse; corpus performed CTC transcripts", recordings=index)
    (out / "index.json").write_text(json.dumps(provenance, indent=2))


def train(args):
    import copy
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    root, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((root / "index.json").read_text())
    initial = torch.load(root / "initial.pt", weights_only=False)
    weight = torch.from_numpy(initial["weight"])
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(torch.from_numpy(initial["bias"]))
    heads = {arm: copy.deepcopy(head) for arm in ("control", "consistent")}
    opts = {arm: torch.optim.Adam(h.parameters(), lr=args.lr) for arm, h in heads.items()}
    groups = {part: [r for r in index["recordings"] if r["partition"] == part]
              for part in ("train", "validation")}
    if not all(groups.values()):
        raise ValueError("both speaker partitions must contain recordings")
    records = {}
    for row in index["recordings"]:
        if digest(root / row["file"]) != row["cache_sha256"]:
            raise ValueError("cached recording checksum mismatch")
        records[row["file"]] = torch.load(root / row["file"], weights_only=False)
    for record in records.values():
        for view in record["views"]:
            view["features"] = torch.from_numpy(view["features"])
            view["log_probs"] = torch.from_numpy(view["log_probs"])
    best, history, updates = float("inf"), [], 0
    deadline = index["started_at"] + min(args.hours, 24) * 3600
    if time.time() >= deadline:
        raise TimeoutError("one-day cache/refit budget exhausted")
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        for i in rng.permutation(len(groups["train"])):
            if time.time() >= deadline:
                break
            record = records[groups["train"][i]["file"]]
            for arm, h in heads.items():
                opts[arm].zero_grad()
                loss = paired_loss(h, record["views"], record["target"], int(arm == "consistent"))["loss"]
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite recording loss; matched run aborted")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(h.parameters(), 5.)
                opts[arm].step()
            updates += 1
        values = {}
        with torch.no_grad():
            for arm, h in heads.items():
                # Same validation criterion for both arms: CTC plus preservation.
                # Choose one common update count by mean across arms.
                values[arm] = sum(paired_loss(h, records[r["file"]]["views"], records[r["file"]]["target"], 0)["loss"].item()
                                  for r in groups["validation"]) / len(groups["validation"])
        mean = sum(values.values()) / len(values)
        history.append(dict(epoch=epoch + 1, updates=updates, validation=values, mean=mean))
        print(json.dumps(history[-1]), flush=True)
        if mean < best:
            best = mean
            for arm, h in heads.items():
                torch.save(dict(head=h.state_dict(), source_sha256=index["source_sha256"],
                                selected_updates=updates, selected_epoch=epoch + 1,
                                arm=arm, cache_sha256=digest(root / "index.json")), out / f"{arm}.pt")
        if time.time() >= deadline:
            break
    (out / "training.json").write_text(json.dumps(dict(history=history, args=vars(args),
                                                      selection="common epoch: mean validation CTC + frozen-output KL"), indent=2))


def export(args):
    import shutil
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    source, out = Path(args.model), Path(args.out)
    encoder = Encoder(source, args.threads)
    fit = torch.load(Path(args.fit) / f"{args.arm}.pt", weights_only=False)
    if fit["source_sha256"] != encoder.manifest["sha256"]:
        raise ValueError("refit and export source models differ")
    for name in args.required_assets:
        if not (source / name).is_file():
            raise ValueError(f"missing required unchanged SDK asset: {name}")
    out.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name not in (encoder.path.name, "manifest.json"):
            shutil.copyfile(path, out / path.name)
            if digest(path) != digest(out / path.name):
                raise ValueError(f"copied SDK asset checksum mismatch: {path.name}")
    graph = onnx.load(encoder.path)
    feature, _, _, old_softmax = readout(graph.graph, len(encoder.manifest["tokens"]) + 1)
    old_softmax.output[0] = "phase_original_log_probs"
    weight = fit["head"]["weight"].numpy().T.copy()
    bias = fit["head"]["bias"].numpy().copy()
    graph.graph.initializer.extend([onnx.numpy_helper.from_array(weight, "phase_head_weight"),
                                    onnx.numpy_helper.from_array(bias, "phase_head_bias")])
    graph.graph.node.extend([
        onnx.helper.make_node("MatMul", [feature, "phase_head_weight"], ["phase_head_linear"]),
        onnx.helper.make_node("Add", ["phase_head_linear", "phase_head_bias"], ["phase_head_logits"]),
        onnx.helper.make_node("LogSoftmax", ["phase_head_logits"], [graph.graph.output[0].name], axis=-1),
    ])
    onnx.checker.check_model(graph)
    onnx.save(graph, out / "model.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(out / "model.onnx"), options, providers=["CPUExecutionProvider"])
    # Forward parity includes state carries across multiple legal streaming windows.
    rng = np.random.default_rng(0)
    states = {s["name"]: np.zeros(s["shape"], np.int64 if s["dtype"] == "int64" else np.float32)
              for s in encoder.states}
    maximum = 0.
    for _ in range(4):
        x = rng.normal(size=(1, encoder.manifest["window"], 80)).astype(np.float32)
        old = encoder.session.run(None, {"x": x, **states})
        new = session.run(None, {"x": x, **states})
        expected = torch.nn.functional.linear(torch.from_numpy(old[-1]),
                                               fit["head"]["weight"], fit["head"]["bias"]).log_softmax(-1).numpy()
        np.testing.assert_allclose(new[0], expected, rtol=1e-5, atol=1e-4)
        maximum = max(maximum, float(np.abs(new[0] - expected).max()))
        for before, after in zip(old[1:-1], new[1:]):
            np.testing.assert_array_equal(before, after)
        states = {s["name"]: value for s, value in zip(encoder.states, new[1:])}
    m = {**encoder.manifest, "id": f'phase-head-{args.arm}', "file": "model.onnx", "sha256": digest(out / "model.onnx"),
         "bytes": (out / "model.onnx").stat().st_size,
         "headRefit": {k: fit[k] for k in ("source_sha256", "selected_updates", "selected_epoch", "arm", "cache_sha256")}}
    (out / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    assets = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in out.iterdir() if p.is_file()}
    (out / "verification.json").write_text(json.dumps(dict(max_head_parity_error=maximum,
                                                          recurrent_states="exact", files=assets), indent=2))
    print(json.dumps(dict(bundle=str(out), verification="passed", files=list(assets))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("cache")
    c.add_argument("--model", required=True)
    c.add_argument("--help-train", required=True)
    c.add_argument("--reference", default="packages/core/assets/ref-v2.json")
    c.add_argument("--corpus-index", help="authorized JSONL: speaker, split, audio, transcript, authorization")
    c.add_argument("--hours", type=float, default=20)
    c.add_argument("--out", required=True)
    t = commands.add_parser("train")
    t.add_argument("--cache", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--hours", type=float, default=20)
    t.add_argument("--lr", type=float, default=1e-4)
    e = commands.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--fit", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--arm", choices=("control", "consistent"), default="consistent")
    e.add_argument("--required-assets", nargs="*", default=["teacher.onnx", "teacher.onnx.data"])
    for command in (c, t, e):
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or (hasattr(args, "hours") and not 0 < args.hours <= 24):
        parser.error("threads must be positive and cache/refit budget must be within 24 hours")
    if args.command == "train" and (args.epochs < 1 or args.lr <= 0):
        parser.error("epochs and learning rate must be positive")
    {"cache": cache, "train": train, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
