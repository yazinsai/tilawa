"""Unchanged-weight robust BiasNorm graphs and training-only calibration.

The Huber M-scale solves E[min((x-bias)^2 / scale^2, c^2)] = b(c).
The original activation, learned bias and learned output gain are retained.
Graph construction requires no source checkpoint or dequantization.
"""
import copy
import hashlib
import math

import numpy as np


RECIPE = "huber-biasnorm-m-scale-v1"
CANDIDATES = (1.5, 2.0, 3.0)
ITERATIONS = 12


def normal_consistency(c):
    if not math.isfinite(c) or c <= 1:
        raise ValueError("Huber scale cutoff must exceed one")
    tail = math.erfc(c / math.sqrt(2)) / 2
    density = math.exp(-c * c / 2) / math.sqrt(2 * math.pi)
    return 1 - 2 * c * density + 2 * (c * c - 1) * tail


def m_energy(energy, axis, c):
    """Reference M-scale squared; the floor protects silent/degenerate vectors."""
    energy = np.asarray(energy)
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError("Expected finite squared channel activations")
    scale = np.maximum(np.median(energy, axis=axis, keepdims=True) / 0.4549364231, 1e-12)
    for _ in range(ITERATIONS):
        scale = np.maximum(np.mean(np.minimum(energy, c * c * scale), axis=axis, keepdims=True)
                           / normal_consistency(c), 1e-12)
    return scale


def training_partition(record):
    """No held-out, excluded or mistake-labelled recording can enter calibration."""
    if (record.get("split") != "train" or record.get("kind") != "none" or
            record.get("exclude") or record.get("excluded") or record.get("unsupported") or
            record.get("events") or record.get("non_errors") or
            record.get("held_out") or record.get("scored") is False or not record.get("speaker")):
        return None
    # Every recording from one speaker is assigned together, independently of text.
    digest = hashlib.sha256(str(record["speaker"]).encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 4 == 0 else "fit"


def calibration(records, cutoff):
    """Equal recording weight; validation observations never determine the gains.

    Each admitted record supplies site -> squared-channel-activation arrays from
    the unchanged shipped stream. Scale restoration is geometric mean RMS ratio.
    Concentration envelopes use the maximum per recording, then the maximum
    across training speakers. Target inspection does not change either quantity.
    """
    ratios, envelope, fit_speakers, validation_speakers = {}, {}, set(), set()
    for record in records:
        partition = training_partition(record)
        if partition is None:
            continue
        (fit_speakers if partition == "fit" else validation_speakers).add(record["speaker"])
        if partition != "fit":
            continue
        for site, observation in record["sites"].items():
            energy, axis = observation["energy"], observation["axis"]
            energy = np.asarray(energy)
            ordinary = np.mean(energy, axis=axis, keepdims=True)
            robust = m_energy(energy, axis, cutoff)
            active = ordinary > 1e-12
            if not active.any():
                continue
            ratios.setdefault(site, []).append(float(np.log(robust[active] / ordinary[active]).mean() / 2))
            share = np.max(energy, axis=axis, keepdims=True) / np.maximum(
                np.sum(energy, axis=axis, keepdims=True), 1e-12)
            envelope[site] = max(envelope.get(site, 0), float(share[active].max()))
    if not fit_speakers or not validation_speakers or not ratios:
        raise ValueError("Calibration requires disjoint fit and validation speakers")
    return {"recipe": RECIPE, "cutoff": cutoff, "iterations": ITERATIONS,
            "gains": {site: math.exp(float(np.mean(values))) for site, values in ratios.items()},
            "concentrationEnvelope": envelope,
            "fitSpeakers": sorted(fit_speakers), "validationSpeakers": sorted(validation_speakers)}


def sites(model):
    """Find only biased squared-energy means that scale their original input."""
    import onnx
    from onnx import helper, numpy_helper

    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None
                      for d in v.type.tensor_type.shape.dim]
              for v in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
    producers = {output: n for n in model.graph.node for output in n.output}
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    constants[n.output[0]] = numpy_helper.to_array(a.t)
        elif n.op_type == "Identity" and n.input[0] in constants:
            constants[n.output[0]] = constants[n.input[0]]
    found = []
    for mean in model.graph.node:
        if mean.op_type != "ReduceMean":
            continue
        square = producers.get(mean.input[0])
        if square is None or not (square.op_type == "Pow" and
                np.array_equal(constants.get(square.input[1]), np.asarray(2)) or
                square.op_type == "Mul" and square.input[0] == square.input[1]):
            continue
        biased = producers.get(square.input[0])
        if biased is None or biased.op_type != "Sub" or biased.input[1] not in constants:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in mean.attribute}
        axes = attrs.get("axes", constants.get(mean.input[1]) if len(mean.input) > 1 else None)
        shape = shapes.get(mean.input[0])
        if shape is None or axes is None or len(axes) != 1 or attrs.get("keepdims", 1) != 1:
            raise ValueError(f"Unsupported biased energy mean: {mean.name}")
        axis = int(axes[0]) % len(shape)
        channels = shape[axis]
        if not channels or channels < 2:
            raise ValueError(f"Unknown BiasNorm channel count: {mean.name}")
        reachable = {mean.output[0]}
        terminal = None
        for _ in range(5):
            for n in model.graph.node:
                if n.op_type == "Mul" and biased.input[0] in n.input and any(x in reachable for x in n.input):
                    terminal = n
                    break
            if terminal is not None:
                break
            reachable.update(output for n in model.graph.node if n.op_type in {"Pow", "Sqrt", "Div", "Mul"}
                             and any(x in reachable for x in n.input) for output in n.output)
        if terminal is None:
            continue
        found.append({"name": mean.name, "mean": mean, "energy": mean.input[0],
                      "axis": axis, "channels": channels, "terminal": terminal})
    if not found:
        raise ValueError("No supported instantaneous BiasNorm sites found")
    return found


def robust_nodes(site, cutoff):
    from onnx import helper, numpy_helper

    prefix = site["name"] + "/robust"
    nodes, initializers = [], []

    def const(label, value, dtype=np.float32):
        name = prefix + "/" + label
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, inputs, label, **attributes):
        name = prefix + "/" + label
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attributes))
        return name

    axis, channels = site["axis"], site["channels"]
    values, indices = prefix + "/ordered", prefix + "/indices"
    half = channels // 2
    nodes.append(helper.make_node("TopK", [site["energy"], const("k", [half + 1], np.int64)],
                                  [values, indices], axis=axis, largest=0, sorted=1, name=values))
    middle = op("Gather", [values, const("middle", [half-1, half] if channels % 2 == 0 else [half], np.int64)],
                "middle_values", axis=axis)
    median = op("ReduceMean", [middle], "median", axes=[axis], keepdims=1)
    floor = const("floor", 1e-12)
    scale = op("Max", [op("Div", [median, const("median_consistency", 0.4549364231)], "initial"), floor], "scale0")
    squared_cut = const("squared_cut", cutoff * cutoff)
    consistency = const("consistency", normal_consistency(cutoff))
    for i in range(ITERATIONS):
        bound = op("Mul", [scale, squared_cut], f"bound{i}")
        clipped = op("Min", [site["energy"], bound], f"clipped{i}")
        mean = op("ReduceMean", [clipped], f"mean{i}", axes=[axis], keepdims=1)
        scale = op("Max", [op("Div", [mean, consistency], f"update{i}"), floor], f"scale{i+1}")
    nodes.append(helper.make_node("Identity", [scale], [site["mean"].output[0]], name=prefix + "/result"))
    return nodes, initializers


def transform(model, calibration_result, gain_only=False):
    """Keep all original weights and state declarations, including learned gain."""
    import onnx
    from onnx import helper, numpy_helper

    model = copy.deepcopy(model)
    selected = sites(model)
    gains = calibration_result["gains"]
    if calibration_result["recipe"] != RECIPE or set(gains) != {s["name"] for s in selected}:
        raise ValueError("Calibration does not cover exactly this graph's BiasNorm sites")
    replacements, after = {}, {}
    for s in selected:
        if not gain_only:
            added, constants = robust_nodes(s, calibration_result["cutoff"])
            replacements[s["name"]] = added
            model.graph.initializer.extend(constants)
        gain = gains[s["name"]]
        if not math.isfinite(gain) or gain <= 0:
            raise ValueError("Invalid restoration scale")
        terminal = s["terminal"]
        original = terminal.output[0]
        terminal.output[0] = original + "/before_restoration"
        constant = original + "/restoration_gain"
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(gain, dtype=np.float32), constant))
        after[terminal.name] = helper.make_node("Mul", [terminal.output[0], constant], [original], name=original + "/restore")
    nodes = []
    for n in model.graph.node:
        nodes.extend(replacements.get(n.name, [n]))
        if n.name in after:
            nodes.append(after[n.name])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    return model


def instrument(model):
    """Expose squared channel activations for training calibration and diagnosis."""
    import onnx
    from onnx import helper
    graph = copy.deepcopy(model)
    for s in sites(graph):
        graph.graph.output.append(helper.make_tensor_value_info(s["energy"], onnx.TensorProto.FLOAT, None))
    return onnx.shape_inference.infer_shapes(graph)


def stream(session, manifest, features):
    """Exact shipped chunk geometry, including warmed independent caches."""
    specs = manifest["states"]
    state = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in specs}
    window, shift = manifest["window"], manifest["shift"]
    frames = ((len(features) - 7) // 2 + 1) // 2
    if frames <= 0:
        raise ValueError("Recording is too short for the listener")
    count = math.ceil(frames / (shift // 4))
    total = window + (count - 1) * shift
    features = np.pad(features, ((0, max(0, total - len(features))), (0, 0)),
                      constant_values=manifest["fbank"]["padValue"])
    names = [s.name for s in session.get_outputs()]
    rows = {name: [] for name in names}
    for start in range(0, total - window + 1, shift):
        output = dict(zip(names, session.run(None, {"x": features[None, start:start+window], **state})))
        for name in names:
            rows[name].append(output[name])
        state = {s["name"]: output["new_" + s["name"]] for s in specs}
    rows["log_probs"] = np.concatenate(rows["log_probs"], axis=1)[0, :frames]
    return rows


def ctc_nll(log_probs, labels):
    """Full-label CTC acceptance; used only on training-speaker validation."""
    seq = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    seq[1::2] = labels
    previous = np.full(len(seq), -np.inf)
    previous[0] = 0
    skip = np.zeros(len(seq), dtype=bool)
    skip[2:] = (seq[2:] != 0) & (seq[2:] != seq[:-2])
    for row in log_probs:
        moved = np.r_[-np.inf, previous[:-1]]
        jumped = np.r_[-np.inf, -np.inf, previous[:-2]]
        previous = np.logaddexp(previous, moved)
        previous = np.logaddexp(previous, np.where(skip, jumped, -np.inf)) + row[seq]
    return -float(np.logaddexp(previous[-1], previous[-2]))


def build_bundle(source, destination, records):
    """Fit/select exclusively within help-train training speakers, then export.

    Records are labels enriched with canonical phone IDs and shipped fbank
    features; no mistake script is converted into a performed transcript.
    The source bundle (including insertion teacher) is copied byte-for-byte.
    """
    import json
    import shutil
    from pathlib import Path
    import onnx
    import onnxruntime as ort

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Export requires a new, separate destination")
    manifest = json.loads((source / "manifest.json").read_text())
    path = source / manifest["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Source graph checksum differs from its manifest")
    # The folded insertion ear is part of the primary bundle, never optional here.
    for name in ("teacher.onnx", "teacher.onnx.data"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Missing existing insertion asset: {name}")
    primary = onnx.load(path)
    geometry = sites(primary)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    diagnostic = ort.InferenceSession(instrument(primary).SerializeToString(), options)
    records = list(records)
    held_out = {r.get("speaker") for r in records if r.get("split") != "train" or r.get("held_out")}
    admitted = []
    for record in records:
        if record.get("speaker") in held_out or training_partition(record) is None:
            continue
        if not record.get("phoneIds") or "features" not in record:
            raise ValueError("None-labelled calibration needs canonical phones and original features")
        features = np.asarray(record["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != manifest["fbank"]["mels"] or not np.isfinite(features).all():
            raise ValueError("Calibration requires finite shipped fbank features")
        if any(not isinstance(p, (int, np.integer)) or not 0 < p <= len(manifest["tokens"])
               for p in record["phoneIds"]):
            raise ValueError("Invalid canonical phone IDs")
        if record.get("inventory", manifest["inventory"]) != manifest["inventory"] or \
                record.get("tokens", manifest["tokens"]) != manifest["tokens"]:
            raise ValueError("Calibration pronunciation inventory differs from the model")
        observed = stream(diagnostic, manifest, features)
        record = {**record, "features": features, "sites": {s["name"]: {
            "axis": s["axis"], "energy": np.concatenate(observed[s["energy"]], axis=0)} for s in geometry}}
        admitted.append(record)
    trials = []
    for c in CANDIDATES:
        fitted = calibration(admitted, c)
        arms = {name: ort.InferenceSession(transform(primary, fitted, gain_only=control).SerializeToString(), options)
                for name, control in (("robust", False), ("gain", True))}
        losses = {name: [] for name in arms}
        for record in admitted:
            if training_partition(record) != "validation":
                continue
            for name, session in arms.items():
                lp = stream(session, manifest, record["features"])["log_probs"]
                loss = ctc_nll(lp, record["phoneIds"]) / len(record["phoneIds"])
                if not math.isfinite(loss):
                    raise ValueError("Nonfinite training-validation sequence score")
                losses[name].append(loss)
        trials.append((float(np.mean(losses["robust"])), fitted,
                       {name: float(np.mean(values)) for name, values in losses.items()}))
    _, selected, _ = min(trials, key=lambda trial: (trial[0], trial[1]["cutoff"]))
    shutil.copytree(source, destination)
    declaration = {**selected, "sourceSha256": digest,
                   "validationTrials": [{"cutoff": fit["cutoff"], **scores} for _, fit, scores in trials]}
    for name, control in (("robust-norm", False), ("robust-norm-gain", True)):
        out = destination / (name + ".onnx")
        onnx.save(transform(primary, selected, gain_only=control), out)
        declaration["gainSha256" if control else "sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (destination / "robust-norm.json").write_text(json.dumps(declaration, indent=2) + "\n")
    return declaration


def learner_records(labels_path, reference_path):
    """Read the admitted original recordings; derive no labels from scripts."""
    import json
    import subprocess
    from pathlib import Path
    import torch
    import torchaudio

    labels_path = Path(labels_path)
    reference = json.loads(Path(reference_path).read_text())
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in reference["ayahs"]}
    ids = {phone: i + 1 for i, phone in enumerate(reference["inventory"]["tokens"])}
    labels = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    held_out = {r.get("speaker") for r in labels if r.get("split") != "train" or r.get("held_out")}
    for record in labels:
        if record.get("speaker") in held_out or training_partition(record) is None:
            continue
        phones = "".join(ayahs[ayah] for ayah in record["ayahs"])
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(labels_path.parent / record["wav"]),
                              "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                             check=True, capture_output=True).stdout
        wave = torch.from_numpy(np.frombuffer(pcm, dtype=np.float32).copy())
        features = torchaudio.compliance.kaldi.fbank(wave.unsqueeze(0), num_mel_bins=80,
            sample_frequency=16000, dither=0.0, energy_floor=1e-10, high_freq=-400).numpy()
        yield {**record, "features": features, "phoneIds": [ids[p] for p in phones],
               "inventory": reference["inventory"]["id"], "tokens": reference["inventory"]["tokens"]}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Complete shipped model bundle")
    parser.add_argument("destination", help="New bundle directory")
    parser.add_argument("records", help="help-train labels JSONL, or prepared features when --reference is absent")
    parser.add_argument("--reference", help="Generic ref-v2.json asset for original help-train audio")
    args = parser.parse_args()
    if args.reference:
        records = learner_records(args.records, args.reference)
    else:
        records_path = Path(args.records)
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        for record in records:
            if training_partition(record) is not None:
                record["features"] = np.load(records_path.parent / record["featureFile"], allow_pickle=False)
    print(json.dumps(build_bundle(args.source, args.destination, records), indent=2))
