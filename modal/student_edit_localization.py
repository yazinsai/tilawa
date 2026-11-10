"""Frozen-encoder contextual edit localization, using training speakers only.

This utility consumes recording-level scripts as weak location/class labels, never
as performed phoneme transcripts. Acoustic word boundaries are marginalized under
the canonical CTC path. The contextual head competes across the recording during
training; its distilled sigmoid head has no one-edit constraint at inference.
The independent control receives exactly the same acoustic/text features.

Run in the supervisor's compute environment, not on a developer laptop:
  python3 modal/student_edit_localization.py --data "$LAB_DATA/help-train" \
    --model "$LAB_DATA/models/spliced-clean-int8" --out /scratch/<experiment>/fit
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def partition(speaker):
    """Stable speaker groups: fitting, model selection, preservation calibration."""
    fold = int(hashlib.sha256(str(speaker).encode()).hexdigest()[:8], 16) % 5
    return "calibration" if fold == 0 else "validation" if fold == 1 else "fit"


def ctc_occupancy(logp, labels):
    """Posterior occupancy of every canonical articulation over all CTC timings.

    Blank states and repeat-label separation are retained. Return None for an
    impossible canonical alignment; absence of evidence cannot authorize a veto.
    """
    from scipy.special import logsumexp

    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels) or len(logp) < len(labels):
        return None
    states = np.zeros(2 * len(labels) + 1, dtype=np.int64)
    states[1::2] = labels
    n, k = len(logp), len(states)
    skip = np.zeros(k, dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    a = np.full((n, k), -np.inf)
    a[0, :2] = logp[0, states[:2]]
    for t in range(1, n):
        prev = a[t - 1]
        one = np.r_[-np.inf, prev[:-1]]
        two = np.r_[-np.inf, -np.inf, prev[:-2]]
        a[t] = logsumexp(np.stack([prev, one, np.where(skip, two, -np.inf)]), axis=0) + logp[t, states]
    z = logsumexp(a[-1, -2:])
    if not np.isfinite(z):
        return None
    b = np.full((n, k), -np.inf)
    b[-1, -2:] = 0
    for t in range(n - 2, -1, -1):
        nxt = b[t + 1] + logp[t + 1, states]
        one = np.r_[nxt[1:], -np.inf]
        two = np.r_[np.where(skip[2:], nxt[2:], -np.inf), -np.inf, -np.inf]
        b[t] = logsumexp(np.stack([nxt, one, two]), axis=0)
    return np.exp(a[:, 1::2] + b[:, 1::2] - z)


def word_features(encoder, logp, pronunciations, tokens):
    """Identical-feature input for contextual and independent heads.

    Posterior occupancy pools frozen encoder means and deviations, acoustic phone
    mass and a text phone histogram. No accused-span backtrace is used.
    """
    ids = {p: i + 1 for i, p in enumerate(tokens)}
    encoded = [[ids[p] for p in s] for s in pronunciations]
    occupancy = ctc_occupancy(logp, [p for word in encoded for p in word])
    if occupancy is None:
        return None
    features, spans, offset = [], [], 0
    for word in encoded:
        if not word:
            return None
        weight = occupancy[:, offset:offset + len(word)].sum(1)
        offset += len(word)
        total = weight.sum()
        if total <= 0:
            return None
        weight /= total
        mean = weight @ encoder
        std = np.sqrt(np.maximum(weight @ (encoder ** 2) - mean ** 2, 0))
        acoustic = weight @ np.exp(logp)
        text = np.bincount(word, minlength=logp.shape[1]) / len(word)
        features.append(np.r_[mean, std, acoustic, text, np.log1p(total), np.log1p(len(word))])
        cumulative = weight.cumsum()
        spans.append([int(np.searchsorted(cumulative, .05)), int(np.searchsorted(cumulative, .95)) + 1])
    return np.asarray(features, dtype=np.float32), spans


def feature_graph(model_dir, out):
    """Expose the frozen pre-CTC encoder from a copy; never modify primary ONNX."""
    import onnx
    from onnx import TensorProto, helper

    manifest = json.loads((model_dir / "manifest.json").read_text())
    graph = onnx.load(str(model_dir / manifest["file"]))
    # Trace backward from log-softmax through the CTC projection. The selected
    # tensor is the projection input, not logits or a state cache.
    producer = {name: node for node in graph.graph.node for name in node.output}
    constants = {v.name: v for v in graph.graph.initializer}
    # Quantized projection outputs pass through scale multiplications with
    # multiple parents. Find the nearest projection whose constant weight has
    # the CTC vocabulary dimension, rather than following an arbitrary parent.
    todo, seen, candidates = ["log_probs"], set(), []
    while todo and not candidates:
        following = []
        for tensor in todo:
            if tensor in seen:
                continue
            seen.add(tensor)
            node = producer.get(tensor)
            if node is None:
                continue
            weight = constants.get(node.input[1]) if len(node.input) > 1 else None
            if node.op_type in {"MatMul", "Gemm", "MatMulInteger"} and weight is not None:
                dims = list(weight.dims)
                vocab_axis = 0 if node.op_type == "Gemm" and any(a.name == "transB" and a.i == 1 for a in node.attribute) else 1
                if len(dims) == 2 and dims[vocab_axis] == len(manifest["tokens"])+1:
                    candidates.append(node)
                    continue
            following.extend(node.input)
        todo = following
    if len(candidates) != 1:
        raise ValueError("Cannot unambiguously locate the frozen CTC projection")
    node = candidates[0]
    name = node.input[0]
    if node.op_type == "MatMulInteger":
        quantizer = producer[name]
        if quantizer.op_type != "DynamicQuantizeLinear":
            raise ValueError("Unsupported encoder quantization boundary")
        name = quantizer.input[0]
    graph.graph.node.append(helper.make_node("Identity", [name], ["encoder_features"]))
    graph.graph.output.append(helper.make_tensor_value_info("encoder_features", TensorProto.FLOAT, [1, None, None]))
    onnx.checker.check_model(graph)
    onnx.save(graph, str(out))
    return manifest, "encoder_features"


def frozen_stream(session, manifest, feature_name, pcm):
    """Same fbank, chunk/cache geometry and final-frame count as the SDK."""
    import torch
    import torchaudio

    feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(pcm).unsqueeze(0),
        num_mel_bins=manifest["fbank"]["mels"], sample_frequency=manifest["sampleRate"],
        dither=0.0, energy_floor=1e-10, high_freq=-400).numpy()
    need = ((len(feats) - 7) // 2 + 1) // 2
    if need <= 0:
        raise ValueError("Recording has no complete encoder frames")
    window, shift = manifest["window"], manifest["shift"]
    windows = (need + shift // 4 - 1) // (shift // 4)
    total = window + (windows - 1) * shift
    feats = np.pad(feats, ((0, max(0, total-len(feats))), (0, 0)), constant_values=manifest["fbank"]["padValue"])
    states = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
              for s in manifest["states"]}
    lp, enc = [], []
    names = [o.name for o in session.get_outputs()]
    for start in range(0, total-window+1, shift):
        outputs = dict(zip(names, session.run(None, {"x": feats[None, start:start+window], **states})))
        lp.append(outputs["log_probs"][0])
        if outputs[feature_name].ndim != 3 or outputs[feature_name].shape[:2] != outputs["log_probs"].shape[:2]:
            raise ValueError("Encoder feature clock differs from primary probabilities")
        enc.append(outputs[feature_name][0])
        states = {s["name"]: outputs["new_" + s["name"]] for s in manifest["states"]}
    return np.concatenate(enc)[:need], np.concatenate(lp)[:need]


def validate_training_labels(labels):
    """Fail closed before reading audio from a release or unspecified partition."""
    if not labels or any(l.get("split") != "train" or not l.get("speaker") for l in labels):
        raise ValueError("Every input must identify a training speaker and split=train")
    if not any(l.get("kind") == "none" for l in labels):
        raise ValueError("None-labelled controls are required")
    if not any(any(e.get("kind") == "repeat" for e in l.get("non_errors", [])) for l in labels):
        raise ValueError("Repeat controls are required")


def preservation_point(values):
    """Preserve every calibration-speaker scripted substitution, including ties."""
    if not values or any(not np.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("Finite calibration substitution scores are required")
    point = float(min(values))
    if point <= 0:
        raise ValueError("Degenerate zero preservation point cannot produce a veto")
    return point


def weak_targets(label, references):
    """Script class/location, with controls explicit and compliance latent."""
    locations = {(s, a, w): i for i, (s, a, w) in enumerate(references)}
    targets = set()
    for e in label.get("events", []):
        r = e["location"]
        at = locations.get((r["surah"], r["ayah"], r["word"]))
        if at is None:
            continue
        cls = 1 if e["errorClass"] == "wrong_word" else 2 if e["errorClass"] in {"omitted_word", "skipped_passage"} else 3
        targets.add((at, cls))
    for e in label.get("non_errors", []):
        if e["kind"] == "repeat":
            r = e["location"]
            at = locations.get((r["surah"], r["ayah"], r["word"]))
            if at is not None:
                targets.add((at, 4))
    return sorted(targets)


def make_head(dim, contextual, distilled=False):
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Sequential(nn.Linear(dim, 48), nn.Tanh())
            self.context = nn.Conv1d(48, 48, 3, padding=1) if contextual else None
            self.output = nn.Linear(48, 1 if distilled else 5)
            self.none = nn.Linear(48, 1)

        def forward(self, x):
            h = self.input(x)
            if self.context is not None:
                h = h + torch.tanh(self.context(h.T.unsqueeze(0))[0].T)
            return self.output(h), self.none(h.mean(0))[0]

    return Head()


def train(records, out):
    """Selection and operating point are frozen before any practice input."""
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(2)
    fit = [r for r in records if r["partition"] == "fit"]
    valid = [r for r in records if r["partition"] == "validation"]
    calibration = [r for r in records if r["partition"] == "calibration"]
    if not fit or not valid or not calibration:
        raise ValueError("All three training-speaker partitions are required")
    for rows in [fit, valid, calibration]:
        if not any(cls == 1 for r in rows for _, cls in r["targets"]):
            raise ValueError("Each speaker partition needs scripted substitutions")
        if not any(not r["targets"] for r in rows) or not any(cls == 4 for r in rows for _, cls in r["targets"]):
            raise ValueError("Each speaker partition needs none and repeat controls")
    dim = fit[0]["x"].shape[1]
    joined = np.concatenate([r["x"] for r in fit])
    mean, std = joined.mean(0), np.maximum(joined.std(0), 1e-3)
    for r in records:
        r["tensor"] = torch.from_numpy((r["x"]-mean)/std)

    def loss(head, r, contextual):
        scores, none = head(r["tensor"])
        if contextual:
            joint = torch.cat([none.reshape(1), scores[:, 1:].reshape(-1)])
            lp = joint.log_softmax(0)
            if not r["targets"]:
                return -lp[0]
            choices = torch.stack([lp[1 + at*4 + cls-1] for at, cls in r["targets"]])
            # A learned recording-level compliance variable has canonical and
            # script alternatives. Neither is imposed as a performed transcript.
            compliance = head.compliance.sigmoid()
            return -torch.logaddexp(lp[0] + torch.log1p(-compliance),
                torch.logsumexp(choices, 0) - np.log(len(choices)) + compliance.log())
        targets = torch.zeros(len(scores), dtype=torch.long)
        for at, cls in r["targets"]:
            targets[at] = cls
        lp = scores.log_softmax(1)
        likelihood = lp[:,0].clone()
        compliance = head.compliance.sigmoid()
        for at, cls in r["targets"]:
            likelihood[at] = torch.logaddexp(lp[at,0]+torch.log1p(-compliance),lp[at,cls]+compliance.log())
        return -likelihood.mean()

    heads, selection = {}, {}
    for contextual, name in [(True, "contextual"), (False, "independent")]:
        candidates = []
        for decay in [1e-3, 1e-2]:
            torch.manual_seed(731)
            head = make_head(dim, contextual)
            head.register_parameter("compliance", torch.nn.Parameter(torch.tensor(2.0)))
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=decay)
            best, saved, epoch_at = float("inf"), None, 0
            for epoch in range(60):
                for i in torch.randperm(len(fit)).tolist():
                    optimizer.zero_grad()
                    value = loss(head, fit[i], contextual)
                    # Weak scripts must not be solved solely by collapsing the
                    # compliance prior to zero; retain a broad Beta(2,2) prior.
                    p = head.compliance.sigmoid()
                    value = value - .05*(p.log()+torch.log1p(-p))
                    value.backward()
                    optimizer.step()
                with torch.no_grad():
                    score = np.mean([float(loss(head, r, contextual)) for r in valid])
                if score < best:
                    best, saved, epoch_at = score, {k:v.detach().clone() for k,v in head.state_dict().items()}, epoch
            candidates.append((best, saved, decay, epoch_at))
        best, saved, decay, epoch = min(candidates, key=lambda x:x[0])
        head = make_head(dim, contextual)
        head.register_parameter("compliance", torch.nn.Parameter(torch.tensor(2.0)))
        head.load_state_dict(saved)
        heads[name] = head.eval()
        selection[name] = dict(loss=best, decay=decay, epoch=epoch)

    # Distill recording competition into independent sigmoid outputs. There is
    # no normalization across words at inference, and multiple changes can pass.
    torch.manual_seed(732)
    distilled = make_head(dim, True, True)
    optimizer = torch.optim.AdamW(distilled.parameters(), lr=1e-3, weight_decay=selection["contextual"]["decay"])
    teacher = heads["contextual"]
    def distill_loss(r):
        with torch.no_grad():
            scores, none = teacher(r["tensor"])
            target = torch.cat([none.reshape(1), scores[:,1:].reshape(-1)]).softmax(0)[1:].reshape(-1,4)[:,0]
        logits, _ = distilled(r["tensor"])
        return F.binary_cross_entropy_with_logits(logits[:,0], target)
    best, saved = float("inf"), None
    for epoch in range(60):
        for i in torch.randperm(len(fit)).tolist():
            optimizer.zero_grad()
            distill_loss(fit[i]).backward()
            optimizer.step()
        with torch.no_grad():
            score = np.mean([float(distill_loss(r)) for r in valid])
        if score < best:
            best, saved = score, {k:v.detach().clone() for k,v in distilled.state_dict().items()}
    distilled.load_state_dict(saved)
    heads["distilled"] = distilled.eval()
    operating = {}
    for name in ["distilled", "independent"]:
        positives = []
        for r in calibration:
            with torch.no_grad():
                logits, _ = heads[name](r["tensor"])
                values = logits[:,0].sigmoid() if name == "distilled" else logits.softmax(1)[:,1]
            positives.extend(float(values[at]) for at, cls in r["targets"] if cls == 1)
        if not positives:
            raise ValueError("Preservation calibration has no scripted substitutions")
        operating[name] = dict(threshold=preservation_point(positives), substitutions=len(positives))
    torch.save(dict(states={k:v.state_dict() for k,v in heads.items()}, mean=mean, std=std, dim=dim), out/"heads.pt")
    report = dict(selection=selection, operating=operating, recordings=len(records),
        partitions={p:sum(r["partition"]==p for r in records) for p in ["fit","validation","calibration"]},
        speakers={p:len({r["speaker"] for r in records if r["partition"]==p}) for p in ["fit","validation","calibration"]})
    (out/"training.json").write_text(json.dumps(report, indent=2))
    export_head(heads, mean, std, dim, out/"edit-head.onnx")
    return report


def export_head(heads, mean, std, dim, path):
    """Normalize raw identical features inside the exported head; verify dynamic
    word counts, including multiple potential changes without a softmax across words."""
    import torch
    import onnxruntime as ort

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.contextual, self.independent = heads["distilled"], heads["independent"]
            self.register_buffer("mean", torch.from_numpy(mean))
            self.register_buffer("std", torch.from_numpy(std))

        def forward(self, x):
            x = (x-self.mean)/self.std
            contextual, _ = self.contextual(x)
            independent, _ = self.independent(x)
            return contextual[:,0].sigmoid(), independent.softmax(1)[:,1]

    module = Export().eval()
    example = torch.from_numpy(np.stack([mean, mean+std, mean-std]).astype(np.float32))
    torch.onnx.export(module, example, str(path), input_names=["x"], output_names=["contextual","independent"],
        dynamic_axes={"x":{0:"words"}, "contextual":{0:"words"}, "independent":{0:"words"}},
        opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for n in [1,2,3,7]:
        x = torch.from_numpy(np.tile(example.numpy(),(3,1))[:n])
        with torch.no_grad():
            wanted = module(x)
        actual = session.run(None,{"x":x.numpy()})
        for a,b in zip(actual,wanted):
            np.testing.assert_allclose(a,b.numpy(),rtol=1e-4,atol=1e-6)


def export_bundle(model_dir, out, manifest, report):
    """Only newly generated bundle manifests change. All supplied files stay fixed.
    Copy the complete source bundle, including inherited side ears, and verify it."""
    bundle = out/"bundle"
    shutil.copytree(model_dir,bundle,dirs_exist_ok=True)
    for source, target in [(out/"encoder.onnx",bundle/"edit-encoder.onnx"),(out/"edit-head.onnx",bundle/"edit-head.onnx")]:
        shutil.copy2(source,target)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(bundle/manifest["file"]) != manifest["sha256"]:
        raise ValueError("Primary ONNX does not match the supplied manifest")
    declaration = {"feature":{"file":"edit-encoder.onnx","sha256":sha(bundle/"edit-encoder.onnx")},
        "head":{"file":"edit-head.onnx","sha256":sha(bundle/"edit-head.onnx")},
        "preservation":report["operating"]["distilled"]["threshold"],
        "independentPreservation":report["operating"]["independent"]["threshold"],
        "featureDim":report["featureDim"]}
    result = {**manifest, "id":manifest["id"]+"-edit-localization", "editLocalization":declaration}
    (bundle/"manifest.json").write_text(json.dumps(result,indent=2))
    (bundle/"edit-training.json").write_text(json.dumps(report,indent=2))
    return bundle


def main():
    import onnxruntime as ort
    import soundfile as sf
    from scipy.signal import resample_poly

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = [json.loads(line) for line in (args.data/"labels.jsonl").read_text().splitlines() if line.strip()]
    validate_training_labels(labels)
    reference = json.loads(Path("packages/core/assets/ref-v2.json").read_text())
    ayahs = {(a["s"],a["a"]):a for a in reference["ayahs"]}
    manifest, feature = feature_graph(args.model, args.out/"encoder.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.out/"encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    primary = ort.InferenceSession(str(args.model/manifest["file"]), sess_options=options, providers=["CPUExecutionProvider"])
    records, excluded = [], []
    for label in labels:
        if not label.get("scored", True):
            excluded.append(dict(id=label["id"], reason="unreviewed extra mistakes"))
            continue
        pronunciations, refs = [], []
        for key in label["ayahs"]:
            s,a = map(int,key.split(":"))
            row = ayahs[s,a]
            offsets = row["o"] + [len(row["p"])]
            for w in range(len(row["w"])):
                pronunciations.append(row["p"][offsets[w]:offsets[w+1]])
                refs.append((s,a,w+1))
        path = args.data / label.get("wav", f"wav/{label['id']}.wav")
        if not path.resolve().is_relative_to(args.data.resolve()):
            raise ValueError("Audio path escapes the authorized training directory")
        pcm, rate = sf.read(path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(1)
        if rate != manifest["sampleRate"]:
            import math
            divisor = math.gcd(rate,manifest["sampleRate"])
            pcm = resample_poly(pcm,manifest["sampleRate"]//divisor,rate//divisor).astype(np.float32)
        enc, lp = frozen_stream(session, manifest, feature, pcm)
        # Check the exposed graph did not change CTC values or cache behavior.
        # A dummy feature alias makes the same stream loop collect logits twice.
        if not records:
            _, baseline = frozen_stream(primary, manifest, "log_probs", pcm)
            np.testing.assert_array_equal(lp,baseline)
        features = word_features(enc,lp,pronunciations,manifest["tokens"])
        if features is None:
            excluded.append(dict(id=label["id"], reason="impossible canonical boundary marginal"))
            continue
        x, spans = features
        targets = weak_targets(label,refs)
        if label.get("kind") != "none" and not targets:
            excluded.append(dict(id=label["id"],reason="script has no in-recording location"))
            continue
        records.append(dict(id=label["id"], speaker=label["speaker"], partition=partition(label["speaker"]),
            x=x, spans=spans, targets=targets))
        if len(records)%25 == 0:
            print("features",len(records),flush=True)
    np.savez_compressed(args.out/"recordings.npz", records=np.asarray(records,dtype=object))
    (args.out/"excluded.json").write_text(json.dumps(excluded,indent=2))
    report = train(records,args.out)
    report["featureDim"] = records[0]["x"].shape[1]
    report["primaryVerified"] = True
    report["excluded"] = len(excluded)
    (args.out/"training.json").write_text(json.dumps(report,indent=2))
    bundle = export_bundle(args.model,args.out,manifest,report)
    print(json.dumps({**report,"bundle":str(bundle)},indent=2),flush=True)


if __name__ == "__main__":
    main()
