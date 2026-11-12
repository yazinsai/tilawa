"""Matched ordinary / feature-hint / attention-routing student refits.

Run directly in the authorized training container, not through a Modal client:
    PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_routing_train.py --recipe <training-only.json> --out <scratch-dir>

The recipe is a frozen, independently audited training-speaker partition. It
must identify the initial checkpoint, selected attention operators and corpus
keys; this runner never selects recordings or checkpoints using practice audio.
"""

import argparse
import hashlib
import io
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn

from student_routing import (
    TEACHER_GRID, HeadCorrespondence, capture_streaming_attention,
    capture_teacher_attention, deterministic_streaming, feature_hint_loss,
    project_teacher, routing_loss,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model):
    """Exact frozen teacher identity, including dtype, shape and state names."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        header = json.dumps([name, list(value.shape), str(value.dtype)]).encode()
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for block in (header, raw):
            digest.update(len(block).to_bytes(8, "little"))
            digest.update(block)
    return digest.hexdigest()


def validate_recipe(recipe):
    """Reject incomplete speaker provenance; do not invent authorization."""
    if recipe.get("schema") != 1 or recipe.get("split") != "train":
        raise ValueError("Routing recipe must describe training-only data")
    speakers = recipe.get("speakers_by_key", {})
    train, valid = recipe.get("training_keys", []), recipe.get("validation_keys", [])
    excluded = set(recipe.get("excluded_speakers", []))
    if (not train or not valid or not excluded or len(set(train)) != len(train)
            or len(set(valid)) != len(valid) or set(train) & set(valid)
            or set(speakers) != set(train) | set(valid)):
        raise ValueError("Need unique disjoint keys, complete speakers and release exclusions")
    train_s, valid_s = {speakers[k] for k in train}, {speakers[k] for k in valid}
    if any(not isinstance(s, str) or not s for s in train_s | valid_s | excluded):
        raise ValueError("Every selected recording and excluded reader needs a speaker identity")
    if train_s & valid_s or (train_s | valid_s) & excluded:
        raise ValueError("Training, validation and release speakers must be disjoint")
    for partition in (train, valid):
        if not any(k.startswith("help:") for k in partition) or not any(not k.startswith("help:") for k in partition):
            raise ValueError("Both partitions need authorized Quran and help none recordings")
    if recipe.get("selection") != "minimum-validation-ctc-plus-preservation":
        raise ValueError("Checkpoint selection must be frozen on training-speaker validation")
    for field in ("checkpoint_sha256", "teacher_sha256", "corpus_sha256"):
        value = recipe.get(field, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing frozen identity: {field}")
    for field in ("checkpoint", "base_bundle"):
        if not isinstance(recipe.get(field), str) or not recipe[field]:
            raise ValueError(f"Missing audited artifact path: {field}")
    if not recipe.get("pairs") or len({p["student"] for p in recipe["pairs"]}) != len(recipe["pairs"]):
        raise ValueError("Need unique audited student/teacher attention operator pairs")
    for field in ("calibration_updates", "updates", "validate_every", "seed"):
        if type(recipe.get(field)) is not int or recipe[field] < (0 if field == "seed" else 1):
            raise ValueError(f"Invalid frozen schedule: {field}")
    for field in ("base_lr", "aux_weight", "preservation_weight", "ctc_weight", "kd_weight"):
        if not np.isfinite(recipe.get(field, float('nan'))) or recipe[field] <= 0:
            raise ValueError(f"Invalid frozen objective weight: {field}")
    if recipe.get("chunk") != 16 or recipe.get("left") != 128:
        raise ValueError("Routing refit must preserve the shipped streaming geometry")
    if recipe.get("real_pause_examples", 0) < 1:
        raise ValueError("Training mix has no audited real pauses")
    return recipe


def load_partition(recipe, partition, data_root, s1_root="/student/s1"):
    """Read authorized whole recordings, including pauses; no mistake positives."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from student import exclusions

    keys = recipe[f"{partition}_keys"]
    data_root = Path(data_root)
    excluded = exclusions(Path("docs/benchmarks/gold-per-v1-exclude.txt").read_text())
    by_shard = {}
    for key in keys:
        if not key.startswith("help:"):
            shard, row = key.rsplit(":", 1)
            if "/" in shard or ".." in shard or (shard, int(row)) in excluded:
                raise ValueError("Unauthorized Quran corpus key")
            by_shard.setdefault(shard, set()).add(int(row))
    selected = {}
    for shard, wanted in by_shard.items():
        for row in pq.read_table(Path(s1_root) / f"{shard}.parquet").to_pylist():
            if row["row"] not in wanted:
                continue
            key = f"{shard}:{row['row']}"
            if row["split"] != "train" or row["bucket"] != "clean-match" or not row["ref"]:
                raise ValueError("Quran example lacks an authorized clean training transcript")
            if key in selected:
                raise ValueError("Duplicate Quran key")
            selected[key] = dict(row, key=key)
    help_root = (data_root / "help-train").resolve()
    refs = {f"{a['s']}:{a['a']}": a["p"] for a in json.loads(Path("packages/core/assets/ref-v2.json").read_text())["ayahs"]}
    wanted_help = {k for k in keys if k.startswith("help:")}
    for line in (help_root / "labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = f"help:{row['id']}"
        if key not in wanted_help:
            continue
        if (key in selected or row.get("split") != "train" or row.get("kind") != "none"
                or row.get("events") != [] or row.get("non_errors") != [] or row.get("unsupported") != []
                or row.get("scored") is not True or row.get("excluded")
                or row.get("speaker") != recipe["speakers_by_key"][key]):
            raise ValueError("Help example is not an audited none-labelled training-speaker take")
        path = (help_root / row["wav"]).resolve()
        if not path.is_relative_to(help_root):
            raise ValueError("Help path leaves the authorized corpus")
        wave, sr = sf.read(path, dtype="float32")
        if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Help PCM must be finite mono 16 kHz")
        transcript = "".join(refs[a] for a in row["ayahs"])
        if not transcript:
            raise ValueError("Help none take lacks a transcript")
        buf = io.BytesIO()
        sf.write(buf, wave, sr, format="WAV", subtype="FLOAT")
        # Ordinary teacher targets are generated live below, on shared PCM.
        selected[key] = dict(key=key, ref=transcript, hyp=transcript, n_samples=len(wave),
                             n_frames=1, flac=buf.getvalue(), top5_ids=bytes(5),
                             top5_p=np.array([1, 0, 0, 0, 0], dtype=np.float16).tobytes())
    if set(selected) != set(keys):
        raise ValueError("Frozen training keys are missing")
    return [selected[k] for k in keys]


def corpus_hash(*partitions):
    digest = hashlib.sha256()
    for rows in partitions:
        for row in sorted(rows, key=lambda r: r["key"]):
            for block in (row["key"].encode(), row["ref"].encode(), row["flac"]):
                digest.update(len(block).to_bytes(8, "little"))
                digest.update(block)
    return digest.hexdigest()


def packed_clips(rows):
    out = {k: [r[k] for r in rows] for k in ("key", "ref", "hyp", "n_samples", "n_frames")}
    for source, target, dtype, offsets in (("flac", "flac", np.uint8, "flac_off"),
                                          ("top5_ids", "ids", np.uint8, "top_off"),
                                          ("top5_p", "probs", np.float16, None)):
        chunks = [np.frombuffer(r[source], dtype=dtype) for r in rows]
        out[target] = np.concatenate(chunks)
        if offsets:
            out[offsets] = np.concatenate([[0], np.cumsum([len(x) for x in chunks])])
    out["target"] = out["ref"]
    out["n_samples"] = np.array(out["n_samples"])
    return out


class Teacher:
    def __init__(self, recipe, device):
        from quran_muaalem import Muaalem
        from student import TOKENS

        self.muaalem = Muaalem("obadx/muaalem-model-v3_2", device=str(device), dtype=torch.float32)
        self.muaalem.model.eval().float().requires_grad_(False)
        if state_hash(self.muaalem.model) != recipe["teacher_sha256"]:
            raise ValueError("Teacher state differs from the frozen training choice")
        vocabulary = self.muaalem.multi_level_tokenizer.id_to_vocab["phonemes"]
        if [vocabulary[i] for i in range(1, len(TOKENS) + 1)] != list(TOKENS):
            raise ValueError("Teacher alphabet differs from the shipped student")
        modules = dict(self.muaalem.model.named_modules())
        pairs = recipe["pairs"]
        self.modules = [modules[p["teacher"]] for p in pairs]
        if any(m.__class__.__name__ != "Wav2Vec2BertSelfAttention" or ".encoder.layers." not in p["teacher"]
               for m, p in zip(self.modules, pairs)):
            raise ValueError("Selected teacher operator is not the pinned temporal attention")
        self.device = device

    def __call__(self, wave):
        inputs = self.muaalem.processor([wave.numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(), capture_teacher_attention(self.modules) as captured:
            output = self.muaalem.model(**inputs, return_dict=False)[0]["phonemes"].float().log_softmax(-1)
        expected = ((len(wave) - 400) // 160 + 1) // 2
        if inputs["input_features"].shape != (1, expected, 160):
            raise ValueError("Teacher processor differs from the audited 20 ms stacked fbank grid")
        if any(captured[m][0].shape[-2:] != (expected, expected) for m in self.modules):
            raise ValueError("Selected teacher attention is not on the pre-adapter acoustic grid")
        return output[0], [(captured[m][0][0], captured[m][1][0]) for m in self.modules]


class RoutingArm:
    def __init__(self, model, recipe, teacher, device):
        self.model, self.recipe, self.device = model, recipe, device
        modules = dict(model.named_modules())
        self.modules = [modules[p["student"]] for p in recipe["pairs"]]
        if any(m.__class__.__name__ != "RelPositionMultiheadAttentionWeights" for m in self.modules):
            raise ValueError("Selected student operator is not the pinned temporal attention")
        self.factors = {m: p["downsample"] for m, p in zip(self.modules, recipe["pairs"])}
        for module, factor in self.factors.items():
            # Verify the configured grid against the stack owning this module.
            stack = next((i for i, enc in enumerate(model.encoder.encoders)
                          if any(x is module for x in enc.modules())), None)
            if stack is None or factor != model.encoder.downsampling_factor[stack]:
                raise ValueError("Recipe attention grid differs from the actual encoder stack")
        self.maps = nn.ModuleList([HeadCorrespondence(s.num_heads, t.num_heads)
                                   for s, t in zip(self.modules, teacher.modules)]).to(device)
        self.hints = nn.ModuleList([nn.Linear(s.embed_dim, t.linear_q.in_features)
                                    for s, t in zip(self.modules, teacher.modules)]).to(device)

    def stream(self, feats, targets=None):
        from student_export import PAD, LOG_EPS, wrap

        recipe = self.recipe
        chunk, left = recipe["chunk"], recipe["left"]
        valid_embed = (len(feats) - 7) // 2
        valid_out = (valid_embed + 1) // 2
        windows = max(1, (valid_out + chunk // 2 - 1) // (chunk // 2))
        total = chunk * 2 + PAD + (windows - 1) * chunk * 2
        f = nn.functional.pad(feats.to(self.device), (0, 0, 0, max(0, total - len(feats))), value=LOG_EPS)
        outputs, routes, hints = [], [], []
        with deterministic_streaming(self.model, chunk, left):
            streaming = wrap(self.model, chunk, left)
            states = [s.to(self.device) for s in streaming.init_states()]
            for w in range(windows):
                with capture_streaming_attention(self.factors, w * chunk) as captured:
                    lp, states = streaming(f[None, w * chunk * 2:w * chunk * 2 + chunk * 2 + PAD], states)
                # Keep the deployed forward caches; auxiliary credit does not
                # introduce an unrelated cross-chunk gradient experiment.
                states = [s.detach() for s in states]
                outputs.append(lp[0])
                if targets is None:
                    continue
                for index, module in enumerate(self.modules):
                    c = captured[module]
                    a, hidden = targets[index]
                    allowed = c.allowed[0] & (c.keys * c.grid.pool < valid_embed)[None]
                    valid_queries = c.queries * c.grid.pool < valid_embed
                    q = c.queries[valid_queries]
                    if not len(q):
                        continue
                    target, paired = project_teacher(a, TEACHER_GRID, c.grid, q, c.keys, allowed[valid_queries])
                    support = allowed[valid_queries][paired]
                    predicted = c.probabilities[:, 0, valid_queries][:, paired]
                    routes.append(routing_loss(predicted, target, self.maps[index], support))
                    bins = c.grid.bins(TEACHER_GRID.centres(torch.arange(len(hidden), device=self.device)))
                    h = torch.stack([hidden[bins == query].mean(0) for query in q[paired]])
                    x = c.features[valid_queries, 0][paired]
                    hints.append(feature_hint_loss(x, h, self.hints[index], torch.ones(len(x), dtype=torch.bool, device=self.device)))
        if targets is not None and not routes:
            raise ValueError("No valid paired routing observations in a training recording")
        return torch.cat(outputs)[:valid_out], (torch.stack(routes).mean() if routes else None), (torch.stack(hints).mean() if hints else None)


def acoustic_objective(arm, anchor, item, teacher_output, recipe, mode):
    """Identical data/teacher passes/streaming work in all three matched arms."""
    import student_train as T

    device = arm.device
    if mode not in ("ordinary", "feature", "routing"):
        raise ValueError("Unknown matched acoustic training arm")
    static_cfg = {**T.DEFAULTS, "ctc_weight": recipe["ctc_weight"], "kd_weight": 0.0}
    ctc = T.losses(arm.model, T.collate([item]), static_cfg, device)["ctc"]
    teacher_lp, targets = teacher_output
    lp, route, hint = arm.stream(item["routing_feats"], targets)
    with torch.no_grad():
        preserved, _, _ = anchor.stream(item["routing_feats"])
    if lp.shape != preserved.shape:
        raise ValueError("Refit changed the shipped output clock")
    preservation = nn.functional.kl_div(lp, preserved.detach(), log_target=True, reduction="batchmean")
    n = min(len(lp), len(teacher_lp))
    ordinary = -(teacher_lp[:n].detach().exp() * lp[:n]).sum(-1).mean()
    aux = {"ordinary": route * 0 + hint * 0, "feature": hint + route * 0, "routing": route + hint * 0}[mode]
    loss = (recipe["ctc_weight"] * ctc + recipe["kd_weight"] * ordinary
            + recipe["preservation_weight"] * preservation + recipe["aux_weight"] * aux)
    return loss, {"ctc": float(ctc.detach()), "kd": float(ordinary.detach()),
                  "preservation": float(preservation.detach()), "routing": float(route.detach()), "feature": float(hint.detach())}


def export_selected(checkpoint, dest, valid_rows, recipe, base_bundle):
    """Export only student weights and verify its recurrent ONNX contract.

Correspondence and feature projections stay in training. Copy and check the
existing declared insertion assets, preserving the shipped verifier. Do not
publish a model selection when any required asset or parity check is missing.
"""
    import student_train as T
    from student_export import PAD, LOG_EPS, export_onnx, fbank, manifest, onnx_deps, stream_onnx, wrap
    onnx_deps()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from student import VOCAB

    checkpoint = Path(checkpoint)
    dest, base = Path(dest), Path(base_bundle)
    if dest.exists():
        raise ValueError("Refuse to overwrite an existing exported bundle")
    dest.mkdir(parents=True)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = T.build_model(ck["cfg"]["size"])
    model.load_state_dict(ck["model"])
    fp = dest / "model-fp32.onnx"
    path = dest / "model.onnx"
    states = export_onnx(model, str(fp), recipe["chunk"], recipe["left"])
    quantize_dynamic(str(fp), str(path), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gemm"])
    metadata = manifest(checkpoint.stem, states, recipe["chunk"], recipe["left"], str(path), ck)
    metadata["format"] = "onnx-int8"
    metadata["quantizedFrom"] = sha256(fp)
    # The registry owns shipped side assets. No trained teacher or projection
    # is deployed, and an absent existing verifier cannot silently change rules.
    # registry.ts adds shipped verifier assets to the primary JSON registry.
    # Read the product's actual registry, rather than copying its constants.
    process = subprocess.run(["pnpm", "--filter", "@tilawa/core", "exec", "tsx", "--eval",
                              "import { MODEL_REGISTRY } from '../sdk/src/registry.ts'; "
                              "process.stdout.write(JSON.stringify(MODEL_REGISTRY.auxiliary ?? []));"],
                             check=True, capture_output=True, text=True)
    for asset in json.loads(process.stdout):
        for entry in [asset, *asset.get("externalData", [])]:
            source = (base / entry["file"]).resolve()
            if not source.is_relative_to(base.resolve()) or not source.is_file():
                raise ValueError("Shipped insertion asset is missing from the primary bundle")
            if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
                raise ValueError("Shipped insertion asset identity differs from the registry")
            target = dest / entry["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != entry["sha256"]:
                raise ValueError("Copied insertion asset is incomplete")
    parity = []
    # Validate identical recurrent calls, rather than comparing a whole-recording
    # chunk-masked pass against streaming and calling that export parity.
    model.eval()
    torch.set_num_threads(2)
    for row in valid_rows:
        feats = fbank(row["flac"])
        chunk, left = recipe["chunk"], recipe["left"]
        n = (((len(feats) - 7) // 2) + 1) // 2
        windows = max(1, (n + chunk // 2 - 1) // (chunk // 2))
        total = chunk * 2 + PAD + (windows - 1) * chunk * 2
        padded = nn.functional.pad(feats, (0, 0, 0, max(0, total - len(feats))), value=LOG_EPS)
        with torch.no_grad(), deterministic_streaming(model, chunk, left):
            streaming = wrap(model, chunk, left)
            current = streaming.init_states()
            outputs = []
            for w in range(windows):
                lp, current = streaming(padded[None, w * chunk * 2:w * chunk * 2 + chunk * 2 + PAD], current)
                outputs.append(lp[0])
            expected = torch.cat(outputs)[:n]
        fp_output = stream_onnx(str(fp), feats, states, recipe["chunk"])[:len(expected)]
        got = stream_onnx(str(path), feats, states, recipe["chunk"])[:len(expected)]
        if fp_output.shape != expected.shape or got.shape != expected.shape or got.shape[1] != VOCAB:
            raise ValueError("Export changed the output inventory or frame count")
        error = float(np.abs(np.exp(fp_output) - expected.exp().numpy()).max())
        if error > 1e-4 or not np.isfinite(got).all():
            raise ValueError("Export failed fp32 recurrent parity or int8 finite-output validation")
        parity.append(dict(key=row["key"], frames=len(expected), fp32_max_probability_error=error,
                           quantized_top1_agreement=float((got.argmax(-1) == fp_output.argmax(-1)).mean())))
    if not parity:
        raise ValueError("Export needs training-speaker validation recordings")
    (dest / "parity.json").write_text(json.dumps(parity, indent=2))
    (dest / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    fp.unlink()
    return metadata


def run(recipe, out, data_root):
    import student_train as T
    from icefall.utils import get_parameter_groups_with_lrs
    from optim import ScaledAdam

    validate_recipe(recipe)
    if not torch.cuda.is_available():
        raise RuntimeError("Corpus training belongs in the authorized GPU container")
    device = torch.device("cuda")
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    checkpoint = Path(recipe["checkpoint"])
    if sha256(checkpoint) != recipe["checkpoint_sha256"]:
        raise ValueError("Initial checkpoint differs from the frozen training choice")
    train_rows, valid_rows = (load_partition(recipe, p, data_root) for p in ("training", "validation"))
    if corpus_hash(train_rows, valid_rows) != recipe["corpus_sha256"]:
        raise ValueError("Authorized training corpus changed after partition selection")
    cfg = {**T.DEFAULTS, "routing_paired": True, "seed": recipe["seed"]}
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    train_ds = T.ClipDataset(packed_clips(train_rows), cfg, train=True, aug=aug)
    valid_ds = T.ClipDataset(packed_clips(valid_rows), cfg, train=False, aug=None)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher = Teacher(recipe, device)

    def make_arm():
        model = T.build_model(ck["cfg"]["size"]).to(device)
        model.load_state_dict(ck["model"])
        return RoutingArm(model, recipe, teacher, device)

    anchor = make_arm()
    anchor.model.requires_grad_(False)
    # Calibrate retrieval-head correspondence on the frozen initial student,
    # then freeze it before any acoustic refit or checkpoint selection.
    calibration = torch.optim.Adam(anchor.maps.parameters(), lr=recipe["base_lr"])
    rng = np.random.default_rng(recipe["seed"])
    for step in range(recipe["calibration_updates"]):
        item = train_ds[int(rng.integers(len(train_ds)))]
        target = teacher(item["wave"])
        calibration.zero_grad(set_to_none=True)
        _, route, _ = anchor.stream(item["routing_feats"], target[1])
        route.backward()
        calibration.step()
    anchor.maps.requires_grad_(False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2))
    torch.save(anchor.maps.state_dict(), out / "correspondence.pt")
    # Sequential arms keep the teacher/anchor footprint fixed. Each arm uses
    # the same order, augmentations, number of successful updates and teacher
    # calls. Common validation CTC+preservation chooses checkpoints.
    for mode in ("ordinary", "feature", "routing"):
        torch.manual_seed(recipe["seed"])
        random.seed(recipe["seed"])
        arm = make_arm()
        arm.maps.load_state_dict(anchor.maps.state_dict())
        arm.maps.requires_grad_(False)
        params = nn.ModuleList([arm.model, arm.hints])
        opt = ScaledAdam(get_parameter_groups_with_lrs(params, lr=recipe["base_lr"], include_names=True),
                         lr=recipe["base_lr"], clipping_scale=2.0)
        rng = np.random.default_rng(recipe["seed"])
        order = rng.permutation(len(train_ds))
        best = float("inf")
        with (out / f"{mode}.jsonl").open("w") as log:
            for step in range(1, recipe["updates"] + 1):
                epoch, pos = divmod(step - 1, len(order))
                train_ds.augmentation_epoch = epoch
                item = train_ds[int(order[pos])]
                arm.model.train()
                arm.model.training_mode()
                for module in arm.model.modules():
                    module.batch_count = (ck["state"]["step"] + step) * ck["cfg"]["max_batch_s"] / 600.0
                opt.zero_grad(set_to_none=True)
                loss, numbers = acoustic_objective(arm, anchor, item, teacher(item["wave"]), recipe, mode)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite matched update; do not silently change effective data")
                loss.backward()
                opt.step()
                entry = dict(mode=mode, step=step, loss=float(loss.detach()), **numbers)
                if step % recipe["validate_every"] == 0 or step == recipe["updates"]:
                    values = []
                    with torch.no_grad(), deterministic_streaming(arm.model, recipe["chunk"], recipe["left"]):
                        for i in range(len(valid_ds)):
                            v = valid_ds[i]
                            # Same criterion for every arm; routing/hint agreement
                            # cannot select its own checkpoint or use targets.
                            _, metrics = acoustic_objective(arm, anchor, v, teacher(v["wave"]), recipe, "ordinary")
                            values.append(recipe["ctc_weight"] * metrics["ctc"]
                                          + recipe["preservation_weight"] * metrics["preservation"])
                    score = float(np.mean(values))
                    entry["validation"] = score
                    if score < best:
                        best = score
                        torch.save(dict(model=arm.model.state_dict(), cfg=ck["cfg"], state={"step": step},
                                        training_recipe=recipe, mode=mode, validation=score), out / f"{mode}.pt")
                print(json.dumps(entry), flush=True)
                log.write(json.dumps(entry) + "\n")
                log.flush()
        del arm, params, opt
        torch.cuda.empty_cache()
    # Release GPU teacher/anchor before CPU ONNX parity. The final model
    # selection file is deliberately left to the caller after SDK validation.
    del anchor, teacher
    torch.cuda.empty_cache()
    for mode in ("ordinary", "feature", "routing"):
        export_selected(out / f"{mode}.pt", out / f"{mode}-bundle", valid_rows, recipe,
                        recipe["base_bundle"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-root", default="/lab/data")
    args = parser.parse_args()
    run(json.loads(Path(args.recipe).read_text()), args.out, args.data_root)
