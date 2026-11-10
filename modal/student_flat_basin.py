"""Matched ordinary/SAM continuations with training-speaker-only selection.

Run directly in the supervisor's training image, without creating a Modal app:

    PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_flat_basin.py audit --out /scratch/.../flat-basin
    PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_flat_basin.py train --out /scratch/.../flat-basin --arm sam-small --rho .001
    PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_flat_basin.py select --out /scratch/.../flat-basin
    PYTHONPATH=modal:tools:$PYTHONPATH python modal/student_flat_basin.py export --out /scratch/.../flat-basin --source-bundle /lab/data/models/<installed-primary-bundle>

Predeclared arms: ordinary (0), sam-small (.001), sam-medium (.003), sam-large
(.01). This utility never reads practice sessions, evidence or release scores.
Selection consumes only losses on reserved groups from the training population.
"""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"ordinary": 0., "sam-small": .001, "sam-medium": .003, "sam-large": .01}
RECIPE = dict(seed=721, steps=1000, eval_every=250, lr=1e-5, preservation=4.,
              max_original=4096, max_validation=512, max_clip_s=30.,
              batch_original=4, batch_none=1, batch_preserve=1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_order(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def reserve_groups(groups):
    """Reserve a fifth of training speakers, keeping every take together."""
    ordered = sorted(set(groups), key=stable_order)
    if len(ordered) < 2:
        raise ValueError("need at least two training speaker groups per source")
    return set(ordered[:max(1, math.ceil(len(ordered) / 5))])


def help_role(label):
    """Only explicit clean training takes have transcripts; scripts never do."""
    if label.get("split") != "train" or label.get("excluded") or label.get("exclude"):
        return None
    if label.get("kind") == "none":
        if label.get("scored") is False or any(label.get(k) for k in ("events", "non_errors", "unsupported")):
            return None
        return "none"
    if label.get("kind"):
        return "preserve"
    return None


def batch_records(records, step):
    rng = random.Random(RECIPE["seed"] + step)
    chosen = []
    for role, count in (("original", RECIPE["batch_original"]),
                        ("none", RECIPE["batch_none"]), ("preserve", RECIPE["batch_preserve"])):
        pool = [r for r in records if r["role"] == role]
        chosen.extend(rng.choices(pool, k=count))
    return chosen


def collect_records():
    """Audit split membership before loading waveforms; fail on missing inputs."""
    import pyarrow.parquet as pq

    files = sorted(Path("/student/s1").glob("*.parquet"))
    if not files:
        raise FileNotFoundError("authorized /student/s1 training recordings are missing")
    excluded = {tuple(line.rsplit(":", 1)) for line in
                (ROOT / "docs/benchmarks/gold-per-v1-exclude.txt").read_text().split()}
    meta = []
    for f in files:
        cols = ["shard", "row", "cluster", "bucket", "split", "n_samples"]
        if not set(cols).issubset(pq.read_schema(f).names):
            raise ValueError(f"{f}: training-speaker split provenance is unavailable")
        for index, r in enumerate(pq.read_table(f, columns=cols).to_pylist()):
            r.update(file=str(f), index=index, group=f"tlog/{r['cluster']}",
                     key=f"{r['shard']}:{r['row']}")
            meta.append(r)
    # Exclude whole held-out and gold speaker clusters, including aliases in
    # other shards, rather than removing only their labelled clips.
    banned = {r["group"] for r in meta if r["split"] != "train" or
              (r["shard"], str(r["row"])) in excluded}
    ordinary = [dict(r, role="original") for r in meta if r["split"] == "train"
                and r["bucket"] == "clean-match" and r["group"] not in banned
                and r["cluster"] is not None and str(r["cluster"]) != "-1"
                and 0 < r["n_samples"] <= RECIPE["max_clip_s"] * 16000]
    held_original = reserve_groups(r["group"] for r in ordinary)

    help_root = Path(os.environ["LAB_DATA"]) / "help-train"
    labels = [json.loads(line) for line in (help_root / "labels.jsonl").read_text().splitlines() if line.strip()]
    banned_help = {str(l["speaker"]) for l in labels if l.get("split") != "train"}
    help_records = []
    import soundfile as sf
    for label in labels:
        role = help_role(label)
        if role is None or str(label["speaker"]) in banned_help:
            continue
        wav = (help_root / label["wav"]).resolve()
        if not wav.is_relative_to(help_root.resolve()) or not wav.is_file():
            raise FileNotFoundError(f"missing or invalid authorized help-train waveform: {label['wav']}")
        info = sf.info(wav)
        if not 0 < info.duration <= RECIPE["max_clip_s"]:
            continue
        r = dict(key=f"help/{label['id']}", group=f"help/{label['speaker']}",
                 role=role, file=str(wav))
        if role == "none":
            # Store canonical ayahs only for the none-labelled takes. Do not
            # even copy a mistake's scripted expected/observed words.
            r["ayahs"] = label["ayahs"]
        help_records.append(r)
    held_help = reserve_groups(r["group"] for r in help_records)
    reserved = held_original | held_help
    all_records = ordinary + help_records
    train, validation = [], []
    for role in ("original", "none", "preserve"):
        rows = sorted([r for r in all_records if r["role"] == role], key=lambda r: stable_order(r["key"]))
        a = [r for r in rows if r["group"] not in reserved]
        b = [r for r in rows if r["group"] in reserved]
        if role == "original":
            a, b = a[:RECIPE["max_original"]], b[:RECIPE["max_validation"]]
        if not a or not b:
            raise ValueError(f"no {role} recordings in a required training-speaker partition")
        train.extend(a)
        validation.extend(b)
    labels_sha = digest(help_root / "labels.jsonl")
    return dict(train=train, validation=validation, reserved=sorted(reserved),
                excluded_groups=sorted(banned | {f"help/{s}" for s in banned_help}),
                help_labels_sha256=labels_sha,
                limits="Tlog speaker identity is an approximate acoustic cluster, not independently verified person identity.")


def prepare(out):
    """Persist one shared plan; each arm must reproduce it before continuing."""
    checkpoint = Path("/student/prior-runs/spliced-clean/ckpt.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError("exact shipped spliced-clean source checkpoint is unavailable")
    records = collect_records()
    plan = dict(schema=1, recipe=RECIPE, arms=ARMS, checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint), **records)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("shared training plan changed; use a new output directory")
    else:
        import uuid

        # Concurrent arms must never observe a partially written shared plan.
        temp = path.with_name(f".plan-{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(plan, ensure_ascii=False, indent=1))
        os.replace(temp, path)
    print(json.dumps({k: {role: sum(r['role'] == role for r in plan[k])
                          for role in ('original', 'none', 'preserve')}
                      for k in ('train', 'validation')}, indent=1), flush=True)
    return plan


class Recordings:
    def __init__(self, records, cfg, aug):
        self.records, self.cfg, self.aug = records, cfg, aug
        self.cache = {}
        self.tables = {}
        ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
        self.ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in ref["ayahs"]}

    def raw(self, r):
        import pyarrow.parquet as pq
        import soundfile as sf
        import torch
        import torchaudio
        from student import spliced_target

        if r["key"] in self.cache:
            return self.cache[r["key"]]
        if r["role"] == "original":
            cols = ["flac", "ref", "hyp", "top5_ids", "top5_p", "n_frames"]
            if r["file"] not in self.tables:
                rows = [t["index"] for t in self.records if t["role"] == "original" and t["file"] == r["file"]]
                self.tables[r["file"]] = dict(zip(rows, pq.read_table(r["file"], columns=cols).take(rows).to_pylist()))
            row = self.tables[r["file"]][r["index"]]
            audio, sr = sf.read(io.BytesIO(row["flac"]), dtype="float32")
            target = spliced_target(row["ref"], row["hyp"])
            teacher = (row["top5_ids"], row["top5_p"], row["n_frames"])
        else:
            audio, sr = sf.read(r["file"], dtype="float32")
            target = "".join(self.ayahs[a] for a in r["ayahs"]) if r["role"] == "none" else None
            teacher = None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
        if len(audio) == 0 or len(audio) > RECIPE["max_clip_s"] * 16000:
            raise ValueError(f"{r['key']}: recording outside the declared duration cap")
        self.cache[r["key"]] = (audio, target, teacher)
        return self.cache[r["key"]]

    def item(self, r, seed, training):
        import numpy as np
        import torch
        import torchaudio
        from student import mix_noise, phoneme_ids
        from student_train import pick, reverb, spec_augment

        audio, target, teacher = self.raw(r)
        wave = audio.copy()
        rng = np.random.default_rng(seed)
        if training:
            if rng.random() < self.cfg["rir_p"]:
                wave = reverb(wave, pick(self.aug["rir"], rng))
            if rng.random() < self.cfg["noise_p"]:
                wave = mix_noise(wave, pick(self.aug["noise"], rng), rng.uniform(*self.cfg["snr_db"]), rng)
            wave *= 10 ** (rng.uniform(-10, 6) / 20)
        feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wave))[None],
            num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
        if training and self.cfg["spec_aug"]:
            feats = spec_augment(feats, rng)
        return dict(record=r, feats=feats, target=phoneme_ids(target) if target is not None else None, teacher=teacher)


def acoustic_batch(items, device):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    x = pad_sequence([i["feats"] for i in items], batch_first=True, padding_value=-23.0259).to(device)
    lens = torch.tensor([len(i["feats"]) for i in items], device=device)
    return x, lens


def objective(model, anchor, items, cfg, device, anchor_rows=None):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss

    x, lens = acoustic_batch(items, device)
    if anchor_rows is None:
        with torch.no_grad():
            anchor_rows = anchor(x, lens)
    lp, out_lens = model(x, lens)
    ref, ref_lens = anchor_rows
    if not torch.equal(out_lens, ref_lens):
        raise ValueError("preservation anchor and candidate have different frame geometry")
    rows = []
    for j, item in enumerate(items):
        if item["record"]["role"] == "preserve" and item["target"] is not None:
            raise ValueError("a scripted mistake must not become a verified transcript target")
        n = int(out_lens[j])
        p, q = lp[j, :n].float(), ref[j, :n].float()
        kl = F.kl_div(p, q, log_target=True, reduction="sum") / max(1, n)
        ctc, kd = p.sum() * 0, p.sum() * 0
        if item["target"] is not None:
            ctc = F.ctc_loss(p[:, None], torch.tensor(item["target"], device=device),
                torch.tensor([n]), torch.tensor([len(item["target"])]),
                blank=0, reduction="sum", zero_infinity=False) / max(1, n)
        if item["teacher"] is not None and cfg["kd_weight"]:
            ids, probs, frames = item["teacher"]
            ids = torch.from_numpy(np.frombuffer(ids, np.uint8).copy().reshape(frames, 5).astype(np.int64)).to(device)
            probs = torch.from_numpy(np.frombuffer(probs, np.float16).copy().reshape(frames, 5).astype(np.float32)).to(device)
            ix = (torch.arange(n, device=device) - cfg["kd_delay"]).clamp(0, frames - 1)
            kd = soft_target_loss(p[None], ids[ix][None], probs[ix][None], torch.ones(1, n, device=device, dtype=torch.bool))
        rows.append(dict(ctc=ctc, kd=kd, preservation=kl,
                         loss=cfg["ctc_weight"] * ctc + cfg["kd_weight"] * kd + RECIPE["preservation"] * kl))
    return {k: torch.stack([r[k] for r in rows]).mean() for k in rows[0]}, anchor_rows


def validate(model, anchor, recordings, cfg, device):
    import torch

    model.eval()
    sums = {role: [] for role in ("original", "none", "preserve")}
    with torch.no_grad():
        for i, r in enumerate(recordings.records):
            for augmented in (False, True):
                item = recordings.item(r, RECIPE["seed"] + i, training=augmented)
                values, _ = objective(model, anchor, [item], cfg, device)
                sums[r["role"]].append(float(values["loss"]))
    means = {role: sum(values) / len(values) for role, values in sums.items()}
    # Exactly the role proportions of the shared six-recording update.
    counts = {"original": RECIPE["batch_original"], "none": RECIPE["batch_none"],
              "preserve": RECIPE["batch_preserve"]}
    score = sum(means[r] * counts[r] for r in counts) / sum(counts.values())
    if not math.isfinite(score):
        raise FloatingPointError("nonfinite training-speaker validation score")
    model.train()
    return dict(score=score, by_role=means)


def train(out, arm, rho):
    import numpy as np
    import torch
    from student_train import DEFAULTS, build_model
    from student_sam import sharpness_step

    if arm not in ARMS or rho != ARMS[arm]:
        raise ValueError("arm/perturbation scale differs from the preregistered training grid")
    plan = prepare(out)
    ck = torch.load(plan["checkpoint"], map_location="cpu", weights_only=False)
    cfg = {**DEFAULTS, **ck["cfg"]}
    if cfg["target"] != "spliced":
        raise ValueError("source checkpoint does not use the shipped spliced transcript objective")
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("rir", "noise")}
    torch.set_num_threads(8)
    torch.manual_seed(RECIPE["seed"])
    random.seed(RECIPE["seed"])
    device = torch.device("cuda")
    model, anchor = build_model(cfg["size"]).to(device), build_model(cfg["size"]).to(device)
    for m in (model, anchor):
        m.load_state_dict(ck["model"], strict=True)
        m.streaming_eval(16, 128)
        for layer in m.modules():
            layer.batch_count = ck["state"]["step"]
    anchor.eval().requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"], weight_decay=0)
    directory = out / arm
    directory.mkdir(exist_ok=True)
    state, best = 0, math.inf
    resume = directory / "latest.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["plan_sha256"] != digest(out / "plan.json"):
            raise ValueError("checkpoint belongs to another data plan")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        state, best = saved["state"]["step"], saved["best"]
    data = Recordings(plan["train"], cfg, aug)
    validation = Recordings(plan["validation"], cfg, aug)
    def save(step, name, metrics=None):
        target = directory / name
        payload = dict(model=model.state_dict(), opt=opt.state_dict(), cfg=cfg,
            state=dict(step=step), best=best, arm=arm, rho=rho, validation=metrics,
            plan_sha256=digest(out / "plan.json"))
        torch.save(payload, str(target) + ".tmp")
        os.replace(str(target) + ".tmp", target)

    started = time.monotonic()
    with (directory / "log.jsonl").open("a") as log:
        for step in range(state, RECIPE["steps"]):
            # All arms draw the same items and augmentation, independently of
            # how many forwards/backwards their optimizer uses or where a run resumes.
            seed = RECIPE["seed"] + step
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            chosen = batch_records(data.records, step)
            items = [data.item(r, seed * 10 + j, training=True) for j, r in enumerate(chosen)]
            # The frozen shipped outputs are computed once on the very same
            # augmented input and reused under the adverse perturbation.
            with torch.no_grad():
                anchor_rows = anchor(*acoustic_batch(items, device))
            values = sharpness_step(model, opt,
                lambda: objective(model, anchor, items, cfg, device, anchor_rows)[0], rho)
            record = dict(step=step + 1, **{k: float(v) for k, v in values.items()})
            if (step + 1) % RECIPE["eval_every"] == 0:
                metrics = validate(model, anchor, validation, cfg, device)
                record["validation"] = metrics
                if metrics["score"] < best:
                    best = metrics["score"]
                    save(step + 1, "best.pt", metrics)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step + 1) % 25 == 0:
                print(json.dumps(record), flush=True)
                save(step + 1, "latest.pt")
            if time.monotonic() - started > 50 * 60:
                save(step + 1, "latest.pt")
                print("Saved continuation; rerun the identical command to finish.", flush=True)
                return


def select(out):
    """Freeze both best continuations before any target/session replay."""
    import torch

    rows = []
    plan_sha = digest(out / "plan.json")
    for arm, rho in ARMS.items():
        latest = torch.load(out / arm / "latest.pt", map_location="cpu", weights_only=False)
        if latest["state"]["step"] != RECIPE["steps"]:
            raise ValueError(f"{arm}: matched continuation is incomplete")
        path = out / arm / "best.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["plan_sha256"] != plan_sha or ck["rho"] != rho:
            raise ValueError(f"{arm}: selection provenance mismatch")
        rows.append(dict(arm=arm, path=str(path), sha256=digest(path), step=ck["state"]["step"],
                         score=ck["validation"]["score"]))
    selected = min((r for r in rows if r["arm"] != "ordinary"), key=lambda r: (r["score"], r["arm"]))
    result = dict(schema=1, plan_sha256=plan_sha, candidate=selected,
                  control=next(r for r in rows if r["arm"] == "ordinary"), all_arms=rows)
    path = out / "selection.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("selection is frozen; a later replay cannot change it")
    path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def check_geometry(manifest, source):
    """Model replacement must preserve the SDK's inventory and streaming ABI."""
    for key in ("arch", "inventory", "tokens", "blank", "sampleRate", "frameMs", "fbank",
                "window", "shift", "chunk", "leftContext", "states", "supported"):
        if manifest.get(key) != source.get(key):
            raise ValueError(f"export changes the shipped {key}")


def export_selected(out, source_bundle):
    """Export and verify candidate AND matched control; never select on replays."""
    import re
    import shutil
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import torch
    from student_train import build_model, DEFAULTS
    import student_export as E

    selection = json.loads((out / "selection.json").read_text())
    if selection["plan_sha256"] != digest(out / "plan.json"):
        raise ValueError("frozen training selection does not match the shared plan")
    source = json.loads((source_bundle / "manifest.json").read_text())
    if digest(source_bundle / source["file"]) != source["sha256"]:
        raise ValueError("installed primary bundle checksum mismatch")
    # Preserve the registry's existing teacher files with no retraining. Missing
    # inherited files must block export instead of silently disabling folded rules.
    registry = (ROOT / "packages/sdk/src/registry.ts").read_text()
    auxiliary = re.findall(r'file:\s*"([^"]+)"\s*,\s*bytes:\s*(\d+)\s*,\s*sha256:\s*"([0-9a-f]{64})"', registry)
    for file, size, sha in auxiliary:
        path = source_bundle / file
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            raise ValueError(f"inherited SDK asset is unavailable or corrupt: {file}")
    plan = json.loads((out / "plan.json").read_text())
    torch.set_num_threads(8)
    reports = {}
    for role in ("candidate", "control"):
        frozen = selection[role]
        if digest(frozen["path"]) != frozen["sha256"]:
            raise ValueError("training-selected checkpoint changed")
        ck = torch.load(frozen["path"], map_location="cpu", weights_only=False)
        model = build_model(ck["cfg"]["size"])
        model.load_state_dict(ck["model"], strict=True)
        model.eval()
        directory = out / f"{role}-int8"
        directory.mkdir(exist_ok=True)
        fp32 = directory / "float.onnx"
        path = directory / "model.onnx"
        states = E.export_onnx(model, str(fp32), source["chunk"], source["leftContext"])
        quantize_dynamic(str(fp32), str(path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
        m = E.manifest(frozen["arm"], states, source["chunk"], source["leftContext"], str(path), ck)
        m.update(format="onnx-int8", quantizedFrom=digest(fp32))
        check_geometry(m, source)
        onnx.checker.check_model(str(path))
        runtime = ort.InferenceSession(str(path), sess_options=ort.SessionOptions(), providers=["CPUExecutionProvider"])
        inputs = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in states}
        inputs["x"] = np.full((1, m["window"], 80), E.LOG_EPS, dtype=np.float32)
        result = runtime.run(None, inputs)
        if result[0].shape != (1, m["chunk"] // 2, len(m["tokens"]) + 1) or not np.isfinite(result[0]).all():
            raise ValueError("exported primary has invalid CTC output")
        for state, value in zip(states, result[1:]):
            if list(value.shape) != state["shape"] or not np.isfinite(value).all():
                raise ValueError("exported primary has invalid streaming state")
        # Training validation clips also verify complete multi-window execution;
        # int8 versus float drift is reported, never used to choose a checkpoint.
        data = Recordings(plan["validation"], {**DEFAULTS, **ck["cfg"]}, None)
        items = [r for r in data.records if r["role"] == "none"][:8]
        parity = []
        for i, r in enumerate(items):
            feats = data.item(r, RECIPE["seed"] + i, False)["feats"]
            a = E.stream_onnx(str(fp32), feats, states, source["chunk"])
            b = E.stream_onnx(str(path), feats, states, source["chunk"])
            if a.shape != b.shape or not np.isfinite(b).all():
                raise ValueError("multi-window int8 execution failed")
            parity.append(dict(key=r["key"], frames=len(b), top1_agree=float((a.argmax(-1) == b.argmax(-1)).mean()),
                               max_probability_difference=float(np.abs(np.exp(a) - np.exp(b)).max())))
        for file, _, _ in auxiliary:
            shutil.copyfile(source_bundle / file, directory / file)
        (directory / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
        report = dict(model_sha256=digest(path), checkpoint_sha256=frozen["sha256"],
                      unchanged_geometry=True, int8_parity=parity,
                      inherited_assets={file: digest(directory / file) for file, _, _ in auxiliary})
        (directory / "verification.json").write_text(json.dumps(report, indent=1))
        reports[role] = report
    (out / "export-verification.json").write_text(json.dumps(reports, indent=1))
    print(json.dumps(reports, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("audit", "train", "select", "export"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--rho", type=float)
    ap.add_argument("--source-bundle", type=Path)
    args = ap.parse_args()
    if args.command == "audit":
        prepare(args.out)
    elif args.command == "train":
        train(args.out, args.arm, args.rho)
    elif args.command == "select":
        select(args.out)
    else:
        if args.source_bundle is None:
            ap.error("export requires the complete installed --source-bundle")
        export_selected(args.out, args.source_bundle)


if __name__ == "__main__":
    main()
