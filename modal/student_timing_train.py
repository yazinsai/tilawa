"""Training-only preparation, group selection and primary phone-head export.

Run inside the supplied supervisor's training image, never as a new Modal app:

  python modal/student_timing_train.py prepare --model /lab/data/models/spliced-clean-int8 \
    --help-labels /lab/data/help-train/labels.jsonl --out <experiment-scratch>/prepared
  python modal/student_timing_train.py fit --prepared <...>/prepared --out <...>/trained

An optional --transcribed JSONL supplies independently transcribed authorized
recordings with speaker, split=train, verified=true, authorized=true, phones,
wav, and sha256. A script is never a performed transcript. The preparation
command consumes training inputs only; it has no practice-session interface.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import ctc_collapse, edit_distance, mix_noise, phoneme_ids, TOKENS
from student_timing_head import heads, objective, partition, phone_log_probs
from student_timing_export import digest, expose, graft, insertion_assets, session, stream

SEED, EPOCHS, FOLDS = 1729, 18, 3
CHECKPOINTS = (6, 12, 18)
PRESERVATION = 1.
ARMS = ("separated", "shared")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def admitted_help(labels):
    """Only full none-labelled takes from consistently training-only speakers."""
    excluded = {r.get("speaker") for r in labels if r.get("split") != "train"}
    return [r for r in labels if r.get("speaker") and r["speaker"] not in excluded
            and r.get("split") == "train" and r.get("kind") == "none"
            and not any(r.get(k) for k in ("events", "non_errors", "unsupported", "exclude", "excluded"))
            and r.get("scored", True) is not False]


def features(wave, augmentation, rng):
    """Same RIR/noise/gain/SpecAugment recipe as student_train. No new channel.

    Original views receive no augmentation. Banks must be the supplied shipped
    training banks; their hashes are recorded before extraction.
    """
    import torchaudio
    if augmentation is not None:
        from scipy.signal import fftconvolve
        def pick(bank):
            i = int(rng.integers(len(bank["offsets"]) - 1))
            return bank["data"][bank["offsets"][i]:bank["offsets"][i + 1]].astype("float32")
        if rng.random() < .2:
            rir = pick(augmentation["rir"])
            altered = fftconvolve(wave, rir[int(np.argmax(np.abs(rir))):])[:len(wave)]
            wave = (altered * (np.sqrt(np.mean(wave ** 2)) /
                               max(1e-8, np.sqrt(np.mean(altered ** 2))))).astype("float32")
        if rng.random() < .5:
            wave = mix_noise(wave, pick(augmentation["noise"]), rng.uniform(5., 25.), rng)
        wave = wave * 10 ** (rng.uniform(-10., 6.) / 20)
    f = torchaudio.compliance.kaldi.fbank(torch.tensor(wave).unsqueeze(0), num_mel_bins=80,
            sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400)
    if augmentation is not None:
        for _ in range(2):
            width = int(rng.integers(0, 28))
            start = int(rng.integers(0, 80 - width + 1))
            f[:, start:start + width] = 0
        for _ in range(int(rng.integers(0, 11))):
            width = int(rng.integers(0, max(1, int(len(f) * .05)) + 1))
            start = int(rng.integers(0, max(1, len(f) - width)))
            f[start:start + width] = 0
    return f.numpy()


def prepare(args):
    import soundfile as sf
    import torchaudio
    from student_timing_export import load_source
    started = time.time()
    deadline = time.monotonic() + 24 * 60 * 60
    torch.set_num_threads(1)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    source = Path(args.model).resolve()
    m, _, _, _ = load_source(source)
    auxiliary_hashes = insertion_assets(source)
    if m["tokens"] != TOKENS:
        raise ValueError("Ordered phone alphabet differs from the shipped training inventory")
    exclusion_list = json.loads(Path(args.exclusions).read_text()) if args.exclusions else []
    if not isinstance(exclusion_list, list) or any(not isinstance(s, str) or not s for s in exclusion_list):
        raise ValueError("Exclusions must be an authorized list of speaker IDs")
    excluded = set(exclusion_list)
    labels = [json.loads(s) for s in Path(args.help_labels).read_text().splitlines() if s]
    excluded.update(r.get("speaker") for r in labels if r.get("split") != "train")
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    recordings = []
    for r in admitted_help(labels):
        if r["speaker"] in excluded:
            continue
        if not r.get("wav") or not r.get("ayahs"):
            raise ValueError("None take lacks WAV or complete canonical ayahs")
        recordings.append(dict(speaker=r["speaker"], phones="".join(ayahs[a] for a in r["ayahs"]),
                         wav=str((Path(args.help_labels).parent / r["wav"]).resolve()), source="help-train-none"))
    if args.transcribed:
        for line in Path(args.transcribed).read_text().splitlines():
            if not line: continue
            r = json.loads(line)
            if (r.get("split") != "train" or r.get("verified") is not True or
                    r.get("authorized") is not True or not r.get("speaker") or r["speaker"] in excluded):
                raise ValueError("Unverified, unauthorized or excluded additional transcript")
            path = (Path(args.transcribed).parent / r["wav"]).resolve()
            if not r.get("sha256") or digest(path) != r["sha256"]:
                raise ValueError("Additional transcript's audio identity is not verified")
            recordings.append(dict(speaker=r["speaker"], phones=r["phones"], wav=str(path), source="authorized-transcribed"))
    groups = partition([r["speaker"] for r in recordings], FOLDS)
    bank_dir = Path(args.augmentation) if args.augmentation else None
    banks = {k: np.load(bank_dir / (k + ".npz")) for k in ("noise", "rir")} if bank_dir else {}
    for bank in banks.values():
        if (bank["data"].ndim != 1 or not np.isfinite(bank["data"]).all() or
                bank["offsets"].ndim != 1 or len(bank["offsets"]) < 2 or
                bank["offsets"][0] != 0 or bank["offsets"][-1] != len(bank["data"]) or
                not np.issubdtype(bank["offsets"].dtype, np.integer) or
                np.any(np.diff(bank["offsets"]) <= 0)):
            raise ValueError("Invalid shipped augmentation bank")
    bank_hashes = {k: digest(bank_dir / (k + ".npz")) for k in banks}
    m, feature, width, extractor = expose(source, out)
    original, frozen = session(source / m["file"]), session(extractor)
    rng = np.random.default_rng(SEED)
    # Adding the feature output must leave every primary/cache output unchanged.
    names = [v.name for v in original.get_outputs()]
    feeds = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in m["states"]}
    for _ in range(8):
        feeds["x"] = rng.normal(-10., 2., (1, m["window"], 80)).astype("float32")
        a, b = original.run(names, feeds), frozen.run(names, feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_array_equal(x, y, err_msg="Frozen extraction changed " + name)
        feeds.update({s["name"]: b[names.index("new_" + s["name"])] for s in m["states"]})
    rows, seen = [], set()
    for i, r in enumerate(recordings):
        if time.monotonic() > deadline:
            raise TimeoutError("One-day preparation/refit budget exhausted")
        audio_hash = digest(r["wav"])
        if audio_hash in seen:
            raise ValueError("Duplicate training audio would leak across groups or weight the loss twice")
        seen.add(audio_hash)
        target = np.asarray(phoneme_ids(r["phones"]), dtype=np.int64)
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or non-finite training recording")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=audio_hash,
                   transcriptSha256=hashlib.sha256(r["phones"].encode()).hexdigest(), views=[])
        # Full natural recordings include their real leading/trailing/internal pauses.
        # When supplied, shipped-bank draws are cached once for BOTH arms.
        for view in range(3 if banks else 1):
            f = features(wave.numpy(), banks if view else None, np.random.default_rng(SEED + i * 3 + view))
            lp, x = stream(frozen, m, f, feature)
            minimum = len(target) + sum(a == b for a, b in zip(target, target[1:]))
            if not len(target) or len(lp) < minimum:
                raise ValueError("None-labelled full transcript cannot align to its recording")
            file = f"recording-{i:05d}-view-{view}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, target=target, fbank=f)
            row["views"].append(file)
        row["viewHashes"] = [digest(out / file) for file in row["views"]]
        rows.append(row)
        print(json.dumps(dict(prepared=i + 1, total=len(recordings))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, budgetStartedAt=started, source=str(source), sourceSha256=m["sha256"],
        tokens=m["tokens"], width=width, rows=rows, groups=groups, excludedSpeakers=sorted(s for s in excluded if s),
        augmentationBankHashes=bank_hashes, insertionAssetHashes=auxiliary_hashes,
        featureOutput=feature, parityWindows=8))


def normalization(rows):
    # Fit only to original views of the training side of each speaker partition.
    raw = torch.cat([r["views"][0]["features"] for r in rows])
    return raw.mean(0), raw.std(0, unbiased=False).clamp_min(.01)


def fit_pair(rows, width, vocab, epochs, on_checkpoint, deadline):
    pair = heads(width, vocab)
    center, scale = normalization(rows)
    optimizers = {arm: torch.optim.AdamW(h.parameters(), lr=.001, weight_decay=.01)
                  for arm, h in pair.items()}
    rng = random.Random(SEED)
    updates = 0
    for epoch in range(1, epochs + 1):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i in order:
            if time.monotonic() > deadline:
                raise TimeoutError("One-day preparation/refit budget exhausted")
            for arm, head in pair.items():
                optimizers[arm].zero_grad()
                values = []
                for view in rows[i]["views"]:
                    x, shipped, target = [view[k] for k in ("features", "shipped", "target")]
                    lp = phone_log_probs(head, x, shipped, center, scale)
                    values.append(objective(lp, target, shipped, PRESERVATION))
                value = torch.stack(values).mean()
                if not torch.isfinite(value):
                    raise ValueError("Non-finite head training loss; neither arm may skip an update")
                value.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                optimizers[arm].step()
            updates += 1
        if epoch in CHECKPOINTS and on_checkpoint:
            on_checkpoint(epoch, pair, center, scale, updates)
    return pair, center, scale, updates


def validate(head, rows, center, scale):
    speakers = {}
    with torch.no_grad():
        for r in rows:
            # Clean held-out takes measure ordinary-reading preservation. Augmented
            # held-out views measure discrimination on the identical channel recipe.
            for view in r["views"]:
                lp = phone_log_probs(head, view["features"], view["shipped"], center, scale)
                hyp = ctc_collapse(lp.argmax(-1).tolist())
                per = edit_distance(hyp, view["target"].tolist()) / len(view["target"])
                anchor = view["shipped"].log_softmax(-1)
                kl = float((anchor.exp() * (anchor - lp)).sum(-1).mean())
                speakers.setdefault(r["speaker"], []).append(dict(per=per, kl=kl))
    values = [{k: float(np.mean([v[k] for v in rows])) for k in ("per", "kl")}
              for rows in speakers.values()]
    return dict(speakers=len(values), per=float(np.mean([v["per"] for v in values])),
                kl=float(np.mean([v["kl"] for v in values])))


def fit(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    info = json.loads((prepared / "prepared.json").read_text())
    remaining = 24 * 60 * 60 - (time.time() - info["budgetStartedAt"])
    if not 0 < remaining <= 24 * 60 * 60:
        raise TimeoutError("One-day preparation/refit budget exhausted")
    deadline = time.monotonic() + remaining
    if out.exists() and any(out.iterdir()):
        raise ValueError("Training/export directory must be empty")
    from student_timing_export import load_source
    source_manifest, _, _, width = load_source(info["source"])
    if source_manifest["sha256"] != info["sourceSha256"] or width != info["width"]:
        raise ValueError("Shipped graph changed after preparation")
    if insertion_assets(info["source"]) != info["insertionAssetHashes"]:
        raise ValueError("Shipped insertion assets changed after preparation")
    if info["tokens"] != TOKENS:
        raise ValueError("Prepared ordered alphabet mismatch")
    rows = []
    for r in info["rows"]:
        views = []
        if not r["views"] or len(r["views"]) != len(r["viewHashes"]):
            raise ValueError("Prepared recording lacks its identical cached views")
        for file, checksum in zip(r["views"], r["viewHashes"]):
            if digest(prepared / file) != checksum:
                raise ValueError("Prepared training data changed after extraction")
            data = np.load(prepared / file, allow_pickle=False)
            if (data["shipped"].ndim != 2 or data["shipped"].shape[1] != len(TOKENS) + 1 or
                    data["features"].shape != (len(data["shipped"]), info["width"])):
                raise ValueError("Feature/posterior clock mismatch")
            if any(not np.isfinite(data[k]).all() for k in ("features", "shipped", "fbank")):
                raise ValueError("Non-finite prepared training data")
            if (data["target"].ndim != 1 or data["target"].dtype != np.int64 or
                    not len(data["target"]) or np.any(data["target"] < 1) or
                    np.any(data["target"] > len(TOKENS))):
                raise ValueError("Invalid prepared complete phone transcript")
            views.append({k: torch.tensor(data[k]) for k in ("features", "shipped", "target")})
        if any(not torch.equal(views[0]["target"], v["target"]) for v in views[1:]):
            raise ValueError("Augmented views changed their performed transcript")
        rows.append(dict(r, views=views))
    groups = partition([r["speaker"] for r in rows], FOLDS)
    if groups != info["groups"] or set(groups) & set(info["excludedSpeakers"]):
        raise ValueError("Prepared speaker partition/provenance mismatch")
    # CPU deterministic CTC; no CUDA atomics. One shared checkpoint epoch is
    # selected jointly, so both final arms have identical optimizer update counts.
    selection = {epoch: [] for epoch in CHECKPOINTS}
    out.mkdir(parents=True, exist_ok=True)
    for fold in range(FOLDS):
        training = [r for r in rows if groups[r["speaker"]] != fold]
        held = [r for r in rows if groups[r["speaker"]] == fold]
        def record(epoch, pair, center, scale, updates):
            metrics = {arm: validate(head, held, center, scale) for arm, head in pair.items()}
            entry = dict(fold=fold, updatesPerArm=updates, arms=metrics)
            selection[epoch].append(entry)
            print(json.dumps(dict(epoch=epoch, validation=entry)), flush=True)
        fit_pair(training, info["width"], len(TOKENS) + 1, EPOCHS, record, deadline)
    candidates = []
    for epoch, folds in selection.items():
        count = sum(v["arms"][ARMS[0]]["speakers"] for v in folds)
        score = sum((m["per"] + m["kl"]) * m["speakers"]
                    for v in folds for m in v["arms"].values()) / (2 * count)
        candidates.append(dict(epoch=epoch, folds=folds, score=score))
    selected = min(candidates, key=lambda v: (v["score"], v["epoch"]))["epoch"]
    pair, center, scale, updates = fit_pair(rows, info["width"], len(TOKENS) + 1, selected, None, deadline)
    report = dict(schema=1, maxEpochs=EPOCHS, selectedEpoch=selected, seed=SEED,
                  preservation=PRESERVATION, groups=groups, checkpointSelection=candidates,
                  updatesPerArm=updates, arms={}, sourceSha256=info["sourceSha256"],
                  augmentationBankHashes=info["augmentationBankHashes"])
    for arm, head in pair.items():
        path = graft(info["source"], head, center.numpy(), scale.numpy(), out / arm, arm)
        if insertion_assets(out / arm) != info["insertionAssetHashes"]:
            raise ValueError("Export changed the shipped insertion assets")
        torch.save(dict(state=head.state_dict(), center=center, scale=scale,
                        epoch=selected, updates=updates), out / (arm + ".pt"))
        exported = session(path, optimize=True)
        new_m = json.loads((out / arm / "manifest.json").read_text())
        worst = 0.
        # Compare every cached full recording/view to its ACTUAL primary graph.
        # This uses carried streaming state, normalization and end-of-stream drain.
        for recording, record in zip(rows, info["rows"]):
            for view, file in zip(recording["views"], record["views"]):
                data = np.load(prepared / file, allow_pickle=False)
                got, _ = stream(exported, new_m, data["fbank"])
                with torch.no_grad():
                    want = phone_log_probs(head, view["features"], view["shipped"], center, scale).numpy()
                if not np.isfinite(got).all() or got.shape != want.shape:
                    raise ValueError("Exported primary frame/alphabet contract mismatch")
                worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
                np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(np.exp(got).sum(-1), 1., atol=2e-5)
        report["arms"][arm] = dict(parameters=sum(p.numel() for p in head.parameters()),
            parityMaxProbabilityError=worst, parityViews=sum(len(r["views"]) for r in rows),
            sha256=digest(path))
    if report["arms"]["separated"]["parameters"] != report["arms"]["shared"]["parameters"]:
        raise ValueError("Control parameter count mismatch")
    write_json(out / "training-report.json", report)
    print(json.dumps(report), flush=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help-labels", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--transcribed")
    p.add_argument("--augmentation", help="Optional supplied shipped noise/RIR bank directory")
    p.add_argument("--exclusions", help="Optional supplied additional excluded-speaker JSON list")
    p = commands.add_parser("fit")
    p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = arguments()
    (prepare if args.command == "prepare" else fit)(args)
