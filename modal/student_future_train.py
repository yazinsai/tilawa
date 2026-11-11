"""Future-marginal teacher cache, matched frozen-head refits and diagnostics.

Use the authorized supervisor's training image. Preparation reads train/none
help takes only; fit and checkpoint selection read only their cache. Diagnostics
accept externally enrolled cases (wav, start, stop, phones), never benchmark
identifiers in this code. They do not train, select checkpoints or score warnings.
"""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance
from student_future import (admitted_none, speaker_partition, continuation_pool,
                            input_deadline, output_chunk, marginal_targets,
                            probabilities, disagreement_rows)
from student_future_export import digest, expose, session, stream, insertion_assets, graft

SEED, DRAWS, CHUNKS, EPOCHS, LR = 1729, 4, 8, 12, .02


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Recording path leaves its authorized corpus")
    return path


def read_wave(path):
    import soundfile as sf
    import torch
    import torchaudio
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Empty or nonfinite audio")
    wave = torch.tensor(wave.mean(1))
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    return wave.numpy()


def fbank(wave):
    import torch
    import torchaudio
    return torchaudio.compliance.kaldi.fbank(torch.tensor(wave)[None], num_mel_bins=80,
        sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()


def teacher_rows(runtime, wave):
    # SDK teacherFeatures: recompute whole-crop normalization on every draw.
    import torch
    import torchaudio
    f = torchaudio.compliance.kaldi.fbank(torch.tensor(wave * 32768)[None],
        num_mel_bins=80, sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=8000)
    if len(f) < 2:
        raise ValueError("Teacher needs at least two feature frames")
    f = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=True) + 1e-7)
    odd = len(f) % 2
    if odd:
        f = torch.cat([f, torch.ones_like(f[:1])])
    mask = np.ones((1, len(f) // 2), dtype=np.int64)
    if odd:
        mask[0, -1] = 0
    return runtime.run(["log_probs"], {"input_features": f.numpy().reshape(1, -1, 160),
                                      "attention_mask": mask})[0][0]


def enrollment(help_path):
    path = Path(help_path)
    labels = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    ayahs = {f'{a["s"]}:{a["a"]}': a["p"] for a in reference["ayahs"]}
    rows, seen = [], set()
    for label in admitted_none(labels):
        wav = contained(path.parent, label["wav"])
        checksum = digest(wav)
        if checksum in seen:
            raise ValueError("Duplicate audio could leak across speaker partitions")
        seen.add(checksum)
        phones = "".join(ayahs[a] for a in label["ayahs"])
        target = np.asarray(phoneme_ids(phones), dtype=np.int64)
        if not len(target):
            raise ValueError("None take has no complete canonical transcript")
        rows.append(dict(id=label["id"], speaker=label["speaker"], wav=str(wav),
                         audioSha256=checksum, target=target, wave=read_wave(wav)))
    groups = speaker_partition(r["speaker"] for r in rows)
    return rows, groups


def donors_for(rows, groups, recipient, needed, seed):
    eligible = [r for r in continuation_pool(rows, groups, recipient) if len(r["wave"]) >= needed]
    if len(eligible) < DRAWS:
        raise ValueError("Not enough distinct same-partition real continuations")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(eligible), DRAWS, replace=False)
    waves, provenance = [], []
    for i in chosen:
        r = eligible[int(i)]
        offset = int(rng.integers(0, len(r["wave"]) - needed + 1))
        waves.append(r["wave"][offset:offset + needed])
        provenance.append(dict(id=r["id"], speaker=r["speaker"], offset=offset,
                               length=needed, audioSha256=r["audioSha256"]))
    return waves, provenance


def prepare(args):
    import torch
    torch.set_num_threads(1)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Cache directory must be empty")
    rows, groups = enrollment(args.help)
    auxiliary = insertion_assets(args.model)
    m, feature, width, path = expose(args.model, out)
    if m["tokens"] != TOKENS:
        raise ValueError("Ordered inventory differs from the deployed listener")
    frozen, teacher = session(path), session(Path(args.model) / "teacher.onnx")
    original = session(Path(args.model) / m["file"])
    cached = []
    for i, r in enumerate(rows):
        wave = r["wave"]
        fb = fbank(wave)
        lp, x = stream(frozen, m, fb, feature)
        baseline, _ = stream(original, m, fb)
        np.testing.assert_array_equal(lp, baseline, err_msg="Feature exposure changed student output")
        if len(lp) < len(r["target"]) + int((r["target"][1:] == r["target"][:-1]).sum()):
            raise ValueError("Complete none transcript cannot align on native clock")
        eligible = [s for s in range(0, len(lp), output_chunk(m))
                    if sum(len(d["wave"]) >= max(1, len(wave) - input_deadline(s, m))
                           for d in continuation_pool(rows, groups, r)) >= DRAWS]
        if not eligible:
            raise ValueError("None take lacks independently replaceable futures")
        # Bounded, evenly spaced chunks selected by geometry, never identities.
        selected = [eligible[int(j)] for j in np.linspace(0, len(eligible) - 1,
                                                        min(CHUNKS, len(eligible)), dtype=int)]
        ordinary = probabilities(teacher_rows(teacher, wave))[:len(lp)].astype("float32")
        if len(ordinary) != len(lp) or ordinary.shape[1] != lp.shape[1]:
            raise ValueError("Teacher cannot cover valid student rows")
        marginal, mask, audits = ordinary.copy(), np.zeros(len(lp), dtype=bool), []
        for start in selected:
            deadline = input_deadline(start, m)
            stop = min(len(lp), start + output_chunk(m))
            mask[start:stop] = True
            if deadline >= len(wave):
                continue
            donors, provenance = donors_for(rows, groups, r, len(wave) - deadline, SEED + i * 1000 + start)
            _, changed, audit = marginal_targets(lambda w: teacher_rows(teacher, w), wave, donors,
                len(lp), m, chunks=[start], student_rows=lp,
                infer_student=lambda w: stream(original, m, fbank(w))[0])
            marginal[start:stop] = changed[start:stop]
            audits.append(dict(**audit[0], donors=provenance))
        file = f"recording-{i:05d}.npz"
        np.savez_compressed(out / file, features=x, shipped=lp, target=r["target"],
                            ordinary=ordinary, marginal=marginal, mask=mask, fbank=fb)
        cached.append(dict(id=r["id"], speaker=r["speaker"], audioSha256=r["audioSha256"],
                           file=file, sha256=digest(out / file), audits=audits))
        print(json.dumps(dict(prepared=i + 1, total=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(source=str(Path(args.model).resolve()),
        sourceSha256=m["sha256"], auxiliaryHashes=auxiliary, width=width, groups=groups,
        rows=cached, helpSha256=digest(args.help), draws=DRAWS, maxChunks=CHUNKS))


def phone_head(width, vocab):
    import torch
    head = torch.nn.Linear(width, vocab)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return head


def log_probs(head, v):
    return (v["shipped"].detach() + head(v["features"].detach())).log_softmax(-1)


def objectives(lp, v, arm):
    import torch
    import torch.nn.functional as F
    ctc = F.ctc_loss(lp[:, None], v["target"], torch.tensor([len(lp)]),
                     torch.tensor([len(v["target"])]), reduction="sum") / len(v["target"])
    target = v[arm].detach()
    kd = -(target[v["mask"]] * lp[v["mask"]]).sum(-1).mean()
    shipped = v["shipped"].detach()
    preserve = (shipped.exp() * (shipped - lp)).sum(-1).mean()
    return ctc, kd, preserve


def validate(head, rows):
    import torch
    speakers = {}
    with torch.no_grad():
        for r in rows:
            v = r["data"]
            lp = log_probs(head, v)
            ctc, _, preserve = objectives(lp, v, "ordinary")
            per = edit_distance(ctc_collapse(lp.argmax(-1).tolist()), v["target"].tolist()) / len(v["target"])
            speakers.setdefault(r["speaker"], []).append([float(ctc + preserve), per])
    if not speakers:
        raise ValueError("Selection requires separate training speakers")
    result = np.mean([np.mean(values, axis=0) for values in speakers.values()], axis=0)
    return dict(objective=float(result[0]), per=float(result[1]))


def fit(args):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit directory must be empty")
    info = json.loads((prepared / "prepared.json").read_text())
    from student_future_export import load_source
    m, _, _, width = load_source(info["source"])
    if m["sha256"] != info["sourceSha256"] or insertion_assets(info["source"]) != info["auxiliaryHashes"]:
        raise ValueError("Source bundle changed since preparation")
    groups = speaker_partition(r["speaker"] for r in info["rows"])
    if groups != info["groups"] or width != info["width"]:
        raise ValueError("Cached speaker partition or feature width differs")
    rows = []
    for r in info["rows"]:
        path = contained(prepared, r["file"])
        if digest(path) != r["sha256"]:
            raise ValueError("Cached authorized training data changed")
        with np.load(path, allow_pickle=False) as data:
            v = {k: torch.tensor(data[k]) for k in data.files}
        if v["features"].shape != (len(v["shipped"]), width) or not v["mask"].any():
            raise ValueError("Invalid cached feature clock or supervision mask")
        for arm in ("ordinary", "marginal"):
            if v[arm].shape != v["shipped"].shape or not torch.isfinite(v[arm]).all():
                raise ValueError("Target clock or distribution differs")
            torch.testing.assert_close(v[arm].sum(-1), torch.ones(len(v[arm])))
        for a in r["audits"]:
            for donor in a["donors"]:
                if donor["id"] == r["id"] or groups[donor["speaker"]] != groups[r["speaker"]]:
                    raise ValueError("Continuation provenance leaked across partitions")
        rows.append(dict(r, data=v))
    fit_rows = [r for r in rows if groups[r["speaker"]] == "fit"]
    selection = [r for r in rows if groups[r["speaker"]] == "selection"]
    out.mkdir(parents=True, exist_ok=True)
    report = dict(seed=SEED, epochs=EPOCHS, learningRate=LR, preparedSha256=digest(prepared / "prepared.json"), arms={})
    for arm in ("ordinary", "marginal"):
        head = phone_head(width, len(TOKENS) + 1)
        optimizer = torch.optim.SGD(head.parameters(), lr=LR)
        history = [dict(epoch=0, **validate(head, selection))]
        chosen, best_score, best = 0, history[0]["objective"], copy.deepcopy(head.state_dict())
        for epoch in range(1, EPOCHS + 1):
            order = list(range(len(fit_rows)))
            random.Random(SEED + epoch).shuffle(order)
            for i in order:
                optimizer.zero_grad(set_to_none=True)
                v = fit_rows[i]["data"]
                loss = sum(objectives(log_probs(head, v), v, arm))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite phone-head loss")
                loss.backward()
                if any(not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError("Nonfinite phone-head gradient")
                optimizer.step()
            metrics = dict(epoch=epoch, **validate(head, selection))
            history.append(metrics)
            if metrics["objective"] < best_score:
                chosen, best_score, best = epoch, metrics["objective"], copy.deepcopy(head.state_dict())
            print(json.dumps(dict(arm=arm, **metrics)), flush=True)
        head.load_state_dict(best)
        head.eval()
        torch.save(dict(state=best, epoch=chosen, sourceSha256=m["sha256"]), out / (arm + ".pt"))
        path = graft(info["source"], head, out / arm, arm)
        runtime = session(path)  # Preserve native quantization/execution for parity.
        manifest = json.loads((out / arm / "manifest.json").read_text())
        worst = 0.
        for r in selection:
            v = r["data"]
            got, _ = stream(runtime, manifest, v["fbank"].numpy())
            with torch.no_grad():
                want = log_probs(head, v).numpy()
            np.testing.assert_allclose(np.exp(got), np.exp(want), atol=2e-4, rtol=2e-4)
            worst = max(worst, float(np.abs(np.exp(got) - np.exp(want)).max()))
        report["arms"][arm] = dict(history=history, selectedEpoch=chosen,
            sha256=digest(path), parityMaxProbabilityError=worst, parityRecordings=len(selection))
        write_json(out / "training-report.json", report)


def diagnose(args):
    import torch
    torch.set_num_threads(1)
    from student_future_export import load_source
    m, _, _, _ = load_source(args.model)
    original, teacher = session(Path(args.model) / m["file"]), session(Path(args.model) / "teacher.onnx")
    rows, groups = enrollment(args.help)
    # Diagnostic draws use fit speakers only; no checkpoint is selected here.
    pool = [r for r in rows if groups[r["speaker"]] == "fit"]
    results = []
    for i, case in enumerate(json.loads(Path(args.cases).read_text())):
        wave = read_wave(case["wav"])
        full, _ = stream(original, m, fbank(wave))
        start, stop = int(case["start"] / .04), int(np.ceil(case["stop"] / .04))
        if not 0 <= start < stop <= len(full):
            raise ValueError("Diagnostic case exceeds valid audio frames")
        # Crop on a deployed chunk boundary; retain real student history via
        # full-session replay. Teacher gets two seconds of original warm audio.
        crop_frame = max(0, ((start - 50) // output_chunk(m)) * output_chunk(m))
        crop_sample = crop_frame * 640
        end = min(len(wave), (stop + 75) * 640)
        crop = wave[crop_sample:end]
        frames = min(len(full) - crop_frame, ((len(fbank(crop)) - 7) // 2 + 1) // 2)
        selected = sorted(set((f - crop_frame) // output_chunk(m) * output_chunk(m)
                              for f in range(start, stop)))
        needed = max(1, len(crop) - input_deadline(selected[0], m))
        dummy = dict(id="diagnostic-query", speaker=pool[0]["speaker"])
        donors, provenance = donors_for(pool, groups, dummy, needed, SEED + i)
        def observed(changed):
            altered = wave.copy()
            altered[crop_sample:end] = changed
            return stream(original, m, fbank(altered))[0][crop_frame:crop_frame + frames]
        ordinary, marginal, audit = marginal_targets(lambda w: teacher_rows(teacher, w), crop,
            donors, frames, m, chunks=selected, infer_student=observed,
            student_rows=full[crop_frame:crop_frame + frames])
        a, b = start - crop_frame, stop - crop_frame
        student = probabilities(full[start:stop])
        consonants = {t for t in phoneme_ids(case["phones"]) if t <= 28}
        s_ids = ctc_collapse(student.argmax(-1).tolist())
        t_ids = ctc_collapse(ordinary[a:b].argmax(-1).tolist())
        missing = consonants - set(s_ids)
        resolved = disagreement_rows(student, ordinary[a:b], marginal[a:b], missing)
        result = dict(case=case, sourceSha256=m["sha256"], donorProvenance=provenance,
            cropStartSample=crop_sample, audit=audit, missingPlantedConsonants=sorted(missing),
            teacherRetainedConsonants=sorted(missing & set(t_ids)),
            resolvedRows=[start + f for f in resolved], premiseMet=bool(resolved))
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(args.out, dict(cases=results, premiseIncidents=sum(r["premiseMet"] for r in results),
                             caveat="Correlated same-frame acoustic diagnostic; not recovered warnings or a practice score"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("fit")
    for name in ("prepared", "out"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("diagnose")
    for name in ("model", "help", "cases", "out"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit, "diagnose": diagnose}[args.command](args)
