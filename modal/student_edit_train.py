"""Training-only preparation, speaker-cross-fitted edits, and matched CTC control.

Run prepare/fit/verify through the supplied supervisor. Preparation requires
enrolled, authorized performed transcripts, help-train none takes and recorded
pause negatives. Scripted mistakes, practice evidence, and teacher guesses are
never enrollment substitutes. Each arm receives identical fixed acoustic views
and an equal wall-clock compute allocation on the same machine; the direct CTC
control may perform more updates because its sequence loss is cheaper.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from student_edit_data import recordings, augmented, fbank, contained, write_json
from student_edit_native import load_source, expose, session, head_parameters, digest, insertion_assets
from student_edit_export import graft
from student_edit_transducer import (COPY, CrossFittedCorruptions, build_head, decode,
    marginal_log_probability, source_events, speaker_folds, valid_actions)
from student import TOKENS, phoneme_ids, edit_distance

SEED = 1729


def timed_target(log_probs, target):
    """CTC Viterbi alignment for acoustic enrollment, not the edit loss.

    Assign target occurrences to chunk clocks. Equivalent edit scripts remain
    marginalized within those chunks; this initial transcript timing does not
    provide the model with expected text at inference.
    """
    frames = len(log_probs)
    labels = [0]
    for phone in target:
        labels.extend((int(phone), 0))
    if frames < len(target) + sum(a == b for a, b in zip(target, target[1:])):
        raise ValueError("Performed transcript cannot align on the native clock")
    previous = np.full(len(labels), -np.inf)
    previous[0] = log_probs[0, 0]
    if len(labels) > 1:
        previous[1] = log_probs[0, labels[1]]
    back = np.zeros((frames, len(labels)), dtype=np.int32)
    for t in range(1, frames):
        current = np.full(len(labels), -np.inf)
        for j, phone in enumerate(labels):
            options = [j]
            if j:
                options.append(j - 1)
            if j > 1 and phone and phone != labels[j - 2]:
                options.append(j - 2)
            best = max(options, key=lambda k: previous[k])
            back[t, j], current[j] = best, previous[best] + log_probs[t, phone]
        previous = current
    state = max(range(max(0, len(labels) - 2), len(labels)), key=lambda k: previous[k])
    if not np.isfinite(previous[state]):
        raise ValueError("Nonfinite performed-transcript alignment")
    alignment = [state]
    for t in range(frames - 1, 0, -1):
        state = int(back[t, state])
        alignment.append(state)
    alignment.reverse()
    timed = []
    for j, phone in enumerate(target):
        owned = [t for t, k in enumerate(alignment) if k == j * 2 + 1]
        if not owned:
            raise ValueError("Missing complete target occurrence in alignment")
        timed.append([int(phone), min(owned), max(owned)])
    return timed


def stream_full(runtime, manifest, feats, feature=None):
    """Preserve final padding for neural context, but mark its rows unscored."""
    state = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in manifest["states"]}
    valid = ((len(feats) - 7) // 2 + 1) // 2
    if valid < 1:
        raise ValueError("Recording contains no complete acoustic frames")
    names = [v.name for v in runtime.get_outputs()]
    outputs = {name: [] for name in names if not name.startswith("new_")}
    consumed, emitted = 0, 0
    while emitted < valid:
        x = np.full((1, manifest["window"], 80), manifest["fbank"]["padValue"], np.float32)
        view = feats[consumed:consumed + manifest["window"]]
        x[0, :len(view)] = view
        count = {manifest["editFrameCountInput"]: np.array([min(valid - emitted, manifest["shift"] // 4)], np.int64)} \
            if manifest.get("editFrameCountInput") else {}
        result = dict(zip(names, runtime.run(names, {"x": x, **state, **count})))
        for name in outputs:
            outputs[name].append(result[name][0].copy())
        state = {s["name"]: result["new_" + s["name"]] for s in manifest["states"]}
        consumed += manifest["shift"]
        emitted += len(result["log_probs"][0])
    return outputs, valid


def prepare(args):
    import soundfile as sf
    import torchaudio
    torch.set_num_threads(1)
    out, source = Path(args.out), Path(args.model)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Preparation directory must be empty")
    m, graph, _, width = load_source(source)
    if m["tokens"] != TOKENS or m.get("editDecoder"):
        raise ValueError("Need the shipped ordinary phone decoder and exact alphabet")
    head_parameters(graph)
    auxiliary = insertion_assets(source)
    excluded = json.loads(Path(args.exclusions).read_text())
    if not isinstance(excluded, list) or any(not isinstance(s, str) or not s for s in excluded):
        raise ValueError("Need an enrolled release-excluded-speaker list")
    reference = json.loads((Path(__file__).resolve().parents[1] / "packages/core/assets/ref-v2.json").read_text())
    rows, excluded = recordings(args.help, args.transcribed, excluded, reference)
    folds = speaker_folds([r["speaker"] for r in rows], excluded)
    for selection in (True, False):
        kinds = {r["source"] for r in rows if (folds[r["speaker"]] == 0) == selection}
        if not {"transcribed", "help-none", "silence"} <= kinds:
            raise ValueError("Fit and selection both require performed transcripts, none takes and real pauses")
    banks = {k: dict(np.load(Path(args.augmentation) / (k + ".npz"), allow_pickle=False))
             for k in ("noise", "rir")}
    _, feature, _, feature_path = expose(source, out)
    original, frozen = session(source / m["file"], optimize=True), session(feature_path, optimize=True)
    feeds = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32)
             for s in m["states"]}
    names = [v.name for v in original.get_outputs()]
    rng = np.random.default_rng(SEED)
    for _ in range(4):
        feeds["x"] = rng.normal(-10, 2, (1, m["window"], 80)).astype(np.float32)
        a, b = original.run(names, feeds), frozen.run(names, feeds)
        for name, x, y in zip(names, a, b):
            np.testing.assert_allclose(x, y, atol=2e-4, rtol=2e-4, err_msg="Feature exposure changed " + name)
            if name == "log_probs":
                np.testing.assert_array_equal(x.argmax(-1), y.argmax(-1))
        feeds.update({s["name"]: b[names.index("new_" + s["name"])] for s in m["states"]})
    cached = []
    for i, r in enumerate(rows):
        wave, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Empty or invalid enrolled recording")
        wave = torch.tensor(wave.mean(1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        if not r["phones"] and not torch.any(wave != 0):
            raise ValueError("Pause negatives must be real recorded audio")
        row = dict(speaker=r["speaker"], source=r["source"], audioSha256=r["audioSha256"],
                   target=phoneme_ids(r["phones"]), views=[])
        for v in range(3):
            seed = SEED + i * 3 + v
            audio = augmented(wave.numpy(), banks if v else None, seed)
            feats = fbank(audio, seed if v else None)
            outputs, valid = stream_full(frozen, m, feats, feature)
            lp, x = np.concatenate(outputs["log_probs"]), np.concatenate(outputs[feature])
            timed = timed_target(lp[:valid], row["target"])
            file = f"recording-{i:05d}-view-{v}.npz"
            np.savez_compressed(out / file, features=x, shipped=lp, valid=valid,
                target=np.asarray(row["target"], np.int64), timed=np.asarray(timed, np.int64).reshape(-1, 3), fbank=feats)
            row["views"].append(dict(file=file, sha256=digest(out / file)))
        cached.append(row)
        print(json.dumps(dict(prepared=i + 1, recordings=len(rows))), flush=True)
    write_json(out / "prepared.json", dict(schema=1, source=str(source), sourceSha256=m["sha256"],
        width=width, frames=m["shift"] // 4, featureOutput=feature, rows=cached, folds=folds,
        excludedSpeakers=excluded, auxiliaryHashes=auxiliary,
        augmentationHashes={k: digest(Path(args.augmentation) / (k + ".npz")) for k in banks}))


def load_prepared(path):
    info = json.loads((path / "prepared.json").read_text())
    m, _, _, width = load_source(info["source"])
    if (m["sha256"] != info["sourceSha256"] or width != info["width"] or
            insertion_assets(info["source"]) != info["auxiliaryHashes"]):
        raise ValueError("Shipped model or verification assets changed")
    if speaker_folds([r["speaker"] for r in info["rows"]], info["excludedSpeakers"]) != info["folds"]:
        raise ValueError("Speaker cross-fitting provenance changed")
    rows, hashes = [], set()
    for r in info["rows"]:
        if r["audioSha256"] in hashes:
            raise ValueError("Duplicated training audio")
        hashes.add(r["audioSha256"])
        views = []
        for v in r["views"]:
            file = contained(path, v["file"])
            if digest(file) != v["sha256"]:
                raise ValueError("Acoustic cache changed")
            with np.load(file, allow_pickle=False) as data:
                view = {k: data[k].copy() for k in data.files}
            if (view["features"].shape != (len(view["shipped"]), width) or
                    not np.isfinite(view["features"]).all() or not np.isfinite(view["shipped"]).all() or
                    view["target"].tolist() != r["target"]):
                raise ValueError("Cached acoustic clock or transcript differs")
            views.append(view)
        if len(views) != 3:
            raise ValueError("Need identical three-view enrollment in both arms")
        rows.append(dict(r, data=views))
    return info, rows


def chunk_sources(lp, frames, valid):
    chunks, previous = [], 0
    for t in range(0, len(lp), frames):
        ids = lp[t:min(t + frames, valid)].argmax(-1).tolist()
        chunks.append(source_events(ids, previous))
        previous = ids[-1]
    return chunks


def head_inputs(events, frames, device):
    source = torch.zeros(1, frames + 1, dtype=torch.int64, device=device)
    starts = torch.full_like(source, frames)
    for i, (phone, start, _) in enumerate(events):
        source[0, i], starts[0, i] = phone, start
    return source, starts


def view_outputs(model, view, frames, device, direct=False, sources=None):
    sources = sources if sources is not None else chunk_sources(view["shipped"], frames, int(view["valid"]))
    result = []
    for chunk, start in enumerate(range(0, len(view["features"]), frames)):
        inputs = head_inputs(sources[chunk], frames, device)
        x = torch.tensor(view["features"][start:start + frames], device=device)[None]
        result.append(model(x, *inputs, direct=direct))
    return result, sources


def copy_loss(actions, events, target):
    terms = []
    for i, (phone, start, end) in enumerate(events):
        if any(p == phone and start <= at <= end for p, at, _ in target):
            valid = valid_actions(start, i, events)
            terms.append(-actions[start, i, COPY] + torch.logsumexp(actions[start, i, list(valid)], 0))
    return torch.stack(terms).mean() if terms else actions.sum() * 0


def loss(model, view, frames, device, direct=False, sources=None):
    outputs, sources = view_outputs(model, view, frames, device, direct, sources)
    valid, target = int(view["valid"]), view["target"].tolist()
    support = torch.cat([x[2][0] for x in outputs])[:valid]
    wanted = torch.tensor(target, dtype=torch.int64, device=device)
    lengths = torch.tensor([valid], dtype=torch.int64)
    target_lengths = torch.tensor([len(target)], dtype=torch.int64)
    aux = F.ctc_loss(support[:, None], wanted, lengths, target_lengths, blank=0,
                     reduction="sum", zero_infinity=False) / valid
    shipped = torch.tensor(view["shipped"][:valid], device=device).log_softmax(-1)
    preserve = (shipped.exp() * (shipped - support)).sum(-1).mean()
    if direct:
        primary = torch.cat([x[3][0] for x in outputs])[:valid]
        recognition = F.ctc_loss(primary[:, None], wanted, lengths, target_lengths,
            blank=0, reduction="sum", zero_infinity=False) / valid
        return recognition + aux + preserve
    edits, copies = [], []
    for chunk, (actions, phones, _, _) in enumerate(outputs):
        start = chunk * frames
        n = min(frames, valid - start)
        timed = [[int(p), int(a - start), min(n - 1, int(b - start))]
                 for p, a, b in view["timed"] if start <= a < start + n]
        events = [[p, a, min(b, n - 1)] for p, a, b in sources[chunk] if a < n]
        edits.append(-marginal_log_probability(actions[0, :n], phones[0, :n], events,
                                              [p for p, _, _ in timed]) / n)
        copies.append(copy_loss(actions[0, :n], events, timed))
    return torch.stack(edits).mean() + aux + preserve + torch.stack(copies).mean()


def validate(model, rows, frames, device, direct):
    from student import ctc_collapse
    speakers = {}
    model.eval()
    with torch.no_grad():
        for r in rows:
            per, pauses = [], []
            for view in r["data"]:
                outputs, sources = view_outputs(model, view, frames, device, direct)
                if direct:
                    lp = torch.cat([x[3][0] for x in outputs])[:int(view["valid"])]
                    hyp = ctc_collapse(lp.argmax(-1).tolist())
                else:
                    hyp = []
                    for chunk, output in enumerate(outputs):
                        n = min(frames, int(view["valid"]) - chunk * frames)
                        events = [[p, a, min(b, n - 1)] for p, a, b in sources[chunk] if a < n]
                        _, script = decode(output[0][0, :n].cpu().numpy(), output[1][0, :n].cpu().numpy(), events)
                        hyp.extend(phone for _, phone, _, _ in script if phone is not None)
                per.append(edit_distance(hyp, view["target"].tolist()) / max(1, len(view["target"])))
                if not len(view["target"]):
                    pauses.append(len(hyp))
            speakers.setdefault(r["speaker"], []).append([float(np.mean(per)), float(np.mean(pauses)) if pauses else 0.])
    values = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(per=float(values[0]), pausePhones=float(values[1]), speakers=len(speakers))


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    prepared, out = Path(args.prepared), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit output directory must be empty")
    info, rows = load_prepared(prepared)
    frames, device = info["frames"], torch.device(args.device)
    fit_rows = [r for r in rows if info["folds"][r["speaker"]] != 0]
    selection = [r for r in rows if info["folds"][r["speaker"]] == 0]
    observations = []
    for r in fit_rows:
        for v in r["data"]:
            observations.append(dict(speaker=r["speaker"], target=v["target"].tolist(),
                source=[e[0] for e in source_events(v["shipped"][:int(v["valid"])].argmax(-1))]))
    channels = CrossFittedCorruptions(observations, info["folds"])
    _, graph, _, _ = load_source(info["source"])
    weight, bias = head_parameters(graph)
    initial = build_head(info["width"], len(TOKENS) + 1)
    with torch.no_grad():
        initial.support.weight.copy_(torch.tensor(weight))
        initial.support.bias.copy_(torch.tensor(bias))
        initial.phones.weight.copy_(torch.tensor(weight[1:]))
        initial.phones.bias.copy_(torch.tensor(bias[1:]))
        initial.direct_blank.weight.copy_(torch.tensor(weight[:1]))
        initial.direct_blank.bias.copy_(torch.tensor(bias[:1]))
        initial.correction.weight.zero_(); initial.correction.bias.zero_()
        initial.state_correction.weight.zero_(); initial.state_correction.bias.zero_()
        initial.actions.weight.zero_()
        initial.actions.bias.copy_(torch.tensor([2., -2., 2., -2., -2.]))
    out.mkdir(parents=True, exist_ok=True)
    report = dict(seed=SEED, secondsPerArm=args.seconds, learningRate=args.lr,
        folds=info["folds"], excludedSpeakers=info["excludedSpeakers"],
        corruptionMembers={str(k): sorted(v) for k, v in channels.members.items()},
        preparedSha256=digest(prepared / "prepared.json"),
        control="same-neural-network-and-fixed-acoustic-views-source-free-direct-CTC-under-equal-time-budget", arms={})
    if args.seconds <= 0 or args.lr <= 0:
        raise ValueError("Need positive compute allocation and training rate")
    for direct in (False, True):
        arm = "direct" if direct else "edit"
        model = copy.deepcopy(initial).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = validate(model, selection, frames, device, direct)
        state, selected, history = copy.deepcopy(model.state_dict()), 0, [dict(step=0, **best)]
        steps, epoch, seconds = 0, 0, 0.
        budget = report["arms"]["edit"]["trainingSeconds"] if direct else args.seconds
        while seconds < budget:
            order = list(range(len(fit_rows)))
            random.Random(SEED + epoch).shuffle(order)
            for index in order:
                begin = time.monotonic()
                model.train()
                r = fit_rows[index]
                view = r["data"][steps % 3]
                opt.zero_grad(set_to_none=True)
                objective = loss(model, view, frames, device, direct)
                if not direct:
                    rng = random.Random(SEED + steps)
                    corrupt = channels.draw(view["target"].tolist(), view["timed"].tolist(), r["speaker"], info["folds"], rng)
                    sources = [[[p, a - start, min(frames - 1, b - start)] for p, a, b in corrupt if start <= a < start + frames]
                               for start in range(0, len(view["features"]), frames)]
                    objective = (objective + loss(model, view, frames, device, False, sources)) / 2
                if not torch.isfinite(objective):
                    raise ValueError("Nonfinite training loss; no batch silently discarded")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                seconds += time.monotonic() - begin
                steps += 1
                if seconds >= budget:
                    break
            epoch += 1
            metrics = validate(model, selection, frames, device, direct)
            history.append(dict(step=steps, seconds=seconds, **metrics))
            print(json.dumps(dict(arm=arm, **history[-1])), flush=True)
            if metrics["per"] < best["per"]:
                best, state, selected = metrics, copy.deepcopy(model.state_dict()), steps
        model.load_state_dict(state)
        torch.save(dict(state=state, width=info["width"], frames=frames, selectedStep=selected,
            sourceSha256=info["sourceSha256"]), out / (arm + ".pt"))
        report["arms"][arm] = dict(history=history, selectedStep=selected, trainingSteps=steps,
            allocatedSeconds=budget, trainingSeconds=seconds,
            parameters=sum(p.numel() for p in model.parameters()), selected=best)
        graft(info["source"], model, out / arm, direct)
    write_json(out / "training-report.json", report)


def verify(args):
    """Check both complete bundles against the fitted neural heads and clock."""
    torch.set_num_threads(1)
    prepared, out = Path(args.prepared), Path(args.out)
    info, rows = load_prepared(prepared)
    report = {}
    for direct in (False, True):
        arm = "direct" if direct else "edit"
        ck = torch.load(out / (arm + ".pt"), map_location="cpu", weights_only=False)
        if ck["sourceSha256"] != info["sourceSha256"]:
            raise ValueError("Checkpoint has different frozen encoder")
        head = build_head(info["width"], len(TOKENS) + 1)
        head.load_state_dict(ck["state"]); head.eval()
        path = out / arm
        m = json.loads((path / "manifest.json").read_text())
        if digest(path / m["file"]) != m["sha256"] or insertion_assets(path) != info["auxiliaryHashes"]:
            raise ValueError("Export bundle is missing or corrupt")
        runtime = session(path / m["file"], optimize=True)
        worst, count = 0., 0
        for r in rows:
            if info["folds"][r["speaker"]] != 0:
                continue
            for view in r["data"]:
                got, valid = stream_full(runtime, m, view["fbank"])
                with torch.no_grad():
                    expected, sources = view_outputs(head, view, info["frames"], "cpu", direct)
                for chunk, output in enumerate(expected):
                    for name, wanted in [("edit_actions", output[0]), ("edit_phones", output[1]),
                                         ("support_log_probs", output[2]), ("direct_log_probs", output[3])]:
                        actual, wanted = got[name][chunk], wanted[0].numpy()
                        np.testing.assert_allclose(actual, wanted, atol=2e-4, rtol=2e-4)
                        worst = max(worst, float(np.abs(actual - wanted).max()))
                    original = view["shipped"][chunk * info["frames"]:(chunk + 1) * info["frames"]]
                    np.testing.assert_allclose(got["source_log_probs"][chunk], original, atol=2e-4, rtol=2e-4)
                    np.testing.assert_array_equal(got["source_log_probs"][chunk].argmax(-1), original.argmax(-1))
                if valid != int(view["valid"]):
                    raise ValueError("Export changed posterior clock")
                count += 1
        if not count:
            raise ValueError("No selection-speaker export parity views")
        report[arm] = dict(views=count, maximumLogitError=worst, modelSha256=m["sha256"],
            auxiliaryHashes=insertion_assets(path), selectedStep=ck["selectedStep"])
    # Declaring a model is a separate supervisor step, after full-session SDK
    # target decoding. No main-model fallback is written by this runner.
    write_json(out / "verified.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    for name in ("model", "help", "transcribed", "exclusions", "augmentation", "out"):
        p.add_argument("--" + name, required=True)
    for command in ("fit", "verify"):
        p = commands.add_parser(command)
        p.add_argument("--prepared", required=True); p.add_argument("--out", required=True)
        if command == "fit":
            p.add_argument("--device", default="cpu"); p.add_argument("--seconds", type=float, default=600.)
            p.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    {"prepare": prepare, "fit": fit, "verify": verify}[args.command](args)
