"""Cache, pair-fit and export a frozen-encoder phone/Arabic-letter experiment.

Run only on authorized training data through the supplied Modal supervisor.
The letter head and boundary lattice are training-only. Export replaces the
native phone head and copies the existing insertion teacher unchanged.
"""

import argparse
import copy
import json
import sys
import time
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from student import TOKENS, phoneme_ids, ctc_collapse, edit_distance
from student_boundary import (BoundaryCTC, authorized_none, objectives, recording_update,
                              speaker_partition, spelling)
from student_boundary_export import (digest, load_source, expose, session, stream,
                                     head_parameters, insertion_assets, graft)


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Training audio escapes its authorized root")
    return path


def training_rows(help_root, reference, corpus_index=None, excluded=()):
    labels = [json.loads(line) for line in (help_root / "labels.jsonl").read_text().splitlines() if line]
    additional = ([json.loads(line) for line in Path(corpus_index).read_text().splitlines() if line]
                  if corpus_index else [])
    excluded = set(excluded) | {r.get("speaker") for r in labels + additional if r.get("split") != "train"}
    ayahs = {f'{a["s"]}:{a["a"]}': a for a in reference["ayahs"]}
    rows = []
    for label in labels:
        if not authorized_none(label) or label["speaker"] in excluded:
            continue
        words = []
        for key in label["ayahs"]:
            a = ayahs[key]
            for i, word in enumerate(a["w"]):
                stop = a["o"][i + 1] if i + 1 < len(a["o"]) else len(a["p"])
                # Expanded held-sound notation is one CTC occurrence. Collapse
                # within each word, retaining repeated occurrences across words.
                phones = "".join(c for c, _ in groupby(a["p"][a["o"][i]:stop]))
                words.append(dict(phones=phones, letters=spelling(word)))
        rows.append(dict(id=label["id"], speaker=label["speaker"], source="help-train/none",
                         audio=contained(help_root, label["wav"]), words=words))
    for row in additional:
        if row.get("split") != "train" or row.get("speaker") in excluded or row.get("excluded"):
            continue
        if (not row.get("speaker") or row.get("authorized") is not True or row.get("verified") is not True
                or not row.get("words") or not row.get("sha256")):
            raise ValueError("Additional transcript must be authorized, performed and word-segmented")
        audio = contained(Path(corpus_index).parent, row["audio"])
        if digest(audio) != row["sha256"]:
            raise ValueError("Performed transcript audio checksum differs")
        rows.append(dict(id=row["id"], speaker=row["speaker"], source="authorized-performed-transcript",
                         audio=audio, words=[dict(phones=w["phones"], letters=spelling(w["letters"]))
                                             for w in row["words"]]))
    seen = set()
    for row in rows:
        row["audio_sha256"] = digest(row["audio"])
        if row["audio_sha256"] in seen:
            raise ValueError("Duplicate recording could leak speakers or duplicate gradients")
        seen.add(row["audio_sha256"])
        for word in row["words"]:
            if not word["phones"] or not word["letters"]:
                raise ValueError("Both alphabets need every word")
            phoneme_ids(word["phones"])
    if not rows:
        raise ValueError("No authorized training-speaker none recordings or transcripts")
    return rows, sorted(s for s in excluded if s)


def cache(args):
    import numpy as np
    import torch
    import torchaudio
    import soundfile as sf

    torch.set_num_threads(args.threads)
    started = time.time()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Cache output must be empty")
    reference = json.loads(Path(args.reference).read_text())
    rows, excluded = training_rows(Path(args.help_train), reference, args.corpus_index,
                                  json.loads(Path(args.exclusions).read_text()) if args.exclusions else ())
    speakers = speaker_partition([r["speaker"] for r in rows], args.seed)
    # Fixed orthographic alphabet from the enrolled reference, not validation or
    # practice targets. Additional performed transcripts must use this alphabet.
    letters = "".join(sorted(set(c for a in reference["ayahs"] for w in a["w"] for c in spelling(w))))
    letter_ids = {c: i + 1 for i, c in enumerate(letters)}
    manifest, native, _, _ = load_source(args.model)
    if manifest["tokens"] != TOKENS:
        raise ValueError("Cannot change the shipped phone alphabet")
    auxiliary = insertion_assets(args.model)
    weight, bias = head_parameters(native)
    manifest, feature, width, model_path = expose(args.model, out)
    encoder, original = session(model_path), session(Path(args.model) / manifest["file"])
    feeds = {s["name"]: np.zeros(s["shape"], np.int64 if s["dtype"] == "int64" else np.float32)
             for s in manifest["states"]}
    names = [o.name for o in original.get_outputs()]
    rng = np.random.default_rng(args.seed)
    for _ in range(4):
        feeds["x"] = rng.normal(-10, 2, (1, manifest["window"], 80)).astype(np.float32)
        before, after = original.run(names, feeds), encoder.run(names, feeds)
        for name, a, b in zip(names, before, after):
            np.testing.assert_array_equal(a, b, err_msg="Feature exposure changed " + name)
        feeds.update({s["name"]: after[names.index("new_" + s["name"])] for s in manifest["states"]})
    index = []
    for i, row in enumerate(rows):
        if time.time() - started >= args.hours * 3600:
            raise TimeoutError("One-day cache/refit budget exhausted")
        wave, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Invalid enrolled audio")
        wave = torch.tensor(wave.mean(1))
        if sr != manifest["sampleRate"]:
            wave = torchaudio.functional.resample(wave, sr, manifest["sampleRate"])
        features = torchaudio.compliance.kaldi.fbank(wave[None], num_mel_bins=80,
            sample_frequency=16000, dither=0., energy_floor=1e-10, high_freq=-400).numpy()
        lp, hidden = stream(encoder, manifest, features, feature)
        pw = [phoneme_ids(w["phones"]) for w in row["words"]]
        lw = [[letter_ids[c] for c in w["letters"]] for w in row["words"]]
        file = f"record-{i:05d}.pt"
        torch.save(dict(features=torch.tensor(hidden), shipped=torch.tensor(lp),
                        phone_words=pw, letter_words=lw, fbank=features), out / file)
        index.append(dict(id=row["id"], speaker=row["speaker"], source=row["source"],
                          partition=speakers[row["speaker"]], audio_sha256=row["audio_sha256"],
                          words=row["words"], file=file, cache_sha256=digest(out / file), frames=len(lp)))
        print(json.dumps(dict(cached=i + 1, total=len(rows), frames=len(lp), words=len(pw))), flush=True)
    torch.save(dict(weight=weight, bias=bias), out / "initial.pt")
    json_write(out / "index.json", dict(schema=1, source=str(Path(args.model).resolve()),
        source_sha256=manifest["sha256"], auxiliary_sha256=auxiliary, reference_sha256=digest(args.reference),
        labels_sha256=digest(Path(args.help_train) / "labels.jsonl"),
        corpus_index_sha256=digest(args.corpus_index) if args.corpus_index else None,
        excluded_speakers=excluded, speakers=speakers, letters=letters, width=width,
        initial_sha256=digest(out / "initial.pt"), feature_output=feature, started_at=started,
        seed=args.seed, hours=args.hours, augmentation="none; original full recordings and real pauses",
        transcript_policy="word-local duration run collapse; undiacritized Arabic spelling",
        recordings=index))


def validate(heads, records):
    import numpy as np
    import torch
    speakers = {}
    with torch.no_grad():
        for row, record, lattice in records:
            values = objectives(heads, record["features"], record["shipped"], lattice, record["occurrences"])
            phones = heads["phone"](record["features"]).log_softmax(-1)
            target = sum(record["phone_words"], [])
            per = edit_distance(ctc_collapse(phones.argmax(-1).tolist()), target) / len(target)
            speakers.setdefault(row["speaker"], []).append([values["independent"].item(), per])
    value = np.mean([np.mean(v, axis=0) for v in speakers.values()], axis=0)
    return dict(objective=float(value[0]), phone_per=float(value[1]), speakers=len(speakers))


def fit(args):
    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    root, out = Path(args.cache), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fit output must be empty")
    info = json.loads((root / "index.json").read_text())
    deadline = info["started_at"] + info["hours"] * 3600
    if digest(root / "initial.pt") != info["initial_sha256"]:
        raise ValueError("Initial head checksum differs from cache")
    initial = torch.load(root / "initial.pt", weights_only=False)
    weight, bias = torch.tensor(initial["weight"]), torch.tensor(initial["bias"])
    phone = torch.nn.Linear(weight.shape[1], weight.shape[0])
    letter = torch.nn.Linear(weight.shape[1], len(info["letters"]) + 1)
    with torch.no_grad():
        phone.weight.copy_(weight)
        phone.bias.copy_(bias)
        # Seed compositional letters from corresponding shipped sound identities
        # when available, with deterministic initialization for orthographic-only
        # letters. Exactly the same two heads initialize both arms.
        letter.weight[0].copy_(weight[0])
        letter.bias[0].copy_(bias[0])
        for i, c in enumerate(info["letters"], 1):
            if c in TOKENS:
                j = TOKENS.index(c) + 1
                letter.weight[i].copy_(weight[j])
                letter.bias[i].copy_(bias[j])
    initial_heads = torch.nn.ModuleDict(dict(phone=phone, letter=letter))
    heads = {arm: copy.deepcopy(initial_heads) for arm in ("shared", "independent")}
    opts = {arm: torch.optim.Adam(h.parameters(), lr=args.lr) for arm, h in heads.items()}
    records = []
    for row in info["recordings"]:
        path = contained(root, row["file"])
        if digest(path) != row["cache_sha256"]:
            raise ValueError("Cached training recording changed")
        record = torch.load(path, weights_only=False)
        record["occurrences"] = sum(len(w) for w in record["phone_words"] + record["letter_words"])
        lattice = BoundaryCTC(record["phone_words"], record["letter_words"], len(TOKENS) + 1, len(info["letters"]) + 1)
        records.append((row, record, lattice))
    groups = {part: [r for r in records if r[0]["partition"] == part] for part in ("train", "validation")}
    if not all(groups.values()) or speaker_partition([r[0]["speaker"] for r in records], info["seed"]) != info["speakers"]:
        raise ValueError("Invalid speaker-grouped selection")
    if set(info["speakers"]) & set(info["excluded_speakers"]):
        raise ValueError("Excluded speaker in cache")
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    history, best, updates = [], float("inf"), 0
    for epoch in range(args.epochs):
        if time.time() >= deadline:
            raise TimeoutError("One-day cache/refit budget exhausted")
        for index in rng.permutation(len(groups["train"])):
            if time.time() >= deadline:
                raise TimeoutError("One-day cache/refit budget exhausted")
            _, record, lattice = groups["train"][index]
            for arm, h in heads.items():
                # Both arms compute both forward sums AND both objective gradients.
                # Discard the unused gradient, keeping compute/data/update counts
                # matched without contaminating the selected acoustic objective.
                recording_update(h, opts[arm], arm, record["features"], record["shipped"],
                                 lattice, record["occurrences"])
            updates += 1
            print(json.dumps(dict(epoch=epoch + 1, update=updates, of=len(groups["train"]) * args.epochs)), flush=True)
        validation = {arm: validate(h, groups["validation"]) for arm, h in heads.items()}
        mean = sum(v["objective"] for v in validation.values()) / 2
        history.append(dict(epoch=epoch + 1, updates=updates, validation=validation, mean=mean))
        print(json.dumps(history[-1]), flush=True)
        if mean < best:
            best = mean
            for arm, h in heads.items():
                torch.save(dict(heads=h.state_dict(), arm=arm, selected_epoch=epoch + 1, selected_updates=updates,
                                source_sha256=info["source_sha256"], cache_sha256=digest(root / "index.json")),
                           out / (arm + ".pt"))
        json_write(out / "training.json", dict(args=vars(args), history=history, seed=args.seed,
            selection="one common epoch: mean speaker-grouped validation independent-boundary NLL + shipped KL",
            compute="each arm: both marginal forward sums and both gradient computations; same record order/updates",
            objective="boundary-normalized phone/letter likelihood per occurrence + original full-recording posterior KL"))
    for arm, h in heads.items():
        selected = torch.load(out / (arm + ".pt"), weights_only=False)
        h.load_state_dict(selected["heads"])
        path = graft(info["source"], h["phone"], out / arm, arm)
        runtime = session(path, optimize=True)
        manifest = json.loads((out / arm / "manifest.json").read_text())
        # Validate optimized runtime against the fitted head on every cached real
        # recording, including state carry. Only phone weights go into the graph.
        maximum = 0.
        for _, record, _ in records:
            got, _ = stream(runtime, manifest, record["fbank"])
            with torch.no_grad():
                expected = h["phone"](record["features"]).log_softmax(-1).numpy()
            np.testing.assert_allclose(got, expected, atol=2e-3, rtol=2e-4)
            maximum = max(maximum, float(np.abs(np.exp(got) - np.exp(expected)).max()))
        if insertion_assets(out / arm) != info["auxiliary_sha256"]:
            raise ValueError("Export changed insertion verification")
        json_write(out / arm / "verification.json", dict(sha256=digest(path),
            max_probability_parity_error=maximum, recordings=len(records),
            letter_head_exported=False, auxiliary_sha256=info["auxiliary_sha256"],
            selected_epoch=selected["selected_epoch"], selected_updates=selected["selected_updates"]))
        print(json.dumps(dict(bundle=str(out / arm), verification="passed", parity=maximum)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("cache")
    for key in ("model", "help-train", "out"):
        c.add_argument("--" + key, required=True)
    c.add_argument("--reference", default="packages/core/assets/ref-v2.json")
    c.add_argument("--corpus-index")
    c.add_argument("--exclusions")
    c.add_argument("--hours", type=float, default=20.)
    f = commands.add_parser("fit")
    for key in ("cache", "out"):
        f.add_argument("--" + key, required=True)
    f.add_argument("--epochs", type=int, default=6)
    f.add_argument("--lr", type=float, default=1e-4)
    for cmd in (c, f):
        cmd.add_argument("--threads", type=int, default=1)
        cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or (args.command == "cache" and not 0 < args.hours <= 24):
        parser.error("Positive threads and budget within one day required")
    if args.command == "fit" and (args.epochs < 1 or args.lr <= 0):
        parser.error("Positive epochs and learning rate required")
    {"cache": cache, "fit": fit}[args.command](args)


if __name__ == "__main__":
    main()
