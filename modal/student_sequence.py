"""Training-only forward/reverse transcription supervision of the primary encoder.

The acoustic encoder retains its original streaming mask. Only the auxiliary
decoder sees the complete utterance, and it never enters a deployment checkpoint.
No practice labels or audio are accepted by the training entry point.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def ordered_targets(sequences, vocab, order, device):
    """Reverse phones within each recording, retaining repeated phone occurrences."""
    if order not in ("forward", "reverse"):
        raise ValueError("Auxiliary order must be forward or reverse")
    bos, eos, pad = vocab, vocab + 1, vocab + 2
    if not sequences or any(not s or any(type(p) is not int or not 0 < p < vocab for p in s)
                            for s in sequences):
        raise ValueError("Need nonempty performed phone sequences without CTC blank")
    length = max(map(len, sequences)) + 1
    history = torch.full((len(sequences), length), pad, dtype=torch.long, device=device)
    labels = history.clone()
    for i, phones in enumerate(sequences):
        phones = list(phones if order == "forward" else reversed(phones))
        history[i, :len(phones) + 1] = torch.tensor([bos, *phones], device=device)
        labels[i, :len(phones) + 1] = torch.tensor([*phones, eos], device=device)
    return history, labels


def positional_encoding(n, dim, device, dtype):
    position = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000) / dim))
    encoding = torch.zeros(n, dim, device=device)
    encoding[:, 0::2] = torch.sin(position * frequency)
    encoding[:, 1::2] = torch.cos(position * frequency)
    return encoding.to(dtype)


class TranscriptionDecoder(nn.Module):
    """Causal label history cross-attends to non-detached streaming acoustic frames."""

    def __init__(self, acoustic_dim, vocab, dim=128, heads=4, layers=2, history_dropout=0.5):
        super().__init__()
        if dim % 2 or dim % heads or not 0 <= history_dropout <= 1:
            raise ValueError("Invalid auxiliary dimensions or history dropout")
        self.vocab, self.pad, self.mask = vocab, vocab + 2, vocab + 3
        self.dim, self.history_dropout = dim, history_dropout
        self.embedding = nn.Embedding(vocab + 4, dim, padding_idx=self.pad)
        self.acoustics = nn.Linear(acoustic_dim, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.output = nn.Linear(dim, vocab + 2)

    def forward(self, encoded, lengths, history):
        if (encoded.ndim != 3 or history.ndim != 2 or encoded.shape[0] != history.shape[0]
                or lengths.shape != (encoded.shape[0],)
                or (lengths <= 0).any() or (lengths > encoded.shape[1]).any()):
            raise ValueError("Invalid acoustic batch lengths")
        padding = history.eq(self.pad)
        if self.training and self.history_dropout:
            drop = torch.rand(history.shape, device=history.device) < self.history_dropout
            # BOS and padding are structural; only previous observed phones drop.
            history = history.masked_fill(drop & history.gt(0) & history.lt(self.vocab), self.mask)
        label = self.embedding(history) * math.sqrt(self.dim)
        label = label + positional_encoding(label.shape[1], self.dim, label.device, label.dtype)
        memory = self.acoustics(encoded)
        memory = memory + positional_encoding(memory.shape[1], self.dim, memory.device, memory.dtype)
        causal = torch.ones(history.shape[1], history.shape[1], device=history.device, dtype=torch.bool).triu(1)
        acoustic_padding = torch.arange(encoded.shape[1], device=lengths.device)[None] >= lengths[:, None]
        hidden = self.decoder(label, memory, tgt_mask=causal, tgt_key_padding_mask=padding,
                              memory_key_padding_mask=acoustic_padding)
        return self.output(hidden)

    def loss(self, encoded, lengths, sequences, order):
        history, labels = ordered_targets(sequences, self.vocab, order, encoded.device)
        logits = self(encoded, lengths, history)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=self.pad)


@contextmanager
def fixed_streaming(model, chunk, left):
    """Preservation passes have no dropout and restore every original training flag."""
    flags = [(m, m.training) for m in model.modules()]
    context = model.encoder.chunk_size, model.encoder.left_context_frames
    try:
        model.eval()
        model.streaming_eval(chunk, left)
        yield
    finally:
        model.encoder.chunk_size, model.encoder.left_context_frames = context
        for module, flag in flags:
            module.training = flag


def preservation_loss(predicted, anchor, lengths):
    valid = torch.arange(predicted.shape[1], device=lengths.device)[None] < lengths[:, None]
    if predicted.shape != anchor.shape or not valid.any():
        raise ValueError("Preservation requires identical primary acoustic frames")
    return F.kl_div(predicted[valid], anchor.detach()[valid], log_target=True, reduction="batchmean")


def auxiliary_loss(decoder, encoded, lengths, sequences, order, seed):
    if order == "continuation":
        return encoded.sum() * 0
    devices = [encoded.device.index] if encoded.is_cuda else []
    # Auxiliary dropout must not change the next primary dropout/augmentation draw.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return decoder.loss(encoded, lengths, sequences, order)


def recipe_contract(record):
    """Check a frozen training audit; this does not establish its factual accuracy."""
    if record.get("schema") != 1 or record.get("selection") != "minimum-primary-validation-objective":
        raise ValueError("Need a frozen training-only selection policy")
    train, valid, excluded = (set(record.get(k, [])) for k in
                              ("training_speakers", "validation_speakers", "excluded_speakers"))
    if not train or not valid or not excluded or train & valid or (train | valid) & excluded:
        raise ValueError("Require disjoint audited training, validation and excluded speakers")
    seen = set()
    for partition, speakers in (("training", train), ("validation", valid)):
        rows = record.get(partition, [])
        if not rows or not any(r["source"] == "quran" for r in rows) or not any(r["source"] == "help-none" for r in rows):
            raise ValueError("Both training-speaker partitions require Quran and help-none recordings")
        for row in rows:
            if row["key"] in seen or row["speaker"] not in speakers or row.get("original_split") != "train":
                raise ValueError("Duplicate, cross-partition or original held-out training item")
            if row["source"] not in ("quran", "help-none"):
                raise ValueError("Unsupported training source")
            if row.get("auxiliary") and (not row.get("transcript") or row.get("transcript_status") != "performed"):
                raise ValueError("Auxiliary supervision requires a performed transcript")
            if row["source"] == "help-none" and not row.get("auxiliary"):
                raise ValueError("Help-none must supply its authorized transcript")
            seen.add(row["key"])
    for key in ("anchor_sha256", "corpus_sha256"):
        value = record.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing exact training corpus/checkpoint hash")
    cfg = record["config"]
    for key in ("epochs", "batch_seconds", "seed", "validation_every", "chunk", "left", "aux_dim", "aux_heads", "aux_layers"):
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid fixed training option: {key}")
    for key in ("learning_rate", "aux_weight", "preservation_weight"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"Invalid fixed loss option: {key}")
    if not 0 < cfg["history_dropout"] <= 1 or cfg["chunk"] != 16 or cfg["left"] != 128:
        raise ValueError("Retain deployed streaming geometry and positive history dropout")
    return cfg


def help_none_rows(labels, selected):
    """Mistake scripts and empty error lists are never inferred to be transcripts."""
    by_id = {}
    for row in labels:
        if row["id"] in by_id:
            raise ValueError("Duplicate help recording")
        by_id[row["id"]] = row
    out = []
    for item in selected:
        if item["source"] != "help-none":
            continue
        if not item["key"].startswith("help:"):
            raise ValueError("Invalid help key")
        row = by_id[item["key"][5:]]
        if (row.get("kind") != "none" or row.get("split") != "train"
                or row.get("speaker") != item["speaker"] or row.get("scored") is not True
                or row.get("excluded") or row.get("events") != [] or row.get("non_errors") != []
                or row.get("unsupported") != [] or not row.get("ayahs")):
            raise ValueError("Help take is not an eligible none-labelled training recording")
        out.append((item, row))
    return out


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_checkpoint(model, base_cfg, state):
    """Original loader/exporter contract: no auxiliary parameters or optimizer state."""
    return {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "cfg": dict(base_cfg), "state": dict(state)}


def _recording_corpus(record, anchor_cfg, help_root, reference_path):
    """Retain the original s1 mixture; partition by audited speaker, add only none takes.

    Keys, phones, exact source audio and cached KD bytes form one corpus hash.
    This loader never uses original test/eval splits for checkpoint selection.
    """
    import numpy as np
    import student_train as T
    from student import phoneme_ids, spliced_target

    recipe_contract(record)
    c = T.load_clips(anchor_cfg["buckets"], "train", anchor_cfg["max_hours"],
                     anchor_cfg["max_clip_s"], anchor_cfg["seed"])
    by_key = {key: i for i, key in enumerate(c["key"])}
    selected = record["training"] + record["validation"]
    if {r["key"] for r in selected if r["source"] == "quran"} != set(by_key):
        raise ValueError("The audit must retain the complete original training mixture")
    root = Path(help_root).resolve()
    labels = [json.loads(s) for s in (root / "labels.jsonl").read_text().splitlines() if s.strip()]
    help_rows = {item["key"]: row for item, row in help_none_rows(labels, selected)}
    reference = json.loads(Path(reference_path).read_text())
    ayahs = {f"{a['s']}:{a['a']}": a["p"] for a in reference["ayahs"]}
    digest, partitions = hashlib.sha256(), []
    for partition in ("training", "validation"):
        items = []
        for item in sorted(record[partition], key=lambda r: r["key"]):
            if item["source"] == "quran":
                i = by_key[item["key"]]
                a, z = c["flac_off"][i:i + 2]
                audio = c["flac"][a:z].tobytes()
                a, z = c["top_off"][i:i + 2]
                top_ids = c["ids"][a:z].reshape(-1, 5).astype(np.int64)
                top_p = c["probs"][a:z].reshape(-1, 5).astype(np.float32)
                ref, hyp = c["ref"][i], c["hyp"][i]
                if anchor_cfg["target"] == "teacher":
                    target = hyp
                elif anchor_cfg["target"] == "spliced":
                    target = spliced_target(ref, hyp)
                else:
                    raise ValueError("Unsupported original primary target recipe")
                samples = int(c["n_samples"][i])
            else:
                import soundfile as sf

                row = help_rows[item["key"]]
                path = (root / row["wav"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Help audio leaves its authorized corpus")
                audio = path.read_bytes()
                info = sf.info(path)
                if info.samplerate != T.SR or info.channels != 1:
                    raise ValueError("Help recordings require original mono model-rate PCM")
                ref = target = "".join(ayahs[a] for a in row["ayahs"])
                if item["transcript"] != ref:
                    raise ValueError("None-labelled transcript differs from its authorized ayahs")
                samples = info.frames
                # The shipped mixture retains its cached teacher KD. New none
                # takes have transcript CTC and full-PCM output preservation.
                top_ids, top_p = np.zeros((0, 5), dtype=np.int64), np.zeros((0, 5), dtype=np.float32)
            auxiliary = phoneme_ids(item["transcript"]) if item.get("auxiliary") else None
            payload = {**item, "partition": partition, "ref": ref, "target": target, "samples": samples}
            for block in (json.dumps(payload, sort_keys=True, ensure_ascii=False).encode(), audio,
                          top_ids.tobytes(), top_p.tobytes()):
                digest.update(len(block).to_bytes(8, "little"))
                digest.update(block)
            items.append(dict(key=item["key"], audio=audio, samples=samples, target=phoneme_ids(target),
                              auxiliary=auxiliary, top_ids=top_ids, top_p=top_p))
        partitions.append(items)
    return partitions, digest.hexdigest()


def audit_corpus(record, anchor_cfg, help_root, reference_path):
    """Fingerprint an independently audited draft before freezing it for all arms.

    Draft hashes must be well-formed (e.g. 64 zeroes). This only fingerprints
    the selected bytes; it cannot verify the producer's speaker/transcript audit.
    """
    partitions, digest = _recording_corpus(record, anchor_cfg, help_root, reference_path)
    return dict(corpus_sha256=digest, partitions=[dict(recordings=len(rows),
                samples=sum(r["samples"] for r in rows),
                auxiliary_recordings=sum(r["auxiliary"] is not None for r in rows))
                for rows in partitions])


def load_recordings(record, anchor_cfg, help_root, reference_path):
    partitions, digest = _recording_corpus(record, anchor_cfg, help_root, reference_path)
    if digest != record["corpus_sha256"]:
        raise ValueError(f"Exact training corpus differs from its frozen audit: {digest}")
    return partitions


class RecordingDataset:
    """The shipped augmentation chain, with matched item/epoch random draws."""

    def __init__(self, rows, base_cfg, augmentation, seed, epoch=0, training=True):
        self.rows, self.cfg, self.augmentation = rows, base_cfg, augmentation
        self.seed, self.epoch, self.training = seed, epoch, training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import io
        import numpy as np
        import soundfile as sf
        import torchaudio
        import student_train as T
        from student import mix_noise

        r = self.rows[i]
        wave, sr = sf.read(io.BytesIO(r["audio"]), dtype="float32")
        if sr != T.SR or wave.ndim != 1 or len(wave) != r["samples"] or not np.isfinite(wave).all():
            raise ValueError("Invalid recording PCM")
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.epoch}:{r['key']}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        if self.training and self.augmentation:
            if rng.random() < self.cfg["rir_p"]:
                wave = T.reverb(wave, T.pick(self.augmentation["rir"], rng))
            if rng.random() < self.cfg["noise_p"]:
                wave = mix_noise(wave, T.pick(self.augmentation["noise"], rng), rng.uniform(*self.cfg["snr_db"]), rng)
            wave = wave * 10 ** (rng.uniform(-10, 6) / 20)
        features = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(np.ascontiguousarray(wave))[None], num_mel_bins=80, sample_frequency=T.SR,
            dither=0.0, energy_floor=1e-10, high_freq=-400)
        if self.training and self.cfg["spec_aug"]:
            features = T.spec_augment(features, rng)
        return dict(i=i, feats=features, target=torch.tensor(r["target"], dtype=torch.long),
                    auxiliary=r["auxiliary"], top_ids=torch.from_numpy(r["top_ids"]), top_p=torch.from_numpy(r["top_p"]))


def collate_recordings(items):
    import student_train as T

    batch = T.collate(items)
    batch["auxiliary"] = [x["auxiliary"] for x in items]
    return batch


def primary_objective(log_probs, lengths, batch, base_cfg):
    """Original transcript CTC and cached teacher KD on the original mixture."""
    from student import soft_target_loss

    device = log_probs.device
    # The old mixture retains its zero-infinity behavior. Newly transcribed
    # none takes must not silently lose CTC supervision through an impossible
    # label/frame alignment (adjacent equal phones need an intervening blank).
    offset = 0
    for i, n in enumerate(batch["target_lens"].tolist()):
        seq = batch["targets"][offset:offset + n]
        required = n + int((seq[1:] == seq[:-1]).sum())
        if batch["n_teacher"][i] == 0 and required > int(lengths[i]):
            raise ValueError("New performed transcript cannot align to its acoustic frames")
        offset += n
    ctc = F.ctc_loss(log_probs.transpose(0, 1), batch["targets"].to(device), lengths,
                     batch["target_lens"].to(device), blank=0, reduction="sum", zero_infinity=True)
    ctc = ctc / lengths.sum().clamp_min(1)
    kd = log_probs.sum() * 0
    if base_cfg["kd_weight"] and batch["n_teacher"].max() > 0:
        B, T, _ = log_probs.shape
        n = batch["n_teacher"].to(device)
        idx = (torch.arange(T, device=device)[None] - base_cfg["kd_delay"]).clamp_min(0)
        idx = torch.minimum(idx, (n.clamp_min(1) - 1)[:, None])
        ids = batch["top_ids"].to(device).gather(1, idx[..., None].expand(B, T, 5))
        probs = batch["top_p"].to(device).gather(1, idx[..., None].expand(B, T, 5))
        valid = (torch.arange(T, device=device)[None] < lengths[:, None]) & n[:, None].gt(0)
        kd = soft_target_loss(log_probs, ids, probs, valid)
    return base_cfg["ctc_weight"] * ctc + base_cfg["kd_weight"] * kd, ctc, kd


def joint_objective(model, anchor, decoder, batch, base_cfg, cfg, order, step):
    device = next(model.parameters()).device
    feats, feat_lens = batch["feats"].to(device), batch["feat_lens"].to(device)
    encoded, lengths = model.encode(feats, feat_lens)
    lp = model.ctc_output(encoded).float().log_softmax(-1)
    primary, ctc, kd = primary_objective(lp, lengths, batch, base_cfg)
    indices = [i for i, seq in enumerate(batch["auxiliary"]) if seq is not None]
    aux = encoded.sum() * 0
    if indices:
        aux = auxiliary_loss(decoder, encoded[indices], lengths[indices],
                             [batch["auxiliary"][i] for i in indices], order, cfg["seed"] + step)
    with fixed_streaming(model, cfg["chunk"], cfg["left"]):
        predicted, preserved_lengths = model(feats, feat_lens)
    with torch.no_grad(), fixed_streaming(anchor, cfg["chunk"], cfg["left"]):
        reference, anchor_lengths = anchor(feats, feat_lens)
    if not torch.equal(preserved_lengths, anchor_lengths):
        raise ValueError("Anchor and primary output clocks differ")
    preserved = preservation_loss(predicted, reference, preserved_lengths)
    return {"ctc": ctc, "kd": kd, "auxiliary": aux, "preservation": preserved,
            "loss": primary + cfg["aux_weight"] * aux + cfg["preservation_weight"] * preserved}


def validation_objective(model, anchor, loader, base_cfg, cfg):
    """One selection metric for all arms, excluding the order-dependent auxiliary."""
    device, total, frames = next(model.parameters()).device, 0.0, 0
    with torch.no_grad(), fixed_streaming(model, cfg["chunk"], cfg["left"]), fixed_streaming(anchor, cfg["chunk"], cfg["left"]):
        for batch in loader:
            feats, lens = batch["feats"].to(device), batch["feat_lens"].to(device)
            lp, lengths = model(feats, lens)
            reference, anchor_lengths = anchor(feats, lens)
            if not torch.equal(lengths, anchor_lengths):
                raise ValueError("Validation anchor clock differs")
            primary, _, _ = primary_objective(lp, lengths, batch, base_cfg)
            preserved = preservation_loss(lp, reference, lengths)
            n = int(lengths.sum())
            total += float(primary + cfg["preservation_weight"] * preserved) * n
            frames += n
    if not frames:
        raise ValueError("Empty training-speaker validation partition")
    return total / frames


def train(recipe_path, anchor_path, help_root, reference_path, name, order):
    """Direct training entry point for the supervisor's train image; no Modal client.

    Run separately for reverse, forward and continuation with identical recipe
    and anchor. The recipe and actual corpus audit must be completed beforehand.
    """
    import numpy as np
    import student_train as T

    if order not in ("reverse", "forward", "continuation") or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Need an explicit arm and single-component run name")
    record = json.loads(Path(recipe_path).read_text())
    cfg = recipe_contract(record)
    if file_sha256(anchor_path) != record["anchor_sha256"]:
        raise ValueError("Initial checkpoint differs from the frozen audit")
    base = torch.load(anchor_path, map_location="cpu", weights_only=False)
    base_cfg = {**T.DEFAULTS, **base["cfg"]}
    rows, validation = load_recordings(record, base_cfg, help_root, reference_path)
    aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    model = T.build_model(base_cfg["size"]).to(device)
    model.load_state_dict(base["model"], strict=True)
    anchor = T.build_model(base_cfg["size"]).to(device).eval().requires_grad_(False)
    anchor.load_state_dict(base["model"], strict=True)
    decoder = TranscriptionDecoder(model.ctc_output[-1].in_features, model.ctc_output[-1].out_features,
                                   cfg["aux_dim"], cfg["aux_heads"], cfg["aux_layers"], cfg["history_dropout"]).to(device)
    # The continuation arm constructs the same decoder but applies no auxiliary
    # gradient; neither decoder contributes capacity to the exported recognizer.
    system = nn.ModuleDict({"primary": model, "auxiliary": decoder})
    optimizer = torch.optim.AdamW(system.parameters(), lr=cfg["learning_rate"])
    run = Path("/student/runs") / name
    run.mkdir(parents=True, exist_ok=True)
    state = dict(epoch=0, batch=0, step=0, best=math.inf)
    if (run / "training.pt").exists():
        saved = torch.load(run / "training.pt", map_location="cpu", weights_only=False)
        if saved["recipe"] != record or saved["order"] != order:
            raise ValueError("Resume cannot change the frozen matched arm")
        system.load_state_dict(saved["system"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
    valid_loader = torch.utils.data.DataLoader(
        RecordingDataset(validation, base_cfg, None, cfg["seed"], training=False),
        batch_sampler=T.batches(np.array([r["samples"] for r in validation]), cfg["batch_seconds"], 0, shuffle=False),
        collate_fn=collate_recordings, num_workers=0)

    def save():
        torch.save(dict(system=system.state_dict(), optimizer=optimizer.state_dict(), state=state,
                        recipe=record, order=order), run / "training.tmp")
        (run / "training.tmp").replace(run / "training.pt")

    def select():
        value = validation_objective(model, anchor, valid_loader, base_cfg, cfg)
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite validation objective")
        if value < state["best"]:
            state["best"] = value
            exported_cfg = {**base_cfg, "name": name}
            checkpoint_state = {"step": base["state"]["step"] + state["step"], "epoch": state["epoch"], "batch": state["batch"]}
            torch.save(primary_checkpoint(model, exported_cfg, checkpoint_state), run / "selected.tmp")
            (run / "selected.tmp").replace(run / "ckpt.pt")
            (run / "selection.json").write_text(json.dumps(dict(order=order, recipe=record, state=state, objective=value)))
        emit(event="validation", value=value, **state)

    def emit(**value):
        line = json.dumps(value)
        print(line, flush=True)
        with (run / "log.jsonl").open("a") as f:
            f.write(line + "\n")

    emit(event="setup", order=order, config=cfg, original_config=base_cfg, train=len(rows), validation=len(validation))
    while state["epoch"] < cfg["epochs"]:
        model.train()
        model.training_mode()
        decoder.train()
        plan = T.batches(np.array([r["samples"] for r in rows]), cfg["batch_seconds"], cfg["seed"] + state["epoch"])
        loader = torch.utils.data.DataLoader(
            RecordingDataset(rows, base_cfg, aug, cfg["seed"], state["epoch"]),
            batch_sampler=plan[state["batch"]:], collate_fn=collate_recordings, num_workers=0)
        for batch in loader:
            torch.manual_seed(cfg["seed"] + state["step"])
            for module in model.modules():
                module.batch_count = base["state"]["step"] + state["step"]
            values = joint_objective(model, anchor, decoder, batch, base_cfg, cfg, order, state["step"])
            if not all(torch.isfinite(v).all() for v in values.values()):
                raise RuntimeError("Nonfinite training loss; do not silently diverge matched arms")
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            # Decoder gradient magnitude must not scale the primary update in
            # either auxiliary arm relative to the continuation control.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 2.0)
            optimizer.step()
            state["batch"] += 1
            state["step"] += 1
            if state["step"] % 20 == 0:
                emit(event="train", **state, **{k: float(v.detach()) for k, v in values.items()})
            if state["step"] % cfg["validation_every"] == 0:
                select()
                save()
        state["epoch"] += 1
        state["batch"] = 0
        select()
        save()
    return {"name": name, "order": order, "objective": state["best"], "selected": str(run / "ckpt.pt")}
