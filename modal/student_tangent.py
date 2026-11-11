"""Acoustic response distillation; no inference-time architecture change.

Perturb shared PCM, not either model's already-normalized features. Every
response is evaluated deterministically in fp32. The teacher is frozen and
the student's response passes retain gradients through their *outputs* only;
there is no second-order differentiation through an acoustic input gradient.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class PerturbationGeometry:
    n_fft: int
    hop: int
    window: int
    time_frames: int
    frequency_bins: int
    epsilon: float

    def __post_init__(self):
        for name in ("n_fft", "hop", "window", "time_frames", "frequency_bins"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hop >= self.window or self.window > self.n_fft:
            raise ValueError("Require hop < window <= n_fft")
        if self.frequency_bins > self.n_fft // 2 + 1:
            raise ValueError("Frequency patch exceeds the spectrum")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be finite and between zero and one")


def training_recipe(record):
    """Require a completed training-only choice, never invent its geometry.

This checks provenance record completeness, not the truth of its corpus audit.
The producer must verify speaker authorization and exclusion independently.
Key lists identify training examples, never runtime recitations or verses.
"""
    if record.get("schema") != 1 or record.get("split") != "train":
        raise ValueError("Acoustic response recipe must be training-only")
    train = set(record.get("training_speakers", []))
    valid = set(record.get("validation_speakers", []))
    excluded = set(record.get("excluded_speakers", []))
    if not train or not valid or not excluded or train & valid or (train | valid) & excluded:
        raise ValueError("Require disjoint audited training, validation and excluded speakers")
    tk, vk = record.get("training_keys", []), record.get("validation_keys", [])
    if not tk or not vk or set(tk) & set(vk) or len(set(tk)) != len(tk) or len(set(vk)) != len(vk):
        raise ValueError("Require disjoint unique training and validation keys")
    speakers = record.get("speakers_by_key", {})
    if set(speakers) != set(tk) | set(vk):
        raise ValueError("Every selected training example needs audited speaker provenance")
    for keys, partition in ((tk, train), (vk, valid)):
        if any(speakers[k] not in partition for k in keys):
            raise ValueError("Example speaker differs from its frozen training partition")
        if not any(k.startswith("help:") for k in keys) or not any(not k.startswith("help:") for k in keys):
            raise ValueError("Both partitions must include Quran corpus and help-train none takes")
    if record.get("geometry_validation_examples", 0) <= 0:
        raise ValueError("Perturbation geometry has not been validated on training speakers")
    if record.get("real_pause_examples", 0) <= 0:
        raise ValueError("Real pause examples have not been audited in the training mix")
    if record.get("selection") != "minimum-validation-objective":
        raise ValueError("Checkpoint selection must be frozen within training-speaker partitions")
    for name in ("teacher_sha256", "student_sha256", "training_data_sha256"):
        value = record.get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing artifact/data hash: {name}")
    if record.get("sample_rate") != 16000 or record.get("frame_ms") != 40:
        raise ValueError("Recipe must retain the primary listener's sample and output clock")
    return PerturbationGeometry(**record["geometry"])


def localized_direction(wave, geometry, seed):
    """One compact, signed STFT patch, synthesized as a real PCM direction.

The perturbation is a relative change to existing spectral energy, not new
speech or a synthesized phoneme. Patch positions use a private RNG so matched
arms do not change one another's augmentation or dropout draws. Zero-energy
audio has a zero direction and hence a zero teacher/student response.
"""
    import torch

    if wave.ndim != 1 or not wave.is_floating_point() or not torch.isfinite(wave).all():
        raise ValueError("Expected finite mono PCM")
    if wave.numel() < geometry.n_fft:
        raise ValueError("Audio is too short for the frozen perturbation geometry")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    window = torch.hann_window(geometry.window, device=wave.device, dtype=wave.dtype)
    spectrum = torch.stft(wave, geometry.n_fft, hop_length=geometry.hop,
                          win_length=geometry.window, window=window, return_complex=True,
                          center=True, pad_mode="constant")
    nf, nt = spectrum.shape
    wt, wf = min(geometry.time_frames, nt), geometry.frequency_bins
    t = int(torch.randint(nt - wt + 1, (), generator=rng))
    f = int(torch.randint(nf - wf + 1, (), generator=rng))
    sign = 2 * int(torch.randint(2, (), generator=rng)) - 1
    # Nonzero tapered edges also give a meaningful one-bin/one-frame patch.
    a = torch.hann_window(wf + 2, periodic=False, device=wave.device, dtype=wave.dtype)[1:-1]
    b = torch.hann_window(wt + 2, periodic=False, device=wave.device, dtype=wave.dtype)[1:-1]
    mask = torch.zeros_like(spectrum.real)
    mask[f:f + wf, t:t + wt] = sign * a[:, None] * b[None, :]
    direction = torch.istft(spectrum * mask, geometry.n_fft, hop_length=geometry.hop,
                            win_length=geometry.window, window=window, center=True,
                            length=wave.numel())
    return direction.detach()


def paired_audio(wave, geometry, seed):
    direction = localized_direction(wave, geometry, seed)
    return wave, wave + geometry.epsilon * direction, wave - geometry.epsilon * direction


@contextmanager
def deterministic_student(model, chunk=16, left=128):
    """Turn off stochastic layers without changing the caller's training state."""
    flags = [(m, m.training) for m in model.modules()]
    encoder = getattr(model, "encoder", None)
    context = (encoder.chunk_size, encoder.left_context_frames) if encoder is not None else None
    try:
        model.eval()
        if hasattr(model, "streaming_eval"):
            model.streaming_eval(chunk, left)
        yield
    finally:
        if context is not None:
            encoder.chunk_size, encoder.left_context_frames = context
        for module, training in flags:
            module.training = training


def _check_triplet(triplet):
    import torch

    if len(triplet) != 3:
        raise ValueError("Need unperturbed, plus and minus responses")
    shape = triplet[0].shape
    if len(shape) != 3 or shape[-1] < 2 or any(x.shape != shape for x in triplet):
        raise ValueError("Responses must share batch, frame and full inventory dimensions")
    if any(not x.is_floating_point() for x in triplet):
        raise ValueError("Responses must be floating point log probabilities")
    return torch


def response_objective(student, teacher, valid, epsilon, mode):
    """Match symmetric derivatives of nonblank phone-versus-blank log odds.

Both signs first subtract their own unperturbed response. Antisymmetrizing
these changes cancels static priors and the even (curvature) part. Any model
specific constant offset therefore contributes no tangent loss.

The identical-input control matches ordinary teacher probabilities on plus
and minus audio. Both modes consume all classes and the same valid rows.
"""
    torch = _check_triplet(student)
    _check_triplet(teacher)
    if teacher[0].shape != student[0].shape:
        raise ValueError("Teacher/student frame clocks or inventories differ")
    if valid.shape != student[0].shape[:2] or valid.dtype != torch.bool or not valid.any():
        raise ValueError("Need nonempty explicit valid-frame mask")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Need positive finite epsilon")
    # Select before arithmetic: padded rows, including -inf sentinels, must
    # contribute neither NaNs nor gradients to the objective.
    s = [x[valid] for x in student]
    t = [x.detach()[valid] for x in teacher]
    if any(not torch.isfinite(x).all() for x in s + t):
        raise ValueError("Non-finite response on a valid frame")
    if mode == "ordinary":
        return sum(-(q.exp() * p.log_softmax(-1)).sum(-1).mean()
                   for p, q in zip(s[1:], t[1:])) / 2
    if mode != "tangent":
        raise ValueError("Response mode must be tangent or ordinary")

    def derivative(rows):
        odds = [x[:, 1:] - x[:, :1] for x in rows]
        change_plus, change_minus = odds[1] - odds[0], odds[2] - odds[0]
        return (change_plus - change_minus) / (2 * epsilon)

    return (derivative(s) - derivative(t)).square().mean()


def preservation_objective(student, anchor, valid):
    """Ordinary reading and real-pause preservation; no transcript-derived mask."""
    import torch
    import torch.nn.functional as F

    if (student.shape != anchor.shape or valid.shape != student.shape[:2]
            or valid.dtype != torch.bool or not valid.any()):
        raise ValueError("Preservation must use identical valid frames and inventories")
    q = anchor.detach()[valid]
    if not torch.isfinite(student[valid]).all() or not torch.isfinite(q).all():
        raise ValueError("Non-finite preservation target on a valid frame")
    return F.kl_div(student[valid], q, log_target=True, reduction="batchmean")


def state_sha256(model):
    """Teacher identity: names, shapes, dtypes and exact contiguous state bytes."""
    import json

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        header = json.dumps([name, list(value.shape), str(value.dtype)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        # View bytes so bfloat16 buffers also have an exact representation.
        import torch
        data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


class AcousticResponseProvider:
    """Fresh frozen-teacher targets on shared, already channel-augmented PCM.

The callbacks implement each model's own front end and must return singleton
full-inventory log-probability tensors on the original 40 ms grid. Only their
shared valid prefix is used: unpaired front-end edge rows are never repeated,
clamped or interpolated. Unequal output lengths are recorded for the audit.
"""

    def __init__(self, teacher, features, geometry, seed=0):
        self.teacher, self.features, self.geometry, self.seed = teacher, features, geometry, seed
        self.edge_rows = 0

    def loss(self, model, waves, keys, step, mode, device):
        import torch

        if not waves or len(waves) != len(keys):
            raise ValueError("Need one stable training key for each PCM item")
        values, counts = [], []
        with deterministic_student(model), torch.autocast(device.type, enabled=False):
            for wave, key in zip(waves, keys):
                seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{step}:{key}".encode()).digest()[:8], "little")
                audio = paired_audio(wave.float(), self.geometry, seed)
                with torch.no_grad():
                    target = [self.teacher(w).detach().float() for w in audio]
                predicted = []
                for w in audio:
                    f = self.features(w).to(device)
                    lp, lengths = model(f[None], torch.tensor([len(f)], device=device))
                    predicted.append(lp[:, :int(lengths[0])].float())
                if any(x.ndim != 3 or x.shape[0] != 1 for x in target + predicted):
                    raise ValueError("Response callbacks must return singleton frame tensors")
                if len({x.shape[1] for x in target}) != 1 or len({x.shape[1] for x in predicted}) != 1:
                    raise ValueError("A perturbation changed a model's frame count")
                n = min(target[0].shape[1], predicted[0].shape[1])
                self.edge_rows += abs(target[0].shape[1] - predicted[0].shape[1])
                valid = torch.ones((1, n), dtype=torch.bool, device=device)
                values.append(response_objective([x[:, :n] for x in predicted], [x[:, :n] for x in target],
                                                 valid, self.geometry.epsilon, mode))
                counts.append(n)
        return sum(v * n for v, n in zip(values, counts)) / sum(counts)


def muaalem_provider(record, device, seed):
    """Load the frozen teacher only for the explicit training opt-in."""
    import torch
    import torchaudio
    from quran_muaalem import Muaalem
    from student import TOKENS

    geometry = training_recipe(record)
    teacher = Muaalem("obadx/muaalem-model-v3_2", device=str(device), dtype=torch.float32)
    teacher.model.eval().float().requires_grad_(False)
    if state_sha256(teacher.model) != record["teacher_sha256"]:
        raise ValueError("Frozen teacher differs from the audited model state")
    alphabet = teacher.multi_level_tokenizer.id_to_vocab["phonemes"]
    if [alphabet[i] for i in range(1, len(TOKENS) + 1)] != list(TOKENS):
        raise ValueError("Frozen teacher alphabet differs from the primary listener")

    def infer(wave):
        inputs = teacher.processor([wave.cpu().numpy()], sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inputs["attention_mask"]
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lp = teacher.model(**inputs, return_dict=False)[0]["phonemes"].float().log_softmax(-1)
        n = round(int(mask.sum()) * lp.shape[1] / mask.shape[1])
        return lp[:, :n]

    def features(wave):
        return torchaudio.compliance.kaldi.fbank(wave.cpu()[None], num_mel_bins=80,
                                                sample_frequency=16000, dither=0.0,
                                                energy_floor=1e-10, high_freq=-400)

    return AcousticResponseProvider(infer, features, geometry, seed)


def help_none_rows(labels, record, partition):
    """Select only explicitly none-labelled takes of original training speakers.

An empty error list alone is insufficient: repeats and unsupported planted
mistakes can also have empty event lists. The original split stays authoritative
even when the training-only recipe lists the speaker as allowed.
"""
    training_recipe(record)
    keys = set(record[f"{partition}_keys"])
    wanted = {k for k in keys if k.startswith("help:")}
    selected = {}
    for row in labels:
        key = f"help:{row['id']}"
        if key not in wanted:
            continue
        if key in selected:
            raise ValueError("Duplicate help-train key")
        if (row.get("split") != "train" or row.get("kind") != "none" or row.get("scored") is not True
                or row.get("events") != [] or row.get("non_errors") != [] or row.get("unsupported") != []
                or row.get("excluded") or row.get("speaker") != record["speakers_by_key"][key]):
            raise ValueError("Selected help take is not an audited none-labelled training-speaker recording")
        if not row.get("ayahs"):
            raise ValueError("Selected help take lacks its authorized transcript")
        selected[key] = row
    if set(selected) != wanted:
        raise ValueError("Frozen help-train examples are missing")
    return [(key, selected[key]) for key in sorted(selected)]


def append_help_none(clips, root, reference_index, record, partition, teacher):
    """Append real help PCM and frozen static KD targets to the s1 clip layout.

No crops, synthetic pauses, splices, repeats or planted mistakes enter this
path. All original PCM, including pauses, is retained. A changed source file
or transcript is detected by the combined corpus fingerprint before training.
"""
    import io
    import json
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch
    from student import VOCAB, ctc_collapse, ids_phonemes

    root = Path(root).resolve()
    rows = help_none_rows([json.loads(s) for s in (root / "labels.jsonl").read_text().splitlines() if s.strip()],
                          record, partition)
    out = {k: list(clips[k]) for k in ("key", "ref", "hyp", "n_samples", "n_frames")}
    references = {f"{a['s']}:{a['a']}": a["p"] for a in reference_index["ayahs"]}
    flac, ids, probs = [clips["flac"]], [clips["ids"]], [clips["probs"]]
    flac_off, top_off = list(clips["flac_off"]), list(clips["top_off"])
    for key, row in rows:
        path = (root / row["wav"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Help recording path leaves its authorized corpus")
        wave, sr = sf.read(path, dtype="float32")
        if sr != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Help ingestion requires finite mono 16 kHz PCM")
        if len(wave) < record["geometry"]["n_fft"]:
            raise ValueError("Help take is shorter than the frozen perturbation geometry")
        transcript = "".join(references[a] for a in row["ayahs"])
        if not transcript:
            raise ValueError("Help take has no transcript phones")
        with torch.no_grad():
            lp = teacher(torch.from_numpy(wave)).detach().float().cpu()
        if (lp.ndim != 3 or lp.shape[0] != 1 or not lp.shape[1] or lp.shape[-1] != VOCAB
                or not torch.isfinite(lp).all()):
            raise ValueError("Invalid static teacher response for help take")
        top_p, top_i = lp[0].exp().topk(5, dim=-1)
        # Float FLAC would quantize: WAV/FLOAT preserves these source samples.
        audio = io.BytesIO()
        sf.write(audio, wave, sr, format="WAV", subtype="FLOAT")
        data = np.frombuffer(audio.getvalue(), dtype=np.uint8)
        flac.append(data)
        ids.append(top_i.numpy().astype(np.uint8).reshape(-1))
        probs.append(top_p.numpy().astype(np.float16).reshape(-1))
        flac_off.append(flac_off[-1] + len(data))
        top_off.append(top_off[-1] + top_i.numel())
        out["key"].append(key)
        out["ref"].append(transcript)
        out["hyp"].append(ids_phonemes(ctc_collapse(lp[0].argmax(-1).tolist())))
        out["n_samples"].append(len(wave))
        out["n_frames"].append(lp.shape[1])
    return {**out, "n_samples": np.asarray(out["n_samples"]), "n_frames": np.asarray(out["n_frames"]),
            "flac": np.concatenate(flac), "flac_off": np.asarray(flac_off),
            "ids": np.concatenate(ids), "probs": np.concatenate(probs), "top_off": np.asarray(top_off)}


def corpus_sha256(*partitions):
    """Fingerprint exact selected PCM, references and cached static KD targets."""
    import json

    digest = hashlib.sha256()
    for clips in partitions:
        digest.update(len(clips["key"]).to_bytes(8, "little"))
        for i in sorted(range(len(clips["key"])), key=lambda i: clips["key"][i]):
            header = json.dumps([clips["key"][i], clips["ref"][i], int(clips["n_samples"][i]),
                                 int(clips["n_frames"][i])],
                                ensure_ascii=False, separators=(",", ":")).encode()
            lo, hi = clips["flac_off"][i:i + 2]
            audio = clips["flac"][lo:hi].tobytes()
            lo, hi = clips["top_off"][i:i + 2]
            for block in (header, audio, clips["ids"][lo:hi].tobytes(), clips["probs"][lo:hi].tobytes()):
                digest.update(len(block).to_bytes(8, "little"))
                digest.update(block)
    return digest.hexdigest()
