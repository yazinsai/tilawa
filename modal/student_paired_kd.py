"""Teacher targets bound to the materialized waveform, never a recording key.

Waveform augmentation runs once in ClipDataset. Student fbank and the teacher's
own processor read this same immutable observation. SpecAugment remains a
student-only feature mask; it neither changes nor re-materializes the waveform.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import struct


@dataclass(frozen=True)
class AcousticInput:
    pcm: bytes
    sample_rate: int
    valid_samples: int

    def __post_init__(self):
        if not isinstance(self.pcm, bytes) or len(self.pcm) % 4:
            raise ValueError("PCM must be immutable float32 bytes")
        if self.sample_rate <= 0 or not 0 < self.valid_samples <= len(self.pcm) // 4:
            raise ValueError("Invalid acoustic extent")

    @classmethod
    def materialize(cls, wave, sample_rate, valid_samples=None):
        import numpy as np

        wave = np.asarray(wave, dtype="<f4")
        if wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Expected finite mono PCM")
        return cls(wave.tobytes(), sample_rate, len(wave) if valid_samples is None else valid_samples)

    def waveform(self):
        import numpy as np

        # The same extent feeds both front ends; artificial batch extension is
        # not audio. bytes owns the storage and cannot be mutated by a worker.
        return np.frombuffer(self.pcm, dtype="<f4")[:self.valid_samples]

    def valid_mask(self):
        import numpy as np

        return np.arange(len(self.pcm) // 4) < self.valid_samples

    def cache_key(self, teacher_identity):
        h = hashlib.sha256(b"tilawa-paired-waveform-v1\0")
        h.update(teacher_identity.encode())
        h.update(struct.pack("<QQ", self.sample_rate, self.valid_samples))
        h.update(self.valid_mask().tobytes())
        h.update(self.pcm)
        return h.hexdigest()


@dataclass(frozen=True)
class TopTargets:
    ids: bytes
    probabilities: bytes
    frames: int
    k: int

    @classmethod
    def materialize(cls, ids, probabilities):
        import numpy as np

        ids = np.asarray(ids, dtype="<i8")
        p = np.asarray(probabilities, dtype="<f4")
        if ids.ndim != 2 or ids.shape != p.shape or not ids.size:
            raise ValueError("Teacher targets must have matching non-empty time and class axes")
        if (ids < 0).any() or not np.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
            raise ValueError("Invalid teacher targets")
        return cls(ids.tobytes(), p.tobytes(), *ids.shape)

    def tensors(self):
        import numpy as np
        import torch

        shape = (self.frames, self.k)
        return (torch.from_numpy(np.frombuffer(self.ids, dtype="<i8").reshape(shape).copy()),
                torch.from_numpy(np.frombuffer(self.probabilities, dtype="<f4").reshape(shape).copy()))


class TargetCache:
    """Bounded, process-local cache, with explicit model and processor identity."""

    def __init__(self, infer, teacher_identity, capacity=64):
        if capacity < 1 or not teacher_identity:
            raise ValueError("Cache needs a teacher identity and positive capacity")
        self.infer, self.identity, self.capacity = infer, teacher_identity, capacity
        self.entries = OrderedDict()

    def get(self, observation):
        key = observation.cache_key(self.identity)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        targets = self.infer(observation)
        if not isinstance(targets, TopTargets):
            raise TypeError("Teacher must return materialized targets")
        self.entries[key] = targets
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return targets


def muaalem_targets(device="cpu", capacity=64):
    """Load one frozen teacher. No Modal launch or path access at import time."""
    import torch
    from quran_muaalem import Muaalem
    from student import TOKENS

    # Match TeacherPass's arithmetic: waveform pairing must not also change
    # the teacher's precision or the persisted target precision.
    teacher = Muaalem("obadx/muaalem-model-v3_2", device=device, dtype=torch.bfloat16)
    teacher.model.eval().requires_grad_(False)
    vocab = teacher.multi_level_tokenizer.id_to_vocab["phonemes"]
    if [vocab[i] for i in range(1, len(TOKENS) + 1)] != list(TOKENS):
        raise ValueError("Teacher alphabet differs from student")
    # Include actual parameters, rather than a mutable Hub model name alone.
    h = hashlib.sha256()
    for name, tensor in sorted(teacher.model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    h.update(json.dumps(teacher.processor.feature_extractor.to_dict(), sort_keys=True).encode())
    h.update(b"singleton-bf16-valid-mask-v1")

    def infer(observation):
        # Singleton inference makes the valid extent independent of other
        # recordings. The processor derives its attention mask from this same
        # trimmed observation, including its intentional paired-frame padding.
        feats = teacher.processor([observation.waveform().copy()], sampling_rate=observation.sample_rate,
                                  return_tensors="pt", padding=True)
        mask = feats["attention_mask"]
        with torch.no_grad():
            logits = teacher.model(**{k: v.to(device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                                      for k, v in feats.items()}, return_dict=False)[0]["phonemes"]
        n = max(1, round(int(mask.sum()) * logits.shape[1] / mask.shape[1]))
        p = logits[0, :n].float().softmax(-1)
        top_p, top_ids = p.topk(5, dim=-1)
        return TopTargets.materialize(top_ids.cpu().numpy(), top_p.cpu().numpy().astype("float16"))

    return TargetCache(infer, h.hexdigest(), capacity)


@dataclass(frozen=True)
class HeadExample:
    """Frozen encoder features from one admitted training-speaker observation.

    Callers admit only authorized transcripts and none-labelled help-train
    takes; neither benchmark audio nor scripted mistake transcripts belong here.
    Both arms use this exact feature snapshot and CTC transcript.
    """
    speaker: str
    source: str
    label: str
    observation: AcousticInput
    features: object
    transcript: object
    current_targets: TopTargets
    paired_targets: TopTargets

    def __post_init__(self):
        if self.source not in ("authorized-quran", "help-train"):
            raise ValueError("Only authorized training recordings may enter a head refit")
        if self.source == "help-train" and self.label != "none":
            raise ValueError("Scripted mistake labels are not performed transcripts")


def cache_head_example(model, item, teacher, *, speaker, source, label):
    """Extract one frozen snapshot from the actual student input and paired PCM.

    The caller sets the fixed streaming chunk regime before caching. The
    existing encoder/head interface is retained, so the fitted Linear state
    loads back into ctc_output[1] and uses the ordinary student_export path.
    """
    import torch

    if model.training:
        raise ValueError("Freeze dropout and encoder inference before caching")
    captured = []
    def capture(module, args):
        captured.append(args[0].detach().cpu().float().clone())
    handle = model.ctc_output.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            _, lengths = model(item["feats"][None], torch.tensor([len(item["feats"])]))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected one phone-head input per recording")
    n = int(lengths[0])
    features = captured[0][0, :n]
    current = TopTargets.materialize(item["top_ids"].numpy(), item["top_p"].numpy())
    return HeadExample(speaker, source, label, item["observation"], features, item["target"].clone(),
                       current, teacher.get(item["observation"]) if item["acoustic_mismatch"] else current)


def refit_heads(initial_head, train, validation, *, training_speakers, held_out_speakers,
                steps=128, seed=0, lr=1e-4, kd_weight=2.0, anchor_weight=2.0, delay=0):
    """Bounded CPU phone-head refit and identical-data current-target control.

    All waveform materialization, encoder inference and teacher inference must
    finish before this call. There is no encoder update or data re-augmentation.
    One fixed optimization budget, no practice-dependent constants or selection.
    The shared checkpoint selector uses transcript loss plus ordinary-output
    preservation on the same speaker-disjoint training-only validation items.
    """
    import copy
    import numpy as np
    import torch
    import torch.nn.functional as F
    from student import soft_target_loss

    if not train or not validation or not 1 <= steps <= 512:
        raise ValueError("Need train/validation items and a bounded update count")
    train_speakers = {e.speaker for e in train}
    validation_speakers = {e.speaker for e in validation}
    if train_speakers & validation_speakers:
        raise ValueError("Checkpoint selection must be speaker-disjoint")
    admitted = train_speakers | validation_speakers
    if not admitted <= set(training_speakers) or admitted & set(held_out_speakers):
        raise ValueError("Unadmitted or release-held-out training speaker")
    original = copy.deepcopy(initial_head).cpu().eval().requires_grad_(False)
    examples = list(train) + list(validation)
    # Own cached CPU tensors; caller edits cannot alter one arm's observations.
    features = [e.features.detach().cpu().float().clone() for e in examples]
    transcripts = [e.transcript.detach().cpu().long().clone() for e in examples]
    if any(x.ndim != 2 or len(x) == 0 for x in features):
        raise ValueError("Expected valid, unpadded encoder frames")
    with torch.no_grad():
        anchors = [original(x).log_softmax(-1) for x in features]
    order = np.random.default_rng(seed).integers(len(train), size=steps).tolist()
    results = {}

    def objective(head, i, target=None):
        lp = head(features[i]).log_softmax(-1)
        n = len(lp)
        ctc = F.ctc_loss(lp[:, None, :], transcripts[i], torch.tensor([n]),
                         torch.tensor([len(transcripts[i])]), blank=0, reduction="sum", zero_infinity=False) / n
        anchor = F.kl_div(lp, anchors[i], log_target=True, reduction="sum") / n
        loss = ctc + anchor_weight * anchor
        if target is not None:
            ids, p = target.tensors()
            idx = (torch.arange(n) - delay).clamp(0, len(ids) - 1)
            loss = loss + kd_weight * soft_target_loss(lp[None], ids[idx][None], p[idx][None],
                                                       torch.ones(1, n, dtype=torch.bool))
        return loss

    for arm in ("current", "paired"):
        head = copy.deepcopy(original).requires_grad_(True)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        history, best, best_step, best_score = [], None, None, float("inf")
        for step in range(steps + 1):
            with torch.no_grad():
                score = sum(float(objective(head, i)) for i in range(len(train), len(examples))) / len(validation)
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation loss")
            history.append(score)
            if score < best_score:
                best_score, best_step = score, step
                best = {k: v.detach().clone() for k, v in head.state_dict().items()}
            if step == steps:
                break
            i = order[step]
            target = examples[i].current_targets if arm == "current" else examples[i].paired_targets
            optimizer.zero_grad(set_to_none=True)
            loss = objective(head, i, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite refit loss")
            loss.backward()
            optimizer.step()
        head.load_state_dict(best)
        results[arm] = dict(head=head.eval(), step=best_step, validation=best_score, history=history)
    return results
