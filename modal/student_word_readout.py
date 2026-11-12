"""Frozen word-ear readout training, independent of passage or warning labels.

The parent ear adapter supplies decoder states and its original output projection.
No ear, tokenizer, crop recipe or generation policy is substituted here. Cached
rollouts are made before either matched arm is fitted. This module deliberately
does not load practice audio or infer performed text from mistake scripts.
"""

from dataclasses import dataclass
import hashlib
from typing import Protocol, Sequence


def _pinned_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class Correspondence:
    # Recorded token to predict at each generated-prefix state, including the
    # final state before EOS. A deletion at a boundary leaves that token as the
    # recovery target rather than silently advancing past it.
    next_recorded: tuple[int, ...]
    operations: tuple[str, ...]


def correspondence(recorded: Sequence[int], generated: Sequence[int]) -> Correspondence:
    """Global edit correspondence; repeated tokens keep occurrence identity.

    Tokens exclude the fixed decoder prompt and EOS. Ties prefer a matching
    diagonal, then a substitution, then an insertion, then a deletion. Inserted
    output tokens do not advance the recorded continuation. Deleted recorded
    tokens are supervised at the state preceding the next generated token.
    """
    n, m = len(recorded), len(generated)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], back[i][0] = i, "delete"
    for j in range(1, m + 1):
        costs[0][j], back[0][j] = j, "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = recorded[i - 1] == generated[j - 1]
            choices = [(costs[i - 1][j - 1] + (not same), "match" if same else "replace"),
                       (costs[i][j - 1] + 1, "insert"), (costs[i - 1][j] + 1, "delete")]
            costs[i][j], back[i][j] = min(choices, key=lambda x: x[0])
    ops = []
    i, j = n, m
    while i or j:
        op = back[i][j]
        ops.append(op)
        if op != "insert":
            i -= 1
        if op != "delete":
            j -= 1
    ops.reverse()
    positions = [None] * (m + 1)
    i = j = 0
    for op in ops:
        if positions[j] is None:
            positions[j] = i
        if op != "insert":
            i += 1
        if op != "delete":
            j += 1
    if positions[j] is None:
        positions[j] = i
    return Correspondence(tuple(positions), tuple(ops))


@dataclass(frozen=True)
class TrainingRecording:
    identity: str
    speaker: str
    recorded_tokens: tuple[int, ...]
    source: str
    transcript_basis: str

    def validate(self, excluded_speakers: set[str]):
        if not self.identity or not self.speaker:
            raise ValueError("A recording and speaker identity are required")
        if self.speaker in excluded_speakers:
            raise ValueError("Release-held-out speaker is not eligible for fitting or selection")
        if self.source not in {"authorized-transcribed", "help-train"}:
            raise ValueError("Only authorized training recordings are eligible")
        permitted = {"verified-performed"} if self.source == "authorized-transcribed" else {"none-labelled"}
        if self.transcript_basis not in permitted:
            raise ValueError("Mistake scripts and teacher guesses are not performed transcripts")
        if not self.recorded_tokens:
            raise ValueError("An exact recorded continuation is required")


def speaker_partition(speaker: str) -> str:
    """Fixed speaker-grouped training partitions; never stratified by targets."""
    bucket = int.from_bytes(hashlib.sha256(speaker.encode()).digest()[:8], "big") % 10
    return "selection" if bucket < 2 else "fit"


class FrozenEar(Protocol):
    eos: int
    artifact_sha256: str
    generation_sha256: str

    def generate(self, audio): ...
    def states(self, audio, history: Sequence[int]): ...


def cache_recording(ear: FrozenEar, recording: TrainingRecording, audio,
                    excluded_speakers: set[str]) -> dict:
    """Cache both histories with the same frozen ear and the same actual audio.

    The adapter owns the unchanged fixed prompt, suppression policy and maximum
    length. `states` returns one state for each prefix, including the empty and
    complete history. Generation must return content tokens without prompt/EOS.
    """
    recording.validate(excluded_speakers)
    if not _pinned_digest(ear.artifact_sha256) or not _pinned_digest(ear.generation_sha256):
        raise ValueError("Pinned ear artifact and generation policy hashes are required")
    generated = tuple(ear.generate(audio))
    if ear.eos in generated or ear.eos in recording.recorded_tokens:
        raise ValueError("Content histories must not contain EOS")
    teacher_states = ear.states(audio, recording.recorded_tokens)
    rollout_states = ear.states(audio, generated)
    if len(teacher_states) != len(recording.recorded_tokens) + 1 or len(rollout_states) != len(generated) + 1:
        raise ValueError("Decoder states must correspond to every prefix, including the final one")
    aligned = correspondence(recording.recorded_tokens, generated)
    next_tokens = recording.recorded_tokens + (ear.eos,)
    return dict(identity=recording.identity, speaker=recording.speaker,
                source=recording.source, transcript_basis=recording.transcript_basis, eos=ear.eos,
                partition=speaker_partition(recording.speaker),
                ear_sha256=ear.artifact_sha256, generation_sha256=ear.generation_sha256,
                recorded=recording.recorded_tokens, generated=generated, operations=aligned.operations,
                teacher=dict(states=teacher_states, targets=next_tokens),
                rollout=dict(states=rollout_states, targets=tuple(next_tokens[i] for i in aligned.next_recorded)))


def validate_caches(caches, excluded_speakers: set[str]):
    """Recheck saved provenance and correspondence before fitting either arm."""
    if not caches:
        raise ValueError("Nonempty paired histories are required")
    for c in caches:
        TrainingRecording(c["identity"], c["speaker"], tuple(c["recorded"]),
                          c["source"], c["transcript_basis"]).validate(excluded_speakers)
        if not _pinned_digest(c["ear_sha256"]) or not _pinned_digest(c["generation_sha256"]):
            raise ValueError("Pinned ear artifact and generation policy hashes are required")
        if c["eos"] in c["recorded"] or c["eos"] in c["generated"]:
            raise ValueError("Content histories must not contain EOS")
        aligned = correspondence(c["recorded"], c["generated"])
        next_tokens = tuple(c["recorded"]) + (c["eos"],)
        if tuple(c["operations"]) != aligned.operations:
            raise ValueError("Saved operations differ from sequence correspondence")
        expected = {"teacher": next_tokens,
                    "rollout": tuple(next_tokens[i] for i in aligned.next_recorded)}
        for arm, targets in expected.items():
            if tuple(c[arm]["targets"]) != targets or len(c[arm]["states"]) != len(targets):
                raise ValueError("Cached states and targets differ from recorded continuation")
    if len({c["identity"] for c in caches}) != len(caches):
        raise ValueError("Duplicate recording in the training cache")
    if len({(c["ear_sha256"], c["generation_sha256"]) for c in caches}) != 1:
        raise ValueError("Matched arms must use one pinned frozen ear and generation policy")
    if any(c["partition"] != speaker_partition(c["speaker"]) for c in caches):
        raise ValueError("Cache partition differs from the fixed speaker split")
    fit = [c for c in caches if c["partition"] == "fit"]
    selection = [c for c in caches if c["partition"] == "selection"]
    if not fit or not selection or {c["speaker"] for c in fit} & {c["speaker"] for c in selection}:
        raise ValueError("Disjoint nonempty speaker groups are required for fitting and selection")
    return fit, selection


def fit_readouts(caches, frozen_projection, *, excluded_speakers: set[str], rank=32, epochs=8, seed=0):
    """Matched residual heads; checkpoint selection uses speaker-held-out caches.

    Both arms receive every paired recording and identical updates, initialization,
    regularization, optimizer and selection rule. The rollout arm averages teacher
    and generated-history losses per recording; the control uses teacher history.
    The base projection and every cached decoder state stay frozen. Hyperparameters
    must be fixed before target evaluation. Heavy fitting runs on the supervisor.
    """
    import copy
    import torch
    from torch import nn
    from torch.nn import functional as F

    fit, selection = validate_caches(caches, excluded_speakers)
    if rank <= 0 or epochs <= 0:
        raise ValueError("Positive residual rank and fitting epoch count are required")
    frozen_projection.eval().requires_grad_(False)
    device = next(frozen_projection.parameters()).device
    d = caches[0]["teacher"]["states"].shape[-1]
    vocab = frozen_projection(caches[0]["teacher"]["states"][:1].to(device).float()).shape[-1]
    torch.manual_seed(seed)
    initial = nn.Sequential(nn.Linear(d, rank), nn.Tanh(), nn.Linear(rank, vocab, bias=False)).to(device)
    nn.init.zeros_(initial[-1].weight)

    def loss(head, history):
        h = history["states"].detach().to(device).float()
        with torch.no_grad():
            base = frozen_projection(h)
        logits = base + head(h)
        target = torch.as_tensor(history["targets"], device=device)
        # Identity preservation is the same in both arms, on the same history.
        preservation = F.kl_div(logits.log_softmax(-1), base.softmax(-1), reduction="batchmean")
        return F.cross_entropy(logits, target) + preservation

    results = {}
    for arm in ("teacher", "rollout"):
        head = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        best = None
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            for pos in torch.randperm(len(fit), generator=generator).tolist():
                c = fit[pos]
                optimizer.zero_grad()
                value = loss(head, c["teacher"])
                if arm == "rollout":
                    value = (value + loss(head, c["rollout"])) / 2
                value.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
            # Identical selection objective on the same recorded continuations.
            # Speaker averaging keeps prolific speakers from dominating selection.
            scores = {}
            with torch.no_grad():
                for c in selection:
                    value = (loss(head, c["teacher"]) + loss(head, c["rollout"])) / 2
                    scores.setdefault(c["speaker"], []).append(float(value))
            score = sum(sum(v) / len(v) for v in scores.values()) / len(scores)
            if best is None or score < best[0]:
                best = (score, epoch + 1, copy.deepcopy(head.state_dict()))
        if best is None:
            raise ValueError("At least one fitting epoch is required")
        head.load_state_dict(best[2])
        results[arm] = dict(head=head.eval(), selection_loss=best[0], epoch=best[1])
    return results
