"""Acoustic, monotone edits of an already decoded, timestamped phone sequence.

The lattice state is (acoustic frame, consumed source events, emitted target
phones). HOLD advances the clock, DELETE consumes a source event, COPY and
SUBSTITUTE consume both, and INSERT advances the clock without consuming the
source. Source-consuming actions are confined to that event's original span.
Every output has a real acoustic frame; deletion cannot advance the clock.
Equivalent scripts are summed, including substitute versus delete + insert.
There is no reference text, lexicon, or output language model at inference.
"""
import hashlib
import math

HOLD, DELETE, COPY, SUBSTITUTE, INSERT = range(5)


def source_events(frame_ids, previous=0):
    events = []
    for t, phone in enumerate(frame_ids):
        if phone and phone != previous:
            events.append([int(phone), t, t])
        elif phone and events and events[-1][0] == phone:
            events[-1][2] = t
        previous = phone
    return events


def valid_actions(t, i, source):
    if i == len(source):
        return (HOLD, INSERT)
    _, start, end = source[i]
    if t < start:
        return (HOLD, INSERT)
    if t > end:
        return ()
    return (DELETE, COPY, SUBSTITUTE) if t == end else tuple(range(5))


def _lse(values):
    # Keep the training graph intact, while allowing dependency-free oracles.
    if hasattr(values[0], "detach"):
        import torch
        return torch.logsumexp(torch.stack(values), 0)
    maximum = max(values)
    return maximum + math.log(sum(math.exp(v - maximum) for v in values))


def transitions(actions, phones, source, t, i):
    allowed = tuple(a for a in valid_actions(t, i, source)
                    if len(phones[t][i]) > 1 or a != SUBSTITUTE)
    if not allowed:
        return []
    norm = _lse([actions[t][i][a] for a in allowed])
    out = []
    for action in allowed:
        score = actions[t][i][action] - norm
        if action == HOLD:
            out.append((t + 1, i, None, score, action))
        elif action == DELETE:
            out.append((t, i + 1, None, score, action))
        elif action == COPY:
            out.append((t + 1, i + 1, source[i][0], score, action))
        else:
            payloads = [k for k in range(1, len(phones[t][i]) + 1)
                        if action == INSERT or k != source[i][0]]
            if not payloads:
                continue
            payload_norm = _lse([phones[t][i][k - 1] for k in payloads])
            for k in payloads:
                out.append((t + 1, i + (action == SUBSTITUTE), k,
                            score + phones[t][i][k - 1] - payload_norm, action))
    return out


def marginal_log_probability(actions, phones, source, target):
    """Forward sum over ALL legal scripts with exactly the performed target.

    Unlike CTC, adjacent equal output phones are distinct articulation events.
    Source spans, not a chosen target alignment, restrict source consumption.
    The lattice includes timing alternatives and equivalent edit operations.
    """
    frames = len(actions)
    initial = actions[0][0][HOLD] * 0
    forward = {(0, 0, 0): initial}
    for t in range(frames):
        for i in range(len(source) + 1):
            for j in range(len(target) + 1):
                key = (t, i, j)
                if key not in forward:
                    continue
                for nt, ni, phone, score, _ in transitions(actions, phones, source, t, i):
                    if phone is not None and (j == len(target) or phone != target[j]):
                        continue
                    destination = (nt, ni, j + (phone is not None))
                    value = forward[key] + score
                    forward[destination] = (_lse([forward[destination], value])
                                            if destination in forward else value)
    if (frames, len(source), len(target)) not in forward:
        raise ValueError("Performed target has no legal timed edit script")
    return forward[(frames, len(source), len(target))]


def decode(actions, phones, source):
    """Global Viterbi script, with one emitted event per acoustic frame at most."""
    frames = len(actions)
    best = {(0, 0): (0., [])}
    for t in range(frames):
        for i in range(len(source) + 1):
            if (t, i) not in best:
                continue
            value, script = best[(t, i)]
            for nt, ni, phone, score, action in transitions(actions, phones, source, t, i):
                candidate = value + float(score)
                if (nt, ni) not in best or candidate > best[(nt, ni)][0]:
                    best[(nt, ni)] = (candidate, script + [(action, phone, t, float(score))])
    if (frames, len(source)) not in best:
        raise ValueError("Source has no complete timed edit script")
    return best[(frames, len(source))]


def speaker_folds(speakers, excluded=(), count=5):
    if count < 3 or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Need identified speakers and at least three folds")
    if set(speakers) & set(excluded):
        raise ValueError("Release-held-out speaker entered training")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < count:
        raise ValueError("Too few distinct training speakers for cross-fitting")
    return {s: i % count for i, s in enumerate(unique)}


def aligned_edits(source, target):
    """Minimum-edit alignment for estimating a corruption channel, NOT training.

    The transducer objective above never conditions on this arbitrary script.
    Channel observations retain correct stretches as well as actual errors.
    """
    costs = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(len(source) + 1):
        costs[i][0] = i
    for j in range(len(target) + 1):
        costs[0][j] = j
    for i, x in enumerate(source, 1):
        for j, y in enumerate(target, 1):
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1,
                              costs[i - 1][j - 1] + (x != y))
    edits, i, j = [], len(source), len(target)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (source[i - 1] != target[j - 1]):
            edits.append((j - 1, target[j - 1], (source[i - 1],)))
            i, j = i - 1, j - 1
        elif j and costs[i][j] == costs[i][j - 1] + 1:
            edits.append((j - 1, target[j - 1], ()))
            j -= 1
        else:
            edits.append((j, None, (source[i - 1],)))
            i -= 1
    return list(reversed(edits))


class CrossFittedCorruptions:
    """Empirical channels estimated only from OTHER fit speakers' real errors.

    Channel inputs are a performed phone, its left neighbour and the shipped
    recognizer's aligned output (including empty/fragmented outputs). Selection
    speakers enter neither channel estimation nor augmentation. Each recording
    still trains on its own actual shipped decode as well as this cross-fit draw.
    """
    def __init__(self, observations, folds, selection=0):
        self.channels, self.members = {}, {}
        for held in range(max(folds.values()) + 1):
            channel, members = {}, set()
            for row in observations:
                speaker = row["speaker"]
                if folds[speaker] in (selection, held):
                    continue
                members.add(speaker)
                for position, wanted, heard in aligned_edits(row["source"], row["target"]):
                    left = row["target"][position - 1] if position else 0
                    channel.setdefault((left, wanted), []).append(heard)
                    channel.setdefault((None, wanted), []).append(heard)
            if not members:
                raise ValueError("Empty out-of-speaker corruption model")
            self.channels[held], self.members[held] = channel, members

    def draw(self, target, timed_target, speaker, folds, rng):
        fold = folds[speaker]
        if speaker in self.members[fold]:
            raise ValueError("Corruption model saw its recipient speaker")
        channel, events = self.channels[fold], []
        for j, (phone, start, end) in enumerate(timed_target):
            left = target[j - 1] if j else 0
            choices = channel.get((left, phone), channel.get((None, phone), [(phone,)]))
            heard = rng.choice(choices)
            fragments = channel.get((left, None), ())
            # Fragment frequency is estimated against ALL phone observations.
            # No practice-dependent insertion rate or handwritten phone model.
            if fragments and rng.randrange(len(choices) + len(fragments)) < len(fragments):
                heard = rng.choice(fragments) + heard
            for k, h in enumerate(heard):
                at = start + (end - start) * k // max(1, len(heard))
                if events and at <= events[-1][2]:
                    at = events[-1][2] + 1
                if at <= end:
                    events.append([h, at, at])
        return events


def build_head(width, vocabulary, hidden=128):
    """Shared acoustic representation, edit scores, and independent CTC support.

    A fixed-width source prefix representation consumes actual decoded events,
    not posterior rows. Its causal cumulative embedding changes only when a
    source event is consumed. The direct-CTC arm uses the same network with the
    source context zeroed; both arms evaluate every lattice-state representation.
    """
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.acoustic = nn.Linear(width, hidden)
            self.embedding = nn.Embedding(vocabulary, hidden, padding_idx=0)
            self.position = nn.Linear(1, hidden, bias=False)
            self.correction = nn.Linear(hidden, width)
            self.state_correction = nn.Linear(hidden, width)
            self.actions = nn.Linear(hidden, 5)
            self.phones = nn.Linear(width, vocabulary - 1)
            self.support = nn.Linear(width, vocabulary)
            self.direct_blank = nn.Linear(width, 1)

        def forward(self, features, source, starts, direct=False):
            # source/start have T+1 padded entries, covering every cursor i.
            embed = self.embedding(source)
            prefix = torch.cat((torch.zeros_like(embed[:, :1]), embed.cumsum(1)[:, :-1]), 1)
            denominator = torch.arange(1, source.shape[1] + 1, device=features.device)[None, :, None]
            context = embed + prefix / denominator
            if direct:
                context = context * 0
            time = torch.arange(features.shape[1], device=features.device)[None, :, None]
            delta = (time - starts[:, None, :]).to(features.dtype)[..., None] / features.shape[1]
            if direct:
                delta = delta * 0
            acoustic = self.acoustic(features)
            cursor = (starts[:, None, :] < time).sum(-1).clamp(max=source.shape[1] - 1)
            local = context.gather(1, cursor[..., None].expand(-1, -1, context.shape[-1]))
            corrected = features + self.correction(torch.tanh(acoustic + local))
            joint = torch.tanh(self.acoustic(corrected)[:, :, None, :] + context[:, None, :, :] + self.position(delta))
            support = self.support(corrected).log_softmax(-1)
            actions = self.actions(joint)
            states = corrected[:, :, None, :] + self.state_correction(joint)
            phones = self.phones(states)
            direct_lp = torch.cat((self.direct_blank(states), phones), -1).mean(2).log_softmax(-1)
            return actions, phones, support, direct_lp

    return Head()
