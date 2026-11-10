"""Occurrence-constrained, mass-preserving temporal CTC distillation.

Each teacher row is transported exactly once to a student row. Correspondences
are nondecreasing, remain inside the same forced transcript-phone occurrence,
and satisfy calibrated delay bounds. Several teacher rows may share a student
row, but their mass is added, never copied or renormalized at the destination.
The normalized forward sum marginalizes correspondences; its gradient gives
the posterior mixture of the transported teacher targets. No inference change
or additional phone class is required.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DelayBounds:
    """Inclusive student-minus-teacher displacement on the output frame clock."""

    lower: int
    upper: int

    def __post_init__(self):
        if type(self.lower) is not int or type(self.upper) is not int or self.lower > self.upper:
            raise ValueError("Invalid temporal correspondence bounds")


def input_deadline_samples(output_frame, geometry, sample_rate):
    """Latest real input sample available for a deployed output chunk.

    Includes the feature analysis window, not just the encoder's attention
    mask. Prefix recomputation also limits teacher convolutions and utterance
    normalization to that same deadline.
    """
    output_per_chunk = geometry["shift"] * geometry["feature_frame_ms"] / geometry["output_frame_ms"]
    if output_per_chunk <= 0 or output_per_chunk != int(output_per_chunk):
        raise ValueError("Input advance must emit an integral number of output rows")
    chunk = output_frame // int(output_per_chunk)
    last_feature = chunk * geometry["shift"] + geometry["window"] - 1
    deadline_ms = last_feature * geometry["feature_frame_ms"] + geometry["analysis_frame_ms"]
    samples = deadline_ms * sample_rate / 1000
    if samples != int(samples):
        raise ValueError("Feature deadline is not on the sample clock")
    return int(samples)


def restrict_teacher_prefix(infer, waveform, output_frames, geometry, sample_rate):
    """Teacher rows with exactly the deployed student's real-input deadlines.

    infer must compute its own features and normalization from the supplied PCM
    prefix. Passing precomputed full-utterance features defeats the restriction.
    No cropped-grid reset, attention-only approximation or row clamping occurs.
    The caller supplies the actual student's valid output count.
    """
    pieces, start = [], 0
    output_per_chunk = geometry["shift"] * geometry["feature_frame_ms"] / geometry["output_frame_ms"]
    if output_per_chunk != int(output_per_chunk) or output_per_chunk <= 0:
        raise ValueError("Invalid deployed output advance")
    while start < output_frames:
        end = min(output_frames, start + int(output_per_chunk))
        deadline = min(len(waveform), input_deadline_samples(start, geometry, sample_rate))
        rows = infer(waveform[:deadline])
        if len(rows) < end:
            raise ValueError("Restricted teacher output cannot cover the student's clock")
        pieces.append(rows[start:end])
        start = end
    return pieces


def matched_emissions(frame_labels, transcript, blank=0):
    """Exact greedy runs matched to distinct transcript occurrences by edit DP.

    Used for timing measurements, not training labels. Unheard or substituted
    phones return None; forced alignment must not turn their tiny probability
    into a claim that restricting lookahead merely delayed an emission.
    """
    runs, previous = [], blank
    for frame, label in enumerate(frame_labels):
        if label != blank:
            if label == previous:
                runs[-1][2] = frame
            else:
                runs.append([label, frame, frame])
        previous = label
    labels = list(transcript)
    table = [list(range(len(runs) + 1))]
    for i, label in enumerate(labels, 1):
        row = [i]
        for j, run in enumerate(runs, 1):
            row.append(min(table[-1][j] + 1, row[-1] + 1, table[-1][j - 1] + (label != run[0])))
        table.append(row)
    result, i, j = [None] * len(labels), len(labels), len(runs)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (labels[i - 1] != runs[j - 1][0]):
            if labels[i - 1] == runs[j - 1][0]:
                result[i - 1] = tuple(runs[j - 1][1:])
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return result


def derive_calibration(records, *, geometry, teacher_sha256, student_sha256, training_data_sha256):
    """Envelope of exact occurrence timing measured on training records only.

    Each record contains one transcript and full teacher, deployed student and
    prefix-restricted teacher argmax trajectories. Use run centres for timing;
    missing emissions are excluded from *timing measurements*, not relabelled
    as delayed consonants. The ordinary envelope is also saved for the card's
    premise test. Permitted transport includes the identity and the envelopes
    of both ordinary and receptive-field-restricted emission differences.
    """
    ordinary, restricted = [], []
    for record in records:
        if record.get("split") != "train":
            raise ValueError("Calibration cannot consume evaluation or practice records")
        transcript = record["transcript"]
        teacher = matched_emissions(record["teacher"], transcript)
        student = matched_emissions(record["student"], transcript)
        limited = matched_emissions(record["restricted_teacher"], transcript)
        for full, heard, cut in zip(teacher, student, limited):
            if full is not None and heard is not None:
                ordinary.append((sum(heard) - sum(full)) / 2)
            if full is not None and cut is not None:
                restricted.append((sum(cut) - sum(full)) / 2)
    if not ordinary or not restricted:
        raise ValueError("No exact occurrence timing available for calibration")
    record = dict(schema=1, split="train", occurrences=len(ordinary), restricted_occurrences=len(restricted),
                  frame_ms=geometry["output_frame_ms"], receptive_field=dict(geometry),
                  teacher_sha256=teacher_sha256, student_sha256=student_sha256, training_data_sha256=training_data_sha256,
                  ordinary_lower=math.floor(min(ordinary)), ordinary_upper=math.ceil(max(ordinary)),
                  lower=math.floor(min(0, *ordinary, *restricted)), upper=math.ceil(max(0, *ordinary, *restricted)))
    calibrated_bounds(record)
    return record


def ctc_occurrences(log_probs, transcript, blank=0):
    """Viterbi CTC occurrence IDs, including intervening alignment blanks.

    This is an alignment to a *transcript*, not a greedy-phone alignment. Equal
    phones at different transcript positions remain different occurrences.
    A blank before a phone belongs to that following occurrence; trailing blank
    belongs to the last occurrence. Boundaries are detached from the KD gradient.
    Impossible transcripts raise instead of dropping a row or a training item.
    """
    labels = [int(x) for x in transcript]
    if not labels or blank in labels or min(labels) < 0:
        raise ValueError("Temporal KD requires a nonempty, blank-free transcript")
    states = [blank]
    for label in labels:
        states.extend((label, blank))
    rows = log_probs.tolist()
    if not rows or any(len(row) <= max(states) for row in rows):
        raise ValueError("Empty trajectory or transcript outside the alphabet")
    previous = [-math.inf] * len(states)
    previous[0], previous[1] = rows[0][blank], rows[0][labels[0]]
    history = []
    for row in rows[1:]:
        current, parents = [], []
        for s, label in enumerate(states):
            allowed = [s]
            if s:
                allowed.append(s - 1)
            if s > 1 and label != blank and label != states[s - 2]:
                allowed.append(s - 2)
            parent = max(allowed, key=lambda a: previous[a])
            current.append(previous[parent] + row[label])
            parents.append(parent)
        previous = current
        history.append(parents)
    state = max((len(states) - 2, len(states) - 1), key=lambda s: previous[s])
    if not math.isfinite(previous[state]):
        raise ValueError("Transcript has no complete CTC alignment")
    path = [state]
    for parents in reversed(history):
        state = parents[state]
        path.append(state)
    path.reverse()
    return [min(s // 2, len(labels) - 1) for s in path]


def calibrated_bounds(record):
    """Reject guessed bounds and incomplete calibration records at training entry.

    The producer must measure both ordinary emission delays and the change
    under prefix-restricted teacher inference on training recordings. This
    validator does not establish that measurement; the experiment audit must
    verify the source records before using the returned bounds.
    """
    if record.get("schema") != 1 or record.get("split") != "train":
        raise ValueError("Temporal bounds must come from a training-only calibration")
    if record.get("occurrences", 0) <= 0 or record.get("restricted_occurrences", 0) <= 0:
        raise ValueError("Both ordinary and lookahead-restricted delays must be measured")
    for key in ("teacher_sha256", "student_sha256", "training_data_sha256"):
        value = record.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Calibration must identify its models and training data")
    geometry = record.get("receptive_field", {})
    needed = ("window", "shift", "feature_frame_ms", "analysis_frame_ms", "output_frame_ms")
    if any(type(geometry.get(k)) not in (int, float) or geometry[k] <= 0 for k in needed):
        raise ValueError("Calibration requires the deployed feature and receptive-field geometry")
    if geometry["window"] < geometry["shift"]:
        raise ValueError("Receptive field cannot be shorter than the input advance")
    if geometry["output_frame_ms"] != record.get("frame_ms"):
        raise ValueError("Calibration delay clock differs from the deployed output clock")
    return DelayBounds(record["lower"], record["upper"])


def correspondence_rows(teacher_occurrences, student_occurrences, bounds):
    """Sparse legal destinations. Never clip a source frame onto an edge row."""
    if not teacher_occurrences or not student_occurrences:
        raise ValueError("Empty temporal trajectory")
    for owners in (teacher_occurrences, student_occurrences):
        if owners[0] != 0 or any(b < a or b > a + 1 for a, b in zip(owners, owners[1:])):
            raise ValueError("Occurrence IDs must be contiguous and monotone")
    if teacher_occurrences[-1] != student_occurrences[-1]:
        raise ValueError("Trajectories have different transcript occurrences")
    return [
        [s for s in range(max(0, t + bounds.lower), min(len(student_occurrences), t + bounds.upper + 1))
         if student_occurrences[s] == occurrence]
        for t, occurrence in enumerate(teacher_occurrences)
    ]


def transport_partition(costs, destinations):
    """Differentiable log partition and uniform-prior normalizer over legal maps.

    costs[t][j] is the CE cost for source t at destinations[t][j]. A path chooses
    exactly one destination for *every* source row. No horizontal/vertical DTW
    moves that duplicate or drop supervision are used. The prior normalizer
    prevents wide timing bands from winning merely by counting more paths.
    """
    import torch

    if not costs or len(costs) != len(destinations):
        raise ValueError("Invalid correspondence costs")
    previous = {}
    counts = {}
    for t, (row, allowed) in enumerate(zip(costs, destinations)):
        if len(row) != len(allowed) or allowed != sorted(set(allowed)):
            raise ValueError("Invalid correspondence row")
        current, next_counts = {}, {}
        for j, s in enumerate(allowed):
            if t == 0:
                current[s], next_counts[s] = -row[j], 0.0
            else:
                sources = [a for a in previous if a <= s]
                if not sources:
                    continue
                current[s] = torch.logsumexp(torch.stack([previous[a] for a in sources]), 0) - row[j]
                n = [counts[a] for a in sources]
                top = max(n)
                next_counts[s] = top + math.log(sum(math.exp(x - top) for x in n))
        if not current:
            raise ValueError("No mass-conserving occurrence correspondence inside the calibrated bounds")
        previous, counts = current, next_counts
    n = list(counts.values())
    top = max(n)
    return torch.logsumexp(torch.stack(list(previous.values())), 0), top + math.log(sum(math.exp(x - top) for x in n))


def temporal_target_loss(student_log_probs, teacher_top_ids, teacher_top_p, transcript, bounds):
    """Single-utterance KD numerator and teacher mass; caller batches numerators.

    The teacher's original renormalized top-k distribution is retained on every
    row, including blank and alternative phones. Only its timing correspondence
    changes. Alignment uses a full-alphabet tensor with omitted probabilities
    floored solely for the detached forced-alignment calculation.
    """
    import torch

    if student_log_probs.ndim != 2 or teacher_top_ids.shape != teacher_top_p.shape:
        raise ValueError("Invalid posterior shapes")
    if teacher_top_p.ndim != 2 or teacher_top_p.shape[0] == 0:
        raise ValueError("Empty teacher trajectory")
    p = teacher_top_p.detach().float()
    if not torch.isfinite(p).all() or (p < 0).any() or (p.sum(-1) <= 0).any():
        raise ValueError("Invalid teacher probability mass")
    p = p / p.sum(-1, keepdim=True)
    ids = teacher_top_ids.long()
    vocab = student_log_probs.shape[-1]
    if (ids < 0).any() or (ids >= vocab).any():
        raise ValueError("Teacher alphabet differs from the student")
    dense = p.new_zeros((p.shape[0], vocab)).scatter_add(1, ids, p)
    teacher_owners = ctc_occurrences(dense.clamp_min(torch.finfo(p.dtype).tiny).log().cpu(), transcript)
    student_owners = ctc_occurrences(student_log_probs.detach().cpu(), transcript)
    destinations = correspondence_rows(teacher_owners, student_owners, bounds)
    costs = []
    for t, allowed in enumerate(destinations):
        selected = student_log_probs[allowed][:, ids[t]]
        costs.append(-(selected * p[t]).sum(-1))
    partition, prior = transport_partition(costs, destinations)
    # Teacher entropy is invariant under every correspondence. Omitting it,
    # like framewise soft_target_loss, does not alter the training gradient.
    return prior - partition, p.shape[0]
