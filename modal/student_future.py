"""Teacher supervision averaged over audio beyond a streaming input deadline.

The student sees the original recording. Only teacher queries receive other
training recordings' continuations, at the original sample origin and length.
No emission shift, transcript edit, or masking of available past is performed.
"""
import hashlib

import numpy as np


def output_chunk(manifest):
    f = manifest["fbank"]
    for value in (manifest["window"], manifest["shift"], manifest["sampleRate"],
                  manifest["frameMs"], f["frameShiftMs"], f["frameLengthMs"]):
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid deployed streaming geometry")
    if any(type(manifest[k]) is not int for k in ("window", "shift", "sampleRate")):
        raise ValueError("Streaming windows must have integral geometry")
    rows = manifest["shift"] * f["frameShiftMs"] / manifest["frameMs"]
    if rows <= 0 or int(rows) != rows or manifest["window"] < manifest["shift"]:
        raise ValueError("Invalid deployed streaming geometry")
    return int(rows)


def input_deadline(frame, manifest):
    """Exclusive raw-sample bound, including the last feature analysis window.

    This conservatively protects the entire ONNX input window, including
    embedding lookahead and all cached history. Sample deadline itself is the
    first sample strictly beyond that window's receptive field.
    """
    if type(frame) is not int or frame < 0:
        raise ValueError("Invalid output frame")
    chunk = frame // output_chunk(manifest)
    f = manifest["fbank"]
    if not f["snipEdges"]:
        raise ValueError("Centered feature padding needs a separate sample contract")
    ms = (chunk * manifest["shift"] + manifest["window"] - 1) * f["frameShiftMs"] + f["frameLengthMs"]
    samples = ms * manifest["sampleRate"] / 1000
    if samples != int(samples):
        raise ValueError("Deadline is not on the input sample clock")
    return int(samples)


def replace_future(wave, donor, deadline):
    """Preserve all observable samples bit for bit; no splice fades into them."""
    wave, donor = np.asarray(wave), np.asarray(donor)
    if (wave.ndim != 1 or donor.ndim != 1 or not len(donor) or
            not np.isfinite(wave).all() or not np.isfinite(donor).all() or
            type(deadline) is not int or deadline < 0):
        raise ValueError("Invalid continuation input")
    result = wave.copy()
    if deadline < len(wave):
        # Every draw is a real contiguous training continuation. Refuse a
        # short donor instead of looping it or changing the recording length.
        if len(donor) < len(wave) - deadline:
            raise ValueError("Training continuation is too short")
        result[deadline:] = donor[:len(wave) - deadline]
    return result


def probabilities(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or not len(rows) or not rows.shape[1] or not np.isfinite(rows).all():
        raise ValueError("Invalid complete teacher log probabilities")
    p = np.exp(rows - rows.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def marginal_targets(infer_teacher, wave, donors, frames, manifest, *,
                     infer_student=None, student_rows=None, chunks=None):
    """Return ordinary and equal-probability-mixture targets on the same clock.

    Each chunk gets independent teacher encodes with futures starting strictly
    beyond its deployed input window. infer_teacher must extract features and
    normalization from each supplied waveform, never reuse full-recording
    features. The original future is not privileged with an extra mixture vote.
    A runtime audit can require identical supervised student rows for every
    intervention. At EOF, where no exclusive future exists, targets coincide.
    """
    if type(frames) is not int or frames <= 0 or not donors:
        raise ValueError("Need valid rows and alternative training continuations")
    if (infer_student is None) != (student_rows is None):
        raise ValueError("Student invariance audit requires its paired baseline")
    if student_rows is not None and len(student_rows) < frames:
        raise ValueError("Student baseline cannot cover the supervised rows")
    ordinary = probabilities(infer_teacher(wave))
    if len(ordinary) < frames:
        raise ValueError("Teacher cannot cover valid student rows; do not clamp time")
    ordinary = ordinary[:frames]
    marginal = ordinary.copy()
    audit = []
    chunk = output_chunk(manifest)
    starts = list(range(0, frames, chunk)) if chunks is None else list(chunks)
    if (not starts or len(set(starts)) != len(starts) or
            any(type(s) is not int or s < 0 or s >= frames or s % chunk for s in starts)):
        raise ValueError("Supervision must select distinct native output chunks")
    for start in starts:
        stop = min(frames, start + output_chunk(manifest))
        deadline = input_deadline(start, manifest)
        if deadline >= len(wave):
            continue
        total = np.zeros_like(ordinary[start:stop])
        for donor in donors:
            changed = replace_future(wave, donor, deadline)
            np.testing.assert_array_equal(changed[:deadline], wave[:deadline])
            if infer_student is not None:
                observed = infer_student(changed)
                np.testing.assert_array_equal(observed[:stop], student_rows[:stop],
                                              err_msg="Intervention changed student-observable evidence")
            p = probabilities(infer_teacher(changed))
            if p.shape[1] != ordinary.shape[1] or len(p) < stop:
                raise ValueError("Continuation changed the teacher alphabet or frame clock")
            total += p[start:stop]
        marginal[start:stop] = total / len(donors)
        audit.append(dict(start=start, stop=stop, deadline=deadline, draws=len(donors)))
    return ordinary.astype(np.float32), marginal.astype(np.float32), audit


def disagreement_rows(student, ordinary, marginal, consonants):
    """Same-frame identity disagreements removed by future averaging.

    This is acoustic diagnostic evidence only. A caller must additionally check
    that the consonant was planted and missing from the student token sequence;
    these frame counts alone never establish a recovered engine warning.
    """
    s, t, m = (np.asarray(p).argmax(-1) for p in (student, ordinary, marginal))
    if s.shape != t.shape or t.shape != m.shape:
        raise ValueError("Diagnostic distributions must share the native frame clock")
    return np.flatnonzero(np.isin(t, list(consonants)) & (s != t) & (m == s)).tolist()


def speaker_partition(speakers):
    speakers = list(speakers)
    if any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Need identified training speakers")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < 5:
        raise ValueError("Need at least five identified training speakers")
    return {s: "selection" if i % 5 == 0 else "fit" for i, s in enumerate(unique)}


def admitted_none(rows):
    """No script is converted to a performed transcript; train none takes only."""
    excluded = {r.get("speaker") for r in rows if r.get("split") != "train"}
    admitted = [r for r in rows if r.get("split") == "train" and r.get("kind") == "none"
            and r.get("speaker") and r["speaker"] not in excluded
            and r.get("scored") is True and not r.get("exclude") and not r.get("excluded")
            and all(r.get(k) == [] for k in ("events", "non_errors", "unsupported"))]
    ids = [r.get("id") for r in admitted]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate training recording identity")
    return admitted


def continuation_pool(records, groups, recipient):
    """Draw only from the recipient's training partition, never its recording."""
    return [r for r in records if r["id"] != recipient["id"]
            and groups[r["speaker"]] == groups[recipient["speaker"]]]
