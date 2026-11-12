"""Training admission and speaker grouping, independent of acoustic runtimes."""
import hashlib
import math
import random

SEED = 1729


def admitted_help(labels):
    """A mistake script is not a transcript; admit complete clean train takes."""
    excluded = {r.get("speaker") for r in labels if r.get("split") != "train"}
    return [r for r in labels if r.get("speaker") and r["speaker"] not in excluded
            and r.get("split") == "train" and r.get("kind") == "none"
            and not any(r.get(k) for k in ("events", "non_errors", "unsupported", "exclude", "excluded"))
            and r.get("scored", True) is not False]


def partition(speakers, folds=3):
    if folds < 2 or any(not isinstance(s, str) or not s for s in speakers):
        raise ValueError("Speaker grouping requires identified speakers and at least two folds")
    unique = sorted(set(speakers), key=lambda s: hashlib.sha256(s.encode()).digest())
    if len(unique) < folds:
        raise ValueError("Not enough training speakers for grouped selection")
    return {speaker: i % folds for i, speaker in enumerate(unique)}


def recording_orders(count, epoch, seed=SEED):
    """Each head sees every recording once, in its own reproducible order."""
    if count < 1 or epoch < 0:
        raise ValueError("Recording orders require nonempty data and a valid epoch")
    result = []
    for peer in range(3):
        order = list(range(count))
        random.Random(seed + 100003 * peer + epoch).shuffle(order)
        result.append(order)
    return result


def deployment_candidates(strength, folds, preservation=.1):
    """Score each initialization using only its grouped training validation."""
    if not folds or any(len(fold) != 3 for fold in folds):
        raise ValueError("Selection requires three readouts in every training fold")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Invalid agreement strength")
    result = []
    for peer in range(3):
        values = [fold[peer] for fold in folds]
        if any(v["speakers"] < 1 or any(not math.isfinite(v[k]) or v[k] < 0 for k in ("per", "kl"))
               for v in values):
            raise ValueError("Empty or invalid training validation fold")
        score = sum((v["per"] + preservation * v["kl"]) * v["speakers"] for v in values) / sum(v["speakers"] for v in values)
        result.append(dict(strength=strength, head=peer, folds=values, score=score))
    return result


def select_deployment(candidates):
    if not candidates:
        raise ValueError("No training-validated deployment candidate")
    return min(candidates, key=lambda v: (v["score"], v["strength"], v["head"]))
