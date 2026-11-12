"""Training-only policies for assistant distillation; no practice-set inputs."""

import hashlib
import math
from pathlib import Path


def draw_seed(seed: int, key: str, visit: int) -> int:
    """A recording's augmentation is independent of workers and the supervisory arm."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}:{visit}".encode()).digest()[:8], "little")


def speaker_partition(records: list[dict], excluded: set[str], seed: int,
                      selection_share: float = 0.1) -> tuple[list[dict], list[dict]]:
    if not 0 < selection_share < 1:
        raise ValueError("selection_share must be between zero and one")
    if any(not isinstance(r.get("speaker"), str) or not r["speaker"].strip() for r in records):
        raise ValueError("Every recording needs a known speaker group")
    # Deny the whole speaker, including their nominally train-labelled recordings.
    denied = excluded | {r["speaker"] for r in records if r["split"] != "train" or r.get("gold", False)}
    eligible = [r for r in records if r["speaker"] not in denied and r.get("supervised", False)]
    speakers = sorted({r["speaker"] for r in eligible}, key=lambda s: draw_seed(seed, s, 0))
    if len(speakers) < 2:
        raise ValueError("Need at least two authorized training speaker groups")
    selection = set(speakers[:max(1, min(len(speakers) - 1, math.ceil(len(speakers) * selection_share)))])
    train = sorted((r for r in eligible if r["speaker"] not in selection), key=lambda r: r["key"])
    val = sorted((r for r in eligible if r["speaker"] in selection), key=lambda r: r["key"])
    if len({r["key"] for r in eligible}) != len(eligible):
        raise ValueError("Duplicate recording key")
    return train, val


def help_none(label: dict) -> bool:
    """A planted script/repeat is never a performed transcript."""
    return (label.get("kind") == "none" and label.get("split") == "train"
            and label.get("scored") is True
            and all(not label.get(k) for k in ("events", "non_errors", "unsupported", "exclude", "excluded", "extra_mistake")))


def release_speakers(manifest: dict) -> set[str]:
    """Release metadata is used only to deny speakers, never to train or select."""
    denied = set()
    for rows in manifest["slices"].values():
        for r in rows:
            speaker = r.get("speaker")
            if not isinstance(speaker, str) or not speaker:
                raise ValueError("Unknown release speaker; cannot certify training exclusions")
            if speaker.startswith("tlog-cluster-"):
                denied.add("s1:" + speaker.removeprefix("tlog-cluster-"))
            elif r["source"].startswith("help"):
                denied.add("help:" + speaker)
            else:
                raise ValueError("Unrecognized release speaker namespace")
    return denied


def confined_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-root asset: {relative}")
    return path


def assistant_config(shipped: dict) -> dict:
    """Threefold FF width, same causal geometry and phoneme head dimensions."""
    return {**shipped, "feedforward_dim": ",".join(str(3 * int(n)) for n in shipped["feedforward_dim"].split(","))}


def budget_remaining(spent: float, target: float) -> bool:
    if not math.isfinite(spent) or not math.isfinite(target) or spent < 0 or target <= 0:
        raise ValueError("Compute budgets must be finite, positive GPU seconds")
    return spent < target


def require_matched_compute(bridge: dict, direct: dict) -> dict:
    """One indivisible optimizer update is the allowed budget rounding error."""
    target = sum(bridge[k] for k in ("assistant", "assistant_targets", "student"))
    spent = direct["student"]
    tolerance = direct["last_update"]
    if (not all(math.isfinite(x) and x >= 0 for x in (*bridge.values(), spent, tolerance))
            or target <= 0 or tolerance <= 0 or not target <= spent <= target + tolerance + 1e-6):
        raise ValueError(f"Unmatched training compute: bridge={target}, direct={spent}, tolerance={tolerance}")
    return {"bridgeGpuSeconds": target, "directGpuSeconds": spent, "roundingGpuSeconds": tolerance}
