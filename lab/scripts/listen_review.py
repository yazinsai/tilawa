"""Local listen-and-tag review for correction slips.

One command, stdlib + modal. Audio, clip ids and tags stay outside the repo
and off git. The page is blind: it never shows which model or rule flagged
an item.

  python scripts/listen_review.py
  python scripts/listen_review.py --summarize

Queues
------
A  Every held-out TLOG candidate the new rules flag, shipped model and a0w.
   Precision of those flags. This is the live-library risk.
B  Seeded stratified sample of 100 TLOG candidates (by locator kind).
   Candidate validity, and verified recall of shipped vs a0w, old vs new.
C  The located self-reported help slips (word the two models agreed on).

  python scripts/listen_review.py --queue tlog_mined
  python scripts/listen_review.py --queue clean_guard
  python scripts/listen_review.py --queue self_repair

Four sessions. A/B/C uses tags.json and queue.json. tlog_mined keeps that
same tags.json (tag ids are tlog_mined|...) and writes queue_mined.json.
Do not rename those two files: a hosted listen may already be mid-queue.
clean_guard uses clean_guard_tags.json. self_repair uses
self_repair_tags.json and queue_self_repair.json.

clean_guard is the jitter-stable same-phoneme edits from help clean dev/test
plus a v1 control pool, mixed with non-edited clean words. The page is blind.
Tags are "real lapse" vs "model mishear (recitation correct)". TLOG is not in
this queue. The manifest lives on the volume at help_slips/clean_guard/ and
is not rebuilt from audio here.

tlog_mined serves help_slips/tlog_mined/queue.json in manifest order. The
page button is M. --summarize writes provisional acted-schema rows.

self_repair is at most 30 blind clips (candidates plus controls) from
help_slips/self_repair/. The page does not show role, mined kind, or clip id.
--summarize exports verified slips as natural labels.

Held-out is the odd sha1 half of the clip id, same split as correction_eval.py.
New-rule flags and the help word spots are read from the volume
(help_slips/rule_flags.json, help_slips/help_located.jsonl), not recomputed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VOLUME = "zipformer-ctc-training"
SAMPLE_N = 100
SUB_SAMPLE = 400
SUB_SEED = 0
WINDOW_S = 1.5
QUEUES = ("A", "B", "C")
MINED_QUEUE = "tlog_mined"
# A/B/C and tlog_mined share tags.json. The mined queue file is queue_mined.json.
QUEUE_ORDER = ("A", "B", "C", MINED_QUEUE)
SCOREBOARD_KINDS = ("skip_word", "substitution", "vowel", "tajweed", "repeat")
SELF_REPAIR_QUEUE = "self_repair"
SELF_REPAIR_N = 30
SELF_REPAIR_CONTROLS = 10
SELF_REPAIR_TAGS = "self_repair_tags.json"
SELF_REPAIR_QUEUE_FILE = "queue_self_repair.json"
_SOURCE_RANK = {"help-clean": 0, "everyayah-dev": 1, "tlog": 2}
_HEARD_EXPORT = frozenset({"skipped", "wrong_word", "tajweed_only"})
LOCATOR_KINDS = ("omitted", "repeated", "restarted", "substituted")
REAL_KINDS = ("skipped", "wrong_word", "repeat", "restart", "tajweed_only")
VERDICTS = ("real", "not_slip", "unsure", "audio_bad")
# clean_guard is a separate listen. Verdicts are what was heard, not a locator kind.
GUARD_QUEUE = "clean_guard"
GUARD_VERDICTS = ("real_lapse", "model_mishear", "unsure", "audio_bad")
EDIT_CLASSES = ("vowel", "consonant_substitution", "deletion", "insertion")
# Point estimate on help edits only. Controls and v1 do not move the call.
HELP_LAPSE_HIGH = 0.50
HELP_LAPSE_LOW = 0.25
CLEAN_GUARD_HELP_N = 40
CLEAN_GUARD_V1_N = 10
CLEAN_GUARD_CONTROL_N = 10
OPS_K4_DRAWS = ("m3", "p3", "m5", "p5")
# Harakat, madd marks, and the tajweed vowel letters used by the phoneme vocab.
# A replace whose consonant skeleton is unchanged is a vowel edit.
_VOWEL_CHARS = set("َُِّْٰٕٓٔٲڇںۜۥۦ۪۾ـ") | set(chr(c) for c in (
    list(range(0x064B, 0x0653)) + [0x0670, 0x0640] + list(range(0x06D6, 0x06EE))
    + [0x0653, 0x0654, 0x0655, 0x0656, 0x0657, 0x0658]
))
AUDIO_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}
# Tie-break when a word's ops split evenly: a consonant change wins, then a vowel.
_CLASS_TIE = ("consonant_substitution", "vowel", "deletion", "insertion")
RULES = ("shipped_old", "shipped_new", "a0w_old", "a0w_new")
NEW_RULES = ("shipped_new", "a0w_new")
CORPUS_URL = "https://github.com/yazinsai/tilawa/releases/download/v0.3.0/zipformer_quran.json"
Z = 1.96

def repo_root() -> Path | None:
    """Repo that contains the public mushaf and lab/, whether or not this file sits in it."""
    start = Path(__file__).resolve()
    for parent in (start.parent, *start.parents):
        if (parent / "web" / "frontend" / "public" / "quran.json").is_file() and (parent / "lab").is_dir():
            return parent
    return None


REPO = repo_root()


def tlog_half(clip_id: str) -> str:
    """Even sha1 prefix → tune, odd → held. Matches correction_eval.tlog_half."""
    import hashlib

    n = int(hashlib.sha1(str(clip_id).encode()).hexdigest()[:8], 16)
    return "tune" if n % 2 == 0 else "held"


def _slip_sort_key(row: dict) -> tuple:
    return (str(row.get("id")), int(row["surah"]), int(row["ayah"]), int(row["word_index"]))


def slip_key(row: dict) -> tuple:
    return (str(row.get("id")), int(row["surah"]), int(row["ayah"]), int(row["word_index"]))


def select_eval_slips(rows: list[dict], n_sub: int = SUB_SAMPLE, seed: int = SUB_SEED) -> list[dict]:
    """All omitted / repeated / restarted rows, plus ``n_sub`` substituted rows."""
    keep = [row for row in rows if row.get("kind") in ("omitted", "repeated", "restarted")]
    subs = [row for row in rows if row.get("kind") == "substituted"]
    subs.sort(key=_slip_sort_key)
    if n_sub < len(subs):
        subs = random.Random(seed).sample(subs, n_sub)
    return keep + subs


def _largest_remainder(groups: dict[str, list], n: int, order: tuple[str, ...]) -> dict[str, int]:
    """Largest-remainder allocation. A tie goes to the smaller group."""
    total = sum(len(groups[key]) for key in order)
    alloc = {key: 0 for key in order}
    if total == 0 or n <= 0:
        return alloc
    n = min(n, total)
    raw = {key: n * len(groups[key]) / total for key in order}
    alloc = {key: min(int(math.floor(raw[key])), len(groups[key])) for key in order}
    left = n - sum(alloc.values())
    rank = sorted(
        order,
        key=lambda key: (raw[key] - math.floor(raw[key]), -len(groups[key])),
        reverse=True,
    )
    for key in rank:
        if left <= 0:
            break
        if alloc[key] >= len(groups[key]):
            continue
        alloc[key] += 1
        left -= 1
    return alloc


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Proportional sample by locator kind. Largest remainder, then a seeded draw.

    Groups are sorted before sampling so the draw does not depend on file order.
    """
    groups: dict[str, list[dict]] = {kind: [] for kind in LOCATOR_KINDS}
    for row in rows:
        kind = row.get("kind")
        if kind in groups:
            groups[kind].append(row)
    for kind in groups:
        groups[kind].sort(key=_slip_sort_key)
    alloc = _largest_remainder(groups, n, LOCATOR_KINDS)
    rng = random.Random(seed)
    out: list[dict] = []
    for kind in LOCATOR_KINDS:
        take = alloc[kind]
        if take:
            out.extend(rng.sample(groups[kind], take))
    return out


def _skeleton(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch not in _VOWEL_CHARS)


def _op_parts(op) -> tuple[str, str, str]:
    if isinstance(op, dict):
        return str(op.get("op") or ""), str(op.get("ref") or ""), str(op.get("hyp") or "")
    name = str(op[0]) if len(op) > 0 else ""
    ref = str(op[1]) if len(op) > 1 else ""
    hyp = str(op[2]) if len(op) > 2 else ""
    return name, ref, hyp


def edit_class(ops) -> str | None:
    """One class for a word's phoneme ops.

    A replace is a vowel when the consonant skeleton does not change.
    Mixed ops take a strict majority. An even split uses
    consonant substitution, then vowel, then deletion, then insertion.
    """
    counts: dict[str, int] = {name: 0 for name in EDIT_CLASSES}
    for op in ops or []:
        name, ref, hyp = _op_parts(op)
        if name == "delete":
            counts["deletion"] += 1
        elif name == "insert":
            counts["insertion"] += 1
        elif name == "replace":
            if _skeleton(ref) == _skeleton(hyp):
                counts["vowel"] += 1
            else:
                counts["consonant_substitution"] += 1
    best = max(counts.values())
    if best <= 0:
        return None
    winners = [name for name in _CLASS_TIE if counts[name] == best]
    return winners[0]


def _ops_key(row: dict) -> tuple:
    ops = tuple(tuple(_op_parts(op)) for op in (row.get("ops") or []))
    return (int(row["s"]), int(row["a"]), int(row["w"]), row.get("kind"), ops)


def stable_same_phone(clip: dict, draws: tuple[str, ...] = OPS_K4_DRAWS) -> list[dict]:
    """Base edits whose phoneme ops also appear in every named draw."""
    views = clip.get("views") or {}
    base = views.get("base")
    if not isinstance(base, list):
        return []
    kept = {_ops_key(row): row for row in base}
    for name in draws:
        rows = views.get(name)
        if not isinstance(rows, list):
            return []
        found = {_ops_key(row) for row in rows}
        kept = {key: row for key, row in kept.items() if key in found}
    return list(kept.values())


def _stratified_take(rows: list[dict], n: int, seed: int, class_of) -> list[dict]:
    """Equal share per edit class, then leftover slots to the smaller classes.

    A class smaller than its share is taken in full, so a rare vowel edit is
    not dropped in favour of the deletion mass.
    """
    groups: dict[str, list[dict]] = {name: [] for name in EDIT_CLASSES}
    for row in rows:
        kind = class_of(row)
        if kind in groups:
            groups[kind].append(row)
    for name in groups:
        groups[name].sort(key=_slip_sort_key)
    present = [name for name in EDIT_CLASSES if groups[name]]
    alloc = {name: 0 for name in EDIT_CLASSES}
    if present and n > 0:
        share = n // len(present)
        for name in present:
            alloc[name] = min(len(groups[name]), share)
        left = min(n, sum(len(groups[name]) for name in present)) - sum(alloc.values())
        # Smaller pools first. An equal size prefers vowel, then consonant, then deletion.
        order = sorted(present, key=lambda name: (len(groups[name]), EDIT_CLASSES.index(name)))
        while left > 0:
            moved = False
            for name in order:
                if left <= 0:
                    break
                if alloc[name] < len(groups[name]):
                    alloc[name] += 1
                    left -= 1
                    moved = True
            if not moved:
                break
    rng = random.Random(seed)
    out: list[dict] = []
    for name in EDIT_CLASSES:
        take = alloc[name]
        if take:
            out.extend(rng.sample(groups[name], take))
    return out


def _split_quota(n_dev: int, n_test: int, n: int) -> tuple[int, int]:
    """Even split, then give the unused quota to the split that still has rows."""
    if n <= 0 or n_dev + n_test <= 0:
        return 0, 0
    n = min(n, n_dev + n_test)
    half = n // 2
    extra = n - half
    # The odd item goes to whichever split is larger, so a thin split is not forced over quota.
    if n_test >= n_dev:
        take_test = min(n_test, extra)
        take_dev = min(n_dev, n - take_test)
    else:
        take_dev = min(n_dev, extra)
        take_test = min(n_test, n - take_dev)
    if take_dev + take_test < n:
        take_test = min(n_test, n - take_dev)
        take_dev = min(n_dev, n - take_test)
    return take_dev, take_test


def _edit_item(clip: dict, edit: dict, audio: dict) -> dict:
    start = int(edit["w"])
    end = int(edit.get("wend") or start + 1)
    if end <= start:
        end = start + 1
    row = {
        "id": str(clip["id"]),
        "surah": int(edit["s"]),
        "ayah": int(edit["a"]),
        "word_index": start,
    }
    pool = "help" if clip.get("set") == "help_clean" else "v1"
    return {
        "queue": GUARD_QUEUE,
        "tag_id": tag_id(GUARD_QUEUE, row),
        "id": row["id"],
        "surah": row["surah"],
        "ayah": row["ayah"],
        "word_index": start,
        "highlight": [start, end],
        "span": None,
        "pool": pool,
        "split": clip.get("split"),
        "role": "edit",
        "edit_class": edit_class(edit.get("ops")),
        "locator_kind": edit.get("kind"),
        "ops": [list(_op_parts(op)) for op in (edit.get("ops") or [])],
        "audio": audio,
    }


def _control_item(word: dict) -> dict:
    row = {
        "id": str(word["id"]),
        "surah": int(word["surah"]),
        "ayah": int(word["ayah"]),
        "word_index": int(word["word_index"]),
    }
    return {
        "queue": GUARD_QUEUE,
        "tag_id": tag_id(GUARD_QUEUE, row),
        "id": row["id"],
        "surah": row["surah"],
        "ayah": row["ayah"],
        "word_index": row["word_index"],
        "highlight": [row["word_index"], row["word_index"] + 1],
        "span": None,
        "pool": word["pool"],
        "split": word.get("split"),
        "role": "control",
        "edit_class": None,
        "locator_kind": None,
        "ops": [],
        "audio": word["audio"],
    }


def assemble_clean_guard(
    clips: list[dict],
    catalog: list[dict],
    *,
    seed: int = 0,
    help_n: int = CLEAN_GUARD_HELP_N,
    v1_n: int = CLEAN_GUARD_V1_N,
    control_n: int = CLEAN_GUARD_CONTROL_N,
) -> list[dict]:
    """Stable same-phoneme edits plus blind clean-word controls.

    ``clips`` are jitter dump rows (help_clean and v1 only; TLOG is ignored).
    ``catalog`` is every word of those clips, with ``audio``. A control is a
    word that has no base-decode slip. Help edits are split about evenly
    between dev and test. Within a split, each edit class gets an equal
    share and a class smaller than that share is taken in full. v1 uses the
    same equal split across classes. The returned list is shuffled so role
    and pool are not blocked.
    """
    audio_by_id: dict[str, dict] = {}
    words_by_id: dict[str, list[dict]] = {}
    for word in catalog:
        audio_by_id[str(word["id"])] = word["audio"]
        words_by_id.setdefault(str(word["id"]), []).append(word)

    help_dev: list[dict] = []
    help_test: list[dict] = []
    v1_rows: list[dict] = []
    base_slip_keys: set[tuple] = set()
    usable_ids: set[str] = set()
    for clip in clips:
        if clip.get("error") or clip.get("set") not in ("help_clean", "v1"):
            continue
        ident = str(clip["id"])
        if ident not in audio_by_id:
            continue
        usable_ids.add(ident)
        audio = audio_by_id[ident]
        for edit in (clip.get("views") or {}).get("base") or []:
            base_slip_keys.add((ident, int(edit["s"]), int(edit["a"]), int(edit["w"])))
        for edit in stable_same_phone(clip):
            item = _edit_item(clip, edit, audio)
            if item["edit_class"] not in EDIT_CLASSES:
                continue
            if item["pool"] == "help" and clip.get("split") == "dev":
                help_dev.append(item)
            elif item["pool"] == "help" and clip.get("split") == "test":
                help_test.append(item)
            elif item["pool"] == "v1":
                v1_rows.append(item)

    n_dev, n_test = _split_quota(len(help_dev), len(help_test), help_n)
    edits = []
    edits.extend(_stratified_take(help_dev, n_dev, seed, lambda row: row["edit_class"]))
    edits.extend(_stratified_take(help_test, n_test, seed + 1, lambda row: row["edit_class"]))
    edits.extend(_stratified_take(v1_rows, v1_n, seed + 2, lambda row: row["edit_class"]))
    chosen = {(row["id"], row["surah"], row["ayah"], row["word_index"]) for row in edits}

    def _clean_words(pool: str, split: str | None) -> list[dict]:
        out = []
        for ident, words in words_by_id.items():
            if ident not in usable_ids:
                continue
            for word in words:
                if word.get("pool") != pool:
                    continue
                if split is not None and word.get("split") != split:
                    continue
                key = (str(word["id"]), int(word["surah"]), int(word["ayah"]), int(word["word_index"]))
                if key in base_slip_keys or key in chosen:
                    continue
                out.append(word)
        out.sort(key=lambda word: (str(word["id"]), int(word["surah"]), int(word["ayah"]), int(word["word_index"])))
        return out

    help_ctrl_dev = _clean_words("help", "dev")
    help_ctrl_test = _clean_words("help", "test")
    v1_ctrl = _clean_words("v1", None)
    # 8 help + 2 v1 when asking for 10, so studio audio is not itself the tell.
    v1_ctrl_n = 0 if control_n <= 0 else min(len(v1_ctrl), max(1, round(control_n * CLEAN_GUARD_V1_N / (CLEAN_GUARD_HELP_N + CLEAN_GUARD_V1_N))))
    if control_n > 0 and not v1_ctrl:
        v1_ctrl_n = 0
    help_ctrl_n = min(control_n - v1_ctrl_n, len(help_ctrl_dev) + len(help_ctrl_test))
    if help_ctrl_n + v1_ctrl_n < control_n and len(v1_ctrl) > v1_ctrl_n:
        v1_ctrl_n = min(len(v1_ctrl), control_n - help_ctrl_n)
    n_cdev, n_ctest = _split_quota(len(help_ctrl_dev), len(help_ctrl_test), help_ctrl_n)
    rng = random.Random(seed + 3)
    controls = []
    if n_cdev:
        controls.extend(_control_item(word) for word in rng.sample(help_ctrl_dev, n_cdev))
    if n_ctest:
        controls.extend(_control_item(word) for word in rng.sample(help_ctrl_test, n_ctest))
    if v1_ctrl_n:
        controls.extend(_control_item(word) for word in rng.sample(v1_ctrl, v1_ctrl_n))
    items = edits + controls
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        if item["tag_id"] in seen:
            continue
        seen.add(item["tag_id"])
        unique.append(item)
    random.Random(seed + 4).shuffle(unique)
    return unique


def clean_guard_counts(items: list[dict]) -> dict:
    """Aggregate counts only. No clip ids."""
    out: dict = {"n": len(items), "by_role": {}, "edits": {}}
    for item in items:
        role = item.get("role") or "?"
        out["by_role"][role] = out["by_role"].get(role, 0) + 1
        if role != "edit":
            pool = item.get("pool") or "?"
            bucket = out.setdefault("controls", {})
            bucket[pool] = bucket.get(pool, 0) + 1
            continue
        pool = item.get("pool") or "?"
        kind = item.get("edit_class") or "?"
        cell = out["edits"].setdefault(pool, {})
        cell[kind] = cell.get(kind, 0) + 1
        if pool == "help":
            split = item.get("split") or "?"
            splits = out.setdefault("help_splits", {})
            splits[split] = splits.get(split, 0) + 1
    return out


def highlight_span(row: dict) -> tuple[int, int]:
    """Inclusive word index, exclusive end. Evidence may cover a short run."""
    start = int(row["word_index"])
    end = start + 1
    for side in (row.get("evidence") or {}).values():
        if isinstance(side, dict) and side.get("word_end"):
            end = max(end, int(side["word_end"]))
    if end <= start:
        end = start + 1
    return start, end


def tag_id(queue: str, row: dict) -> str:
    ident, surah, ayah, word = slip_key(row)
    return f"{queue}|{ident}|{surah}|{ayah}|{word}"


def wilson(k: int, n: int, z: float = Z) -> list[float] | None:
    """Wilson score interval for k successes in n trials. None when n is 0."""
    if n <= 0:
        return None
    phat = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / den
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / den
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "p": (k / n) if n else None, "ci95": wilson(k, n)}


def _tag_of(tags: dict, queue: str, item: dict) -> dict | None:
    tag = tags.get(item["tag_id"])
    if not isinstance(tag, dict):
        return None
    if tag.get("verdict") not in VERDICTS:
        return None
    return tag


def _decided(tag: dict | None) -> str | None:
    if not tag:
        return None
    verdict = tag.get("verdict")
    if verdict in ("real", "not_slip"):
        return verdict
    return None


def precision_of(items: list[dict], tags: dict, pred) -> dict:
    k = n = 0
    flagged = 0
    for item in items:
        if not pred(item):
            continue
        flagged += 1
        verdict = _decided(_tag_of(tags, item["queue"], item))
        if verdict is None:
            continue
        n += 1
        if verdict == "real":
            k += 1
    out = _rate(k, n)
    out["flagged"] = flagged
    return out


def recall_of(items: list[dict], tags: dict, rule: str) -> dict:
    """Verified recall: among decided real slips that were scored, fraction this rule caught."""
    k = n = unscored = 0
    for item in items:
        tag = _tag_of(tags, item["queue"], item)
        if _decided(tag) != "real":
            continue
        flags = item.get("flags")
        if not item.get("scored") or not isinstance(flags, dict) or rule not in flags:
            unscored += 1
            continue
        n += 1
        if flags.get(rule):
            k += 1
    out = _rate(k, n)
    out["unscored_real"] = unscored
    return out


def _rule_on(item: dict, rule: str) -> bool:
    flags = item.get("flags") or {}
    return bool(flags.get(rule))


def verified_slip_label(item: dict, tag: dict) -> dict | None:
    """Acted-label row for a heard real slip. The listener's key wins.

    ``wrong_word`` does not split vowel from consonant substitution, so a
    vowel candidate confirmed as a wrong word is stored as ``vowel``.
    """
    if tag.get("verdict") != "real":
        return None
    heard = tag.get("kind")
    mined = item.get("mined_kind")
    if heard == "skipped":
        label = "skip_word"
    elif heard == "wrong_word":
        label = "vowel" if mined == "vowel" else "substitution"
    elif heard == "tajweed_only":
        label = "tajweed"
    elif heard == "repeat":
        label = "repeat"
    elif heard == "restart":
        label = "restart"
    else:
        return None
    qc_pass = item.get("qc_pass") is True
    # A heard control is a blind-check failure, not a new eval word.
    in_scoreboard = qc_pass and label in SCOREBOARD_KINDS and item.get("role") != "control"
    return {
        "id": item.get("id"),
        "split": MINED_QUEUE,
        "label_kind": label,
        "surah": int(item["surah"]),
        "ayah": int(item["ayah"]),
        "word_index": int(item["word_index"]),
        "span_s": item.get("span"),
        "mapped": True,
        "source": MINED_QUEUE,
        "heard_kind": heard,
        "mined_kind": mined,
        "rule": item.get("rule") if label == "tajweed" else None,
        "stratum": item.get("stratum") or ("control" if item.get("role") == "control" else "slip"),
        "control": item.get("role") == "control",
        "qc_pass": qc_pass,
        # TLOG stays provisional until inference QC and a real listen both pass.
        "provisional": not in_scoreboard,
        "train_ready": False,
        "in_scoreboard": in_scoreboard,
    }


def summarize_mined(items: list[dict], tags: dict) -> tuple[dict, list[dict]]:
    counts: Counter = Counter()
    labels = []
    control_real = control_decided = 0
    for item in items:
        tag = _tag_of(tags, MINED_QUEUE, item)
        verdict = tag.get("verdict") if tag else "untagged"
        counts[verdict] += 1
        if item.get("role") == "control" and verdict in ("real", "not_slip"):
            control_decided += 1
            if verdict == "real":
                control_real += 1
        if tag and verdict == "real":
            label = verified_slip_label(item, tag)
            if label is not None:
                labels.append(label)
    by_kind = Counter(label["label_kind"] for label in labels if label["in_scoreboard"])
    block = {
        "n": len(items),
        "tagged": sum(counts[v] for v in VERDICTS),
        "untagged": counts["untagged"],
        "real": counts["real"],
        "not_slip": counts["not_slip"],
        "unsure": counts["unsure"],
        "audio_bad": counts["audio_bad"],
        "controls": sum(1 for item in items if item.get("role") == "control"),
        "sister_ayah": sum(1 for item in items if item.get("stratum") == "sister_ayah"),
        "control_check": _rate(control_real, control_decided),
        "provisional_labels": sum(1 for label in labels if label["provisional"]),
        "scoreboard_labels": dict(by_kind),
        "scoreboard_n": sum(by_kind.values()),
    }
    return block, labels


def summarize_queues(queues: dict[str, list[dict]], tags: dict) -> dict:
    report: dict = {"queues": {}}
    for name in QUEUES:
        if name not in queues:
            continue
        items = queues.get(name) or []
        counts = Counter()
        real_kinds: Counter = Counter()
        locator_real: Counter = Counter()
        locator_decided: Counter = Counter()
        for item in items:
            tag = _tag_of(tags, name, item)
            verdict = tag.get("verdict") if tag else "untagged"
            counts[verdict] += 1
            if verdict == "real":
                kind = tag.get("kind") if tag else None
                if kind in REAL_KINDS:
                    real_kinds[kind] += 1
                locator_real[item.get("locator_kind") or "?"] += 1
            if verdict in ("real", "not_slip"):
                locator_decided[item.get("locator_kind") or "?"] += 1
        block: dict = {
            "n": len(items),
            "tagged": sum(counts[v] for v in VERDICTS),
            "untagged": counts["untagged"],
            "real": counts["real"],
            "not_slip": counts["not_slip"],
            "unsure": counts["unsure"],
            "audio_bad": counts["audio_bad"],
            "real_kinds": dict(real_kinds),
        }
        if name == "A":
            union = precision_of(items, tags, lambda item: True)
            by_rule = {rule: precision_of(items, tags, lambda item, rule=rule: _rule_on(item, rule)) for rule in RULES}
            block["precision_new"] = union
            block["precision_by_rule"] = by_rule
            block["gate_shipped"] = _precision_gate(by_rule["shipped_new"], by_rule["shipped_old"])
            block["gate_a0w"] = _precision_gate(by_rule["a0w_new"], by_rule["a0w_old"])
        else:
            decided_n = counts["real"] + counts["not_slip"]
            block["validity"] = _rate(counts["real"], decided_n)
            validity_by_kind = {}
            for kind in LOCATOR_KINDS:
                validity_by_kind[kind] = _rate(locator_real[kind], locator_decided[kind])
            block["validity_by_locator_kind"] = validity_by_kind
            block["recall"] = {rule: recall_of(items, tags, rule) for rule in RULES}
        report["queues"][name] = block
    if "A" in report["queues"]:
        report["pass"] = report["queues"]["A"]["gate_shipped"] == "pass"
    if MINED_QUEUE in queues:
        block, labels = summarize_mined(queues[MINED_QUEUE], tags)
        report["queues"][MINED_QUEUE] = block
        report["verified_labels"] = labels
    else:
        report["verified_labels"] = []
    return report


def _precision_gate(new: dict, old: dict) -> str:
    if not new["n"] or not old["n"]:
        return "incomplete"
    if new["p"] + 1e-12 < old["p"]:
        return "fail"
    return "pass"


def format_report(report: dict) -> str:
    lines = []
    a = report["queues"].get("A")
    if a is not None:
        lines.append(
            f"A  items {a['n']}  tagged {a['tagged']}  "
            f"real {a['real']}  not_slip {a['not_slip']}  unsure {a['unsure']}  audio_bad {a['audio_bad']}"
        )
        lines.append("   new-flag precision " + _fmt_rate(a["precision_new"]))
        for rule in RULES:
            lines.append(f"   {rule:12} precision " + _fmt_rate(a["precision_by_rule"][rule]))
        lines.append(f"   gate shipped new vs shipped old: {a['gate_shipped']}")
        lines.append(f"   gate a0w new vs a0w old:         {a['gate_a0w']}")
    for name in ("B", "C"):
        if name not in report["queues"]:
            continue
        block = report["queues"][name]
        lines.append(
            f"{name}  items {block['n']}  tagged {block['tagged']}  "
            f"real {block['real']}  not_slip {block['not_slip']}  "
            f"unsure {block['unsure']}  audio_bad {block['audio_bad']}"
        )
        lines.append("   candidate validity " + _fmt_rate(block["validity"]))
        for kind in LOCATOR_KINDS:
            lines.append(f"   validity {kind:12} " + _fmt_rate(block["validity_by_locator_kind"][kind]))
        if block["real_kinds"]:
            kinds = " ".join(f"{k}={v}" for k, v in sorted(block["real_kinds"].items()))
            lines.append(f"   real kinds {kinds}")
        for rule in RULES:
            rec = block["recall"][rule]
            extra = f"  unscored_real {rec['unscored_real']}" if rec["unscored_real"] else ""
            lines.append(f"   recall {rule:12} " + _fmt_rate(rec) + extra)
    mined = report["queues"].get(MINED_QUEUE)
    if mined is not None:
        lines.append(
            f"M  items {mined['n']}  tagged {mined['tagged']}  "
            f"real {mined['real']}  not_slip {mined['not_slip']}  "
            f"unsure {mined['unsure']}  audio_bad {mined['audio_bad']}"
        )
        lines.append("   control slips " + _fmt_rate(mined["control_check"]))
        kinds = " ".join(f"{k}={v}" for k, v in sorted(mined["scoreboard_labels"].items())) or "none"
        lines.append(
            f"   provisional {mined['provisional_labels']}  "
            f"scoreboard labels {mined['scoreboard_n']}  {kinds}"
        )
        lines.append(
            f"   sister_ayah {mined['sister_ayah']}  "
            "(TLOG labels stay provisional until QC and a real listen both pass)"
        )
    if a is not None:
        lines.append(
            f"decision: {a['gate_shipped']} "
            "(shipped new-rule precision must be at least the shipped baseline)"
        )
    return "\n".join(lines) + "\n"


def _fmt_rate(stat: dict) -> str:
    if not stat["n"]:
        return f"—  (0 decided, {stat.get('flagged', '')})".rstrip()
    ci = stat["ci95"]
    ci_txt = "—" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    return f"{stat['k']}/{stat['n']} = {stat['p']:.3f}  {ci_txt}"


def _guard_verdict(tags: dict, item: dict) -> str | None:
    tag = tags.get(item["tag_id"])
    if not isinstance(tag, dict):
        return None
    verdict = tag.get("verdict")
    if verdict not in GUARD_VERDICTS:
        return None
    return verdict


def _lapse_rate(items: list[dict], tags: dict) -> dict:
    """Real lapses over decided items. Unsure and bad audio stay out."""
    k = n = 0
    for item in items:
        verdict = _guard_verdict(tags, item)
        if verdict not in ("real_lapse", "model_mishear"):
            continue
        n += 1
        if verdict == "real_lapse":
            k += 1
    return _rate(k, n)


def help_lapse_decision(rate: dict) -> str:
    """Point estimate on help edits.

    ≥ 50% real lapses: the clean guard is mislabelled. Re-score the killed
    high-sensitivity rows against a listening-verified guard.
    < 25%: the slips are model habits. That supports the confusion-prior filter.
    The band in between is inconclusive. An empty decided set is incomplete.
    """
    if not rate["n"] or rate["p"] is None:
        return "incomplete"
    if rate["p"] + 1e-12 >= HELP_LAPSE_HIGH:
        return "mislabelled_guard"
    if rate["p"] < HELP_LAPSE_LOW:
        return "model_habits"
    return "inconclusive"


_DECISION_TEXT = {
    "mislabelled_guard": (
        "≥50% real lapses on help: the guard is mislabelled; "
        "re-score killed high-sensitivity rows against a listening-verified guard"
    ),
    "model_habits": (
        "<25% real lapses on help: the slips are model habits; "
        "this supports the confusion-prior approach"
    ),
    "inconclusive": "25–50% real lapses on help: inconclusive, do not relabel the guard yet",
    "incomplete": "no decided help edits yet",
}


def _ci_crosses_cuts(rate: dict) -> bool:
    ci = rate.get("ci95")
    if not ci or rate.get("p") is None:
        return False
    lo, hi = ci
    return (lo < HELP_LAPSE_LOW < hi) or (lo < HELP_LAPSE_HIGH < hi)


def summarize_clean_guard(items: list[dict], tags: dict) -> dict:
    """Real-lapse fraction for help and v1, per edit class, plus blind controls."""
    edits = [item for item in items if item.get("role") == "edit"]
    controls = [item for item in items if item.get("role") == "control"]
    counts = Counter()
    for item in items:
        counts[_guard_verdict(tags, item) or "untagged"] += 1

    def block(subset: list[dict]) -> dict:
        rate = _lapse_rate(subset, tags)
        by_class = {}
        for kind in EDIT_CLASSES:
            chosen = [item for item in subset if item.get("edit_class") == kind]
            by_class[kind] = _lapse_rate(chosen, tags)
            by_class[kind]["items"] = len(chosen)
        rate["items"] = len(subset)
        rate["by_class"] = by_class
        return rate

    help_items = [item for item in edits if item.get("pool") == "help"]
    help_rate = block(help_items)
    help_rate["by_split"] = {
        split: _lapse_rate([item for item in help_items if item.get("split") == split], tags)
        for split in ("dev", "test")
    }
    v1_rate = block([item for item in edits if item.get("pool") == "v1"])
    decision = help_lapse_decision(help_rate)
    return {
        "n": len(items),
        "edits": len(edits),
        "controls": len(controls),
        "tagged": sum(counts[v] for v in GUARD_VERDICTS),
        "untagged": counts["untagged"],
        "real_lapse": counts["real_lapse"],
        "model_mishear": counts["model_mishear"],
        "unsure": counts["unsure"],
        "audio_bad": counts["audio_bad"],
        "help": help_rate,
        "v1": v1_rate,
        "control": {
            "all": _lapse_rate(controls, tags),
            "help": _lapse_rate([item for item in controls if item.get("pool") == "help"], tags),
            "v1": _lapse_rate([item for item in controls if item.get("pool") == "v1"], tags),
        },
        "decision": decision,
        "ci_crosses_cut": _ci_crosses_cuts(help_rate),
    }


def format_clean_guard(report: dict) -> str:
    lines = [
        f"clean_guard  items {report['n']}  edits {report['edits']}  controls {report['controls']}  "
        f"tagged {report['tagged']}  real_lapse {report['real_lapse']}  "
        f"model_mishear {report['model_mishear']}  unsure {report['unsure']}  "
        f"audio_bad {report['audio_bad']}  untagged {report['untagged']}"
    ]
    lines.append("   help real-lapse " + _fmt_rate(report["help"]))
    for split in ("dev", "test"):
        lines.append(f"   help {split:12} " + _fmt_rate(report["help"]["by_split"][split]))
    for kind in EDIT_CLASSES:
        lines.append(f"   help {kind:24} " + _fmt_rate(report["help"]["by_class"][kind]))
    lines.append("   v1   real-lapse " + _fmt_rate(report["v1"]))
    for kind in EDIT_CLASSES:
        lines.append(f"   v1   {kind:24} " + _fmt_rate(report["v1"]["by_class"][kind]))
    lines.append("   control false-lapse " + _fmt_rate(report["control"]["all"]))
    lines.append("   control help          " + _fmt_rate(report["control"]["help"]))
    lines.append("   control v1            " + _fmt_rate(report["control"]["v1"]))
    if report["ci_crosses_cut"]:
        lines.append("   help CI crosses 25% or 50%; the call still uses the point estimate")
    lines.append(f"decision: {report['decision']} ({_DECISION_TEXT[report['decision']]})")
    return "\n".join(lines) + "\n"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_words(corpus: dict) -> dict[tuple[int, int], list[str]]:
    out = {}
    for surah in corpus.get("surahs") or []:
        sn = int(surah["n"])
        for ayah in surah.get("ayahs") or []:
            words = []
            for word in ayah.get("w") or []:
                if len(word) > 2 and word[2]:
                    words.append(word[2])
                elif len(word) > 1:
                    words.append(word[1])
                else:
                    words.append(str(word[0]))
            out[(sn, int(ayah["n"]))] = words
    return out


def load_surah_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    names = {}
    for row in json.loads(path.read_text(encoding="utf-8")):
        sn = int(row["surah"])
        names.setdefault(sn, row.get("surah_name") or "")
    return names


def _flags_index(payload: dict) -> dict[tuple, dict]:
    out = {}
    for row in payload.get("slips") or []:
        flags = {rule: bool(row.get(rule)) for rule in RULES}
        out[slip_key(row)] = {"flags": flags, "scored": bool(row.get("scored", True))}
    return out


def _item_from_slip(queue: str, row: dict, audio: dict, flags: dict | None) -> dict:
    start, end = highlight_span(row)
    span = row.get("span_s")
    if not (isinstance(span, list) and len(span) == 2):
        span = None
    else:
        span = [float(span[0]), float(span[1])]
    scored = bool(flags and flags.get("scored"))
    return {
        "queue": queue,
        "tag_id": tag_id(queue, row),
        "id": str(row["id"]),
        "surah": int(row["surah"]),
        "ayah": int(row["ayah"]),
        "word_index": int(row["word_index"]),
        "highlight": [start, end],
        "span": span,
        "locator_kind": row.get("kind"),
        "audio": audio,
        "flags": (flags or {}).get("flags"),
        "scored": scored,
    }


def build_queues(
    candidates: list[dict],
    flags_payload: dict,
    help_rows: list[dict],
    help_audio: dict[str, str],
    seed: int,
) -> dict[str, list[dict]]:
    """Three shuffled queues. ``help_audio`` maps a help clip id to its wav name."""
    flags = _flags_index(flags_payload)
    held = [row for row in select_eval_slips(candidates) if tlog_half(row["id"]) == "held"]
    queue_a = []
    for row in held:
        info = flags.get(slip_key(row))
        if not info:
            continue
        if not any(info["flags"].get(rule) for rule in NEW_RULES):
            continue
        queue_a.append(
            _item_from_slip( "A", row, {"source": "tlog", "name": f"{row['id']}.flac"}, info)
        )
    queue_b = []
    # The volume flags were scored for this seed. Shuffle seed is separate.
    for row in stratified_sample(candidates, SAMPLE_N, SUB_SEED):
        info = flags.get(slip_key(row))
        queue_b.append(
            _item_from_slip("B", row, {"source": "tlog", "name": f"{row['id']}.flac"}, info)
        )
    queue_c = []
    for row in help_rows:
        name = help_audio.get(str(row["id"]))
        if not name:
            continue
        info = flags.get(slip_key(row))
        queue_c.append(_item_from_slip("C", row, {"source": "help", "name": name}, info))
    rng_salt = (seed + 1) * 1009
    out = {}
    for i, (name, rows) in enumerate((("A", queue_a), ("B", queue_b), ("C", queue_c))):
        random.Random(rng_salt + i).shuffle(rows)
        out[name] = rows
    return out


def _one_per_clip(rows: list[dict]) -> list[dict]:
    """Highest edit_tokens wins. Ties keep the lowest word index."""
    best: dict[str, dict] = {}
    ordered = sorted(rows, key=lambda row: (str(row.get("id")), int(row["surah"]), int(row["ayah"]), int(row["word_index"])))
    for row in ordered:
        key = str(row.get("id"))
        prev = best.get(key)
        if prev is None or int(row.get("edit_tokens") or 0) > int(prev.get("edit_tokens") or 0):
            best[key] = row
    return list(best.values())


def _repair_item(row: dict, role: str) -> dict:
    span = row.get("span_s")
    if not (isinstance(span, list) and len(span) == 2):
        span = None
    else:
        span = [float(span[0]), float(span[1])]
    return {
        "queue": SELF_REPAIR_QUEUE,
        "tag_id": tag_id(SELF_REPAIR_QUEUE, row),
        "id": str(row["id"]),
        "surah": int(row["surah"]),
        "ayah": int(row["ayah"]),
        "word_index": int(row["word_index"]),
        "highlight": [int(row["word_index"]), int(row["word_index"]) + 1],
        "span": span,
        "locator_kind": row.get("kind"),
        "audio": row["audio"],
        "role": role,
        "mined_kind": row.get("kind"),
        "source_set": row.get("source_set"),
        "flags": None,
        "scored": False,
    }


def select_self_repair_queue(
    candidates: list[dict],
    controls: list[dict],
    *,
    n: int = SELF_REPAIR_N,
    n_controls: int = SELF_REPAIR_CONTROLS,
    seed: int = 0,
) -> list[dict]:
    """≤ n clips. Help-clean candidates first, then controls, then a blind shuffle.

    One word per clip. Controls never outnumber candidates and never share a clip
    with a queued candidate. Unseen candidates only.
    """
    unseen = [row for row in candidates if row.get("unseen", True) and row.get("audio")]
    pool = _one_per_clip(unseen)
    by_source: dict[int, list[dict]] = {}
    for row in pool:
        rank = _SOURCE_RANK.get(str(row.get("source_set")), 9)
        by_source.setdefault(rank, []).append(row)
    for rank in by_source:
        by_source[rank].sort(key=_slip_sort_key)
    rng = random.Random(seed)
    n_cand = min(len(pool), max(0, n - n_controls))
    chosen: list[dict] = []
    for rank in sorted(by_source):
        if len(chosen) >= n_cand:
            break
        group = by_source[rank]
        need = n_cand - len(chosen)
        take = group if need >= len(group) else rng.sample(group, need)
        chosen.extend(take)
    n_ctrl = min(n_controls, len(chosen), n - len(chosen))
    used = {str(row["id"]) for row in chosen}
    ctrl_pool = _one_per_clip(
        [row for row in controls if row.get("audio") and str(row.get("id")) not in used]
    )
    ctrl_pool.sort(key=_slip_sort_key)
    rng_c = random.Random(seed + 1)
    picked_ctrl = ctrl_pool if n_ctrl >= len(ctrl_pool) else rng_c.sample(ctrl_pool, n_ctrl)
    items = [_repair_item(row, "candidate") for row in chosen]
    items.extend(_repair_item(row, "control") for row in picked_ctrl)
    random.Random(seed + 2).shuffle(items)
    return items


def natural_label(item: dict, tag: dict | None) -> dict | None:
    """Acted-label schema for a verified candidate. Controls and restarts stay out."""
    if item.get("role") != "candidate" or not tag or tag.get("verdict") != "real":
        return None
    heard = tag.get("kind")
    if heard not in _HEARD_EXPORT:
        return None
    mined = item.get("mined_kind")
    if heard == "skipped" or mined == "skip":
        kind = "skip_word"
    elif heard == "tajweed_only":
        kind = "tajweed"
    elif mined == "vowel":
        kind = "vowel"
    else:
        kind = "substitution"
    return {
        "kind": kind,
        "surah": int(item["surah"]),
        "ayah": int(item["ayah"]),
        "word": int(item["word_index"]) + 1,
        "elicitation": "natural",
        "origin": "natural",
        "grain": mined,
        "heard_kind": heard,
    }


def summarize_self_repair(items: list[dict], tags: dict) -> dict:
    def _real(group: list[dict]) -> dict:
        k = n = 0
        for item in group:
            verdict = _decided(_tag_of(tags, SELF_REPAIR_QUEUE, item))
            if verdict is None:
                continue
            n += 1
            if verdict == "real":
                k += 1
        return _rate(k, n)

    candidates = [item for item in items if item.get("role") == "candidate"]
    controls = [item for item in items if item.get("role") == "control"]
    labels = []
    for item in candidates:
        label = natural_label(item, _tag_of(tags, SELF_REPAIR_QUEUE, item))
        if label:
            labels.append(label)
    kinds: dict[str, int] = {}
    for label in labels:
        kinds[label["kind"]] = kinds.get(label["kind"], 0) + 1
    return {
        "n": len(items),
        "candidates": len(candidates),
        "controls": len(controls),
        "candidate_validity": _real(candidates),
        "control_false_real": _real(controls),
        "natural_n": len(labels),
        "natural_kinds": kinds,
        "natural_labels": labels,
    }


def format_self_repair(report: dict) -> str:
    lines = [
        f"self_repair items {report['n']}  candidates {report['candidates']}  controls {report['controls']}",
        "   candidate real " + _fmt_rate(report["candidate_validity"]),
        "   control real " + _fmt_rate(report["control_false_real"]),
        f"   natural labels {report['natural_n']}",
    ]
    if report.get("natural_kinds"):
        kinds = " ".join(f"{k}={v}" for k, v in sorted(report["natural_kinds"].items()))
        lines.append(f"   natural kinds {kinds}")
    return "\n".join(lines) + "\n"


def _outside_repo(path: Path) -> None:
    if REPO is None:
        return
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO.resolve())
    except ValueError:
        return
    raise SystemExit(f"{resolved} is inside the repo; audio and tags must stay outside it")


def _read_volume_file(remote: str) -> bytes:
    import modal

    vol = modal.Volume.from_name(VOLUME)
    data = bytearray()
    for chunk in vol.read_file(remote):
        data.extend(chunk)
    if not data:
        raise FileNotFoundError(remote)
    return bytes(data)


def _download(remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(_read_volume_file(remote))
    tmp.replace(dest)


def _volume():
    import modal

    return modal.Volume.from_name(VOLUME)


class Cache:
    def __init__(self, root: Path, *, tags_name: str = "tags.json", queue_name: str = "queue.json"):
        _outside_repo(root)
        self.root = root
        self.meta = root / "meta"
        self.audio = root / "audio"
        self.meta.mkdir(parents=True, exist_ok=True)
        self.audio.mkdir(parents=True, exist_ok=True)
        self.tags_path = root / tags_name
        self.queue_path = root / queue_name
        # Fixed name. A tlog_mined listen reads and writes this path only.
        self.mined_queue_path = root / "queue_mined.json"
        self._fetch_lock = threading.Lock()
        self._file_locks: dict[str, threading.Lock] = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._fetch_lock:
            lock = self._file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[key] = lock
            return lock

    def ensure_meta(self) -> dict[str, Path]:
        files = {
            "candidates": ("help_slips/tlog_candidates.jsonl", self.meta / "tlog_candidates.jsonl"),
            "manifest": ("help/manifest.json", self.meta / "manifest.json"),
            "located": ("help_slips/help_located.jsonl", self.meta / "help_located.jsonl"),
            "flags": ("help_slips/rule_flags.json", self.meta / "rule_flags.json"),
        }
        for _key, (remote, dest) in files.items():
            _download(remote, dest)
        return {key: dest for key, (_remote, dest) in files.items()}

    def ensure_mined(self, rebuild: bool = False) -> Path:
        dest = self.meta / "tlog_mined_queue.json"
        if rebuild and dest.is_file():
            dest.unlink()
        _download("help_slips/tlog_mined/queue.json", dest)
        return dest

    def ensure_self_repair(self) -> dict[str, Path]:
        files = {
            "candidates": ("help_slips/self_repair/candidates.jsonl", self.meta / "self_repair_candidates.jsonl"),
            "controls": ("help_slips/self_repair/controls.jsonl", self.meta / "self_repair_controls.jsonl"),
        }
        for _key, (remote, dest) in files.items():
            _download(remote, dest)
        return {key: dest for key, (_remote, dest) in files.items()}

    def audio_path(self, item: dict) -> Path:
        audio = item["audio"]
        # Names come from the manifest / clip id, never from the request path.
        name = Path(audio["name"]).name
        return self.audio / f"{audio['source']}__{name}"

    def remote_audio(self, item: dict) -> str:
        audio = item["audio"]
        remote = audio.get("remote")
        if isinstance(remote, str) and ".." not in remote.split("/"):
            if remote.startswith(("help/", "audio/tlog/", "phase0/corpora/")):
                return remote
        name = Path(audio["name"]).name
        if audio["source"] == "tlog":
            stem = name[:-5] if name.endswith(".flac") else name
            return f"audio/tlog/{stem}.flac"
        if audio["source"] == "v1":
            return f"help_slips/clean_guard/v1/{name}"
        if audio["source"] == "everyayah":
            raise FileNotFoundError("everyayah item has no remote")
        return f"help/{name}"

    def _local_v1(self, item: dict) -> Path | None:
        if item["audio"].get("source") != "v1" or REPO is None:
            return None
        path = REPO / "lab" / "benchmark" / "test_corpus" / Path(item["audio"]["name"]).name
        return path if path.is_file() else None

    def ensure_audio(self, item: dict) -> Path:
        dest = self.audio_path(item)
        lock = self._lock_for(str(dest))
        with lock:
            if dest.is_file() and dest.stat().st_size > 0:
                return dest
            local = self._local_v1(item)
            if local is not None:
                dest.write_bytes(local.read_bytes())
                return dest
            _download(self.remote_audio(item), dest)
        return dest

    def load_tags(self) -> dict:
        if not self.tags_path.is_file():
            return {}
        data = json.loads(self.tags_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def save_tags(self, tags: dict) -> None:
        tmp = self.tags_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(tags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.tags_path)


def ensure_corpus(cache: Cache) -> Path:
    if REPO is not None:
        local = REPO / "lab" / "data" / "zipformer" / "quran.json"
        if local.is_file():
            return local
    dest = cache.meta / "zipformer_quran.json"
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, dest)  # noqa: S310 — public release asset
    return dest


def _help_audio_map(manifest: dict) -> dict[str, str]:
    samples = manifest["samples"] if isinstance(manifest, dict) else manifest
    if isinstance(samples, dict):
        samples = list(samples.values())
    return {str(row["id"]): str(row["file"]) for row in samples if row.get("file")}


def build_from_cache(cache: Cache, seed: int, rebuild: bool) -> dict:
    if cache.queue_path.is_file() and not rebuild:
        saved = json.loads(cache.queue_path.read_text(encoding="utf-8"))
        if saved.get("seed") == seed and all(name in saved.get("queues", {}) for name in QUEUES):
            return saved
    paths = cache.ensure_meta()
    candidates = load_jsonl(paths["candidates"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    located = load_jsonl(paths["located"])
    flags = json.loads(paths["flags"].read_text(encoding="utf-8"))
    queues = build_queues(candidates, flags, located, _help_audio_map(manifest), seed)
    corpus = json.loads(ensure_corpus(cache).read_text(encoding="utf-8"))
    saved = {
        "seed": seed,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queues": queues,
        "counts": {name: len(rows) for name, rows in queues.items()},
        "words": {f"{s}:{a}": words for (s, a), words in load_words(corpus).items()},
        "surah_names": {
            str(k): v
            for k, v in load_surah_names(
                (REPO / "web" / "frontend" / "public" / "quran.json") if REPO else Path("/no/such/quran.json")
            ).items()
        },
    }
    tmp = cache.queue_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cache.queue_path)
    return saved


def build_mined(cache: Cache, seed: int, rebuild: bool) -> dict:
    """Load the mined queue from the volume. Order is the manifest order."""
    if cache.mined_queue_path.is_file() and not rebuild:
        saved = json.loads(cache.mined_queue_path.read_text(encoding="utf-8"))
        if saved.get("focus") == MINED_QUEUE and MINED_QUEUE in saved.get("queues", {}):
            return saved
    raw = json.loads(cache.ensure_mined(rebuild).read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, dict) else raw
    corpus = json.loads(ensure_corpus(cache).read_text(encoding="utf-8"))
    saved = {
        "seed": seed,
        "focus": MINED_QUEUE,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queues": {MINED_QUEUE: items},
        "counts": {MINED_QUEUE: len(items)},
        "words": {f"{s}:{a}": words for (s, a), words in load_words(corpus).items()},
        "surah_names": {
            str(k): v
            for k, v in load_surah_names(
                (REPO / "web" / "frontend" / "public" / "quran.json") if REPO else Path("/no/such/quran.json")
            ).items()
        },
    }
    tmp = cache.mined_queue_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(cache.mined_queue_path)
    return saved


def build_self_repair(cache: Cache, seed: int, rebuild: bool) -> dict:
    """Blind queue on queue_self_repair.json. Tags stay in self_repair_tags.json."""
    if cache.tags_path.name != SELF_REPAIR_TAGS:
        raise SystemExit("self_repair tags file must be self_repair_tags.json")
    path = cache.root / SELF_REPAIR_QUEUE_FILE
    if path.is_file() and not rebuild:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("seed") == seed and SELF_REPAIR_QUEUE in (saved.get("queues") or {}):
            return saved
    paths = cache.ensure_self_repair()
    items = select_self_repair_queue(
        load_jsonl(paths["candidates"]), load_jsonl(paths["controls"]), seed=seed
    )
    corpus = json.loads(ensure_corpus(cache).read_text(encoding="utf-8"))
    saved = {
        "seed": seed,
        "focus": SELF_REPAIR_QUEUE,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "queues": {SELF_REPAIR_QUEUE: items},
        "counts": {SELF_REPAIR_QUEUE: len(items)},
        "words": {f"{s}:{a}": words for (s, a), words in load_words(corpus).items()},
        "surah_names": {
            str(k): v
            for k, v in load_surah_names(
                (REPO / "web" / "frontend" / "public" / "quran.json") if REPO else Path("/no/such/quran.json")
            ).items()
        },
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(saved, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return saved


def _counts(saved: dict, tags: dict, queues: tuple[str, ...] = QUEUES, verdicts: tuple[str, ...] = VERDICTS) -> dict:
    counts = {}
    for name in queues:
        rows = saved["queues"][name]
        done = sum(
            1 for row in rows
            if row["tag_id"] in tags and (tags[row["tag_id"]] or {}).get("verdict") in verdicts
        )
        counts[name] = {"done": done, "n": len(rows)}
    return counts


def public_item(saved: dict, tags: dict, queue: str, index: int, mode=None) -> dict:
    verdicts = mode.verdicts if mode is not None else VERDICTS
    queues = mode.queues if mode is not None else QUEUES
    items = saved["queues"][queue]
    if not items:
        return {
            "queue": queue, "index": 0, "n": 0, "surah": 0, "ayah": 0, "surah_name": "",
            "words": [], "highlight": [0, 0], "span": None, "window_s": WINDOW_S,
            "tag": None, "counts": _counts(saved, tags, queues, verdicts), "audio": "", "empty": True,
        }
    item = items[index]
    key = f"{item['surah']}:{item['ayah']}"
    words = (saved.get("words") or {}).get(key) or []
    start, end = item["highlight"]
    if words and end > len(words):
        end = len(words)
    if words and start >= len(words):
        start, end = 0, 0
    tag = tags.get(item["tag_id"])
    public_tag = None
    if isinstance(tag, dict) and tag.get("verdict") in verdicts:
        public_tag = {
            "verdict": tag["verdict"],
            "kind": tag.get("kind"),
            "notes": tag.get("notes") or "",
        }
    return {
        "queue": queue,
        "index": index,
        "n": len(items),
        "surah": item["surah"],
        "ayah": item["ayah"],
        "surah_name": (saved.get("surah_names") or {}).get(str(item["surah"])) or "",
        "words": words,
        "highlight": [start, end],
        "span": item.get("span"),
        "window_s": WINDOW_S,
        "tag": public_tag,
        "counts": _counts(saved, tags, queues, verdicts),
        "audio": f"/audio?queue={queue}&i={index}",
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listen</title>
<style>
  :root { color-scheme: dark; --bg:#14120e; --fg:#f3ead8; --dim:#b3a48c; --mark:#e6c15a; --line:#3a3428; --ok:#8fbf88; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 "IBM Plex Sans", "Segoe UI", sans-serif; }
  main { max-width: 820px; margin: 0 auto; padding: 24px 20px 80px; }
  header { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  button, .ghost { background: #241f18; color: var(--fg); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; font: inherit; cursor: pointer; }
  button.on { border-color: var(--mark); color: var(--mark); }
  button:focus-visible { outline: 2px solid var(--mark); }
  .meta { color: var(--dim); margin: 14px 0; }
  .ayah { font: 34px/1.8 "Amiri", "Noto Naskh Arabic", "Scheherazade New", serif; direction: rtl; text-align: right; }
  mark { background: transparent; color: var(--mark); border-bottom: 3px solid var(--mark); padding: 0 2px; }
  audio { width: 100%; margin: 12px 0; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
  .kind { min-width: 7.5rem; }
  label { display: block; color: var(--dim); margin-top: 16px; }
  textarea { width: 100%; min-height: 72px; background: #1c1914; color: var(--fg); border: 1px solid var(--line); border-radius: 8px; padding: 8px; font: inherit; }
  kbd { font: 12px/1 ui-monospace, monospace; border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; color: var(--dim); }
  .status { color: var(--ok); min-height: 1.2em; }
  .err { color: #e07a7a; }
</style>
</head>
<body>
<main>
  <header id="queues"></header>
  <p class="meta" id="meta"></p>
  <p class="ayah" id="ayah" dir="rtl"></p>
  <audio id="player" preload="auto" controls></audio>
  <div class="row">
    <button id="play-window" type="button">Play window <kbd>space</kbd></button>
    <button id="play-full" type="button">Play full clip <kbd>f</kbd></button>
  </div>
  <div class="row" id="reals"></div>
  <div class="row">
    <button data-verdict="not_slip" type="button">Not a slip <kbd>n</kbd></button>
    <button data-verdict="unsure" type="button">Unsure <kbd>u</kbd></button>
    <button data-verdict="audio_bad" type="button">Audio bad <kbd>b</kbd></button>
  </div>
  <label>Notes <textarea id="notes" placeholder="optional"></textarea></label>
  <p class="status" id="status"></p>
  <div class="row">
    <button id="prev" type="button">Prev <kbd>←</kbd></button>
    <button id="next" type="button">Next <kbd>→</kbd></button>
  </div>
  <p class="meta">Real slip: <kbd>1</kbd> skipped <kbd>2</kbd> wrong word <kbd>3</kbd> repeat <kbd>4</kbd> restart <kbd>5</kbd> tajweed-only. Tags save on their own.</p>
</main>
<script>
const ORDER = /*ORDER*/;
const LABEL = {A: "A", B: "B", C: "C", tlog_mined: "M"};
const REALS = [
  ["skipped", "Skipped", "1"],
  ["wrong_word", "Wrong word", "2"],
  ["repeat", "Repeat", "3"],
  ["restart", "Restart", "4"],
  ["tajweed_only", "Tajweed only", "5"],
];
const VERDICT_KEYS = { n: "not_slip", u: "unsure", b: "audio_bad" };
let queue = ORDER[0];
let index = 0;
let item = null;
let stopAt = null;

const $ = (id) => document.getElementById(id);
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function renderQueues(counts) {
  $("queues").innerHTML = ORDER.map((name) => {
    const c = counts[name] || { done: 0, n: 0 };
    const label = LABEL[name] || name;
    return `<button type="button" data-queue="${name}" class="${name === queue ? "on" : ""}">${label} ${c.done}/${c.n}</button>`;
  }).join("");
}
function renderItem() {
  if (!item) return;
  renderQueues(item.counts);
  if (!item.n) {
    $("meta").textContent = queue + " is empty";
    $("ayah").textContent = "";
    return;
  }
  const ref = item.surah + ":" + item.ayah;
  const name = item.surah_name ? " · " + item.surah_name : "";
  $("meta").textContent = (LABEL[queue] || queue) + "  " + (index + 1) + "/" + item.n + "   " + ref + name;
  const [a, b] = item.highlight || [0, 0];
  $("ayah").innerHTML = (item.words || []).map((w, i) => (i >= a && i < b) ? "<mark>" + esc(w) + "</mark>" : esc(w)).join(" ");
  const player = $("player");
  const src = item.audio;
  if (player.dataset.src !== src) {
    player.dataset.src = src;
    player.src = src;
  }
  const tag = item.tag || {};
  $("notes").value = tag.notes || "";
  document.querySelectorAll("button[data-verdict], button[data-kind]").forEach((btn) => {
    const on = (btn.dataset.kind && tag.verdict === "real" && tag.kind === btn.dataset.kind)
      || (btn.dataset.verdict && tag.verdict === btn.dataset.verdict);
    btn.classList.toggle("on", !!on);
  });
}
async function load(q, i) {
  queue = q;
  const res = await fetch("/api/item?queue=" + q + "&i=" + i);
  if (!res.ok) {
    $("status").textContent = "Could not load this item.";
    $("status").className = "err";
    return;
  }
  item = await res.json();
  index = item.index;
  $("status").textContent = "";
  $("status").className = "status";
  renderItem();
}
async function save(patch, advance) {
  const body = {
    queue, index,
    verdict: patch.verdict,
    kind: patch.kind || null,
    notes: $("notes").value,
  };
  const res = await fetch("/api/tag", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    $("status").textContent = "Save failed.";
    $("status").className = "err";
    return;
  }
  const data = await res.json();
  $("status").textContent = "Saved.";
  $("status").className = "status";
  if (advance && data.next_untagged != null && data.next_untagged !== index) {
    await load(queue, data.next_untagged);
  } else {
    item.tag = data.tag;
    item.counts = data.counts;
    renderItem();
  }
}
function playWindow() {
  const player = $("player");
  stopAt = null;
  if (!item || !item.n) return;
  if (!item.span) { player.play(); return; }
  const start = Math.max(0, item.span[0] - item.window_s);
  const end = item.span[1] + item.window_s;
  stopAt = end;
  const go = () => { player.currentTime = start; player.play(); };
  if (player.readyState >= 1) go();
  else player.addEventListener("loadedmetadata", go, { once: true });
}
function playFull() {
  if (!item || !item.n) return;
  stopAt = null;
  const player = $("player");
  player.currentTime = 0;
  player.play();
}
$("player").addEventListener("timeupdate", () => {
  if (stopAt != null && $("player").currentTime >= stopAt) {
    $("player").pause();
    stopAt = null;
  }
});
$("play-window").addEventListener("click", playWindow);
$("play-full").addEventListener("click", playFull);
$("prev").addEventListener("click", () => { if (item && item.n) load(queue, Math.max(0, index - 1)); });
$("next").addEventListener("click", () => { if (item && item.n) load(queue, Math.min(item.n - 1, index + 1)); });
$("queues").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-queue]");
  if (btn) load(btn.dataset.queue, 0);
});
const reals = $("reals");
for (const [kind, label, key] of REALS) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "kind";
  btn.dataset.kind = kind;
  btn.innerHTML = esc(label) + " <kbd>" + key + "</kbd>";
  btn.addEventListener("click", () => save({ verdict: "real", kind }, true));
  reals.appendChild(btn);
}
document.querySelectorAll("button[data-verdict]").forEach((btn) => {
  btn.addEventListener("click", () => save({ verdict: btn.dataset.verdict, kind: null }, true));
});
let noteTimer = 0;
$("notes").addEventListener("input", () => {
  clearTimeout(noteTimer);
  noteTimer = setTimeout(() => {
    if (!item || !item.tag) return;
    save({ verdict: item.tag.verdict, kind: item.tag.kind || null }, false);
  }, 400);
});
document.addEventListener("keydown", (ev) => {
  if (ev.target === $("notes")) {
    if (ev.key === "Escape") $("notes").blur();
    return;
  }
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key;
  if (k === " ") { ev.preventDefault(); playWindow(); return; }
  if (k === "f") { playFull(); return; }
  if (k === "ArrowRight") { if (item && item.n) load(queue, Math.min(item.n - 1, index + 1)); return; }
  if (k === "ArrowLeft") { if (item && item.n) load(queue, Math.max(0, index - 1)); return; }
  if (REALS.length) {
    const n = "12345".indexOf(k);
    if (n >= 0 && REALS[n]) { save({ verdict: "real", kind: REALS[n][0] }, true); return; }
  }
  if (VERDICT_KEYS[k]) save({ verdict: VERDICT_KEYS[k], kind: null }, true);
});
fetch("/api/resume").then((r) => r.json()).then((data) => load(data.queue, data.index));
</script>
</body>
</html>
"""


def clean_guard_page() -> str:
    """Same player as A/B/C. The only new tags are real lapse vs model mishear."""
    page = PAGE.replace("/*ORDER*/", json.dumps([GUARD_QUEUE]), 1)
    page = page.replace(
        """const REALS = [
  ["skipped", "Skipped", "1"],
  ["wrong_word", "Wrong word", "2"],
  ["repeat", "Repeat", "3"],
  ["restart", "Restart", "4"],
  ["tajweed_only", "Tajweed only", "5"],
];
const VERDICT_KEYS = { n: "not_slip", u: "unsure", b: "audio_bad" };""",
        """const REALS = [];
const VERDICT_KEYS = { "1": "real_lapse", m: "model_mishear", u: "unsure", b: "audio_bad" };""",
        1,
    )
    page = page.replace(
        """    <button data-verdict="not_slip" type="button">Not a slip <kbd>n</kbd></button>
    <button data-verdict="unsure" type="button">Unsure <kbd>u</kbd></button>
    <button data-verdict="audio_bad" type="button">Audio bad <kbd>b</kbd></button>""",
        """    <button data-verdict="real_lapse" type="button">Real lapse <kbd>1</kbd></button>
    <button data-verdict="model_mishear" type="button">Model mishear (recitation correct) <kbd>m</kbd></button>
    <button data-verdict="unsure" type="button">Unsure <kbd>u</kbd></button>
    <button data-verdict="audio_bad" type="button">Audio bad <kbd>b</kbd></button>""",
        1,
    )
    page = page.replace(
        "Real slip: <kbd>1</kbd> skipped <kbd>2</kbd> wrong word <kbd>3</kbd> repeat <kbd>4</kbd> restart <kbd>5</kbd> tajweed-only. Tags save on their own.",
        "1 real lapse. m model mishear (recitation correct). u unsure. b bad audio. Tags save on their own.",
        1,
    )
    if "model mishear (recitation correct)" not in page or "/*ORDER*/" in page:
        raise RuntimeError("clean_guard page did not build")
    if 'const ORDER = ["A"' in page:
        raise RuntimeError("clean_guard page still serves A/B/C")
    return page


def self_repair_page() -> str:
    """A/B/C keys. The queue name is the only session label. No role, no clip id."""
    page = PAGE.replace("/*ORDER*/", json.dumps([SELF_REPAIR_QUEUE]), 1)
    if "/*ORDER*/" in page or "self_repair" not in page:
        raise RuntimeError("self_repair page did not build")
    return page


class _Mode:
    def __init__(self, queues: tuple[str, ...], verdicts: tuple[str, ...], real_kinds: tuple[str, ...], page: str):
        self.queues = queues
        self.verdicts = verdicts
        self.real_kinds = real_kinds
        self.page = page


def _page_for(mode: "_Mode") -> str:
    page = mode.page
    if "/*ORDER*/" in page:
        page = page.replace("/*ORDER*/", json.dumps(list(mode.queues)))
    return page


def abc_mode() -> _Mode:
    return _Mode(QUEUES, VERDICTS, REAL_KINDS, PAGE)


def mined_mode() -> _Mode:
    return _Mode((MINED_QUEUE,), VERDICTS, REAL_KINDS, PAGE)


def guard_mode() -> _Mode:
    return _Mode((GUARD_QUEUE,), GUARD_VERDICTS, (), clean_guard_page())


def repair_mode() -> _Mode:
    return _Mode((SELF_REPAIR_QUEUE,), VERDICTS, REAL_KINDS, self_repair_page())


class App:
    def __init__(self, cache: Cache, saved: dict, mode: _Mode | None = None):
        self.cache = cache
        self.saved = saved
        self.mode = mode or abc_mode()
        self.tags = cache.load_tags()
        self.lock = threading.Lock()

    def item(self, queue: str, index: int) -> dict:
        rows = self.saved["queues"][queue]
        if not rows:
            with self.lock:
                return public_item(self.saved, self.tags, queue, 0, self.mode)
        index = max(0, min(index, len(rows) - 1))
        self.cache.ensure_audio(rows[index])
        with self.lock:
            return public_item(self.saved, self.tags, queue, index, self.mode)

    def resume(self) -> dict:
        with self.lock:
            tags = self.tags
        for name in self.mode.queues:
            rows = self.saved["queues"][name]
            for i, row in enumerate(rows):
                tag = tags.get(row["tag_id"]) or {}
                if tag.get("verdict") not in self.mode.verdicts:
                    return {"queue": name, "index": i}
        return {"queue": self.mode.queues[0], "index": 0}

    def set_tag(self, queue: str, index: int, verdict: str, kind: str | None, notes: str) -> dict:
        if queue not in self.mode.queues:
            raise ValueError("queue")
        rows = self.saved["queues"][queue]
        if index < 0 or index >= len(rows):
            raise IndexError("index")
        if verdict not in self.mode.verdicts:
            raise ValueError("verdict")
        if verdict == "real":
            if kind not in self.mode.real_kinds:
                raise ValueError("kind")
        else:
            kind = None
        item = rows[index]
        record = {
            "verdict": verdict,
            "kind": kind,
            "notes": notes or "",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self.lock:
            self.tags[item["tag_id"]] = record
            self.cache.save_tags(self.tags)
            tags = self.tags
        nxt = index
        for i in range(index + 1, len(rows)):
            if (tags.get(rows[i]["tag_id"]) or {}).get("verdict") not in self.mode.verdicts:
                nxt = i
                break
        view = public_item(self.saved, tags, queue, index, self.mode)
        return {"tag": view["tag"], "counts": view["counts"], "next_untagged": nxt}

    def prefetch(self) -> None:
        items = [item for name in self.mode.queues for item in self.saved["queues"][name]]
        # Unique files only. A and B can share a clip.
        seen = set()
        pending = []
        for item in items:
            key = (item["audio"]["source"], item["audio"]["name"])
            if key in seen:
                continue
            seen.add(key)
            pending.append(item)

        def one(item: dict) -> None:
            try:
                self.cache.ensure_audio(item)
            except Exception as exc:
                print(f"fetch failed {item['audio']['source']}: {type(exc).__name__}", file=sys.stderr, flush=True)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, pending))


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send(200, _page_for(app.mode).encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/resume":
                    self._json(200, app.resume())
                    return
                if parsed.path == "/api/item":
                    queue = (qs.get("queue") or ["A"])[0]
                    index = int((qs.get("i") or ["0"])[0])
                    if queue not in app.mode.queues:
                        self._json(400, {"error": "queue"})
                        return
                    self._json(200, app.item(queue, index))
                    return
                if parsed.path == "/audio":
                    self._audio(qs)
                    return
            except Exception as exc:
                self._json(500, {"error": type(exc).__name__})
                return
            self._json(404, {"error": "not found"})

        def _audio(self, qs: dict) -> None:
            queue = (qs.get("queue") or ["A"])[0]
            index = int((qs.get("i") or ["0"])[0])
            rows = app.saved["queues"][queue]
            item = rows[index]
            path = app.cache.ensure_audio(item)
            data = path.read_bytes()
            ctype = AUDIO_TYPES.get(path.suffix.lower(), "application/octet-stream")
            start = 0
            end = len(data) - 1
            status = 200
            extra = {"Accept-Ranges": "bytes"}
            header = self.headers.get("Range")
            if header and header.startswith("bytes="):
                spec = header.split("=", 1)[1].split(",")[0]
                left, _, right = spec.partition("-")
                if left:
                    start = int(left)
                if right:
                    end = int(right)
                end = min(end, len(data) - 1)
                start = max(0, start)
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{len(data)}"
            chunk = data[start : end + 1]
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Cache-Control", "no-store")
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(chunk)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/tag":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
                result = app.set_tag(
                    str(body.get("queue")),
                    int(body.get("index")),
                    str(body.get("verdict")),
                    body.get("kind"),
                    str(body.get("notes") or ""),
                )
            except (ValueError, IndexError, KeyError, TypeError) as exc:
                self._json(400, {"error": type(exc).__name__})
                return
            self._json(200, result)

    return Handler


def upload_tags(cache: Cache, report: dict, remote_dir: str = "help_slips/tags") -> str:
    import io

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote = f"{remote_dir}/{stamp}.json"
    payload = {
        "created": stamp,
        "summary": report,
        "tags": cache.load_tags(),
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    vol = _volume()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(blob), remote)
        labels = report.get("verified_labels") or []
        if labels and remote_dir == "help_slips/tags":
            rows = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels)
            batch.put_file(io.BytesIO(rows.encode("utf-8")), "help_slips/tlog_mined/verified_labels.jsonl")
    return remote


def upload_natural_labels(labels: list[dict]) -> str:
    import io

    remote = "help_slips/self_repair/natural_labels.jsonl"
    blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels).encode("utf-8")
    vol = _volume()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(blob), remote)
    return remote


def _print_counts(saved: dict) -> None:
    order = (*QUEUES, MINED_QUEUE, GUARD_QUEUE, SELF_REPAIR_QUEUE)
    names = [name for name in order if name in saved["queues"]]
    names.extend(name for name in saved["queues"] if name not in names)
    print("queues", " ".join(f"{name}={len(saved['queues'][name])}" for name in names), flush=True)


def _verses_of(row: dict) -> list[tuple[int, int]]:
    verses = [(int(v["surah"]), int(v["ayah"])) for v in (row.get("expected_verses") or [])]
    if not verses and row.get("surah") is not None and row.get("ayah") is not None:
        verses = [(int(row["surah"]), int(row["ayah"]))]
    return verses


def catalog_from_sources(clips: list[dict], help_manifest: Path, v1_root: Path) -> list[dict]:
    """One catalog row per reference word. TLOG clips are skipped."""
    from shared.phoneme_labels import PhonemeCorpus

    corpus = PhonemeCorpus()
    help_rows: dict[str, dict] = {}
    if help_manifest.is_file():
        data = json.loads(help_manifest.read_text(encoding="utf-8"))
        samples = data["samples"] if isinstance(data, dict) else data
        for row in samples:
            if row.get("use") == "clean":
                help_rows[str(row["id"])] = row
    v1_rows: dict[str, dict] = {}
    manifest = v1_root / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        samples = data["samples"] if isinstance(data, dict) else data
        for row in samples:
            v1_rows[str(row["id"])] = row
    catalog: list[dict] = []
    for clip in clips:
        if clip.get("error"):
            continue
        ident = str(clip["id"])
        if clip.get("set") == "help_clean":
            src = help_rows.get(ident)
            if not src:
                continue
            audio = {"source": "help", "name": str(src["file"])}
            pool, split = "help", src.get("split")
        elif clip.get("set") == "v1":
            src = v1_rows.get(ident)
            if not src:
                continue
            audio = {"source": "v1", "name": str(src["file"])}
            pool, split = "v1", None
        else:
            continue
        for surah, ayah in _verses_of(src):
            n_words = len(corpus.word_phonemes(surah, ayah))
            for word in range(n_words):
                catalog.append({
                    "id": ident,
                    "pool": pool,
                    "split": split,
                    "surah": surah,
                    "ayah": ayah,
                    "word_index": word,
                    "audio": audio,
                })
    return catalog


def build_clean_guard_manifest(
    clips: list[dict],
    catalog: list[dict],
    words: dict,
    surah_names: dict,
    *,
    seed: int = 0,
) -> dict:
    items = assemble_clean_guard(clips, catalog, seed=seed)
    used = {f"{item['surah']}:{item['ayah']}" for item in items}
    return {
        "queue": GUARD_QUEUE,
        "seed": seed,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition": (
            "ops k=4 same-phoneme edits, a0w-ep1-a0.5 int8, draws m3/p3/m5/p5. "
            "Help clean dev+test, v1 professional control, blind non-edited words. TLOG excluded."
        ),
        "counts": clean_guard_counts(items),
        "items": items,
        "words": {key: words[key] for key in used if key in words},
        "surah_names": {str(item["surah"]): surah_names.get(str(item["surah"])) or "" for item in items},
    }


def load_clean_guard(cache: Cache, rebuild: bool) -> dict:
    dest = cache.meta / "clean_guard_manifest.json"
    if rebuild or not dest.is_file() or dest.stat().st_size == 0:
        _download("help_slips/clean_guard/manifest.json", dest)
    manifest = json.loads(dest.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("clean_guard manifest has no items")
    return {
        "seed": manifest.get("seed", 0),
        "built": manifest.get("built"),
        "queues": {GUARD_QUEUE: items},
        "words": manifest.get("words") or {},
        "surah_names": {str(k): v for k, v in (manifest.get("surah_names") or {}).items()},
        "counts": manifest.get("counts") or clean_guard_counts(items),
    }


def _write_clean_guard_manifest(args: argparse.Namespace) -> None:
    if args.out is None:
        raise SystemExit("--out is required with --build-clean-guard")
    _outside_repo(args.out)
    clips = load_jsonl(args.build_clean_guard)
    catalog = catalog_from_sources(clips, args.help_manifest, args.v1)
    from shared.phoneme_labels import resolve_quran_json

    word_map = {
        f"{surah}:{ayah}": words
        for (surah, ayah), words in load_words(json.loads(resolve_quran_json().read_text(encoding="utf-8"))).items()
    }
    names_path = (REPO / "web" / "frontend" / "public" / "quran.json") if REPO else Path("/no/such/quran.json")
    surah_names = {str(k): v for k, v in load_surah_names(names_path).items()}
    manifest = build_clean_guard_manifest(clips, catalog, word_map, surah_names, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(args.out)
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True), flush=True)
    print(f"wrote {args.out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summarize", action="store_true", help="precision, validity, recall; upload tags")
    parser.add_argument("--no-upload", action="store_true", help="with --summarize, do not write the volume")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the shuffled queues")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tilawa-listen-review")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--queue",
        choices=("abc", MINED_QUEUE, GUARD_QUEUE, SELF_REPAIR_QUEUE),
        default="abc",
        help="abc is A/B/C. The other three are separate listens.",
    )
    parser.add_argument("--build-clean-guard", type=Path, metavar="DUMP", help="jitter jsonl → private manifest, no server")
    parser.add_argument("--out", type=Path, help="with --build-clean-guard, manifest path outside the repo")
    parser.add_argument("--help-manifest", type=Path, default=Path("/tmp/jitter/help/manifest.json"))
    parser.add_argument("--v1", type=Path, default=None, help="v1 corpus directory (manifest.json + audio)")
    args = parser.parse_args(argv)
    if args.v1 is None:
        args.v1 = (REPO / "lab" / "benchmark" / "test_corpus") if REPO else Path("lab/benchmark/test_corpus")

    if args.build_clean_guard:
        _write_clean_guard_manifest(args)
        return

    if args.queue == GUARD_QUEUE:
        cache = Cache(args.cache, tags_name="clean_guard_tags.json", queue_name="clean_guard_queue.json")
        saved = load_clean_guard(cache, args.rebuild)
        print(
            "queues",
            f"clean_guard={len(saved['queues'][GUARD_QUEUE])}",
            json.dumps(saved.get("counts") or {}, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        if args.summarize:
            report = summarize_clean_guard(saved["queues"][GUARD_QUEUE], cache.load_tags())
            sys.stdout.write(format_clean_guard(report))
            if args.no_upload:
                print("upload skipped", flush=True)
            else:
                remote = upload_tags(cache, report, "help_slips/clean_guard/tags")
                print(f"uploaded {remote}", flush=True)
            return
        app = App(cache, saved, guard_mode())
    elif args.queue == MINED_QUEUE:
        cache = Cache(args.cache)
        if cache.tags_path.name != "tags.json" or cache.mined_queue_path.name != "queue_mined.json":
            raise SystemExit("tlog_mined cache format is tags.json + queue_mined.json")
        saved = build_mined(cache, args.seed, args.rebuild)
        _print_counts(saved)
        if args.summarize:
            report = summarize_queues(saved["queues"], cache.load_tags())
            sys.stdout.write(format_report(report))
            if args.no_upload:
                print("upload skipped", flush=True)
            else:
                remote = upload_tags(cache, report)
                print(f"uploaded {remote}", flush=True)
            return
        app = App(cache, saved, mined_mode())
    elif args.queue == SELF_REPAIR_QUEUE:
        cache = Cache(
            args.cache, tags_name=SELF_REPAIR_TAGS, queue_name=SELF_REPAIR_QUEUE_FILE
        )
        saved = build_self_repair(cache, args.seed, args.rebuild)
        _print_counts(saved)
        if args.summarize:
            report = summarize_self_repair(saved["queues"].get(SELF_REPAIR_QUEUE) or [], cache.load_tags())
            sys.stdout.write(format_self_repair(report))
            if args.no_upload:
                print("upload skipped", flush=True)
            else:
                remote = upload_tags(cache, report, "help_slips/self_repair/tags")
                print(f"uploaded {remote}", flush=True)
                if report["natural_labels"]:
                    labels = upload_natural_labels(report["natural_labels"])
                    print(f"uploaded {labels}", flush=True)
            return
        app = App(cache, saved, repair_mode())
    else:
        cache = Cache(args.cache)
        saved = build_from_cache(cache, args.seed, args.rebuild)
        if args.summarize and cache.mined_queue_path.is_file():
            mined = json.loads(cache.mined_queue_path.read_text(encoding="utf-8"))
            if MINED_QUEUE in mined.get("queues", {}):
                saved["queues"][MINED_QUEUE] = mined["queues"][MINED_QUEUE]
        _print_counts(saved)
        if args.summarize:
            report = summarize_queues(saved["queues"], cache.load_tags())
            sys.stdout.write(format_report(report))
            if args.no_upload:
                print("upload skipped", flush=True)
            else:
                remote = upload_tags(cache, report)
                print(f"uploaded {remote}", flush=True)
            return
        app = App(cache, saved)

    thread = threading.Thread(target=app.prefetch, name="prefetch", daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stop", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
