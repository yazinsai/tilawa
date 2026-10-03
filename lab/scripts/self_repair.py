"""Self-repair slips: a restart whose first rendition deviates and whose second matches.

Free decode plus the reference word alignment. No audio, no acted takes.
A candidate is a word inside a multi-word restart, not the restart itself.
Thresholds are the pre-declared ones in the probe writeup; do not retune them
to chase a count.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

# Frozen with the probe. See docs/self-repair-miner.md.
MIN_REPEAT_WORDS = 2
MIN_OVERLAP_WORDS = 2
MIN_REWIND = 1
MAX_GAP_TOKENS = 6
MAX_ATTEMPT_COST = 0.75
MIN_EQ_WORDS = 2
MAX_PER_CLIP = 3

# Shadda counts as a consonant feature. Other harakat and Quranic marks do not.
_SHADDA = "\u0651"


def _skeleton(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    for ch in text:
        if ch == _SHADDA:
            out.append(ch)
            continue
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch)
    return "".join(out)


def _edit(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify_deviation(ref_tokens: list[str], hyp_tokens: list[str]) -> str:
    """vowel / letter / substitution / skip for one word's first rendition."""
    if not hyp_tokens:
        return "skip"
    ref_s = _skeleton("".join(ref_tokens))
    hyp_s = _skeleton("".join(hyp_tokens))
    if not hyp_s:
        return "skip"
    if ref_s == hyp_s:
        return "vowel"
    if _edit(ref_s, hyp_s) == 1:
        return "letter"
    if len(hyp_s) * 2 < len(ref_s) and len(hyp_tokens) * 2 <= len(ref_tokens):
        return "skip"
    return "substitution"


@dataclass
class SelfRepair:
    """One deviant word. ``hyp_indexes`` are into the hyp passed to the miner."""

    word_index: int
    kind: str
    rewind: int
    overlap: int
    edit_tokens: int
    hyp_indexes: list[int] = field(default_factory=list)


def _fit_span(ref: list[int], hyp: list[int], max_gap: int) -> dict | None:
    """Fit all of ``ref`` into ``hyp``. Leading hyp is free.

    The chosen alignment ends within ``max_gap`` tokens of ``len(hyp)``.
    """
    n, m = len(ref), len(hyp)
    if n == 0 or m == 0:
        return None
    inf = n + m + 5
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0
    for i in range(1, n + 1):
        for j in range(m + 1):
            cand = dp[i - 1][j] + 1
            if cand < dp[i][j]:
                dp[i][j] = cand
                bt[i][j] = ("del", i - 1, None)
            if j == 0:
                continue
            sub = 0 if ref[i - 1] == hyp[j - 1] else 1
            cand = dp[i - 1][j - 1] + sub
            if cand < dp[i][j]:
                dp[i][j] = cand
                bt[i][j] = ("eq" if sub == 0 else "sub", i - 1, j - 1)
            cand = dp[i][j - 1] + 1
            if cand < dp[i][j]:
                dp[i][j] = cand
                bt[i][j] = ("ins", None, j - 1)
    best_j = None
    best_c = inf
    for j in range(max(0, m - max_gap), m + 1):
        if dp[n][j] < best_c:
            best_c = dp[n][j]
            best_j = j
    if best_j is None or best_c >= inf:
        return None
    events: list[tuple[str, int | None, int | None]] = []
    i, j = n, best_j
    guard = 0
    while i > 0:
        step = bt[i][j]
        if step is None:
            return None
        tag, ri, hi = step
        events.append((tag, ri, hi))
        if tag == "del":
            i -= 1
        elif tag == "ins":
            j -= 1
        else:
            i -= 1
            j -= 1
        guard += 1
        if guard > n + m + 2:
            return None
    events.reverse()
    return {"events": events, "cost": best_c, "end": best_j}


def _produced_last(buckets: list[dict], start: int) -> int | None:
    """Last reference word the attempt really produced.

    A trailing scrap (fewer hyp tokens than the word, no equal token) after
    a fully matched word is interregnum between the attempt and the restart,
    not a slip and not part of the attempt.
    """
    last = None
    for w, bucket in enumerate(buckets):
        if bucket["eq"] or bucket["sub"] or bucket["extra"]:
            last = w
    if last is None:
        return None
    if last > 0:
        prev = buckets[last - 1]
        bucket = buckets[last]
        prev_clean = prev["n"] > 0 and prev["eq"] == prev["n"] and prev["sub"] == 0 and prev["dele"] == 0
        scrap = bucket["eq"] == 0 and len(bucket["hyp"]) < bucket["n"]
        if prev_clean and scrap:
            last -= 1
            while last >= 0 and not (buckets[last]["eq"] or buckets[last]["sub"] or buckets[last]["extra"]):
                last -= 1
    if last is None or last < 0:
        return None
    return start + last


def _maximal_exact_runs(hyp: list[int], words: list[list[int]]) -> list[tuple[int, int, int, int]]:
    """Left-maximal exact runs of ≥ ``MIN_REPEAT_WORDS`` reference words.

    Each run is ``(word_start, word_end, hyp_start, hyp_end)``.
    """
    runs: list[tuple[int, int, int, int]] = []
    n = len(words)
    for i in range(len(hyp)):
        for w, seq in enumerate(words):
            if not seq or hyp[i : i + len(seq)] != seq:
                continue
            if w > 0 and words[w - 1]:
                prev = words[w - 1]
                if i >= len(prev) and hyp[i - len(prev) : i] == prev:
                    continue
            j = i + len(seq)
            end_w = w + 1
            while end_w < n and words[end_w] and hyp[j : j + len(words[end_w])] == words[end_w]:
                j += len(words[end_w])
                end_w += 1
            if end_w - w >= MIN_REPEAT_WORDS:
                runs.append((w, end_w, i, j))
    # Identical runs can be proposed twice when two words share a token sequence.
    return sorted(set(runs))


def _word_buckets(
    events: list[tuple[str, int | None, int | None]],
    ref_word: list[int],
    hyp: list[int],
    n_words: int,
    word_start: int,
) -> list[dict]:
    per = [
        {"eq": 0, "sub": 0, "dele": 0, "extra": 0, "n": 0, "hyp": []}
        for _ in range(n_words)
    ]
    # n is filled by the caller for the span; others stay 0.
    next_word: list[int | None] = [None] * len(events)
    prev_word: list[int | None] = [None] * len(events)
    seen: int | None = None
    for idx, (_tag, ri, _hi) in enumerate(events):
        prev_word[idx] = seen
        if ri is not None:
            seen = ref_word[ri]
    seen = None
    for idx in range(len(events) - 1, -1, -1):
        next_word[idx] = seen
        ri = events[idx][1]
        if ri is not None:
            seen = ref_word[ri]
    for idx, (tag, ri, hi) in enumerate(events):
        if tag == "ins":
            if hi is None:
                continue
            dest = prev_word[idx]
            if dest is None:
                dest = next_word[idx]
            if dest is None:
                continue
            bucket = per[dest - word_start]
            bucket["extra"] += 1
            bucket["hyp"].append(hi)
            continue
        if ri is None:
            continue
        bucket = per[ref_word[ri] - word_start]
        if tag == "eq":
            bucket["eq"] += 1
            if hi is not None:
                bucket["hyp"].append(hi)
        elif tag == "sub" and hi is not None:
            bucket["sub"] += 1
            bucket["hyp"].append(hi)
        elif tag == "del":
            bucket["dele"] += 1
    return per


def mine_collapsed(
    words: list[list[int]],
    hyp: list[int],
    tokens: list[str],
) -> list[SelfRepair]:
    """Self-repair words on madd-collapsed token ids.

    ``words[i]`` is the collapsed token-id sequence of reference word ``i``.
    """
    found: dict[int, SelfRepair] = {}
    if len(words) < MIN_REPEAT_WORDS or not hyp:
        return []
    for start, end, hyp_start, _hyp_end in _maximal_exact_runs(hyp, words):
        span_words = words[start:end]
        ref: list[int] = []
        ref_word: list[int] = []
        for w, seq in enumerate(span_words):
            for tid in seq:
                ref.append(tid)
                ref_word.append(start + w)
        if not ref:
            continue
        budget = len(ref) * 2 + MAX_GAP_TOKENS
        region_start = max(0, hyp_start - budget)
        region = hyp[region_start:hyp_start]
        if not region:
            continue
        # The window ends at the restart, so it is the attempt. A short
        # trailing scrap (interregnum) is dropped after the fit, not by
        # leaving a gap the wrong word can hide in.
        fitted = _fit_span(ref, region, 0)
        if fitted is None:
            continue
        events = []
        for tag, ri, hi in fitted["events"]:
            events.append((tag, ri, None if hi is None else hi + region_start))
        buckets = _word_buckets(events, ref_word, hyp, end - start, start)
        for w, seq in enumerate(span_words):
            buckets[w]["n"] = len(seq)
        last = _produced_last(buckets, start)
        if last is None or last - start < MIN_REWIND:
            continue
        eq_words = sum(1 for w in range(last - start + 1) if buckets[w]["eq"] > 0)
        if eq_words < MIN_EQ_WORDS:
            continue
        used = 0
        cost = 0
        for w in range(last - start + 1):
            bucket = buckets[w]
            used += bucket["n"]
            cost += bucket["sub"] + bucket["dele"] + bucket["extra"]
        if used == 0 or cost > MAX_ATTEMPT_COST * used:
            continue
        # Overlap is the words both renditions cover, up through the last
        # word the first attempt produced (trailing deletions were not said).
        overlap = last - start + 1
        if overlap < MIN_OVERLAP_WORDS:
            continue
        for w, bucket in enumerate(buckets):
            word = start + w
            if word > last:
                break
            if bucket["sub"] == 0 and bucket["dele"] == 0:
                continue
            if bucket["eq"] == bucket["n"] and bucket["sub"] == 0 and bucket["dele"] == 0:
                continue
            hyp_ids = [hyp[i] for i in bucket["hyp"] if 0 <= i < len(hyp)]
            ref_tokens = [tokens[tid] for tid in span_words[w]] if span_words[w] else []
            hyp_tokens = [tokens[i] for i in hyp_ids]
            kind = classify_deviation(ref_tokens, hyp_tokens)
            indexes = list(bucket["hyp"])
            if not indexes:
                # Deletion: anchor the listen window on the previous hyp token.
                for prev in range(w - 1, -1, -1):
                    if buckets[prev]["hyp"]:
                        indexes = [buckets[prev]["hyp"][-1]]
                        break
            edit_tokens = bucket["sub"] + bucket["dele"] + bucket["extra"]
            repair = SelfRepair(
                word_index=word,
                kind=kind,
                rewind=last - start,
                overlap=overlap,
                edit_tokens=edit_tokens,
                hyp_indexes=indexes,
            )
            prev = found.get(word)
            if prev is None or repair.edit_tokens > prev.edit_tokens:
                found[word] = repair
    rows = [found[k] for k in sorted(found)]
    if len(rows) > MAX_PER_CLIP:
        return []
    return rows


def _orig_indexes(collapsed_indexes: list[int], back: list[list[int]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for idx in collapsed_indexes:
        if not (0 <= idx < len(back)):
            continue
        for n in back[idx]:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def mine_self_repairs(
    word_phonemes: list[str],
    hyp_ids: list[int],
    tokens: list[str],
) -> list[SelfRepair]:
    """Self-repairs of a passage. Indexes point at ``hyp_ids`` (pre-collapse)."""
    from locate_slips import _collapse_hyp, _collapse_words, encode_tokens

    word_ids = [encode_tokens(w, tokens) for w in word_phonemes]
    words = _collapse_words(word_ids, tokens)
    collapsed, back = _collapse_hyp(hyp_ids, tokens)
    repairs = mine_collapsed(words, collapsed, tokens)
    for repair in repairs:
        repair.hyp_indexes = _orig_indexes(repair.hyp_indexes, back)
    return repairs


def clean_word_spots(
    word_phonemes: list[str],
    hyp_ids: list[int],
    tokens: list[str],
    *,
    limit: int = 3,
) -> list[tuple[int, list[int]]]:
    """Exact reference words in a clip that has no self-repair. Listen controls.

    Returns ``(word_index, original hyp indexes)`` for up to ``limit`` words
    from the longest exact run, skipping the first and last word of the run
    when the run is longer than two.
    """
    from locate_slips import _collapse_hyp, _collapse_words, encode_tokens

    if mine_self_repairs(word_phonemes, hyp_ids, tokens):
        return []
    word_ids = [encode_tokens(w, tokens) for w in word_phonemes]
    words = _collapse_words(word_ids, tokens)
    collapsed, back = _collapse_hyp(hyp_ids, tokens)
    runs = _maximal_exact_runs(collapsed, words)
    if not runs:
        return []
    start, end, hyp_start, _hyp_end = max(runs, key=lambda run: (run[1] - run[0], -run[0]))
    cursor = hyp_start
    spots: list[tuple[int, list[int]]] = []
    chosen_words = list(range(start, end))
    if len(chosen_words) > 2:
        chosen_words = chosen_words[1:-1]
    for w in chosen_words:
        seq = words[w]
        indexes = list(range(cursor, cursor + len(seq)))
        cursor += len(seq)
        spots.append((w, _orig_indexes(indexes, back)))
        if len(spots) >= limit:
            break
    return spots


def span_from_frames(
    indexes: list[int],
    frames: list[int],
    *,
    frame_s: float = 0.04,
    duration: float | None = None,
) -> list[float] | None:
    """``[start, end)`` seconds from CTC frames of hyp tokens."""
    chosen = [frames[i] for i in indexes if 0 <= i < len(frames)]
    if not chosen:
        return None
    start = min(chosen) * frame_s
    end = (max(chosen) + 1) * frame_s
    if duration is not None and duration > 0:
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            end = min(duration, start + frame_s)
    return [round(start, 3), round(end, 3)]


def coincide(
    candidates: list[dict],
    issues: list[dict],
    *,
    tol: int = 0,
) -> dict:
    """How many rule issues sit on a self-repair candidate.

    Match is clip id + surah + ayah + word within ``tol``. Issues missing a
    word are counted only at clip level (``tol`` is ignored for those).
    """
    by_clip: dict[str, list[dict]] = {}
    for row in candidates:
        by_clip.setdefault(str(row.get("id")), []).append(row)
    word_hits = 0
    clip_hits = 0
    seen_issues = 0
    for issue in issues:
        cid = str(issue.get("id"))
        rows = by_clip.get(cid)
        if not rows:
            continue
        seen_issues += 1
        try:
            surah = int(issue["surah"])
            ayah = int(issue["ayah"])
        except (KeyError, TypeError, ValueError):
            clip_hits += 1
            continue
        word = issue.get("word")
        if word is None:
            same = [r for r in rows if int(r["surah"]) == surah and int(r["ayah"]) == ayah]
            if same:
                clip_hits += 1
            continue
        try:
            word_i = int(word)
        except (TypeError, ValueError):
            continue
        hit = False
        for row in rows:
            if int(row["surah"]) != surah or int(row["ayah"]) != ayah:
                continue
            if abs(int(row["word_index"]) - word_i) <= tol:
                hit = True
                break
        if hit:
            word_hits += 1
            clip_hits += 1
    return {
        "issues": len(issues),
        "issues_on_candidate_clips": seen_issues,
        "coincide_clip": clip_hits,
        "coincide_word": word_hits,
        "tol": tol,
    }


def greedy_with_frames(lp, blank: int = 250) -> tuple[list[int], list[int]]:
    """CTC collapse. Returns token ids and the frame index of each kept token."""
    ids: list[int] = []
    frames: list[int] = []
    prev = None
    for t, u in enumerate(lp.argmax(-1).tolist()):
        if u != blank and u != prev:
            ids.append(int(u))
            frames.append(t)
        prev = u
    return ids, frames
