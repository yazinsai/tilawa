"""Word-level locator for genuine recitation slips.

Aligns a decoded phoneme token sequence to ``PhonemeCorpus`` word phonemes
with Levenshtein, maps each reference token back to its word, and classifies
the edit as an omission, a substitution, a repeat, or a restart. Runs of
ا / ۥ / ۦ that differ only in length are a labeling convention and are
collapsed before alignment, so they are not slips.

The alignment is pure (no audio). Time spans are a separate step: CTC Viterbi
frame spans at 40 ms, taken from the hypothesis tokens that carry the edit,
or from the reference word when the word was deleted.

Another worker can import :func:`locate_word_slips` and :func:`locate_clip`
for other real recordings. This module does not synthesize audio.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

MADD_CHARS = frozenset("اۥۦ")
FRAME_S = 0.04
HIGH_CONFIDENCE = 0.85

KIND_OMITTED = "omitted"
KIND_SUBSTITUTED = "substituted"
KIND_REPEATED = "repeated"
KIND_RESTARTED = "restarted"
EXTENT_PARTIAL = "partial"
EXTENT_WHOLE = "whole"


def collapse_madd_runs(text: str) -> str:
    """Shrink each run of one madd letter to a single character."""
    collapsed, _origins = collapse_madd_runs_with_map(text)
    return collapsed


def collapse_madd_runs_with_map(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapsed text, plus the ``[start, end)`` span in ``text`` of each kept char."""
    out: list[str] = []
    origins: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in MADD_CHARS:
            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            out.append(ch)
            origins.append((i, j))
            i = j
        else:
            out.append(ch)
            origins.append((i, i + 1))
            i += 1
    return "".join(out), origins


def encode_tokens(text: str, tokens: list[str]) -> list[int]:
    """Greedy longest-match over ``tokens``. Blank and empty symbols are not used."""
    table = {t: i for i, t in enumerate(tokens) if t and t != "<blank>"}
    if not text:
        return []
    max_len = max(len(t) for t in table)
    ids: list[int] = []
    i, n = 0, len(text)
    while i < n:
        matched = None
        for length in range(min(max_len, n - i), 0, -1):
            tid = table.get(text[i : i + length])
            if tid is not None:
                matched = tid
                i += length
                break
        if matched is None:
            raise ValueError(f"OOV at {i}: {text[i : i + 8]!r}")
        ids.append(matched)
    return ids


def _decode(ids: list[int], tokens: list[str]) -> str:
    return "".join(tokens[i] for i in ids)


def _ids_str(ids: list[int]) -> str:
    return "".join(chr(0x100 + int(i)) for i in ids)


@dataclass
class WordSlip:
    """One slip inside a single ayah."""

    word_index: int
    kind: str
    extent: str
    ops: list[dict] = field(default_factory=list)
    # Indexes into the *original* hyp token-id list passed to locate_word_slips.
    hyp_indexes: list[int] = field(default_factory=list)
    word_end: int = 0  # exclusive; defaults to word_index + 1
    n_ref_tokens: int = 0

    def __post_init__(self) -> None:
        if self.word_end <= self.word_index:
            self.word_end = self.word_index + 1

    @property
    def n_edits(self) -> int:
        return len(self.ops)


def _align_events(ref: list[int], hyp: list[int]) -> list[tuple[str, int | None, int | None]]:
    """Levenshtein opcodes as a flat stream of eq/sub/del/ins.

    Each event is ``(tag, ref_index or None, hyp_index or None)``.
    """
    from Levenshtein import opcodes

    events: list[tuple[str, int | None, int | None]] = []
    if not ref and not hyp:
        return events
    for tag, i1, i2, j1, j2 in opcodes(_ids_str(ref), _ids_str(hyp)):
        nref, nhyp = i2 - i1, j2 - j1
        if tag == "equal":
            for k in range(nref):
                events.append(("eq", i1 + k, j1 + k))
        elif tag == "delete":
            for k in range(nref):
                events.append(("del", i1 + k, None))
        elif tag == "insert":
            for k in range(nhyp):
                events.append(("ins", None, j1 + k))
        else:
            paired = min(nref, nhyp)
            for k in range(paired):
                events.append(("sub", i1 + k, j1 + k))
            for k in range(paired, nref):
                events.append(("del", i1 + k, None))
            for k in range(paired, nhyp):
                events.append(("ins", None, j1 + k))
    return events


def _match_spans(ins: list[int], words: list[list[int]]) -> list[tuple[int, int]]:
    """Contiguous word spans whose token ids equal ``ins``."""
    hits: list[tuple[int, int]] = []
    n = len(words)
    for start in range(n):
        acc: list[int] = []
        for end in range(start, n):
            acc.extend(words[end])
            if len(acc) > len(ins):
                break
            if acc == ins and any(words[k] for k in range(start, end + 1)):
                hits.append((start, end + 1))
    return hits


def _pick_span(hits: list[tuple[int, int]], anchor: int | None) -> tuple[int, int] | None:
    if not hits:
        return None

    def key(span: tuple[int, int]) -> tuple:
        start, end = span
        ends_here = anchor is not None and end == anchor + 1
        dist = abs(start - anchor) if anchor is not None else start
        return (0 if ends_here else 1, dist, -(end - start))

    return min(hits, key=key)


def _span_kind(start: int, end: int, anchor: int | None, n_words: int) -> tuple[str, str] | None:
    """Repeat vs restart for an insertion that equals ``words[start:end]``.

    A local echo of the word just finished, or of that word plus the one
    before it, is a repeat. Going back to word 0, or jumping back two or
    more words, is a restart. A copy inserted before the ayah (false start)
    is a restart at word 0.
    """
    if end <= start:
        return None
    length = end - start
    if anchor is None:
        if start == 0:
            return KIND_RESTARTED, EXTENT_WHOLE
        return None
    if end == anchor + 1 and start == anchor:
        return KIND_REPEATED, EXTENT_WHOLE
    if end == anchor + 1 and start < anchor:
        if start == 0 or anchor - start >= 2:
            return KIND_RESTARTED, EXTENT_WHOLE
        return KIND_REPEATED, EXTENT_WHOLE
    if anchor == n_words - 1 and start < anchor and end <= n_words:
        if length == 1 and start == anchor:
            return KIND_REPEATED, EXTENT_WHOLE
        if start == 0 or anchor - start >= 2:
            return KIND_RESTARTED, EXTENT_WHOLE
        return KIND_REPEATED, EXTENT_WHOLE
    if start < anchor:
        return KIND_RESTARTED, EXTENT_WHOLE
    if start == anchor + 1:
        return KIND_REPEATED, EXTENT_WHOLE
    return None


def _is_suffix(needle: list[int], hay: list[int]) -> bool:
    n = len(needle)
    return n > 0 and n <= len(hay) and hay[-n:] == needle


def _preceding_in_word(
    events: list[tuple[str, int | None, int | None]],
    insert_at: int,
    ref: list[int],
    ref_word: list[int],
    anchor: int,
) -> list[int]:
    got: list[int] = []
    for tag, ri, _hi in reversed(events[:insert_at]):
        if tag == "ins" or ri is None:
            continue
        if ref_word[ri] != anchor:
            break
        got.append(ref[ri])
    got.reverse()
    return got


def _ops_insert(ids: list[int], tokens: list[str]) -> list[dict]:
    return [{"op": "insert", "ref": "", "hyp": tokens[i]} for i in ids]


def _put_slip(sink: dict[int, WordSlip], slip: WordSlip) -> None:
    """One slip per starting word. Restarts and repeats win over token edits."""
    rank = {KIND_RESTARTED: 3, KIND_REPEATED: 2, KIND_OMITTED: 1, KIND_SUBSTITUTED: 0}
    prev = sink.get(slip.word_index)
    if prev is None or rank[slip.kind] > rank[prev.kind]:
        sink[slip.word_index] = slip


def classify_collapsed(
    words: list[list[int]],
    hyp: list[int],
    tokens: list[str],
) -> list[WordSlip]:
    """Classify slips on sequences that are already madd-collapsed."""
    ref: list[int] = []
    ref_word: list[int] = []
    for w, ids in enumerate(words):
        for tid in ids:
            ref.append(tid)
            ref_word.append(w)
    events = _align_events(ref, hyp)
    slips: dict[int, WordSlip] = {}
    consumed: set[int] = set()

    i = 0
    while i < len(events):
        if events[i][0] != "ins":
            i += 1
            continue
        j = i
        while j < len(events) and events[j][0] == "ins":
            j += 1
        hyp_idx = [events[k][2] for k in range(i, j) if events[k][2] is not None]
        ins_ids = [hyp[k] for k in hyp_idx]
        anchor = None
        for tag, ri, _hi in reversed(events[:i]):
            if ri is not None:
                anchor = ref_word[ri]
                break
        after = None
        for _tag, ri, _hi in events[j:]:
            if ri is not None:
                after = ref_word[ri]
                break
        span = _pick_span(_match_spans(ins_ids, words), anchor)
        slip = None
        if span is not None:
            kind_extent = _span_kind(span[0], span[1], anchor, len(words))
            if kind_extent is not None:
                kind, extent = kind_extent
                n_ref = sum(len(words[k]) for k in range(span[0], span[1]))
                slip = WordSlip(
                    word_index=span[0],
                    word_end=span[1],
                    kind=kind,
                    extent=extent,
                    ops=_ops_insert(ins_ids, tokens),
                    hyp_indexes=list(hyp_idx),
                    n_ref_tokens=n_ref,
                )
        elif anchor is not None and words[anchor]:
            preceding = _preceding_in_word(events, i, ref, ref_word, anchor)
            if (
                _is_suffix(ins_ids, preceding)
                and len(ins_ids) < len(words[anchor])
                and (len(ins_ids) >= 2 or len(ins_ids) * 2 >= len(words[anchor]))
            ):
                slip = WordSlip(
                    word_index=anchor,
                    kind=KIND_REPEATED,
                    extent=EXTENT_PARTIAL,
                    ops=_ops_insert(ins_ids, tokens),
                    hyp_indexes=list(hyp_idx),
                    n_ref_tokens=len(words[anchor]),
                )
        if slip is None and after is not None and words[after]:
            # Stutter of the start of the next word, before its matched copy.
            if (
                words[after][: len(ins_ids)] == ins_ids
                and len(ins_ids) < len(words[after])
                and (len(ins_ids) >= 2 or len(ins_ids) * 2 >= len(words[after]))
            ):
                slip = WordSlip(
                    word_index=after,
                    kind=KIND_REPEATED,
                    extent=EXTENT_PARTIAL,
                    ops=_ops_insert(ins_ids, tokens),
                    hyp_indexes=list(hyp_idx),
                    n_ref_tokens=len(words[after]),
                )
        if slip is not None:
            _put_slip(slips, slip)
            consumed.update(hyp_idx)
        i = j

    # Token edits on each reference word. Insertions already explained as a
    # repeat or restart are not also substitutions. An unexplained insertion
    # sticks to the word just before it (the gap's left edge).
    per: list[dict] = [
        {"eq": 0, "sub": 0, "dele": 0, "extra": 0, "ops": [], "hyp": [], "n": len(words[w])}
        for w in range(len(words))
    ]
    next_word: list[int | None] = [None] * len(events)
    prev_word: list[int | None] = [None] * len(events)
    seen_word: int | None = None
    for idx, (_tag, ri, _hi) in enumerate(events):
        prev_word[idx] = seen_word
        if ri is not None:
            seen_word = ref_word[ri]
    seen_word = None
    for idx in range(len(events) - 1, -1, -1):
        next_word[idx] = seen_word
        ri = events[idx][1]
        if ri is not None:
            seen_word = ref_word[ri]
    for idx, (tag, ri, hi) in enumerate(events):
        if tag == "ins":
            if hi is None or hi in consumed:
                continue
            dest = prev_word[idx]
            if dest is None:
                dest = next_word[idx]
            if dest is None:
                continue
            bucket = per[dest]
            bucket["extra"] += 1
            bucket["ops"].append({"op": "insert", "ref": "", "hyp": tokens[hyp[hi]]})
            bucket["hyp"].append(hi)
            continue
        if ri is None:
            continue
        w = ref_word[ri]
        bucket = per[w]
        ref_tok = tokens[ref[ri]]
        if tag == "eq":
            bucket["eq"] += 1
        elif tag == "sub" and hi is not None:
            bucket["sub"] += 1
            bucket["ops"].append({"op": "replace", "ref": ref_tok, "hyp": tokens[hyp[hi]]})
            bucket["hyp"].append(hi)
        elif tag == "del":
            bucket["dele"] += 1
            bucket["ops"].append({"op": "delete", "ref": ref_tok, "hyp": ""})

    for w, bucket in enumerate(per):
        if bucket["n"] == 0:
            continue
        if bucket["sub"] == 0 and bucket["dele"] == 0 and bucket["extra"] == 0:
            continue
        if w in slips and slips[w].kind in (KIND_REPEATED, KIND_RESTARTED):
            continue
        n = bucket["n"]
        if bucket["eq"] == 0 and bucket["sub"] == 0 and bucket["dele"] == n and bucket["extra"] == 0:
            kind, extent = KIND_OMITTED, EXTENT_WHOLE
        elif bucket["eq"] > 0 and bucket["sub"] == 0 and bucket["extra"] == 0 and bucket["dele"] > 0:
            kind, extent = KIND_OMITTED, EXTENT_PARTIAL
        elif bucket["eq"] == 0 and (bucket["sub"] > 0 or bucket["dele"] > 0):
            kind, extent = KIND_SUBSTITUTED, EXTENT_WHOLE
        else:
            kind, extent = KIND_SUBSTITUTED, EXTENT_PARTIAL
        _put_slip(
            slips,
            WordSlip(
                word_index=w,
                kind=kind,
                extent=extent,
                ops=bucket["ops"],
                hyp_indexes=bucket["hyp"],
                n_ref_tokens=n,
            ),
        )
    return [slips[k] for k in sorted(slips)]


def _collapse_hyp(hyp_ids: list[int], tokens: list[str]) -> tuple[list[int], list[list[int]]]:
    """Madd-collapse ``hyp_ids``. Each collapsed token maps to original indexes."""
    text, spans = "", []
    pos = 0
    for tid in hyp_ids:
        piece = tokens[tid]
        text += piece
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    collapsed, origins = collapse_madd_runs_with_map(text)
    c_ids = encode_tokens(collapsed, tokens)
    back: list[list[int]] = []
    cursor = 0
    for tid in c_ids:
        length = len(tokens[tid])
        o0 = origins[cursor][0]
        o1 = origins[cursor + length - 1][1]
        cursor += length
        back.append([i for i, (a, b) in enumerate(spans) if not (b <= o0 or a >= o1)])
    return c_ids, back


def _collapse_words(word_ids: list[list[int]], tokens: list[str]) -> list[list[int]]:
    """Madd-collapse each word, then longest-match the concatenated ayah.

    Tokens are assigned to the word with the greatest character overlap so a
    perfect hypothesis (same collapsed string) aligns with zero edits.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for ids in word_ids:
        piece = collapse_madd_runs(_decode(ids, tokens))
        parts.append(piece)
        spans.append((pos, pos + len(piece)))
        pos += len(piece)
    flat = encode_tokens("".join(parts), tokens)
    grouped: list[list[int]] = [[] for _ in word_ids]
    cursor = 0
    for tid in flat:
        length = len(tokens[tid])
        a, b = cursor, cursor + length
        cursor = b
        best, best_ov = 0, -1
        for w, (c, d) in enumerate(spans):
            ov = max(0, min(b, d) - max(a, c))
            if ov > best_ov or (ov == best_ov and ov > 0 and w < best):
                best, best_ov = w, ov
        if best_ov > 0:
            grouped[best].append(tid)
    return grouped


def locate_word_slips(
    word_ids: list[list[int]],
    hyp_ids: list[int],
    tokens: list[str],
) -> list[WordSlip]:
    """Word slips of ``hyp_ids`` against per-word reference token ids.

    ``word_ids[i]`` is the tokenizer output for reference word ``i``. Madd-length
    runs are ignored. ``WordSlip.hyp_indexes`` indexes ``hyp_ids``.
    """
    words = _collapse_words(word_ids, tokens)
    collapsed_hyp, back = _collapse_hyp(hyp_ids, tokens)
    slips = classify_collapsed(words, collapsed_hyp, tokens)
    for slip in slips:
        orig: list[int] = []
        for idx in slip.hyp_indexes:
            if 0 <= idx < len(back):
                orig.extend(back[idx])
        # Stable unique, keep order.
        seen: set[int] = set()
        uniq: list[int] = []
        for n in orig:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        slip.hyp_indexes = uniq
        # n_ref_tokens was counted in collapsed space; keep it (madd-normalized).
    return slips


def locate_clip(word_phonemes: list[str], hyp_ids: list[int], tokens: list[str]) -> list[WordSlip]:
    """Locate slips from reference word phoneme strings and decoded token ids."""
    word_ids = [encode_tokens(w, tokens) for w in word_phonemes]
    return locate_word_slips(word_ids, hyp_ids, tokens)


def ayah_local_slip(
    slip: WordSlip,
    index_map: list[tuple[int, int, int]],
) -> tuple[tuple[int, int], WordSlip] | None:
    """Move a passage-level slip onto the ayah that contains its first word.

    ``locate_clip`` indexes words in one flat sequence. For a multi-ayah take,
    ``index_map[i]`` is ``(surah, ayah, word_index)`` of passage word ``i``.
    The returned slip's ``word_index`` is the index inside that ayah. A span
    that crosses into the next ayah is clipped at the boundary. ``hyp_indexes``
    still index the clip hypothesis (needed for time spans).
    """
    if slip.word_index < 0 or slip.word_index >= len(index_map):
        return None
    surah, ayah, word = index_map[slip.word_index]
    end = slip.word_index
    limit = max(slip.word_end, slip.word_index + 1)
    while end < limit and end < len(index_map):
        s, a, _w = index_map[end]
        if (s, a) != (surah, ayah):
            break
        end += 1
    if end == slip.word_index:
        return None
    local_end = index_map[end - 1][2] + 1
    local = WordSlip(
        word_index=word,
        kind=slip.kind,
        extent=slip.extent,
        ops=list(slip.ops),
        hyp_indexes=list(slip.hyp_indexes),
        word_end=local_end,
        n_ref_tokens=slip.n_ref_tokens,
    )
    return (surah, ayah), local


def span_seconds(
    token_frames: list[tuple[int, int] | None],
    indexes: list[int],
    *,
    frame_s: float = FRAME_S,
) -> list[float] | None:
    """``[start, end)`` seconds from inclusive CTC Viterbi ``(first, last)`` frames."""
    chosen: list[tuple[int, int]] = []
    for i in indexes:
        if i < 0 or i >= len(token_frames):
            continue
        span = token_frames[i]
        if span is None or span[0] < 0:
            continue
        chosen.append(span)
    if not chosen:
        return None
    first = min(a for a, _b in chosen)
    last = max(b for _a, b in chosen)
    return [round(first * frame_s, 3), round((last + 1) * frame_s, 3)]


def union_spans(spans: list[list[float] | None]) -> list[float] | None:
    ok = [s for s in spans if s and len(s) == 2]
    if not ok:
        return None
    return [round(min(s[0] for s in ok), 3), round(max(s[1] for s in ok), 3)]


def confidence_score(
    *,
    kind_agree: bool,
    extent: str,
    edit_tokens: int,
    word_tokens: int,
    neighbours_clean: bool,
) -> float:
    """Rank an agreed slip.

    Both models already agree on the word. A large (whole-word) edit and
    clean neighbouring words each raise the score. ``HIGH_CONFIDENCE`` is
    0.85: agreement, the same kind, and a large edit.
    """
    score = 0.40
    if kind_agree:
        score += 0.20
    large = extent == EXTENT_WHOLE or (word_tokens > 0 and edit_tokens >= word_tokens)
    if large:
        score += 0.25
    elif word_tokens > 0 and edit_tokens * 2 >= word_tokens:
        score += 0.10
    if neighbours_clean:
        score += 0.15
    # A block much longer than the word was not segmented into a repeat or
    # restart; it got glued onto one word. That is not "a large edit on one word".
    if word_tokens > 0 and edit_tokens > max(8, word_tokens * 2):
        score -= 0.35
    return round(min(max(score, 0.35), 1.0), 2)


def is_review_slip(slip: WordSlip) -> bool:
    """A hand-review candidate, not a one-phone model wobble.

    Whole-word edits, repeats, and restarts always qualify. A partial edit
    qualifies when it covers at least half the word and at least two tokens.
    """
    if slip.kind in (KIND_REPEATED, KIND_RESTARTED) or slip.extent == EXTENT_WHOLE:
        return True
    return slip.n_edits >= 2 and slip.n_ref_tokens > 0 and slip.n_edits * 2 >= slip.n_ref_tokens


def neighbours_clean(slip: WordSlip, slips: list[WordSlip], n_words: int) -> bool:
    """True when the words just outside this slip's region are not themselves slips."""
    occupied: set[int] = set()
    for other in slips:
        if other.word_index == slip.word_index:
            continue
        for w in range(other.word_index, other.word_end):
            occupied.add(w)
    left = slip.word_index - 1
    right = slip.word_end
    left_ok = left < 0 or left not in occupied
    right_ok = right >= n_words or right not in occupied
    return left_ok and right_ok


def same_ayah_suspect(row: dict, dev_ids: set[str]) -> bool:
    """Plan suspect bucket, nearest ayah is the label, not in the reserved dev list."""
    if row.get("bucket") != "suspect" or "per" not in row:
        return False
    if row.get("id") in dev_ids:
        return False
    return row.get("alt_key") == f"{row.get('surah')}:{row.get('ayah')}"


def _evidence(slip: WordSlip) -> dict:
    return {
        "kind": slip.kind,
        "extent": slip.extent,
        "word_end": slip.word_end,
        "n_edits": slip.n_edits,
        "ops": slip.ops[:32],
    }


def merge_agreed(
    left: list[WordSlip],
    right: list[WordSlip],
    *,
    n_words: int,
    left_name: str,
    right_name: str,
) -> list[dict]:
    """Slips whose ``word_index`` both models produced. Spans are attached later."""
    right_by = {s.word_index: s for s in right}
    rows = []
    for a in left:
        b = right_by.get(a.word_index)
        if b is None:
            continue
        kind = a.kind if a.kind == b.kind else (a.kind if a.n_edits >= b.n_edits else b.kind)
        extent = EXTENT_WHOLE if EXTENT_WHOLE in (a.extent, b.extent) else EXTENT_PARTIAL
        word_tokens = max(a.n_ref_tokens, b.n_ref_tokens, 1)
        edit_tokens = max(a.n_edits, b.n_edits)
        clean = neighbours_clean(a, left, n_words) and neighbours_clean(b, right, n_words)
        conf = confidence_score(
            kind_agree=a.kind == b.kind,
            extent=extent,
            edit_tokens=edit_tokens,
            word_tokens=word_tokens,
            neighbours_clean=clean,
        )
        rows.append(
            {
                "word_index": a.word_index,
                "word_end": min(a.word_end, b.word_end) if a.kind == b.kind else a.word_end,
                "kind": kind,
                "extent": extent,
                "confidence": conf,
                "evidence": {left_name: _evidence(a), right_name: _evidence(b)},
                "_a": a,
                "_b": b,
            }
        )
    return rows


def load_dev_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def iter_scores(shard_dir: Path):
    for path in sorted(shard_dir.glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def ref_frames_by_word(word_phonemes: list[str], hyp_lp, tokens: list[str]):
    """Viterbi ``(first, last)`` frames per reference token, grouped by word.

    Uses the uncollapsed ayah so the span is where the label was aligned,
    which is the omission / substitution slot when the hypothesis deleted it.
    """
    from shared.zipformer_score import ctc_viterbi_spans

    text = "".join(word_phonemes)
    spans: list[tuple[int, int]] = []
    pos = 0
    for word in word_phonemes:
        spans.append((pos, pos + len(word)))
        pos += len(word)
    ids = encode_tokens(text, tokens)
    grouped: list[list[int]] = [[] for _ in word_phonemes]
    cursor = 0
    for index, tid in enumerate(ids):
        length = len(tokens[tid])
        a, b = cursor, cursor + length
        cursor = b
        best, best_ov = 0, -1
        for w, (c, d) in enumerate(spans):
            ov = max(0, min(b, d) - max(a, c))
            if ov > best_ov:
                best, best_ov = w, ov
        if best_ov > 0:
            grouped[best].append(index)
    frames = ctc_viterbi_spans(hyp_lp, ids)
    return frames, grouped


def slip_span(slip: WordSlip, hyp_frames, ref_frames, ref_groups: list[list[int]]) -> list[float] | None:
    """Hyp-token span when the edit has hypothesis tokens, else the reference word."""
    if slip.hyp_indexes and hyp_frames:
        got = span_seconds(list(hyp_frames), slip.hyp_indexes)
        if got is not None:
            return got
    if not ref_frames:
        return None
    indexes: list[int] = []
    for w in range(slip.word_index, min(slip.word_end, len(ref_groups))):
        indexes.extend(ref_groups[w])
    return span_seconds(list(ref_frames), indexes)


def clamp_span(span: list[float] | None, duration: float | None) -> list[float] | None:
    """Keep a Viterbi span inside the clip. Drain frames sit past the audio."""
    if not span or duration is None or duration <= 0:
        return span
    start = min(max(span[0], 0.0), duration)
    end = min(max(span[1], 0.0), duration)
    if end <= start:
        end = min(duration, start + FRAME_S)
    return [round(start, 3), round(end, 3)]


def select_rows(shard_dir: Path, dev_ids: set[str]) -> tuple[list[dict], dict]:
    suspects = same = kept = 0
    rows = []
    for row in iter_scores(shard_dir):
        if row.get("bucket") == "suspect" and "per" in row:
            suspects += 1
            if row.get("alt_key") == f"{row.get('surah')}:{row.get('ayah')}":
                same += 1
                if row["id"] not in dev_ids:
                    kept += 1
                    rows.append(row)
    return rows, {"suspects": suspects, "same_ayah": same, "after_dev": kept}


def write_review(path: Path, ranked: list[dict], summary: dict) -> None:
    lines = [
        "# TLOG slip candidates",
        "",
        "Hand-verify these. Audio stays on the volume; this file has ids and spans only.",
        "",
        f"- Suspect clips: {summary.get('suspects')}",
        f"- Same-ayah suspects: {summary.get('same_ayah')}",
        f"- After removing dev ids: {summary.get('after_dev')}",
        f"- Agreed slips: {summary.get('located_slips')} across {summary.get('located_clips')} clips",
        f"- Inter-model agreement (word location): {summary.get('agreement_rate')}",
        f"- High-confidence (>={HIGH_CONFIDENCE}): {summary.get('high_confidence')}",
        "",
        "| rank | id | surah | ayah | word | kind | extent | span_s | confidence |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for i, row in enumerate(ranked, start=1):
        span = row.get("span_s")
        span_txt = "" if not span else f"{span[0]:.2f}-{span[1]:.2f}"
        lines.append(
            f"| {i} | {row['id']} | {row['surah']} | {row['ayah']} | {row['word_index']} "
            f"| {row['kind']} | {row['extent']} | {span_txt} | {row['confidence']:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_audio(path: Path):
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=-1)
    if sr != 16000:
        import torch
        import torchaudio

        audio = torchaudio.functional.resample(
            torch.from_numpy(np.ascontiguousarray(audio)), sr, 16000
        ).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def _fetch(vol, clip_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    data = bytearray()
    for chunk in vol.read_file(f"audio/tlog/{clip_id}.flac"):
        data.extend(chunk)
    if not data:
        raise FileNotFoundError(clip_id)
    dest.write_bytes(data)


def run_models(args: argparse.Namespace) -> dict:
    """Decode shortlisted clips with both ONNX models and write the agreed set."""
    from shared.fbank import compute_fbank
    from shared.phoneme_labels import PhonemeCorpus, load_tokens
    from shared.zipformer_score import StreamingZipformer, ctc_viterbi_spans, greedy_ids

    tokens = load_tokens(args.tokens) if args.tokens else load_tokens()
    corpus = PhonemeCorpus(args.quran) if args.quran else PhonemeCorpus()
    dev_ids = load_dev_ids(Path(args.dev_ids)) if args.dev_ids else set()
    rows, counts = select_rows(Path(args.shards), dev_ids)
    if args.limit:
        rows = rows[: args.limit]

    # v3 hypothesis strings are already on the shards. Shortlist clips that
    # have a non-madd slip under that hypothesis; the other model cannot
    # agree on a clip v3 does not flag.
    short: list[tuple[dict, list[WordSlip]]] = []
    v3_keys: set[tuple[str, int]] = set()
    for row in rows:
        try:
            words = corpus.word_phonemes(int(row["surah"]), int(row["ayah"]))
            hyp_ids = encode_tokens(row.get("hyp") or "", tokens)
            slips = [s for s in locate_clip(words, hyp_ids, tokens) if is_review_slip(s)]
        except (ValueError, KeyError):
            continue
        if not slips:
            continue
        short.append((row, slips))
        for slip in slips:
            v3_keys.add((row["id"], slip.word_index))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    import modal

    vol = modal.Volume.from_name("zipformer-ctc-training")
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    fetch_err: set[str] = set()
    decode_err = 0
    print(f"shortlist {len(short)} clips", flush=True)

    def _one_fetch(row: dict) -> str:
        dest = audio_dir / f"{row['id']}.flac"
        _fetch(vol, row["id"], dest)
        return row["id"]

    # Downloads are network-bound. Inference stays one model, 4 threads.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_one_fetch, row): row["id"] for row, _slips in short}
        done = 0
        for fut in as_completed(futures):
            done += 1
            cid = futures[fut]
            try:
                fut.result()
            except Exception:
                fetch_err.add(cid)
            if done % 100 == 0 or done == len(futures):
                print(f"fetch {done}/{len(futures)} fail {len(fetch_err)}", flush=True)
    # One model resident at a time. 4 intra-op threads, no second process.
    model_paths = [("v3", args.v3_model), ("a0w-ep1-a0.5", args.a0w_model)]
    per_model: dict[str, dict[str, dict]] = {name: {} for name, _p in model_paths}

    for name, model_path in model_paths:
        session = StreamingZipformer(model_path, threads=args.threads)
        for n, (row, _preview) in enumerate(short, start=1):
            cid = row["id"]
            dest = audio_dir / f"{cid}.flac"
            if cid in fetch_err or not dest.is_file():
                fetch_err.add(cid)
                per_model[name][cid] = {"error": "missing audio"}
                continue
            try:
                wave = _load_audio(dest)
                lp = session.log_probs(compute_fbank(wave, sr=16000))
                hyp = greedy_ids(lp)
            except Exception as exc:  # keep going; this clip is not a candidate
                if not dest.is_file():
                    fetch_err.add(cid)
                else:
                    decode_err += 1
                per_model[name][cid] = {"error": f"{type(exc).__name__}: {exc}"[:240]}
                continue
            words = corpus.word_phonemes(int(row["surah"]), int(row["ayah"]))
            try:
                slips = [s for s in locate_clip(words, hyp, tokens) if is_review_slip(s)]
                hyp_frames = ctc_viterbi_spans(lp, hyp) if hyp else []
                ref_frames, ref_groups = ref_frames_by_word(words, lp, tokens)
            except Exception as exc:
                decode_err += 1
                per_model[name][cid] = {"error": f"{type(exc).__name__}: {exc}"[:240]}
                continue
            spans = {}
            for slip in slips:
                spans[slip.word_index] = clamp_span(
                    slip_span(slip, hyp_frames, ref_frames, ref_groups),
                    float(row.get("duration") or 0) or None,
                )
            per_model[name][cid] = {
                "slips": slips,
                "spans": spans,
                "n_words": len(words),
                "surah": int(row["surah"]),
                "ayah": int(row["ayah"]),
                "duration": row.get("duration"),
            }
            del lp
            if n % 25 == 0 or n == len(short):
                print(f"{name} {n}/{len(short)}", flush=True)
        del session
    if not args.keep_audio:
        for row, _preview in short:
            dest = audio_dir / f"{row['id']}.flac"
            if dest.is_file():
                dest.unlink()

    a0w_keys: set[tuple[str, int]] = set()
    for cid, payload in per_model["a0w-ep1-a0.5"].items():
        for slip in payload.get("slips") or []:
            a0w_keys.add((cid, slip.word_index))
    # Fresh v3 locations, not the stored-hyp preview. Agreement is over clips
    # both models actually decoded.
    v3_fresh: set[tuple[str, int]] = set()
    for cid, payload in per_model["v3"].items():
        for slip in payload.get("slips") or []:
            v3_fresh.add((cid, slip.word_index))
    union = v3_fresh | a0w_keys
    inter = v3_fresh & a0w_keys
    agreement = round(len(inter) / len(union), 4) if union else None

    ranked: list[dict] = []
    for cid in sorted({c for c, _w in inter}):
        left = per_model["v3"].get(cid) or {}
        right = per_model["a0w-ep1-a0.5"].get(cid) or {}
        if "slips" not in left or "slips" not in right:
            continue
        merged = merge_agreed(
            left["slips"],
            right["slips"],
            n_words=left["n_words"],
            left_name="v3",
            right_name="a0w-ep1-a0.5",
        )
        for row in merged:
            if (cid, row["word_index"]) not in inter:
                continue
            span = clamp_span(
                union_spans(
                    [
                        (left.get("spans") or {}).get(row["word_index"]),
                        (right.get("spans") or {}).get(row["word_index"]),
                    ]
                ),
                float(left.get("duration") or 0) or None,
            )
            ranked.append(
                {
                    "id": cid,
                    "surah": left["surah"],
                    "ayah": left["ayah"],
                    "word_index": row["word_index"],
                    "kind": row["kind"],
                    "extent": row["extent"],
                    "span_s": span,
                    "evidence": row["evidence"],
                    "confidence": row["confidence"],
                }
            )
    def _rank_key(row: dict) -> tuple:
        edits = max(v["n_edits"] for v in row["evidence"].values())
        return (
            -row["confidence"],
            0 if row["extent"] == EXTENT_WHOLE else 1,
            -min(edits, 8),
            row["id"],
            row["word_index"],
        )

    ranked.sort(key=_rank_key)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in ranked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_kind: dict[str, int] = {}
    for row in ranked:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    high = sum(1 for r in ranked if r["confidence"] >= HIGH_CONFIDENCE)
    summary = {
        **counts,
        "shortlist_clips": len(short),
        "stored_v3_slip_keys": len(v3_keys),
        "decoded_v3": sum(1 for p in per_model["v3"].values() if "slips" in p),
        "decoded_a0w": sum(1 for p in per_model["a0w-ep1-a0.5"].values() if "slips" in p),
        "fetch_errors": len(fetch_err),
        "decode_errors": decode_err,
        "v3_slip_keys": len(v3_fresh),
        "a0w_slip_keys": len(a0w_keys),
        "agreed_keys": len(inter),
        "agreement_rate": agreement,
        "located_slips": len(ranked),
        "located_clips": len({r["id"] for r in ranked}),
        "by_kind": by_kind,
        "high_confidence": high,
        "high_confidence_min": HIGH_CONFIDENCE,
    }
    review = Path(args.review) if args.review else out.with_name("review.md")
    write_review(review, ranked, summary)
    summary_path = out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, default=Path("/tmp/phase0/shards_all/v3"))
    parser.add_argument("--dev-ids", type=Path, default=Path("/tmp/tlog_slips/dev_nonclean_ids.txt"))
    parser.add_argument("--v3-model", type=Path, default=Path("/tmp/models/v3.onnx"))
    parser.add_argument("--a0w-model", type=Path, default=Path("/tmp/models/a0w-ep1-a0.5.onnx"))
    parser.add_argument("--tokens", type=Path, default=None)
    parser.add_argument("--quran", type=Path, default=None)
    parser.add_argument("--audio-dir", type=Path, default=Path("/tmp/tlog_slips/audio"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/tlog_slips/tlog_candidates.jsonl"))
    parser.add_argument("--review", type=Path, default=Path("/tmp/tlog_slips/review.md"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print suspect / same-ayah / dev-excluded counts and exit",
    )
    args = parser.parse_args(argv)
    if args.count_only:
        dev_ids = load_dev_ids(args.dev_ids) if args.dev_ids else set()
        _rows, counts = select_rows(args.shards, dev_ids)
        print(json.dumps(counts))
        return
    run_models(args)


if __name__ == "__main__":
    main()
