# /// script
# requires-python = ">=3.12"
# ///
"""Re-derive listener-dependent checker constants from one student's own decodes.

The formulas are fixed. Run on the shipped student they reproduce today's constants
(the idempotence check). Run on another student they are that student's constants.

    python3 tools/listener_calibration.py                 # decode (Modal train image)
    python3 tools/listener_calibration.py derive A.json B.json

Decodes about 60 gold-excluded training readers under the shipped augmentation
(noise, RIR, gain; not SpecAugment, which is a training mask). Soft costs are the
negative log confusion rate, floored at today's cost. The confidence cut is the
quantile of clean-word confidence that keeps the shipped gate's clean pass rate.
SUPPORT and the tracker's nats-per-edit scale by the ratio of median per-phone CTC
nll on correctly matched words. The parallel-phrase switch penalty stays at today's
4 nats while the 99th percentile of correct-phrase margins is still under it, and
follows a tail that crosses that penalty.
"""

from __future__ import annotations

import json
import math
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "modal")]

from student import TOKENS, exclusions, held_out, phoneme_ids  # noqa: E402

TODAY = {
    "support": 10.0,
    "unsure": 0.8,
    "phraseSwitch": 4.0,
    "natsPerEdit": 3.0,
    "soft": 3,
    "confusable": 5,
    "shortVowel": 5,
    "unit": 10,
}
CONFUSABLE_PAIRS = ["هح", "عء", "صس", "ضد", "طت", "قك", "ذز", "ثس"]
SHORT_VOWELS = ["َ", "ُ", "ِ"]
CONSONANTS = set("ءبتثجحخدذرزسشصضطظعغفقكلمنهوي")
MARKS = set("َُِ۪ۦۥاـؙۜ")
DIACRITICS = __import__("re").compile(r"[\u0640\u064b-\u065f\u0670\u06d6-\u06ed]")
READERS = 60
PER_READER = 8
PHRASE_EXTRA = 4
SCAN_PER_READER = 40
MIN_GAP_FRAMES = 8
TAIL_Q = 0.99
# A pause the folded gap path would look at. Shorter holes are CTC blanks between phones.
SEED = 0


def letter_skeleton(word: str) -> str:
    s = DIACRITICS.sub("", word)
    return "".join("ي" if c == "ى" else "ا" if c in "ٱأإآ" else c for c in s)


def consonants(phonemes: str) -> str:
    return "".join(c for c in phonemes if c not in MARKS)


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def tail(xs: list[float], q: float = TAIL_Q) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[i]


def quantile_cut(confs: list[float], pass_rate: float) -> float:
    """Threshold where the share of values at or above it equals `pass_rate`."""
    xs = sorted(confs)
    n = len(xs)
    if not n:
        return TODAY["unsure"]
    i = int(round(n * (1 - pass_rate)))
    return xs[min(n - 1, max(0, i))]


def pair_rate(stats: dict, a: str, b: str) -> float:
    """Symmetric confusion rate of the unordered pair, over both reference counts."""
    sub = stats["sub"]
    ref = stats["refCount"]
    n = ref.get(a, 0) + ref.get(b, 0)
    if not n:
        return 0.0
    k = sub.get(f"{a}>{b}", 0) + sub.get(f"{b}>{a}", 0)
    return k / n


def vowel_rate(stats: dict) -> float:
    """Substitutions among non-consonants, over non-consonant reference phones."""
    ref = stats["refCount"]
    sub = stats["sub"]
    denom = sum(n for p, n in ref.items() if p not in CONSONANTS)
    if not denom:
        return 0.0
    num = 0
    for key, n in sub.items():
        a, b = key.split(">", 1)
        if a not in CONSONANTS and b not in CONSONANTS:
            num += n
    return num / denom


def soft_cost(rate: float, floor: int, shipped_rate: float) -> int:
    """Negative log confusion rate, in units where the shipped rate costs exactly `floor`.

    A pair the new student confuses more often would price below the floor; it stays at today's
    cost. A pair it confuses less often prices above the floor. An unseen pair stays at the floor.
    """
    if rate <= 0 or shipped_rate <= 0:
        return floor
    raw = floor * math.log(rate) / math.log(shipped_rate)
    # A soft pair never prices above one plain edit. The discount shrinks; it does not become a super-edit.
    return min(TODAY["unit"], max(floor, round_half_up(raw)))


def pair_rates(stats: dict) -> dict[str, float]:
    rates = {}
    for pair in CONFUSABLE_PAIRS:
        rates[pair] = pair_rate(stats, pair[0], pair[1])
    for i, a in enumerate(SHORT_VOWELS):
        for b in SHORT_VOWELS[i + 1 :]:
            rates[a + b] = pair_rate(stats, a, b)
    return rates


def anchor_from(stats: dict) -> dict:
    confs = stats["wordConf"]
    return {
        "passRate": (sum(c >= TODAY["unsure"] - 1e-12 for c in confs) / len(confs)) if confs else 1.0,
        "medianNll": median(stats["phoneNll"]),
        "phraseTail": tail(stats["phraseMargins"]),
        "blankMax": max(stats["blankCosts"]) if stats["blankCosts"] else None,
        "pairRates": pair_rates(stats),
        "vowelRate": vowel_rate(stats),
    }


def derive(stats: dict, anchor: dict) -> dict:
    """Constants for `stats`, anchored so the shipped student's own stats reproduce `TODAY`."""
    scale = 1.0
    if anchor["medianNll"]:
        scale = median(stats["phoneNll"]) / anchor["medianNll"]
    margins = stats["phraseMargins"]
    model_tail = tail(margins) if margins else None
    # Keep today's penalty while the correct-phrase tail stays under it. The gap between
    # two negative tails is not headroom: raising 4 by that gap sat on top of planted swaps.
    switch = TODAY["phraseSwitch"]
    if model_tail is not None and model_tail > TODAY["phraseSwitch"]:
        switch = model_tail
    costs = {}
    shipped_rates = anchor.get("pairRates") or {}
    for pair in CONFUSABLE_PAIRS:
        costs[pair] = soft_cost(pair_rate(stats, pair[0], pair[1]), TODAY["confusable"], shipped_rates.get(pair, 0.0))
        costs[pair[::-1]] = costs[pair]
    for i, a in enumerate(SHORT_VOWELS):
        for b in SHORT_VOWELS[i + 1 :]:
            key = a + b
            costs[key] = soft_cost(pair_rate(stats, a, b), TODAY["shortVowel"], shipped_rates.get(key, 0.0))
            costs[b + a] = costs[key]
    blanks = stats["blankCosts"]
    blank_max = max(blanks) if blanks else None
    # The shipped gate's blank-cost cut is 2× SUPPORT (20 nats). Keep that offset from the
    # shipped clean-pause maximum, so a model whose pauses cost more gets a higher cut.
    blank_cut = None
    if blank_max is not None and anchor.get("blankMax") is not None:
        blank_cut = 2 * TODAY["support"] + (blank_max - anchor["blankMax"])
    return {
        "unsure": quantile_cut(stats["wordConf"], anchor["passRate"]),
        "support": TODAY["support"] * scale,
        "natsPerEdit": TODAY["natsPerEdit"] * scale,
        "phraseSwitch": switch,
        "soft": soft_cost(vowel_rate(stats), TODAY["soft"], anchor.get("vowelRate") or 0.0),
        "costs": costs,
        "blankCut": blank_cut,
        "blankMax": blank_max,
        "scale": scale,
        "phraseTail": tail(margins) if margins else None,
        "medianNll": median(stats["phoneNll"]),
        "passRate": anchor["passRate"],
        "correctWords": stats.get("correctWords", 0),
        "phraseMargins": len(margins),
        "blankPauses": len(blanks),
    }


def within_noise(got: dict, today: dict = TODAY) -> list[str]:
    """Deltas a reviewer would not call a different constant. Costs move by a whole unit."""
    bad = []
    if abs(got["unsure"] - today["unsure"]) > 0.02:
        bad.append(f"unsure {got['unsure']:.4f}")
    if abs(got["support"] - today["support"]) > 0.5:
        bad.append(f"support {got['support']:.3f}")
    if abs(got["natsPerEdit"] - today["natsPerEdit"]) > 0.25:
        bad.append(f"natsPerEdit {got['natsPerEdit']:.3f}")
    if abs(got["phraseSwitch"] - today["phraseSwitch"]) > 0.5:
        bad.append(f"phraseSwitch {got['phraseSwitch']:.3f}")
    if got["soft"] != today["soft"]:
        bad.append(f"soft {got['soft']}")
    for pair in CONFUSABLE_PAIRS:
        if got["costs"].get(pair, today["confusable"]) != today["confusable"]:
            bad.append(f"cost {pair} {got['costs'].get(pair)}")
    for i, a in enumerate(SHORT_VOWELS):
        for b in SHORT_VOWELS[i + 1 :]:
            if got["costs"].get(a + b, today["shortVowel"]) != today["shortVowel"]:
                bad.append(f"cost {a}{b} {got['costs'].get(a + b)}")
    if got["blankCut"] is not None and abs(got["blankCut"] - 2 * today["support"]) > 0.5:
        bad.append(f"blankCut {got['blankCut']:.3f}")
    return bad


def align_ops(ref: str, hyp: str) -> list[tuple]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        prev, row = dp[i - 1], dp[i]
        for j in range(1, m + 1):
            row[j] = min(prev[j - 1] + (ri != hyp[j - 1]), prev[j] + 1, row[j - 1] + 1)
    ops = []
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("eq" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return a + math.log1p(math.exp(b - a)) if a > b else b + math.log1p(math.exp(a - b))


def ctc_nll(logp, labels: list[int]) -> float:
    """Plain CTC negative log-likelihood, no repeat arcs. `logp` is (T, V)."""
    t_len, _ = logp.shape
    length = len(labels)
    if t_len <= 0:
        return math.inf if length else 0.0
    states = 2 * length + 1
    lab = [0 if s % 2 == 0 else labels[(s - 1) // 2] for s in range(states)]
    prev = [-math.inf] * states
    prev[0] = float(logp[0, 0])
    if length:
        prev[1] = float(logp[0, lab[1]])
    for t in range(1, t_len):
        row = logp[t]
        nxt = [0.0] * states
        for s in range(states):
            a = prev[s]
            if s > 0:
                a = logaddexp(a, prev[s - 1])
            if s > 1 and s % 2 == 1 and lab[s] != lab[s - 2]:
                a = logaddexp(a, prev[s - 2])
            nxt[s] = a + float(row[lab[s]])
        prev = nxt
    end = prev[states - 1]
    if length:
        end = logaddexp(end, prev[states - 2])
    return -end if end != -math.inf else math.inf


def collapse_runs(ids) -> list[tuple[int, int, int]]:
    runs = []
    prev = 0
    for t, i in enumerate(ids):
        i = int(i)
        if i != 0 and i == prev and runs:
            runs[-1] = (i, runs[-1][1], t)
        elif i != 0 and i != prev:
            runs.append((i, t, t))
        prev = i
    return runs


def build_index(ref: dict) -> dict:
    words = []
    parts = []
    cursor = 0
    for ayah in ref["ayahs"]:
        p, offs = ayah["p"], ayah["o"]
        parts.append(p)
        for i, surface in enumerate(ayah["w"]):
            g0 = cursor + offs[i]
            g1 = cursor + (offs[i + 1] if i + 1 < len(offs) else len(p))
            words.append({"i": len(words), "s": ayah["s"], "a": ayah["a"], "w": i + 1, "text": surface,
                          "phones": p[offs[i]:g1 - cursor], "g0": g0, "g1": g1})
        cursor += len(p)
    corpus = "".join(parts)
    phone = [w["phones"] for w in words]
    skel = [letter_skeleton(w["text"]) for w in words]
    groups: dict[str, list[int]] = defaultdict(list)
    phrase = 4
    for i in range(len(words) - phrase + 1):
        if words[i]["s"] != words[i + phrase - 1]["s"]:
            continue
        if any(not phone[i + k] for k in range(phrase)):
            continue
        for j in (1, 2):
            key = f"{j}\0" + "\0".join(skel[i:i + j]) + "\0|\0" + "\0".join(skel[i + j + 1:i + phrase])
            groups[key].append(i)
    windows = []
    seen = set()
    for key, starts in groups.items():
        j = int(key.split("\0", 1)[0])
        by: dict[str, set[str]] = defaultdict(set)
        for s in starts:
            by[skel[s + j]].add(phone[s + j])
        if len(by) < 2:
            continue
        for s in starts:
            mine = consonants(phone[s + j])
            written = "".join(phone[s:s + phrase])
            alts = []
            for k, forms in by.items():
                if k == skel[s + j]:
                    continue
                for alt_word in forms:
                    if consonants(alt_word) == mine or alt_word == phone[s + j]:
                        continue
                    ident = (s, j, alt_word)
                    if ident in seen:
                        continue
                    seen.add(ident)
                    alts.append("".join(phone[s:s + j]) + alt_word + "".join(phone[s + j + 1:s + phrase]))
            if not alts:
                continue
            windows.append({"start": s, "word": s + j, "g0": words[s]["g0"], "g1": words[s + phrase - 1]["g1"],
                            "written": written, "alts": alts})
    windows.sort(key=lambda w: w["g0"])
    return {"corpus": corpus, "words": words, "windows": windows, "byStart": {w["g0"]: w for w in words}}


def tiles(index: dict, at: int, ref: str) -> list[dict] | None:
    """Word-aligned cover of corpus[at:at+len(ref)). None when `at` falls mid-word."""
    pos, end = at, at + len(ref)
    out = []
    by = index["byStart"]
    while pos < end:
        w = by.get(pos)
        if w is None or w["g1"] > end:
            return None
        out.append(w)
        pos = w["g1"]
    return out


def locate(index: dict, ref: str) -> tuple[int, list[dict]] | None:
    corpus = index["corpus"]
    start = 0
    while True:
        at = corpus.find(ref, start)
        if at < 0:
            return None
        covered = tiles(index, at, ref)
        if covered is not None:
            return at, covered
        start = at + 1


def covers_phrase(index: dict, ref: str, cache: dict) -> bool:
    hit = cache.get(ref)
    if hit is not None:
        return hit
    found = locate(index, ref)
    ok = False
    if found is not None:
        at, _ = found
        end = at + len(ref)
        wins = index["windows"]
        lo, hi = 0, len(wins)
        while lo < hi:
            mid = (lo + hi) // 2
            if wins[mid]["g0"] < at:
                lo = mid + 1
            else:
                hi = mid
        i = lo
        while i < len(wins) and wins[i]["g0"] < end:
            if wins[i]["g1"] <= end:
                ok = True
                break
            i += 1
    cache[ref] = ok
    return ok


def select_clips(manifest: str, gold_text: str, index: dict) -> list[dict]:
    import numpy as np

    gold = exclusions(gold_text)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for line in open(manifest):
        c = json.loads(line)
        if c.get("bucket") != "clean-match" or c.get("split") != "train":
            continue
        if held_out(gold, c["shard"], int(c["row"]), "train"):
            continue
        if not c.get("ref"):
            continue
        clusters[str(c["cluster"])].append(c)
    eligible = sorted(k for k, rows in clusters.items() if len(rows) >= PER_READER)
    rng = np.random.default_rng(SEED)
    take = min(READERS, len(eligible))
    chosen = [eligible[i] for i in sorted(rng.choice(len(eligible), size=take, replace=False).tolist())]
    cache: dict[str, bool] = {}
    picked = []
    for cluster in chosen:
        rows = clusters[cluster]
        order = rng.permutation(len(rows))
        phrase, other = [], []
        for i in order[:SCAN_PER_READER]:
            c = rows[int(i)]
            (phrase if covers_phrase(index, c["ref"], cache) else other).append(c)
        body = phrase[:5] + other[: max(0, PER_READER - min(5, len(phrase)))]
        extra = phrase[5:5 + PHRASE_EXTRA]
        for c in body + extra:
            picked.append({"shard": c["shard"], "row": int(c["row"]), "cluster": cluster, "ref": c["ref"],
                           "key": f"{c['shard']}:{c['row']}"})
    return picked


def empty_stats() -> dict:
    return {"wordConf": [], "phoneNll": [], "sub": defaultdict(int), "refCount": defaultdict(int),
            "phraseMargins": [], "blankCosts": [], "correctWords": 0, "clips": 0, "located": 0, "unlocated": 0}


def consume(stats: dict, logp, ref: str, at: int, index: dict) -> None:
    import numpy as np

    ids = logp.argmax(-1)
    runs = collapse_runs(ids)
    hyp = "".join(TOKENS[i - 1] for i, _, _ in runs)
    ops = align_ops(ref, hyp)
    eq_hyp = {}
    ins_before: dict[int, list[int]] = defaultdict(list)
    cursor = 0
    for kind, ri, hj in ops:
        if kind in ("eq", "sub"):
            stats["refCount"][ref[ri]] += 1
            if kind == "sub":
                stats["sub"][f"{ref[ri]}>{hyp[hj]}"] += 1
            else:
                eq_hyp[ri] = hj
            cursor = ri + 1
        elif kind == "del":
            stats["refCount"][ref[ri]] += 1
            cursor = ri + 1
        else:
            ins_before[cursor].append(hj)
    found = tiles(index, at, ref)
    local = found or []
    correct = {}
    for w in local:
        s, e = w["g0"] - at, w["g1"] - at
        if e <= s:
            continue
        if any(i not in eq_hyp for i in range(s, e)):
            continue
        if any(ins_before.get(i) for i in range(s + 1, e)):
            continue
        hs = [eq_hyp[i] for i in range(s, e)]
        frame0, frame1 = runs[hs[0]][1], runs[hs[-1]][2] + 1
        labels = phoneme_ids(w["phones"])
        nll = ctc_nll(logp[frame0:frame1], labels)
        if not math.isfinite(nll):
            continue
        confs = []
        for h in hs:
            i, a, b = runs[h]
            confs.append(float(np.exp(logp[a:b + 1, i]).max()))
        correct[w["i"]] = {"frame0": frame0, "frame1": frame1, "conf": sum(confs) / len(confs), "nll": nll / len(labels)}
        stats["correctWords"] += 1
        stats["wordConf"].append(correct[w["i"]]["conf"])
        stats["phoneNll"].append(correct[w["i"]]["nll"])
    for a, b in zip(local, local[1:]):
        if b["i"] != a["i"] + 1 or a["i"] not in correct or b["i"] not in correct:
            continue
        if ins_before.get(b["g0"] - at):
            continue
        f0, f1 = correct[a["i"]]["frame1"], correct[b["i"]]["frame0"]
        if f1 - f0 < MIN_GAP_FRAMES:
            continue
        stats["blankCosts"].append(float((-logp[f0:f1, 0]).sum()))
    for w in index["windows"]:
        if w["g0"] < at or w["g1"] > at + len(ref):
            continue
        idxs = range(w["start"], w["start"] + 4)
        if any(i not in correct for i in idxs):
            continue
        f0 = correct[w["start"]]["frame0"]
        f1 = correct[w["start"] + 3]["frame1"]
        if f1 - f0 < 4:
            continue
        written = ctc_nll(logp[f0:f1], phoneme_ids(w["written"]))
        if not math.isfinite(written):
            continue
        for alt in w["alts"]:
            alt_nll = ctc_nll(logp[f0:f1], phoneme_ids(alt))
            if math.isfinite(alt_nll):
                stats["phraseMargins"].append(written - alt_nll)


def finish(stats: dict) -> dict:
    return {**stats, "sub": dict(stats["sub"]), "refCount": dict(stats["refCount"])}


def load_waves(clips: list[dict], aug: dict | None) -> list[dict]:
    import io
    from collections import defaultdict as dd

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    from student import mix_noise
    from student_train import pick, reverb

    by = dd(list)
    for c in clips:
        by[c["shard"]].append(c)
    raw = {}
    for shard, rows in by.items():
        want = {r["row"] for r in rows}
        table = pq.read_table(f"/student/s1/{shard}.parquet", columns=["row", "flac"])
        for row, blob in zip(table.column("row").to_pylist(), table.column("flac").to_pylist()):
            if row in want:
                raw[(shard, row)] = blob
        del table
    cfg = {"rir_p": 0.2, "noise_p": 0.5, "snr_db": (5.0, 25.0)}
    out = []
    for c in clips:
        blob = raw.get((c["shard"], c["row"]))
        if blob is None:
            continue
        wave, _ = sf.read(io.BytesIO(blob), dtype="float32")
        if getattr(wave, "ndim", 1) > 1:
            wave = wave.mean(axis=1)
        rng = np.random.default_rng(zlib.crc32(c["key"].encode()) & 0xFFFFFFFF)
        if aug is not None:
            if rng.random() < cfg["rir_p"]:
                wave = reverb(wave, pick(aug["rir"], rng))
            if rng.random() < cfg["noise_p"]:
                wave = mix_noise(wave, pick(aug["noise"], rng), float(rng.uniform(*cfg["snr_db"])), rng)
            wave = wave * 10 ** (float(rng.uniform(-10, 6)) / 20)
        out.append({**c, "wave": np.ascontiguousarray(wave, dtype=np.float32)})
    return out


def fbank_of(wave):
    import numpy as np
    import torch
    import torchaudio

    return torchaudio.compliance.kaldi.fbank(
        torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0), num_mel_bins=80, sample_frequency=16000,
        dither=0.0, energy_floor=1e-10, high_freq=-400,
    )


def decode(model, feats: list) -> list:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    model.streaming_eval(16, 128)
    order = sorted(range(len(feats)), key=lambda i: feats[i].shape[0])
    out = [None] * len(feats)
    with torch.inference_mode():
        for k in range(0, len(order), 8):
            idx = order[k:k + 8]
            x = torch.nn.utils.rnn.pad_sequence([feats[i] for i in idx], batch_first=True, padding_value=-23.0259).to(device)
            lens = torch.tensor([feats[i].shape[0] for i in idx], device=device)
            lp, out_lens = model(x, lens)
            lp = lp.float().cpu()
            for j, i in enumerate(idx):
                out[i] = lp[j, : int(out_lens[j])].numpy()
    return out


def run_model(name: str, ckpt: str, size: str, clips: list[dict], index: dict) -> dict:
    import torch

    from student_train import build_model

    print(f"load {name} {ckpt} size={size}", flush=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["cfg"]["size"] == size, (ck["cfg"]["size"], size)
    model = build_model(size)
    model.load_state_dict(ck["model"])
    n = sum(p.numel() for p in model.parameters())
    print(f"{name} params {n}", flush=True)
    if size == "wide":
        assert n == 247690818, n
    feats = [c["feats"] for c in clips]
    lps = decode(model, feats)
    stats = empty_stats()
    for c, lp in zip(clips, lps):
        stats["clips"] += 1
        found = locate(index, c["ref"])
        if found is None:
            stats["unlocated"] += 1
            continue
        stats["located"] += 1
        try:
            consume(stats, lp, c["ref"], found[0], index)
        except Exception as e:  # one bad clip does not drop the reader
            print(f"skip {c['key']}: {e}", flush=True)
        if stats["clips"] % 25 == 0:
            print(f"{name} {stats['clips']}/{len(clips)} correct {stats['correctWords']} phrases {len(stats['phraseMargins'])}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return finish(stats)


def main() -> None:
    import numpy as np

    shard = int(os.environ.get("LAB_SHARD", "0"))
    shards = int(os.environ.get("LAB_SHARDS", "1"))
    out = Path(os.environ.get("LAB_OUT", "/tmp/calibration"))
    out.mkdir(parents=True, exist_ok=True)
    if shard == 0:
        src = Path("/scratch/listener-wider-student-capacity/wide-clean-b-int8")
        dst = Path("/scratch/listener-wide-ear-refit-calibration/wide-clean-int8")
        if src.is_dir() and not (dst / "model.onnx").exists():
            import shutil
            shutil.copytree(src, dst)
            print(f"copied model -> {dst}", flush=True)
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    print("index", flush=True)
    index = build_index(ref)
    print(f"words {len(index['words'])} windows {len(index['windows'])}", flush=True)
    gold_path = Path("/root/gold-per-v1-exclude.txt")
    if not gold_path.exists():
        gold_path = ROOT / "docs/benchmarks/gold-per-v1-exclude.txt"
    clips = select_clips("/student/manifest-s1.jsonl", gold_path.read_text(), index)
    mine = [c for i, c in enumerate(clips) if i % shards == shard]
    print(f"clips {len(clips)} shard {shard}/{shards} -> {len(mine)} readers {len({c['cluster'] for c in clips})}", flush=True)
    aug = None
    if os.path.exists("/student/aug/rir.npz"):
        aug = {k: dict(np.load(f"/student/aug/{k}.npz")) for k in ("noise", "rir")}
    loaded = load_waves(mine, aug)
    print(f"waves {len(loaded)}", flush=True)
    for c in loaded:
        c["feats"] = fbank_of(c.pop("wave"))
    models = {
        "shipped": ("/student/prior-runs/spliced-clean/ckpt.pt", "full"),
        "wide": ("/scratch/listener-wider-student-capacity/student-runs/wide-clean/ckpt.pt", "wide"),
    }
    summary = {}
    for name, (ckpt, size) in models.items():
        stats = run_model(name, ckpt, size, loaded, index)
        (out / f"{name}.json").write_text(json.dumps(stats))
        summary[name] = {"clips": stats["clips"], "located": stats["located"], "correctWords": stats["correctWords"],
                         "phrases": len(stats["phraseMargins"]), "blanks": len(stats["blankCosts"]),
                         "medianNll": median(stats["phoneNll"]), "words": len(stats["wordConf"])}
        print(name, json.dumps(summary[name]), flush=True)
    if (out / "shipped.json").exists() and (out / "wide.json").exists():
        shipped = json.loads((out / "shipped.json").read_text())
        wide = json.loads((out / "wide.json").read_text())
        anchor = anchor_from(shipped)
        idem = derive(shipped, anchor)
        wide_c = derive(wide, anchor)
        noise = within_noise(idem)
        print("IDEMPOTENCE", "ok" if not noise else noise, flush=True)
        print("SHIPPED", json.dumps({k: idem[k] for k in ("unsure", "support", "natsPerEdit", "phraseSwitch", "soft", "blankCut", "scale")}), flush=True)
        print("WIDE", json.dumps({k: wide_c[k] for k in ("unsure", "support", "natsPerEdit", "phraseSwitch", "soft", "blankCut", "scale", "costs")}), flush=True)
    print("DONE", json.dumps(summary), flush=True)


def derive_files(shipped_path: str, wide_path: str) -> None:
    shipped = json.loads(Path(shipped_path).read_text())
    wide = json.loads(Path(wide_path).read_text())
    anchor = anchor_from(shipped)
    idem = derive(shipped, anchor)
    got = derive(wide, anchor)
    print(json.dumps({"idempotence": within_noise(idem), "shipped": idem, "wide": got}, ensure_ascii=False, indent=1))


def stream_onnx(manifest: dict, feats) -> "object":
    """The SDK's streaming loop: one window of fbank frames, the running cache, int8 or fp32."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(manifest["_path"], providers=["CPUExecutionProvider"])
    st = {s["name"]: np.zeros(s["shape"], dtype=np.int64 if s["dtype"] == "int64" else np.float32) for s in manifest["states"]}
    T, shift, chunk = manifest["window"], manifest["shift"], manifest["chunk"]
    f = np.asarray(feats, dtype=np.float32)
    need = ((f.shape[0] - 7) // 2 + 1) // 2
    windows = max(1, -(-need // (chunk // 2)))
    total = T + (windows - 1) * shift
    if total > f.shape[0]:
        f = np.concatenate([f, np.full((total - f.shape[0], 80), -23.0259, np.float32)])
    outs = []
    names = [s["name"] for s in manifest["states"]]
    for a in range(0, total - T + 1, shift):
        res = sess.run(None, {"x": f[None, a:a + T], **st})
        outs.append(res[0][0])
        st = dict(zip(names, res[1:]))
    return np.concatenate(outs)[:need]


def evidence_doc(lp, inventory: str) -> dict:
    import numpy as np

    ids = lp.argmax(-1)
    runs = collapse_runs(ids)
    phonemes = [{"p": TOKENS[i - 1], "start": a, "end": b, "conf": round(float(np.exp(lp[a:b + 1, i]).max()), 4)} for i, a, b in runs]
    k = 5
    part = np.argpartition(-lp, k - 1, axis=1)[:, :k]
    flat_i, flat_p = [], []
    for t in range(lp.shape[0]):
        ix = part[t][np.argsort(-lp[t, part[t]])]
        flat_i.extend(int(i) for i in ix)
        flat_p.extend(round(float(np.exp(lp[t, i])), 4) for i in ix)
    return {"inventory": inventory, "frameMs": 40, "nFrames": int(lp.shape[0]), "phonemes": phonemes,
            "posteriors": {"k": k, "ids": flat_i, "probs": flat_p}}


def decode_onnx_sessions(wavs: list[str], model_dir: str) -> None:
    import subprocess

    import numpy as np

    out = Path(os.environ.get("LAB_OUT", "/tmp/calibration"))
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path("/scratch/listener-wide-ear-refit-calibration/evidence")
    scratch.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    manifest = json.loads(Path(model_dir, "manifest.json").read_text())
    manifest["_path"] = str(Path(model_dir, manifest["file"]))
    for wav in wavs:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True).stdout
        wave = np.frombuffer(pcm, dtype=np.float32).copy()
        lp = stream_onnx(manifest, fbank_of(wave).numpy())
        name = Path(wav).name.replace(".phone.wav", "").replace(".wav", "")
        doc = evidence_doc(lp, ref["inventory"]["id"])
        text = json.dumps(doc)
        (out / f"{name}.evidence.json").write_text(text)
        (scratch / f"{name}.evidence.json").write_text(text)
        print(f"{name} frames {doc['nFrames']} audio_ms {len(wave) / 16:.0f} frame_ms {doc['nFrames'] * 40} phones {len(doc['phonemes'])}", flush=True)


def decode_sessions(wavs: list[str]) -> None:
    """Greedy streaming decode of wavs with the wide student, as session evidence JSON."""
    import subprocess

    import numpy as np
    import torch

    from student_train import build_model

    out = Path(os.environ.get("LAB_OUT", "/tmp/calibration"))
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path("/scratch/listener-wide-ear-refit-calibration/evidence")
    scratch.mkdir(parents=True, exist_ok=True)
    ref = json.loads((ROOT / "packages/core/assets/ref-v2.json").read_text())
    inventory = ref["inventory"]["id"]
    ckpt = "/scratch/listener-wider-student-capacity/student-runs/wide-clean/ckpt.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_model("wide")
    model.load_state_dict(ck["model"])
    feats = []
    metas = []
    for wav in wavs:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True).stdout
        wave = np.frombuffer(pcm, dtype=np.float32).copy()
        metas.append({"wav": wav, "samples": int(wave.shape[0]), "durationS": len(wave) / 16000})
        feats.append(fbank_of(wave))
    lps = decode(model, feats)
    for meta, lp in zip(metas, lps):
        ids = lp.argmax(-1)
        runs = collapse_runs(ids)
        phonemes = []
        for i, a, b in runs:
            phonemes.append({"p": TOKENS[i - 1], "start": a, "end": b,
                             "conf": round(float(np.exp(lp[a:b + 1, i]).max()), 4)})
        k = 5
        part = np.argpartition(-lp, k - 1, axis=1)[:, :k]
        flat_i, flat_p = [], []
        for t in range(lp.shape[0]):
            ix = part[t][np.argsort(-lp[t, part[t]])]
            flat_i.extend(int(i) for i in ix)
            flat_p.extend(round(float(np.exp(lp[t, i])), 4) for i in ix)
        name = Path(meta["wav"]).name.replace(".phone.wav", "").replace(".wav", "")
        doc = {"inventory": inventory, "frameMs": 40, "nFrames": int(lp.shape[0]), "phonemes": phonemes,
               "posteriors": {"k": k, "ids": flat_i, "probs": flat_p}}
        text = json.dumps(doc)
        (out / f"{name}.evidence.json").write_text(text)
        (scratch / f"{name}.evidence.json").write_text(text)
        print(f"{name} frames {doc['nFrames']} audio_ms {meta['durationS'] * 1000:.0f} frame_ms {doc['nFrames'] * 40} phones {len(phonemes)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "derive":
        derive_files(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "sessions":
        decode_sessions(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "onnx-sessions":
        decode_onnx_sessions(sys.argv[2:], "/scratch/listener-wide-ear-refit-calibration/wide-clean-int8")
    else:
        main()
