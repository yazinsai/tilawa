"""Turn the audited help-page clips into a labelled eval set.

    uv run tools/help_import.py build            # .scratch/help/labels.jsonl + split summary
    uv run tools/help_import.py build --swap     # also relabel the version_swap.json clips (only after the audit confirms them)
    uv run tools/help_import.py upload           # wav + labels -> private Modal volume tilawa-help

Audio is converted to 16 kHz mono WAV by `help_audit.py pull`; the originals stay in .scratch/help/raw/audio.
"""

import json
import os
import random
import re
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = Path(os.environ.get("HELP_DIR", ROOT / ".scratch/help"))
RAW, WAV = DIR / "raw", DIR / "wav"
QURAN = Path.home() / "ai/projects/offline-tarteel/web/frontend/public/quran.json"
INDEX = ROOT / "packages/core/assets/ref-v2.json"
VOLUME = "tilawa-help"
# Clips whose planted mistake has no ref-v2 word. Anything else that fails to map stops the build.
EXCLUDED = {"muo3pp0v_f7928cdd": "v1 repeat planted on the basmala word الرحمن before 104:1; ref-v2 has no basmala there"}
ARABIC_LETTER = re.compile(r"[\u0621-\u064a\u0671]")
BASMALA = ["بسم", "الله", "الرحمن", "الرحيم"]
AUDIT_FAIL = {"2": "mistake not made", "3": "other or extra mistake", "4": "wrong or incomplete passage", "5": "audio unusable"}
# Strata whose audit error rate failed: an unanswered clip from one of these is dropped too.
FAILED_STRATA = {"teacher: poor match", "seconds per word out of range"}


def audit_exclusions() -> dict[str, str]:
    answers = {a["id"]: a for a in map(json.loads, (DIR / "answers.jsonl").open())}
    out = {i: f"audit: {AUDIT_FAIL[a['verdict']]}" for i, a in answers.items() if a.get("verdict") in AUDIT_FAIL}
    for s in json.loads((DIR / "sample.json").read_text()):
        if s["id"] not in out and answers.get(s["id"], {}).get("verdict") in (None, "0") and FAILED_STRATA & set(s["why"]):
            out[s["id"]] = "audit: unconfirmed clip in a failed stratum"
    return out


class WordCountMismatch(ValueError):
    pass


def words(text: str) -> list[str]:
    """Words with standalone waqf marks dropped, matching the help page's 1-based indices."""
    return [w for w in text.replace("\ufeff", "").split() if ARABIC_LETTER.search(w)]


def word_ref(quran: dict[str, dict], ref: dict[str, list[str]], ayah: str, word: int) -> dict:
    """The help page's 1-based `text_clean` word index as a ref-v2 WordRef.

    quran.json prepends the basmala to each surah's first ayah and the help page counts it; ref-v2 doesn't.
    """
    clean, target = words(quran[ayah]["text_clean"]), ref[ayah]
    shift = 0
    if len(clean) == len(target) + len(BASMALA) and clean[: len(BASMALA)] == BASMALA:
        shift = len(BASMALA)
    elif len(clean) != len(target):
        raise WordCountMismatch(f"{ayah}: {len(clean)} text_clean words vs {len(target)} ref-v2 words")
    if not shift < word <= len(clean):
        raise WordCountMismatch(f"{ayah}: word {word} has no ref-v2 word (basmala prefix {shift}, {len(clean)} words)")
    s, a = map(int, ayah.split(":"))
    return {"surah": s, "ayah": a, "word": word - shift}


def ayahs_of(r: dict) -> list[str]:
    if r.get("ayahs"):
        return r["ayahs"]
    ref = r["ref"]
    if "-" not in ref:
        return [ref]
    (s1, a1), (s2, a2) = (map(int, x.split(":")) for x in ref.split("-"))
    assert s1 == s2, ref
    return [f"{s1}:{a}" for a in range(a1, a2 + 1)]


TAJWEED = {"no_shadda": "missing_shadda", "madd_short": "madd_natural"}
LIGHT_EMPHATIC = {"ص": "consonant_sad_sin", "ط": "consonant_ta_ta", "ض": "consonant_dad_dal", "ق": "consonant_qaf_kaf"}
# The checker's own threshold: an omission of this many words or more is a skipped passage.
SKIPPED_PASSAGE_WORDS = 3
DIACRITIC = re.compile(r"[\u0640\u064b-\u065f\u0670\u06d6-\u06ed]")
ALEF = str.maketrans("ٱأإآى", "اااای")


def skeleton(word: str) -> str:
    return DIACRITIC.sub("", word).translate(ALEF)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def forms_other_ayah(ref: dict[str, list[str]], ayahs: list[str], recited: list[str]) -> bool:
    """Whether the recited words fit another span at least as well as the prompted ayahs (word edit distance).

    Candidate spans are runs of 1 to len(ayahs) + 1 consecutive whole ayahs that share no ayah with the prompt.
    """
    said = [skeleton(w) for w in recited]
    intended = edit_distance(said, [skeleton(w) for a in ayahs for w in ref[a]])
    if intended == 0 or not said:
        return False
    keys, k, prompt = list(ref), len(ayahs) + 1, set(ayahs)
    for i in range(len(keys)):
        for n in range(1, k + 1):
            window = keys[i:i + n]
            if prompt & set(window):
                break
            words = [skeleton(w) for a in window for w in ref[a]]
            if abs(len(words) - len(said)) > intended:
                continue
            vocab = set(words)
            if sum(w not in vocab for w in said) <= intended and edit_distance(said, words) <= intended:
                return True
    return False


def recited_words(ref: dict[str, list[str]], ayahs: list[str], mistakes: list[dict]) -> list[str]:
    """The prompted ayahs' words with the planted word-level mistakes applied (tajweed and vowels leave words alone)."""
    out = []
    for a in ayahs:
        if any(m["kind"] == "skip_ayah" and m["ayah"] == a for m in mistakes):
            continue
        for i, w in enumerate(ref[a], 1):
            at = [m for m in mistakes if m.get("at") == (a, i)]
            kinds = {m["kind"] for m in at}
            if "skip_word" in kinds:
                continue
            sub = next((m for m in at if m["kind"] == "substitution"), None)
            out.append(sub.get("said_uthmani") or sub.get("said") or "?" if sub else w)
            if "repeat" in kinds:
                out.append(w)
    return out


def label(quran: dict[str, dict], ref: dict[str, list[str]], r: dict) -> dict:
    """One clip's planted mistakes as the checker events a correct engine should end with."""
    events, non_errors, unsupported, placed = [], [], [], []
    for m in r.get("mistakes") or []:
        k = m["kind"]
        if k == "skip_ayah":
            s, a = map(int, m["ayah"].split(":"))
            n = len(ref[m["ayah"]])
            cls = "skipped_passage" if n >= SKIPPED_PASSAGE_WORDS else "omitted_word"
            events.append({"errorClass": cls, "state": "suspected", "location": {"surah": s, "ayah": a, "word": 1}, "words": n})
            placed.append(m)
            continue
        at = word_ref(quran, ref, m["ayah"], m["word"])
        placed.append({**m, "at": (m["ayah"], at["word"])})
        if k == "repeat":
            non_errors.append({"kind": "repeat", "location": at})
            continue
        if k == "tajweed":
            cls = LIGHT_EMPHATIC.get(m.get("letter")) if m["rule"] == "light_emphatic" else TAJWEED.get(m["rule"])
            if cls is None:
                unsupported.append({"rule": m["rule"], "location": at})
                continue
        else:
            cls = {"substitution": "wrong_word", "skip_word": "omitted_word", "vowel": "short_vowel"}[k]
        e = {"errorClass": cls, "state": "confirmed", "location": at, "expected": m["expected"]}
        if m.get("said_uthmani"):
            e["observed"] = m["said_uthmani"]
        events.append(e)
    ayahs = ayahs_of(r)
    other = forms_other_ayah(ref, ayahs, recited_words(ref, ayahs, placed)) if events else False
    events = [{**e, "forms_other_ayah": other} for e in events]
    return {
        "id": r["id"],
        "speaker": r["speaker"],
        "kind": r["kind"],
        "ayahs": ayahs,
        "events": events,
        "non_errors": non_errors,
        "unsupported": unsupported,
        "scored": not r.get("extra_mistake"),
        **({"relabel": r["relabel"]} if r.get("relabel") else {}),
    }


HEAVY_ERROR_CLIPS = 16
EVAL_SHARE = 0.5
SPLIT_SEED = 12


def split_speakers(labels: list[dict], share: float = EVAL_SHARE) -> dict[str, str]:
    """Speaker -> train or eval. Heavy speakers stay in train; the rest are dealt so each kind's eval share is near `share`."""
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for lab in labels:
        by_speaker[lab["speaker"]][lab["kind"]] += 1
    heavy = {s for s, c in by_speaker.items() if sum(n for k, n in c.items() if k != "none") >= HEAVY_ERROR_CLIPS}
    light = [s for s in by_speaker if s not in heavy]
    target = Counter()
    for s in light:
        target.update(by_speaker[s])
    target = {k: share * n for k, n in target.items()}
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(light)
    light.sort(key=lambda s: -sum(by_speaker[s].values()))
    held = Counter()
    split = dict.fromkeys(heavy, "train")
    for s in light:
        c = by_speaker[s]
        gain = sum(abs(target[k] - held[k]) - abs(target[k] - held[k] - n) for k, n in c.items())
        split[s] = "eval" if gain > 0 else "train"
        if gain > 0:
            held.update(c)
    return split


def site_counts(labels: list[dict]) -> dict[str, int]:
    """Scored error sites per class for the >= 15-sites rule: one speaker on one ayah is one site, whatever the seed."""
    sites = defaultdict(set)
    for lab in labels:
        if not lab["scored"]:
            continue
        for e in lab["events"]:
            loc = e["location"]
            sites[e["errorClass"]].add((lab["speaker"], loc["surah"], loc["ayah"]))
    return {k: len(v) for k, v in sites.items()}


def apply_swaps(records: list[dict], swaps: dict[str, dict]) -> list[dict]:
    """Replace the stored v1 label with the v2 prompt the screen showed, for clips the audit confirmed by ear."""
    out = []
    for r in records:
        s = swaps.get(r["id"])
        if s:
            r = {**r, "ref": s["ref"], "ayahs": None, "kind": s["kind"], "mistakes": s.get("mistakes") or [], "prompt_version": 2, "relabel": "version_swap"}
        out.append(r)
    return out


def wav_ok(path: Path) -> bool:
    try:
        with wave.open(str(path)) as w:
            return w.getframerate() == 16000 and w.getnchannels() == 1
    except (wave.Error, FileNotFoundError, EOFError):
        return False


def convert(records: list[dict]) -> None:
    """16 kHz mono WAV per clip under wav/; the originals in raw/audio are left alone."""
    WAV.mkdir(parents=True, exist_ok=True)
    for r in records:
        dst = WAV / f"{r['id']}.wav"
        if wav_ok(dst):
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(RAW / r["file"]), "-ar", "16000", "-ac", "1", str(dst)], check=True)
        if not wav_ok(dst):
            sys.exit(f"{dst} is not 16 kHz mono after conversion")


def build(swap: bool) -> None:
    records = [json.loads(line) for line in (RAW / "submissions.jsonl").open()]
    convert(records)
    quran = {f"{a['surah']}:{a['ayah']}": a for a in json.loads(QURAN.read_text(encoding="utf-8-sig"))}
    ref = {f"{a['s']}:{a['a']}": a["w"] for a in json.loads(INDEX.read_text())["ayahs"]}
    # The split is drawn from the stored labels so --swap never moves a speaker.
    split = split_speakers([{"speaker": r["speaker"], "kind": r["kind"]} for r in records])
    if swap:
        records = apply_swaps(records, json.loads((DIR / "version_swap.json").read_text()))
    excluded = EXCLUDED | audit_exclusions()
    labels = []
    for r in records:
        if r["id"] in excluded:
            labels.append({"id": r["id"], "speaker": r["speaker"], "kind": r["kind"], "ayahs": ayahs_of(r), "events": [], "non_errors": [], "unsupported": [], "scored": False, "excluded": excluded[r["id"]], "split": split[r["speaker"]], "wav": f"wav/{r['id']}.wav"})
            continue
        lab = label(quran, ref, r)
        for a in lab["ayahs"]:
            if a not in ref:
                raise WordCountMismatch(f"{r['id']}: {a} not in ref-v2")
            word_ref(quran, ref, a, len(words(quran[a]["text_clean"])))
        lab |= {"split": split[r["speaker"]], "wav": f"wav/{r['id']}.wav", "prompt_version": r["prompt_version"], "level": r.get("level"), "device": r.get("device")}
        labels.append(lab)
    with (DIR / "labels.jsonl").open("w") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    print(f"{len(labels)} clips -> {DIR / 'labels.jsonl'}{' (version swaps applied)' if swap else ''}")
    for name in ("train", "eval"):
        part = [lab for lab in labels if lab["split"] == name]
        kinds = Counter(lab["kind"] for lab in part)
        print(f"{name}: {len({lab['speaker'] for lab in part})} speakers, {len(part)} clips, {sum(not lab['scored'] for lab in part)} unscored; " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())))
        print("  sites: " + ", ".join(f"{k} {n}" for k, n in sorted(site_counts(part).items())))


def upload() -> None:
    subprocess.run(["modal", "volume", "create", VOLUME], capture_output=True)
    subprocess.run(["modal", "volume", "put", "-f", VOLUME, str(WAV), "/wav"], check=True)
    subprocess.run(["modal", "volume", "put", "-f", VOLUME, str(DIR / "labels.jsonl"), "/labels.jsonl"], check=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build("--swap" in sys.argv)
    elif cmd == "upload":
        upload()
    else:
        sys.exit(__doc__)
