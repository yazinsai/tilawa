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
LOCATOR_KINDS = ("omitted", "repeated", "restarted", "substituted")
REAL_KINDS = ("skipped", "wrong_word", "repeat", "restart", "tajweed_only")
VERDICTS = ("real", "not_slip", "unsure", "audio_bad")
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
    total = sum(len(v) for v in groups.values())
    if total == 0 or n <= 0:
        return []
    raw = {kind: n * len(groups[kind]) / total for kind in groups}
    alloc = {kind: min(int(math.floor(raw[kind])), len(groups[kind])) for kind in groups}
    left = n - sum(alloc.values())
    # Largest remainder. A tie goes to the smaller kind so a rare slip is not dropped.
    order = sorted(
        groups,
        key=lambda kind: (raw[kind] - math.floor(raw[kind]), -len(groups[kind])),
        reverse=True,
    )
    for kind in order:
        if left <= 0:
            break
        if alloc[kind] >= len(groups[kind]):
            continue
        alloc[kind] += 1
        left -= 1
    rng = random.Random(seed)
    out: list[dict] = []
    for kind in LOCATOR_KINDS:
        take = alloc[kind]
        if take:
            out.extend(rng.sample(groups[kind], take))
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


def summarize_queues(queues: dict[str, list[dict]], tags: dict) -> dict:
    report: dict = {"queues": {}}
    for name in QUEUES:
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
    report["pass"] = report["queues"]["A"]["gate_shipped"] == "pass"
    return report


def _precision_gate(new: dict, old: dict) -> str:
    if not new["n"] or not old["n"]:
        return "incomplete"
    if new["p"] + 1e-12 < old["p"]:
        return "fail"
    return "pass"


def format_report(report: dict) -> str:
    lines = []
    a = report["queues"]["A"]
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
    def __init__(self, root: Path):
        _outside_repo(root)
        self.root = root
        self.meta = root / "meta"
        self.audio = root / "audio"
        self.meta.mkdir(parents=True, exist_ok=True)
        self.audio.mkdir(parents=True, exist_ok=True)
        self.tags_path = root / "tags.json"
        self.queue_path = root / "queue.json"
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

    def audio_path(self, item: dict) -> Path:
        audio = item["audio"]
        # Names come from the manifest / clip id, never from the request path.
        name = Path(audio["name"]).name
        return self.audio / f"{audio['source']}__{name}"

    def remote_audio(self, item: dict) -> str:
        audio = item["audio"]
        name = Path(audio["name"]).name
        if audio["source"] == "tlog":
            stem = name[:-5] if name.endswith(".flac") else name
            return f"audio/tlog/{stem}.flac"
        return f"help/{name}"

    def ensure_audio(self, item: dict) -> Path:
        dest = self.audio_path(item)
        lock = self._lock_for(str(dest))
        with lock:
            if dest.is_file() and dest.stat().st_size > 0:
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


def _counts(saved: dict, tags: dict) -> dict:
    counts = {}
    for name in QUEUES:
        rows = saved["queues"][name]
        done = sum(
            1 for row in rows
            if row["tag_id"] in tags and (tags[row["tag_id"]] or {}).get("verdict") in VERDICTS
        )
        counts[name] = {"done": done, "n": len(rows)}
    return counts


def public_item(saved: dict, tags: dict, queue: str, index: int) -> dict:
    items = saved["queues"][queue]
    if not items:
        return {
            "queue": queue, "index": 0, "n": 0, "surah": 0, "ayah": 0, "surah_name": "",
            "words": [], "highlight": [0, 0], "span": None, "window_s": WINDOW_S,
            "tag": None, "counts": _counts(saved, tags), "audio": "", "empty": True,
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
    if isinstance(tag, dict) and tag.get("verdict") in VERDICTS:
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
        "counts": _counts(saved, tags),
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
const REALS = [
  ["skipped", "Skipped", "1"],
  ["wrong_word", "Wrong word", "2"],
  ["repeat", "Repeat", "3"],
  ["restart", "Restart", "4"],
  ["tajweed_only", "Tajweed only", "5"],
];
const VERDICT_KEYS = { n: "not_slip", u: "unsure", b: "audio_bad" };
let queue = "A";
let index = 0;
let item = null;
let stopAt = null;

const $ = (id) => document.getElementById(id);
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function renderQueues(counts) {
  $("queues").innerHTML = ["A", "B", "C"].map((name) => {
    const c = counts[name] || { done: 0, n: 0 };
    return `<button type="button" data-queue="${name}" class="${name === queue ? "on" : ""}">${name} ${c.done}/${c.n}</button>`;
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
  $("meta").textContent = queue + "  " + (index + 1) + "/" + item.n + "   " + ref + name;
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
  const n = "12345".indexOf(k);
  if (n >= 0) { save({ verdict: "real", kind: REALS[n][0] }, true); return; }
  if (VERDICT_KEYS[k]) save({ verdict: VERDICT_KEYS[k], kind: null }, true);
});
fetch("/api/resume").then((r) => r.json()).then((data) => load(data.queue, data.index));
</script>
</body>
</html>
"""


class App:
    def __init__(self, cache: Cache, saved: dict):
        self.cache = cache
        self.saved = saved
        self.tags = cache.load_tags()
        self.lock = threading.Lock()

    def item(self, queue: str, index: int) -> dict:
        rows = self.saved["queues"][queue]
        if not rows:
            with self.lock:
                return public_item(self.saved, self.tags, queue, 0)
        index = max(0, min(index, len(rows) - 1))
        self.cache.ensure_audio(rows[index])
        with self.lock:
            return public_item(self.saved, self.tags, queue, index)

    def resume(self) -> dict:
        with self.lock:
            tags = self.tags
        for name in QUEUES:
            rows = self.saved["queues"][name]
            for i, row in enumerate(rows):
                tag = tags.get(row["tag_id"]) or {}
                if tag.get("verdict") not in VERDICTS:
                    return {"queue": name, "index": i}
        return {"queue": "A", "index": 0}

    def set_tag(self, queue: str, index: int, verdict: str, kind: str | None, notes: str) -> dict:
        if queue not in QUEUES:
            raise ValueError("queue")
        rows = self.saved["queues"][queue]
        if index < 0 or index >= len(rows):
            raise IndexError("index")
        if verdict not in VERDICTS:
            raise ValueError("verdict")
        if verdict == "real":
            if kind not in REAL_KINDS:
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
            if (tags.get(rows[i]["tag_id"]) or {}).get("verdict") not in VERDICTS:
                nxt = i
                break
        view = public_item(self.saved, tags, queue, index)
        return {"tag": view["tag"], "counts": view["counts"], "next_untagged": nxt}

    def prefetch(self) -> None:
        items = [item for name in QUEUES for item in self.saved["queues"][name]]
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
                    self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/resume":
                    self._json(200, app.resume())
                    return
                if parsed.path == "/api/item":
                    queue = (qs.get("queue") or ["A"])[0]
                    index = int((qs.get("i") or ["0"])[0])
                    if queue not in QUEUES:
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
            ctype = "audio/wav" if path.suffix == ".wav" else "audio/flac"
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


def upload_tags(cache: Cache, report: dict) -> str:
    import io

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote = f"help_slips/tags/{stamp}.json"
    payload = {
        "created": stamp,
        "summary": report,
        "tags": cache.load_tags(),
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    vol = _volume()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(blob), remote)
    return remote


def _print_counts(saved: dict) -> None:
    print("queues", " ".join(f"{name}={len(saved['queues'][name])}" for name in QUEUES), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summarize", action="store_true", help="precision, validity, recall; upload tags")
    parser.add_argument("--no-upload", action="store_true", help="with --summarize, do not write the volume")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the shuffled queues")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tilawa-listen-review")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cache = Cache(args.cache)
    saved = build_from_cache(cache, args.seed, args.rebuild)
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
