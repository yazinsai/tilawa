"""Stage clean-license Quran recitation audio to lhotse CutSets on Modal.

Writes 16 kHz FLAC + phoneme-string supervisions to volume `zipformer-ctc-training`
(`/vol/audio/<source>/`, `/vol/manifests/<source>_cuts.jsonl.gz`). Optional Kaldi
fbank features land at `/vol/fbank/<source>/` and
`/vol/manifests/<source>_cuts_fbank.jsonl.gz`.

Default `--sources` is the shipped mix (everyayah, qua, iqra, retasy, tlog).
QuranTTS is NPL-1.2 and is excluded from that mix; pass `--sources qurantts`
explicitly only for an internal ablation.

EveryAyah uses `greentechapps/everyayah_curated_1s_20s` **train + validation**
only, and drops the three q-lab held-out reciters by name: the repo was
re-split after q-lab was cut, so its current train/validation hold the
`everyayah_heldout` reciters (and the exact benchmark clips). Split-only
exclusion is not enough; see `shared/leak_guard.py`.

Tlog clips whose filename stem is in q-lab `tlog_holdout`, and QUA catalog
rows whose slug/name contains `nufais`, are dropped as held-out leaks.

Usage:
  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah,retasy --limit 20 --skip-fbank

  modal run scripts/prepare_zipformer_data_modal.py --summary-only

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources everyayah --limit 20

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --sources qurantts --limit 20 --skip-fbank   # NPL-1.2 ablation, not shipped

  modal run scripts/prepare_zipformer_data_modal.py \\
      --sources qua --fbank-shards 12              # skip prepare; sharded fbank + merge

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --multi-windows 200 --skip-fbank             # B1 smoke: synthetic multi-ayah cuts

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --multi-windows 40000 --force --fbank-shards 4

  modal run --detach scripts/prepare_zipformer_data_modal.py \\
      --flatten-multi   # MonoCut + whole-cut text; no re-extract
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, Path("/app")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from shared.leak_guard import match_heldout_reciter  # noqa: E402
from shared.paths import resolve_data_file  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers (imported by tests; no lhotse)
# ---------------------------------------------------------------------------

ALL_SOURCES = ("everyayah", "qua", "qurantts", "iqra", "retasy", "tlog")
# QuranTTS is NPL-1.2 — keep the prepare function for opt-in ablation, but it
# is not part of the shipped training mix.
DEFAULT_SOURCES = ("everyayah", "qua", "iqra", "retasy", "tlog")
# B1 synthetic source: built from everyayah_cuts.jsonl.gz, not an HF ingest.
MULTI_SOURCE = "everyayah_multi"
MULTI_MIN_AYAHS = 2
MULTI_MAX_AYAHS = 4
MULTI_MAX_DURATION_S = 25.0
MULTI_GAP_MAX_S = 0.8
# In-memory noise recordings do not round-trip through CutSet JSONL without new
# files (inode cap). Skip the 30 % noise-gap mix; silence pad only.
MULTI_NOISE_FRACTION = 0.0
FBANK_SHARD_CUT_THRESHOLD = 60_000
# q-lab everyayah_heldout wavs are curated *test* shards (test-000NN-of-00013_*).
# Ingest train + validation only — never test — so the held-out set stays unseen.
EVERYAYAH_SPLITS = ("train", "validation")
BAD_RETASY_LABELS = {"in_correct", "not_related_quran", "not_match_aya"}
MIN_DURATION_S = 1.0
LONG_DURATION_S = 20.0
MAX_DURATION_S = 60.0
MATCH_MIN_SCORE = 0.95
HF_EVERYAYAH_CURATED = "greentechapps/everyayah_curated_1s_20s"
HF_EVERYAYAH = "tarteel-ai/everyayah"
HF_QUA = "hetchyy/quranic-universal-ayahs"
HF_QURANTTS = "Quran-Lab/QuranTTS"
HF_IQRA = "IqraEval/Iqra_train"
HF_RETASY = "RetaSy/quranic_audio_dataset"
HF_TLOG = "tarteel-ai/tlog"

_SA_NAME_RE = re.compile(
    r"^(\d+)_(\d+)(?:_[^.]+)?\.(?:wav|flac|mp3)$",
    re.IGNORECASE,
)
_SA_SEARCH_RE = re.compile(
    r"(?<![0-9])(\d{1,3})_(\d{1,3})(?:_[^/]+)?\.(?:wav|flac|mp3)",
    re.IGNORECASE,
)
_SURA_AYAH_KEYS = (
    ("surah", "ayah"),
    ("sura", "ayah"),
    ("chapter", "verse"),
    ("sura_number", "aya_number"),
    ("surah_id", "ayah_id"),
    ("chapter_number", "verse_number"),
)
QLAB_RECITER = {
    "qul_alnufais": "alnufais",
    "everyayah_heldout": "everyayah_heldout",
    "tlog_holdout": "tlog",
}
_PERMISSIVE_LICENSE_MARKERS = (
    "mit license",
    "apache license",
    "bsd license",
    "cc-by",
    "cc by",
    "creative commons attribution",
    "cc0",
    "isc license",
    "unlicense",
)
_NONPERMISSIVE_MARKERS = (
    "no-profit",
    "npl-1",
    "not for sale",
    "commercial use prohibited",
    "non-profit license",
)


def parse_surah_ayah_filename(name: str) -> tuple[int, int] | None:
    """Parse `S_A.wav` / `S_A_id.wav` (optionally with a directory prefix)."""
    if not name:
        return None
    text = str(name).replace("\\", "/")
    base = Path(text).name.split("?")[0]
    m = _SA_NAME_RE.match(base)
    if m is None:
        m = _SA_SEARCH_RE.search(base) or _SA_SEARCH_RE.search(text)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def duration_decision(duration_s: float) -> str:
    """`keep` (1–20s), `long` (20–60s, still written), or skip_*."""
    if duration_s < MIN_DURATION_S:
        return "skip_short"
    if duration_s > MAX_DURATION_S:
        return "skip_long"
    if duration_s > LONG_DURATION_S:
        return "long"
    return "keep"


def retasy_keep(label: str | None) -> bool:
    return label == "correct"


def is_tarteel_dupe(**fields: object) -> bool:
    blob = " ".join(str(v).lower() for v in fields.values() if v not in (None, ""))
    return "tarteel" in blob


def is_hafs_riwayah(value: object | None) -> bool:
    if value is None or str(value).strip() == "":
        return True
    return "hafs" in str(value).lower()


def basmala_candidate(
    source: str,
    surah: int,
    ayah: int,
    row: dict | None = None,
) -> bool:
    if ayah == 1 and surah not in (1, 9):
        return True
    if source == "qua" and row:
        ctx = str(row.get("recording_context") or "").lower()
        if "basmala" in ctx:
            return True
    return False


def license_is_permissive(text: str) -> bool:
    t = text.lower()
    if any(m in t for m in _NONPERMISSIVE_MARKERS):
        return False
    return any(m in t for m in _PERMISSIVE_LICENSE_MARKERS)


def categorize_word_count(word_count: int) -> str:
    """Same thresholds as `benchmark/build_v3_corpus.py`."""
    if word_count <= 5:
        return "short"
    if word_count <= 15:
        return "medium"
    return "long"


def qlab_reciter(source: str) -> str:
    return QLAB_RECITER[source]


def qlab_flat_filename(source: str, file_name: str) -> str:
    return f"{source}__{Path(file_name).name}"


def tlog_holdout_key(name: str) -> str:
    """Stem used to match tlog training files against q-lab `tlog_holdout`."""
    base = Path(str(name).replace("\\", "/")).name.split("?")[0]
    if base.startswith("tlog_holdout__"):
        base = base[len("tlog_holdout__") :]
    return Path(base).stem


def is_nufais_holdout(**fields: object) -> bool:
    blob = " ".join(str(v).lower() for v in fields.values() if v not in (None, ""))
    return "nufais" in blob


def is_heldout_reciter(**fields: object) -> bool:
    """Any q-lab held-out reciter (3 EveryAyah + Al-Nufais), by name."""
    return match_heldout_reciter(*fields.values()) is not None


def qlab_exclusions_from_samples(samples: list[dict]) -> dict:
    """Build held-out exclusion sets from a q-lab-style samples list."""
    tlog_ids: set[str] = set()
    for s in samples:
        if (s.get("source") or "") != "tlog_holdout":
            continue
        key = tlog_holdout_key(s.get("file") or s.get("id") or "")
        if key:
            tlog_ids.add(key)
    return {"tlog_ids": tlog_ids}


def qlab_manifest_path() -> Path:
    here = Path(__file__).resolve().parent.parent
    for p in (
        Path("/app/benchmark/test_corpus_qlab/manifest.json"),
        here / "benchmark" / "test_corpus_qlab" / "manifest.json",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError("q-lab manifest.json not found (ship via add_local_file)")


def load_qlab_exclusions(path: str | Path | None = None) -> dict:
    manifest = Path(path) if path is not None else qlab_manifest_path()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    samples = data.get("samples") if isinstance(data, dict) else data
    excl = qlab_exclusions_from_samples(list(samples or []))
    print(f"[qlab] excluding {len(excl['tlog_ids'])} tlog_holdout ids from {manifest}")
    return excl


def make_clip_id(source: str, idx: int, surah: int, ayah: int, split: str | None = None) -> str:
    if split:
        return f"{source}_{split}_{idx:08d}_{surah}_{ayah}"
    return f"{source}_{idx:08d}_{surah}_{ayah}"


def flac_clip_path(source: str, clip_id: str, audio_root: Path | str | None = None) -> Path:
    root = Path(audio_root) if audio_root is not None else Path("/vol/audio")
    return root / source / f"{clip_id}.flac"


def existing_flac_duration(path: Path | str) -> float | None:
    """Duration in seconds if `path` is a non-empty FLAC; else None."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate or 16000)
        if sr <= 0:
            return None
        return float(len(f) / sr)


def progress_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_progress.json"


def partial_cuts_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_cuts.partial.jsonl"


def final_cuts_path(source: str, manifest_root: Path | str | None = None) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_cuts.jsonl.gz"


def empty_progress() -> dict:
    return {"rows_seen": 0, "hours_kept": 0.0, "clips_kept": 0, "split_rows": {}}


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write `text` via `<path>.tmp` + fsync + `os.replace` (no truncate-in-place)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_progress(source: str, manifest_root: Path | str | None = None) -> dict | None:
    """Return parsed progress, or None if missing/corrupt (never silently empty)."""
    p = progress_path(source, manifest_root)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[{source}] corrupt progress {p}; ignoring")
        return None
    if not isinstance(data, dict):
        print(f"[{source}] non-object progress {p}; ignoring")
        return None
    data.setdefault("rows_seen", 0)
    data.setdefault("hours_kept", 0.0)
    data.setdefault("clips_kept", 0)
    data.setdefault("split_rows", {})
    return data


def save_progress(source: str, payload: dict, manifest_root: Path | str | None = None) -> None:
    p = progress_path(source, manifest_root)
    merged = empty_progress()
    merged.update(payload)
    merged["rows_seen"] = int(merged.get("rows_seen") or 0)
    merged["hours_kept"] = float(merged.get("hours_kept") or 0.0)
    merged["clips_kept"] = int(merged.get("clips_kept") or 0)
    merged["split_rows"] = dict(merged.get("split_rows") or {})
    atomic_write_text(p, json.dumps(merged, indent=2) + "\n")


def append_cut_dict(path: Path | str, cut_dict: dict) -> None:
    """Append one lhotse cut-dict JSON line and flush (no lhotse required)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(cut_dict, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def hours_from_cut_dicts(cut_dicts: list[dict]) -> float:
    return sum(float(d.get("duration") or 0.0) for d in cut_dicts) / 3600.0


def load_cut_dicts(path: Path | str) -> list[dict]:
    """Load JSONL cut dicts. A torn *final* line is dropped and the file repaired.

    JSONDecodeError or UnicodeDecodeError (mid-codepoint crash on
    ensure_ascii=False Arabic) on the last non-empty line: drop, warn, truncate.
    The same errors on any earlier line still raise.
    """
    path = Path(path)
    if not path.is_file():
        return []
    data = path.read_bytes()
    records: list[tuple[int, bytes, bool]] = []
    i = 0
    while i < len(data):
        nl = data.find(b"\n", i)
        if nl == -1:
            records.append((i, data[i:], False))
            break
        records.append((i, data[i:nl], True))
        i = nl + 1
    nonempty = [k for k, (_, chunk, _) in enumerate(records) if chunk.strip()]
    out: list[dict] = []
    last_good_end = 0
    for k, (start, chunk, had_nl) in enumerate(records):
        if not chunk.strip():
            if had_nl:
                last_good_end = start + len(chunk) + 1
            continue
        try:
            out.append(json.loads(chunk.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_last = bool(nonempty) and k == nonempty[-1]
            if is_last:
                print(
                    f"[{path.name}] torn JSONL line at byte offset {start} "
                    f"({type(exc).__name__}); dropping last line and truncating to {last_good_end}"
                )
                with path.open("r+b") as f:
                    f.truncate(last_good_end)
                    f.flush()
                    os.fsync(f.fileno())
                return out
            raise
        last_good_end = start + len(chunk) + (1 if had_nl else 0)
    return out


def cut_id_of(cut_dict: dict) -> str:
    return str(cut_dict.get("id") or cut_dict.get("cut_id") or "")


def finalize_cut_dicts(cut_dicts: list[dict]) -> list[dict]:
    """Prefix+suffix in order; first id wins (no duplicates)."""
    seen: set[str] = set()
    merged: list[dict] = []
    for d in cut_dicts:
        cid = cut_id_of(d)
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        merged.append(d)
    return merged


def remove_partial_cuts(source: str, manifest_root: Path | str | None = None) -> None:
    p = partial_cuts_path(source, manifest_root)
    if p.is_file():
        p.unlink()


def consume_row(state: dict, split: str | None = None) -> int:
    """Mark one HF row as seen. Always increment, including skipped rows.

    Returns the clip-id idx for this row: per-`split` count when `split` is
    set (everyayah split / QUA mushaf slug), else global `rows_seen`.
    """
    if split is not None:
        sr = state.setdefault("split_rows", {})
        key = str(split)
        idx = int(sr.get(key) or 0)
        sr[key] = idx + 1
        state["rows_seen"] = int(state.get("rows_seen") or 0) + 1
        return idx
    idx = int(state.get("rows_seen") or 0)
    state["rows_seen"] = idx + 1
    return idx


def persist_progress(source: str, state: dict, manifest_root: Path | str | None = None) -> None:
    extra = {
        k: state[k]
        for k in state
        if k not in {"cut_dicts", "rows_seen", "hours_kept", "clips_kept", "split_rows"}
    }
    payload = {
        "rows_seen": int(state.get("rows_seen") or 0),
        "hours_kept": float(state.get("hours_kept") or 0.0),
        "clips_kept": int(state.get("clips_kept") or 0),
        "split_rows": dict(state.get("split_rows") or {}),
    }
    payload.update(extra)
    save_progress(source, payload, manifest_root)


def restore_partial_state(
    source: str,
    manifest_root: Path | str | None = None,
    force: bool = False,
) -> dict:
    """Load partial JSONL + progress, or wipe both (and the final cuts) if force.

    `hours_kept` is always recomputed from cut durations (never summed on top of
    a stale progress value). Missing/corrupt progress with leftover cuts falls
    back to `rows_seen = clips_kept = len(cuts)`.

    `--force` also deletes `*_cuts_fbank.jsonl.gz`, every
    `*_cuts_fbank.shard-*.jsonl.gz`, and `<vol>/fbank_sharded/<source>/`
    (`vol` = `manifest_root.parent`, so default `/vol/manifests` →
    `/vol/fbank_sharded/<source>/`).
    """
    man = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    partial = partial_cuts_path(source, man)
    prog = progress_path(source, man)
    final = final_cuts_path(source, man)
    if force:
        for p in (
            partial,
            prog,
            Path(str(prog) + ".tmp"),
            final,
            man / f"{source}_stats.json",
            man / f"{source}_cuts_fbank.jsonl.gz",
        ):
            if p.is_file():
                p.unlink()
        for p in man.glob(f"{source}_cuts_fbank.shard-*.jsonl.gz"):
            if p.is_file():
                p.unlink()
        sharded = man.parent / "fbank_sharded" / source
        if sharded.is_dir():
            shutil.rmtree(sharded)
        state = empty_progress()
        state["cut_dicts"] = []
        return state
    cut_dicts = load_cut_dicts(partial)
    progress = load_progress(source, man)
    n = len(cut_dicts)
    hours = hours_from_cut_dicts(cut_dicts)
    state = empty_progress()
    if progress is None:
        if n:
            print(
                f"[{source}] progress missing/corrupt with {n} partial cuts; "
                f"rows_seen fallback to clips_kept={n} "
                f"(skip may be incomplete; duplicates de-duped by id)"
            )
        state["cut_dicts"] = cut_dicts
        state["rows_seen"] = n
        state["clips_kept"] = n
        state["hours_kept"] = hours
        return state
    state.update(progress)
    state["cut_dicts"] = cut_dicts
    rows_seen = int(progress.get("rows_seen") or 0)
    if n and rows_seen < n:
        print(
            f"[{source}] progress rows_seen={rows_seen} < {n} cuts; "
            f"raising rows_seen to {n} (skip may be incomplete)"
        )
        rows_seen = n
    state["rows_seen"] = rows_seen
    state["hours_kept"] = hours
    state["clips_kept"] = n
    state["split_rows"] = dict(progress.get("split_rows") or {})
    return state


def skip_hf_stream(ds, n: int, label: str = ""):
    """Advance a streaming HF dataset by n rows when `.skip` exists."""
    if n <= 0:
        return ds
    skip = getattr(ds, "skip", None)
    if callable(skip):
        print(f"[{label or 'stream'}] checkpoint skip({n})")
        return skip(n)
    print(f"[{label or 'stream'}] no .skip(); scanning from 0, reusing existing FLACs")
    return ds


def parse_sources(csv: str) -> list[str]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    if not parts:
        return list(DEFAULT_SOURCES)
    unknown = [p for p in parts if p not in ALL_SOURCES]
    if unknown:
        raise ValueError(f"unknown sources {unknown}; expected subset of {ALL_SOURCES}")
    return parts


def shard_items(items: list, shard: int, n_shards: int) -> list:
    """Every n-th item by index. Exact partition: union == full list, no overlap."""
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if shard < 0 or shard >= n_shards:
        raise ValueError(f"shard must be in [0, {n_shards}), got {shard}")
    return [item for i, item in enumerate(items) if i % n_shards == shard]


def merge_shard_items(shards: list) -> list:
    """Concatenate shard sequences in shard-index order (0, 1, …)."""
    out = []
    for part in shards:
        out.extend(part)
    return out


def fbank_shard_storage_path(
    source: str, shard: int, root: Path | str | None = None
) -> Path:
    base = Path(root) if root is not None else Path("/vol/fbank_sharded")
    return base / source / f"shard-{shard}"


def fbank_shard_manifest_path(
    source: str, shard: int, manifest_root: Path | str | None = None
) -> Path:
    root = Path(manifest_root) if manifest_root is not None else Path("/vol/manifests")
    return root / f"{source}_cuts_fbank.shard-{shard}.jsonl.gz"


def expected_fbank_cut_count(n_raw: int, no_speed_perturb: bool) -> int:
    return int(n_raw) if no_speed_perturb else int(n_raw) * 3


def consecutive_ayah_spans(
    ayahs: list[int],
    min_len: int = MULTI_MIN_AYAHS,
    max_len: int = MULTI_MAX_AYAHS,
) -> list[tuple[int, int]]:
    """Inclusive (start, end) spans of consecutive ayahs, lengths min_len..max_len.

    Gaps break runs. Duplicate ayah numbers are collapsed. Empty if no run is
    long enough. Order is start-ascending, then length-ascending within a start.
    """
    uniq = sorted({int(a) for a in ayahs})
    if not uniq:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = uniq[0]
    for a in uniq[1:]:
        if a == prev + 1:
            prev = a
        else:
            segments.append((start, prev))
            start = prev = a
    segments.append((start, prev))
    spans: list[tuple[int, int]] = []
    for seg_s, seg_e in segments:
        length = seg_e - seg_s + 1
        for L in range(min_len, min(max_len, length) + 1):
            for i in range(length - L + 1):
                a0 = seg_s + i
                spans.append((a0, a0 + L - 1))
    return spans


def cut_dict_window_record(cut_dict: dict) -> dict | None:
    """Pull speaker/surah/ayah/duration/id from a lhotse cut dict. None if unusable.

    Already-span cuts (``ayah_end != ayah``) are skipped so grouping stays
    single-ayah EveryAyah rows.
    """
    cid = cut_id_of(cut_dict)
    if not cid:
        return None
    duration = float(cut_dict.get("duration") or 0.0)
    if duration <= 0:
        return None
    sups = cut_dict.get("supervisions") or []
    if not sups or not isinstance(sups[0], dict):
        return None
    sup = sups[0]
    custom = sup.get("custom") if isinstance(sup.get("custom"), dict) else {}
    speaker = str(sup.get("speaker") or "unknown")
    try:
        surah = int(custom["surah"])
        ayah = int(custom["ayah"])
    except (KeyError, TypeError, ValueError):
        return None
    ayah_end = int(custom["ayah_end"]) if custom.get("ayah_end") is not None else ayah
    if ayah_end != ayah:
        return None
    return {
        "id": cid,
        "speaker": speaker,
        "surah": surah,
        "ayah": ayah,
        "duration": duration,
        "text": str(sup.get("text") or ""),
    }


def group_window_records(
    records: list[dict],
) -> dict[tuple[str, int], dict[int, dict]]:
    """``(speaker, surah) -> {ayah: record}``. First record wins on duplicate ayah."""
    groups: dict[tuple[str, int], dict[int, dict]] = {}
    for rec in records:
        key = (str(rec["speaker"]), int(rec["surah"]))
        ayah = int(rec["ayah"])
        bucket = groups.setdefault(key, {})
        if ayah not in bucket:
            bucket[ayah] = rec
    return groups


class MultiAyahWindowPick:
    """One candidate 2–4 ayah run. Plain class so tests can exec_module this file."""

    __slots__ = (
        "speaker",
        "surah",
        "ayah",
        "ayah_end",
        "cut_ids",
        "durations",
        "n_ayahs",
    )

    def __init__(
        self,
        speaker: str,
        surah: int,
        ayah: int,
        ayah_end: int,
        cut_ids: tuple[str, ...],
        durations: tuple[float, ...],
        n_ayahs: int,
    ):
        self.speaker = speaker
        self.surah = int(surah)
        self.ayah = int(ayah)
        self.ayah_end = int(ayah_end)
        self.cut_ids = tuple(cut_ids)
        self.durations = tuple(float(d) for d in durations)
        self.n_ayahs = int(n_ayahs)

    def max_duration_s(self, gap_max_s: float = MULTI_GAP_MAX_S) -> float:
        n_gaps = max(self.n_ayahs - 1, 0)
        return float(sum(self.durations)) + n_gaps * float(gap_max_s)


def candidate_multi_ayah_windows(
    groups: dict[tuple[str, int], dict[int, dict]],
    *,
    min_len: int = MULTI_MIN_AYAHS,
    max_len: int = MULTI_MAX_AYAHS,
    max_duration_s: float = MULTI_MAX_DURATION_S,
    gap_max_s: float = MULTI_GAP_MAX_S,
) -> list[MultiAyahWindowPick]:
    """All 2–4 consecutive-ayah windows whose padded duration cannot exceed the cap."""
    picks: list[MultiAyahWindowPick] = []
    for speaker, surah in sorted(groups):
        by_ayah = groups[(speaker, surah)]
        ayahs = sorted(by_ayah)
        for a0, a1 in consecutive_ayah_spans(ayahs, min_len, max_len):
            recs = [by_ayah[a] for a in range(a0, a1 + 1)]
            durs = tuple(float(r["duration"]) for r in recs)
            n = a1 - a0 + 1
            pick = MultiAyahWindowPick(
                speaker=speaker,
                surah=surah,
                ayah=a0,
                ayah_end=a1,
                cut_ids=tuple(str(r["id"]) for r in recs),
                durations=durs,
                n_ayahs=n,
            )
            if pick.max_duration_s(gap_max_s) > max_duration_s:
                continue
            picks.append(pick)
    return picks


def allocate_stratified_counts(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` slots, proportional to ``sizes``.

    Keys in sorted order. Never more than available. If ``n`` exceeds the
    total, returns a copy of ``sizes``.
    """
    keys = sorted(sizes)
    total = sum(int(sizes[k]) for k in keys)
    if not keys or n <= 0 or total <= 0:
        return {k: 0 for k in keys}
    if n >= total:
        return {k: int(sizes[k]) for k in keys}
    raw = {k: n * int(sizes[k]) / total for k in keys}
    alloc = {k: min(int(raw[k]), int(sizes[k])) for k in keys}
    assigned = sum(alloc.values())
    remainders = sorted((-(raw[k] - int(raw[k])), k) for k in keys)
    for _, k in remainders:
        if assigned >= n:
            break
        if alloc[k] < int(sizes[k]):
            alloc[k] += 1
            assigned += 1
    if assigned < n:
        for k in keys:
            while assigned < n and alloc[k] < int(sizes[k]):
                alloc[k] += 1
                assigned += 1
    return alloc


def select_windows_stratified(
    candidates: list[MultiAyahWindowPick],
    n_windows: int,
    seed: int,
) -> list[MultiAyahWindowPick]:
    """Sample ``n_windows`` runs, stratified by reciter, seed-deterministic."""
    rng = random.Random(int(seed))
    by_speaker: dict[str, list[MultiAyahWindowPick]] = {}
    for w in candidates:
        by_speaker.setdefault(w.speaker, []).append(w)
    for sp, items in by_speaker.items():
        items.sort(key=lambda w: (w.surah, w.ayah, w.ayah_end, w.cut_ids))
        rng.shuffle(items)
        by_speaker[sp] = items
    alloc = allocate_stratified_counts(
        {sp: len(items) for sp, items in by_speaker.items()},
        int(n_windows),
    )
    picked: list[MultiAyahWindowPick] = []
    for sp in sorted(by_speaker):
        picked.extend(by_speaker[sp][: alloc.get(sp, 0)])
    rng.shuffle(picked)
    return picked


def sample_window_gaps(
    n_ayahs: int,
    rng: random.Random,
    gap_max_s: float = MULTI_GAP_MAX_S,
) -> tuple[float, ...]:
    n_gaps = max(int(n_ayahs) - 1, 0)
    return tuple(rng.uniform(0.0, float(gap_max_s)) for _ in range(n_gaps))


def multi_ayah_window_text(corpus, surah: int, ayah: int, ayah_end: int) -> str:
    """Phoneme string for ``[ayah, ayah_end]`` with no inserted spaces."""
    text = corpus.span_phonemes(int(surah), int(ayah), int(ayah_end))
    if " " in text:
        raise ValueError(
            f"span_phonemes inserted spaces for {surah}:{ayah}-{ayah_end}"
        )
    return text


def n_ayahs_histogram(picks: list[MultiAyahWindowPick]) -> dict[str, int]:
    hist = {str(k): 0 for k in range(MULTI_MIN_AYAHS, MULTI_MAX_AYAHS + 1)}
    for w in picks:
        key = str(w.n_ayahs)
        hist[key] = int(hist.get(key) or 0) + 1
    return hist


def supervision_covers_window(
    cut_duration: float, sup_start: float, sup_duration: float, tol: float = 0.05
) -> bool:
    """Icefall ASR wants one supervision spanning the whole cut.

    MixedCut.supervisions is derived from tracks; attaching the window text
    on the first ayah track leaves ``sup.duration`` equal to ayah 1 only.
    """
    return abs(float(sup_start)) <= tol and abs(
        float(sup_duration) - float(cut_duration)
    ) <= tol


def whole_cut_supervision_fields(
    *,
    cut_id: str,
    duration: float,
    recording_id: str,
    text: str,
    speaker: str,
    custom: dict,
) -> dict:
    return {
        "id": str(cut_id),
        "recording_id": str(recording_id),
        "start": 0.0,
        "duration": float(duration),
        "text": text,
        "speaker": speaker,
        "custom": dict(custom),
    }


def pick_surah_ayah(row: dict) -> tuple[int, int] | None:
    for s_key, a_key in _SURA_AYAH_KEYS:
        if s_key in row and a_key in row and row[s_key] is not None and row[a_key] is not None:
            try:
                return int(row[s_key]), int(row[a_key])
            except (TypeError, ValueError):
                return None
    return None


def load_token_inventory(path: str | Path) -> list[str]:
    """Load icefall `tokens.txt` (`<sym> <id>` per line)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    by_id: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sym, sep, idx_s = line.rpartition(" ")
        if not sep:
            raise ValueError(f"bad tokens.txt line: {line!r}")
        by_id[int(idx_s)] = sym
    if not by_id:
        raise ValueError(f"no tokens in {path}")
    n = max(by_id) + 1
    missing = [i for i in range(n) if i not in by_id]
    if missing:
        raise ValueError(f"token id gaps in {path}: {missing[:8]}")
    return [by_id[i] for i in range(n)]


def local_quran_json() -> Path:
    return resolve_data_file("quran.json")


# ---------------------------------------------------------------------------
# Modal image / volume
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ZIPFORMER_QURAN = resolve_data_file("zipformer/quran.json")
_QURAN_JSON = local_quran_json()
_TOKENS_TXT = PROJECT_ROOT / "experiments" / "zipformer-ctc" / "tokens.txt"
_SHARED = PROJECT_ROOT / "shared"

app = modal.App("zipformer-ctc-data")
vol = modal.Volume.from_name("zipformer-ctc-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "lhotse",
        "datasets>=4.0,<5.0",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
        "soundfile",
        "numpy",
        "python-Levenshtein",
    )
    .pip_install(
        "torch",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("lilcom", "kaldi-native-fbank")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(str(_SHARED / "paths.py"), remote_path="/app/shared/paths.py")
    .add_local_file(str(_SHARED / "phoneme_labels.py"), remote_path="/app/shared/phoneme_labels.py")
    .add_local_file(str(_SHARED / "normalizer.py"), remote_path="/app/shared/normalizer.py")
    .add_local_file(str(_SHARED / "quran_db.py"), remote_path="/app/shared/quran_db.py")
    .add_local_file(str(_SHARED / "fbank.py"), remote_path="/app/shared/fbank.py")
    .add_local_file(str(_SHARED / "leak_guard.py"), remote_path="/app/shared/leak_guard.py")
    .add_local_file(str(_ZIPFORMER_QURAN), remote_path="/app/data/zipformer/quran.json")
    .add_local_file(str(_QURAN_JSON), remote_path="/app/data/quran.json")
    .add_local_file(str(_TOKENS_TXT), remote_path="/app/tokens.txt")
    .add_local_file(
        str(PROJECT_ROOT / "benchmark" / "test_corpus_qlab" / "manifest.json"),
        remote_path="/app/benchmark/test_corpus_qlab/manifest.json",
    )
)

# Full EveryAyah/QUA ingest is many hours; 24h is Modal's typical max.
# 32 cores match fbank num_jobs. Resume without --force if a source times out.
FBANK_NUM_JOBS = 32
_FN_KW = dict(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface")],
    cpu=32,
    memory=65536,
    timeout=24 * 3600,
)


def _bump_skip(stats: dict, reason: str) -> None:
    skipped = stats.setdefault("skipped", {})
    skipped[reason] = int(skipped.get(reason, 0)) + 1


def _empty_stats(source: str) -> dict:
    return {
        "source": source,
        "clips": 0,
        "hours": 0.0,
        "oov": 0,
        "skipped": {},
        "license_ok": True,
        "features": None,
        "hf_repo": None,
        "split": None,
        "reused_flac": 0,
    }


def patch_hf_list_feature() -> None:
    """Alias datasets-4 `List` so older `datasets` 3.x can read parquet metadata.

    Some QUA mushaf configs (first seen: `ahmed_amer_tvquran`) embed
    `{"_type": "List", ...}` in Arrow schema metadata. No-op on datasets 4+
    where `List` is already registered. Staging image now pins datasets 4.x
    (Audio still `decode=False` so torchcodec is not required).
    """
    from datasets.features import features as feat_mod

    types = feat_mod._FEATURE_TYPES
    if "List" in types:
        return
    seq = types.get("Sequence")
    if seq is None:
        raise RuntimeError("datasets has neither List nor Sequence feature type")
    types["List"] = seq


def _boot_remote() -> None:
    sys.path.insert(0, "/app")
    os.environ.setdefault("HF_HOME", "/vol/hf_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    Path("/vol/hf_cache").mkdir(parents=True, exist_ok=True)
    Path("/vol/audio").mkdir(parents=True, exist_ok=True)
    Path("/vol/manifests").mkdir(parents=True, exist_ok=True)
    Path("/vol/licenses").mkdir(parents=True, exist_ok=True)
    Path("/vol/fbank").mkdir(parents=True, exist_ok=True)
    Path("/vol/fbank_sharded").mkdir(parents=True, exist_ok=True)
    patch_hf_list_feature()


def _print_features(source: str, repo: str, features, splits) -> None:
    print(f"[{source}] repo={repo}")
    print(f"[{source}] features: {features}")
    print(f"[{source}] splits: {splits}")


def _load_labelers():
    from shared.phoneme_labels import OOVError, PhonemeCorpus, PhonemeTokenizer
    from shared.quran_db import QuranDB

    corpus = PhonemeCorpus("/app/data/zipformer/quran.json")
    tokenizer = PhonemeTokenizer(load_token_inventory("/app/tokens.txt"))
    db = QuranDB(Path("/app/data/quran.json"))
    return corpus, tokenizer, db, OOVError


def _stream_ds(repo: str, split: str, name: str | None = None, audio_col: str = "audio"):
    """Streaming HF split with Audio(decode=False) so we never need torchcodec."""
    from datasets import Audio, load_dataset

    patch_hf_list_feature()

    # `split="all"` is a reserved keyword in datasets>=3 and cannot be passed
    # as the split= argument even when the config advertises a split named all.
    if split == "all":
        dd = (
            load_dataset(repo, name, streaming=True)
            if name
            else load_dataset(repo, streaming=True)
        )
        ds = dd["all"] if hasattr(dd, "keys") and "all" in dd else dd
    else:
        ds = (
            load_dataset(repo, name, split=split, streaming=True)
            if name
            else load_dataset(repo, split=split, streaming=True)
        )
    feats = getattr(ds, "features", None) or {}
    if audio_col in feats:
        ds = ds.cast_column(audio_col, Audio(sampling_rate=16000, decode=False))
    return ds


def _ffmpeg_pcm16k(src: bytes | str):
    import subprocess
    import numpy as np

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        "pipe:0" if isinstance(src, (bytes, bytearray)) else str(src),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    if isinstance(src, (bytes, bytearray)):
        proc = subprocess.run(cmd, input=src, capture_output=True, check=True)
    else:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wav.size == 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise ValueError(f"ffmpeg empty audio: {err[:200]}")
    return wav, float(wav.shape[0] / 16000.0)


def _audio_to_16k_mono(audio_obj) -> tuple["object", float]:
    import io

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    def _resample(arr, sr: int):
        wav = np.asarray(arr, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1 if wav.shape[-1] <= 8 else 0)
        wav = np.ascontiguousarray(wav.reshape(-1))
        if int(sr) != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), int(sr), 16000
            ).numpy()
        return wav, float(wav.shape[0] / 16000.0)

    if audio_obj is None:
        raise ValueError("missing audio")
    if isinstance(audio_obj, dict):
        if audio_obj.get("array") is not None:
            return _resample(audio_obj["array"], int(audio_obj.get("sampling_rate") or 16000))
        blob = audio_obj.get("bytes")
        path = audio_obj.get("path")
        if blob is not None:
            raw = bytes(blob)
            try:
                arr, sr = sf.read(io.BytesIO(raw), always_2d=False)
                return _resample(arr, sr)
            except Exception:
                return _ffmpeg_pcm16k(raw)
        if path:
            try:
                arr, sr = sf.read(str(path), always_2d=False)
                return _resample(arr, sr)
            except Exception:
                return _ffmpeg_pcm16k(str(path))
        raise ValueError("audio dict has no array/bytes/path")
    return _resample(audio_obj, 16000)


def _wav_from_file(path: str | Path) -> tuple["object", float]:
    import soundfile as sf

    arr, sr = sf.read(str(path), always_2d=False)
    return _audio_to_16k_mono({"array": arr, "sampling_rate": sr})


def _write_flac(path: Path, wav) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, 16000, format="FLAC")


def _match_ayah(db, text: str):
    if not text or not str(text).strip():
        return None
    hit = db.match_verse(str(text))
    if hit is None or float(hit.get("score") or 0) < MATCH_MIN_SCORE:
        return None
    return hit


def iqra_match_scores(db, sentence: str, tashkeel: str) -> dict:
    """Compare match_verse / search / hamza-stripped scores for one Iqra row."""
    from shared.normalizer import normalize_arabic

    out: dict[str, float] = {}
    for label, raw in (("sentence", sentence or ""), ("tashkeel", tashkeel or "")):
        raw = str(raw)
        if not raw.strip():
            out[f"{label}_match"] = 0.0
            out[f"{label}_search"] = 0.0
            out[f"{label}_hamza"] = 0.0
            continue
        hit = db.match_verse(raw)
        out[f"{label}_match"] = float(hit["score"]) if hit else 0.0
        hits = db.search(raw, top_k=1)
        out[f"{label}_search"] = float(hits[0]["score"]) if hits else 0.0
        hit_h = db.match_verse(normalize_arabic(raw, strip_hamza=True))
        out[f"{label}_hamza"] = float(hit_h["score"]) if hit_h else 0.0
    return out


def iqra_row_keep_flags(scores: dict, threshold: float = MATCH_MIN_SCORE) -> dict:
    """Keep-rate flags for one Iqra score dict from `iqra_match_scores`."""
    baseline = max(float(scores.get("sentence_match") or 0), float(scores.get("tashkeel_match") or 0))
    search = max(float(scores.get("sentence_search") or 0), float(scores.get("tashkeel_search") or 0))
    hamza = max(float(scores.get("sentence_hamza") or 0), float(scores.get("tashkeel_hamza") or 0))
    return {
        "baseline": baseline >= threshold,
        "search": search >= threshold,
        "hamza": hamza >= threshold,
        "baseline_score": baseline,
        "search_score": search,
        "hamza_score": hamza,
    }


def match_iqra_row(db, sentence: str, tashkeel: str):
    """Primary: match_verse on tashkeel then sentence. Keep ≥ 0.95 unless diag says otherwise."""
    return _match_ayah(db, tashkeel or "") or _match_ayah(db, sentence or "")


def _phoneme_text(corpus, tokenizer, OOVError, surah: int, ayah: int, ayah_end: int | None):
    end = ayah_end if ayah_end is not None else ayah
    phonemes = corpus.span_phonemes(surah, ayah, end)
    tokenizer.encode(phonemes)
    return phonemes


def _save_cuts(source: str, cut_dicts: list, manifest_root: Path | str | None = None) -> None:
    from lhotse import CutSet

    merged = finalize_cut_dicts(list(cut_dicts or []))
    cuts = CutSet.from_dicts(merged)
    out = final_cuts_path(source, manifest_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(str(out))
    remove_partial_cuts(source, manifest_root)
    print(f"[{source}] wrote {out} ({len(cuts)} cuts)")


def _write_stats(stats: dict) -> None:
    path = Path(f"/vol/manifests/{stats['source']}_stats.json")
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[{stats['source']}] stats: {json.dumps(stats, default=str)}")


def _sync_stats(stats: dict, state: dict) -> None:
    stats["clips"] = int(state.get("clips_kept") or 0)
    stats["hours"] = float(state.get("hours_kept") or 0.0)


def _begin_or_skip(source: str, force: bool, stats: dict) -> tuple[dict | None, dict | None]:
    """Return (state, None) to ingest, or (None, skip_stats) if already complete.

    A leftover `*_cuts.partial.jsonl` means a crash mid-run: resume even if a
    stale final gzip exists. `--force` deletes partial + progress + final + audio
    + fbank dir + `*_cuts_fbank.jsonl.gz` + `*_cuts_fbank.shard-*.jsonl.gz` +
    `/vol/fbank_sharded/<source>/` (smoke leftovers must not skip fbank).
    """
    if force:
        state = restore_partial_state(source, force=True)
        audio_dir = Path("/vol/audio") / source
        if audio_dir.is_dir():
            shutil.rmtree(audio_dir)
        fbank_dir = Path("/vol/fbank") / source
        if fbank_dir.is_dir():
            shutil.rmtree(fbank_dir)
        sharded_dir = Path("/vol/fbank_sharded") / source
        if sharded_dir.is_dir():
            shutil.rmtree(sharded_dir)
        vol.commit()
        print(
            f"[{source}] --force: cleared partial, progress, final cuts, "
            f"fbank gzip, shard gzip, {audio_dir}, {fbank_dir}, and {sharded_dir}"
        )
        return state, None
    partial = partial_cuts_path(source)
    final = final_cuts_path(source)
    if final.is_file() and not partial.is_file():
        print(f"[{source}] {final} exists; skip (pass --force to redo)")
        existing = Path(f"/vol/manifests/{source}_stats.json")
        if existing.is_file():
            try:
                loaded = json.loads(existing.read_text(encoding="utf-8"))
                loaded["skipped_existing"] = True
                _write_stats(loaded)
                return None, loaded
            except json.JSONDecodeError:
                pass
        stats["skipped_existing"] = True
        _write_stats(stats)
        return None, stats
    state = restore_partial_state(source, force=False)
    n = len(state["cut_dicts"])
    if n:
        _sync_stats(stats, state)
        stats["reused_flac"] = n
        stats["prefix_restored"] = n
        print(
            f"[{source}] resume prefix_cuts={n} rows_seen={state['rows_seen']} "
            f"hours_kept={state['hours_kept']:.4f} clips_kept={state['clips_kept']}"
        )
    return state, None


def _commit(every_n: int, n: int) -> None:
    if n > 0 and n % every_n == 0:
        vol.commit()
        print(f"  volume commit at {n} clips")


def _maybe_crash_after(crash_after: int, limit: int, clips_kept: int) -> None:
    """Hidden test knob: raise after N kept clips when `--limit` is also set."""
    if crash_after > 0 and limit > 0 and int(clips_kept) >= crash_after:
        vol.commit()
        raise RuntimeError(f"crash-after {crash_after} (kept {clips_kept} clips)")


def _finish_row(
    source: str,
    state: dict,
    stats: dict,
    crash_after: int,
    limit: int,
    *,
    kept: bool,
    commit_every: int = 100,
    manifest_root: Path | str | None = None,
) -> None:
    persist_progress(source, state, manifest_root)
    _sync_stats(stats, state)
    if kept:
        _maybe_crash_after(crash_after, limit, int(state.get("clips_kept") or 0))
        _commit(commit_every, int(state.get("clips_kept") or 0))


def _finalize_source(source: str, state: dict, stats: dict) -> dict:
    _sync_stats(stats, state)
    n_prefix = int(stats.get("prefix_restored") or 0)
    n_reused = int(stats.get("reused_flac") or 0)
    if state.get("cut_dicts"):
        _save_cuts(source, state["cut_dicts"])
        print(
            f"[{source}] finalize cuts={stats['clips']} "
            f"prefix_restored={n_prefix} reused_flac={n_reused} "
            f"new_flac={max(int(stats['clips']) - n_reused, 0)}"
        )
    _write_stats(stats)
    vol.commit()
    return stats


def _append_cut(state, source, *, clip_id, flac_path, phonemes, speaker, custom, duration: float, manifest_root=None):
    """Write one MonoCut.to_dict() line immediately, then bump hours/clips."""
    from lhotse import MonoCut, Recording, SupervisionSegment

    rec = Recording.from_file(str(flac_path), recording_id=clip_id)
    cut = MonoCut(
        id=clip_id,
        start=0.0,
        duration=rec.duration,
        channel=0,
        recording=rec,
        supervisions=[
            SupervisionSegment(
                id=clip_id,
                recording_id=clip_id,
                start=0.0,
                duration=rec.duration,
                channel=0,
                text=phonemes,
                language="quran-phonemes",
                speaker=speaker or "unknown",
                custom=custom,
            )
        ],
    )
    cut_dict = cut.to_dict()
    append_cut_dict(partial_cuts_path(source, manifest_root), cut_dict)
    state.setdefault("cut_dicts", []).append(cut_dict)
    state["clips_kept"] = int(state.get("clips_kept") or 0) + 1
    state["hours_kept"] = float(state.get("hours_kept") or 0.0) + float(duration) / 3600.0


def _ingest_clip(
    *,
    source: str,
    idx: int,
    wav,
    duration: float,
    surah: int,
    ayah: int,
    ayah_end: int | None,
    speaker: str,
    condition: str,
    stats: dict,
    corpus,
    tokenizer,
    OOVError,
    state: dict,
    extra_custom: dict | None = None,
    split: str | None = None,
    audio_root: Path | str | None = None,
    manifest_root: Path | str | None = None,
) -> bool:
    decision = duration_decision(duration)
    if decision.startswith("skip_"):
        _bump_skip(stats, decision)
        return False
    try:
        phonemes = _phoneme_text(corpus, tokenizer, OOVError, surah, ayah, ayah_end)
    except OOVError:
        stats["oov"] += 1
        _bump_skip(stats, "oov")
        return False
    except ValueError:
        _bump_skip(stats, "missing_ayah")
        return False
    clip_id = make_clip_id(source, idx, surah, ayah, split=split)
    flac_path = flac_clip_path(source, clip_id, audio_root=audio_root)
    if wav is not None:
        _write_flac(flac_path, wav)
    elif existing_flac_duration(flac_path) is None:
        _bump_skip(stats, "missing_flac")
        return False
    else:
        stats["reused_flac"] = int(stats.get("reused_flac") or 0) + 1
    custom = {
        "surah": surah,
        "ayah": ayah,
        "ayah_end": ayah_end if ayah_end is not None else ayah,
        "source": source,
        "basmala_candidate": basmala_candidate(source, surah, ayah, extra_custom),
        "condition": condition,
        "long": decision == "long",
    }
    if extra_custom:
        custom.update(extra_custom)
    _append_cut(
        state,
        source,
        clip_id=clip_id,
        flac_path=flac_path,
        phonemes=phonemes,
        speaker=str(speaker or "unknown"),
        custom=custom,
        duration=duration,
        manifest_root=manifest_root,
    )
    _sync_stats(stats, state)
    return True


def _audio_for_clip(
    *,
    source: str,
    idx: int,
    surah: int,
    ayah: int,
    audio_obj,
    force: bool,
    stats: dict,
    split: str | None = None,
    audio_root: Path | str | None = None,
):
    """Decode audio, or reuse an existing non-empty FLAC (crash resume)."""
    clip_id = make_clip_id(source, idx, surah, ayah, split=split)
    flac_path = flac_clip_path(source, clip_id, audio_root=audio_root)
    if not force:
        dur = existing_flac_duration(flac_path)
        if dur is not None:
            return None, dur
    try:
        return _audio_to_16k_mono(audio_obj)
    except Exception:
        _bump_skip(stats, "audio_error")
        return None, None


# ---------------------------------------------------------------------------
# Per-source prepare functions
# ---------------------------------------------------------------------------


def _prepare_everyayah(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("everyayah")
    print(
        f"[everyayah] splits={list(EVERYAYAH_SPLITS)} "
        "(never test: q-lab everyayah_heldout is curated test shards)"
    )
    state, skipped = _begin_or_skip("everyayah", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()

    curated_ok = False
    try:
        builder = load_dataset_builder(HF_EVERYAYAH_CURATED)
        feats = builder.info.features or {}
        names = set(feats.keys())
        _print_features("everyayah", HF_EVERYAYAH_CURATED, feats, builder.info.splits)
        curated_ok = bool({"sura", "ayah"} <= names or {"chapter", "verse"} <= names)
    except Exception as e:
        print(f"[everyayah] curated builder failed: {type(e).__name__}: {e}")

    if curated_ok:
        stats["hf_repo"] = HF_EVERYAYAH_CURATED
        stats["split"] = "+".join(EVERYAYAH_SPLITS)
        stats["features"] = str(feats)
        for split in EVERYAYAH_SPLITS:
            if limit and state["clips_kept"] >= limit:
                break
            ds = _stream_ds(HF_EVERYAYAH_CURATED, split)
            already = int((state.get("split_rows") or {}).get(split) or 0)
            ds = skip_hf_stream(ds, already, f"everyayah/{split}")
            for row in ds:
                if limit and state["clips_kept"] >= limit:
                    break
                idx = consume_row(state, split=split)
                sa = pick_surah_ayah(row)
                if sa is None:
                    _bump_skip(stats, "no_surah_ayah")
                    _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                    continue
                surah, ayah = sa
                if is_heldout_reciter(qari=row.get("qari"), reciter=row.get("reciter")):
                    _bump_skip(stats, "heldout_reciter")
                    _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                    continue
                wav, dur = _audio_for_clip(
                    source="everyayah",
                    idx=idx,
                    surah=surah,
                    ayah=ayah,
                    audio_obj=row.get("audio"),
                    force=force,
                    stats=stats,
                    split=split,
                )
                if dur is None:
                    _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                    continue
                kept = _ingest_clip(
                    source="everyayah",
                    idx=idx,
                    wav=wav,
                    duration=dur,
                    surah=surah,
                    ayah=ayah,
                    ayah_end=ayah,
                    speaker=str(row.get("qari") or row.get("reciter") or "everyayah"),
                    condition="studio",
                    stats=stats,
                    corpus=corpus,
                    tokenizer=tokenizer,
                    OOVError=OOVError,
                    state=state,
                    split=split,
                )
                _finish_row("everyayah", state, stats, crash_after, limit, kept=kept)
    else:
        builder = load_dataset_builder(HF_EVERYAYAH)
        _print_features("everyayah", HF_EVERYAYAH, builder.info.features, builder.info.splits)
        stats["hf_repo"] = HF_EVERYAYAH
        stats["split"] = "train"
        stats["features"] = str(builder.info.features)
        already = int(state.get("rows_seen") or 0)
        ds = skip_hf_stream(_stream_ds(HF_EVERYAYAH, "train"), already, "everyayah")
        for row in ds:
            if limit and state["clips_kept"] >= limit:
                break
            idx = consume_row(state)
            hit = _match_ayah(db, row.get("text") or "")
            if hit is None:
                _bump_skip(stats, "low_match")
                _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                continue
            if is_heldout_reciter(reciter=row.get("reciter")):
                _bump_skip(stats, "heldout_reciter")
                _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                continue
            wav, dur = _audio_for_clip(
                source="everyayah",
                idx=idx,
                surah=int(hit["surah"]),
                ayah=int(hit["ayah"]),
                audio_obj=row.get("audio"),
                force=force,
                stats=stats,
            )
            if dur is None:
                _finish_row("everyayah", state, stats, crash_after, limit, kept=False)
                continue
            kept = _ingest_clip(
                source="everyayah",
                idx=idx,
                wav=wav,
                duration=dur,
                surah=int(hit["surah"]),
                ayah=int(hit["ayah"]),
                ayah_end=hit.get("ayah_end") or int(hit["ayah"]),
                speaker=str(row.get("reciter") or "everyayah"),
                condition="studio",
                stats=stats,
                corpus=corpus,
                tokenizer=tokenizer,
                OOVError=OOVError,
                state=state,
            )
            _finish_row("everyayah", state, stats, crash_after, limit, kept=kept)

    return _finalize_source("everyayah", state, stats)


def _prepare_qua(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("qua")
    stats["hf_repo"] = HF_QUA
    print("[qua] dropping catalog rows whose slug/name contains nufais (q-lab qul_alnufais held-out)")
    state, skipped = _begin_or_skip("qua", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()

    catalog_builder = load_dataset_builder(HF_QUA, "mushafs")
    _print_features("qua/mushafs", HF_QUA, catalog_builder.info.features, catalog_builder.info.splits)
    catalog = _stream_ds(HF_QUA, "all", name="mushafs")
    kept = []
    for row in catalog:
        slug = str(row.get("slug") or "")
        if is_tarteel_dupe(
            slug=slug,
            reciter=row.get("name_en") or "",
            reciter_id=row.get("reciter_id") or "",
            channel=row.get("channel") or "",
        ):
            _bump_skip(stats, "tarteel_dupe")
            continue
        if is_heldout_reciter(
            slug=slug,
            reciter=row.get("name_en") or "",
            reciter_id=row.get("reciter_id") or "",
            name_ar=row.get("name_ar") or "",
        ):
            _bump_skip(stats, "nufais_holdout")
            continue
        if not is_hafs_riwayah(row.get("riwayah")):
            _bump_skip(stats, "not_hafs")
            continue
        kept.append(row)
    nufais_n = int((stats.get("skipped") or {}).get("nufais_holdout") or 0)
    print(
        f"[qua] kept {len(kept)} hafs non-tarteel non-nufais mushafs "
        f"(excluded nufais={nufais_n} tarteel={int((stats.get('skipped') or {}).get('tarteel_dupe') or 0)} "
        f"not_hafs={int((stats.get('skipped') or {}).get('not_hafs') or 0)})"
    )
    if kept:
        sample = kept[0]
        print(
            f"[qua] first kept slug={sample.get('slug')} riwayah={sample.get('riwayah')} "
            f"recording_context={sample.get('recording_context')!r} reciter={sample.get('name_en')}"
        )

    printed_ayah_schema = False
    for mushaf in kept:
        slug = str(mushaf["slug"])
        if limit and state["clips_kept"] >= limit:
            break
        if not printed_ayah_schema:
            b = load_dataset_builder(HF_QUA, slug)
            _print_features(f"qua/{slug}", HF_QUA, b.info.features, b.info.splits)
            stats["features"] = str(b.info.features)
            printed_ayah_schema = True
        ctx = str(mushaf.get("recording_context") or "")
        condition = "studio" if "studio" in ctx.lower() else "crowd"
        speaker = str(mushaf.get("name_en") or slug)
        already = int((state.get("split_rows") or {}).get(slug) or 0)
        print(f"[qua] mushaf slug={slug} already={already}", flush=True)
        try:
            b = load_dataset_builder(HF_QUA, slug)
            splits = getattr(b.info, "splits", None) or {}
            n_ex = int(getattr(splits.get("train"), "num_examples", 0) or 0)
            if n_ex and already >= n_ex:
                print(f"[qua/{slug}] already complete ({already}/{n_ex}); skip load", flush=True)
                continue
            ds = skip_hf_stream(
                _stream_ds(HF_QUA, "train", name=slug), already, f"qua/{slug}"
            )
        except Exception as e:
            print(f"[qua/{slug}] load_dataset failed: {type(e).__name__}: {e}", flush=True)
            _bump_skip(stats, "mushaf_load_error")
            failed = stats.setdefault("failed_slugs", [])
            failed.append(slug)
            continue
        for row in ds:
            if limit and state["clips_kept"] >= limit:
                break
            idx = consume_row(state, split=slug)
            sa = pick_surah_ayah(row)
            if sa is None:
                _bump_skip(stats, "no_surah_ayah")
                _finish_row("qua", state, stats, crash_after, limit, kept=False, commit_every=50)
                continue
            surah, ayah = sa
            wav, dur = _audio_for_clip(
                source="qua",
                idx=idx,
                surah=surah,
                ayah=ayah,
                audio_obj=row.get("audio"),
                force=force,
                stats=stats,
                split=slug,
            )
            if dur is None:
                _finish_row("qua", state, stats, crash_after, limit, kept=False, commit_every=50)
                continue
            extra = {
                "recording_context": ctx,
                "slug": slug,
                "riwayah": mushaf.get("riwayah"),
            }
            kept_clip = _ingest_clip(
                source="qua",
                idx=idx,
                wav=wav,
                duration=dur,
                surah=surah,
                ayah=ayah,
                ayah_end=ayah,
                speaker=speaker,
                condition=condition,
                stats=stats,
                corpus=corpus,
                tokenizer=tokenizer,
                OOVError=OOVError,
                state=state,
                extra_custom=extra,
                split=slug,
            )
            _finish_row("qua", state, stats, crash_after, limit, kept=kept_clip, commit_every=50)

    return _finalize_source("qua", state, stats)


def _prepare_qurantts(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder
    from huggingface_hub import hf_hub_download

    stats = _empty_stats("qurantts")
    stats["hf_repo"] = HF_QURANTTS
    state, skipped = _begin_or_skip("qurantts", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()

    lic_path = hf_hub_download(HF_QURANTTS, "LICENSE", repo_type="dataset")
    lic_text = Path(lic_path).read_text(encoding="utf-8")
    Path("/vol/licenses/qurantts.txt").write_text(lic_text, encoding="utf-8")
    stats["license_ok"] = license_is_permissive(lic_text)
    print(f"[qurantts] license_ok={stats['license_ok']}")

    from datasets import get_dataset_config_names

    configs = get_dataset_config_names(HF_QURANTTS)
    cfg = "metadata" if "metadata" in configs else (configs[0] if configs else None)
    print(f"[qurantts] configs={configs} using={cfg}")
    builder = load_dataset_builder(HF_QURANTTS, cfg) if cfg else load_dataset_builder(HF_QURANTTS)
    _print_features("qurantts", HF_QURANTTS, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    split_names = list(builder.info.splits or {"train": None})
    split = "train" if "train" in split_names else split_names[0]
    stats["split"] = f"{cfg}:{split}" if cfg else split
    ds = (
        load_dataset(HF_QURANTTS, cfg, split=split, streaming=True)
        if cfg
        else load_dataset(HF_QURANTTS, split=split, streaming=True)
    )
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(ds, already, "qurantts")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        fn = row.get("file_name") or row.get("path") or ""
        sa = pick_surah_ayah(row) or parse_surah_ayah_filename(fn)
        if sa is None:
            _bump_skip(stats, "no_surah_ayah")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        surah, ayah = sa
        listed = float(row.get("duration_s") or 0)
        if listed > MAX_DURATION_S:
            _bump_skip(stats, "skip_long")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        if not fn:
            _bump_skip(stats, "no_file_name")
            _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
            continue
        clip_id = make_clip_id("qurantts", idx, surah, ayah)
        flac_path = flac_clip_path("qurantts", clip_id)
        wav = None
        dur = None if force else existing_flac_duration(flac_path)
        if dur is None:
            try:
                local = hf_hub_download(HF_QURANTTS, fn, repo_type="dataset")
                wav, dur = _wav_from_file(local)
            except Exception as e:
                print(f"[qurantts] audio fail {fn}: {e}")
                _bump_skip(stats, "audio_error")
                _finish_row("qurantts", state, stats, crash_after, limit, kept=False)
                continue
        kept = _ingest_clip(
            source="qurantts",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=surah,
            ayah=ayah,
            ayah_end=ayah,
            speaker=str(row.get("reciter") or "qurantts"),
            condition="studio",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            state=state,
            extra_custom={"riwaya": row.get("riwaya"), "file_name": fn},
        )
        _finish_row("qurantts", state, stats, crash_after, limit, kept=kept)

    return _finalize_source("qurantts", state, stats)


def _prepare_iqra(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("iqra")
    stats["hf_repo"] = HF_IQRA
    print(
        "[iqra] match_verse ≥ 0.95 on tashkeel_sentence then sentence "
        "(Iqra_train is MSA + Quran mix; non-Quran is the drop, not a norm bug); "
        "first 200 rows also score search(top_k=1) and hamza-stripped normalize"
    )
    state, skipped = _begin_or_skip("iqra", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_IQRA)
    _print_features("iqra", HF_IQRA, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_IQRA, "train"), already, "iqra")
    diag_n = 0
    diag_keep = {"baseline": 0, "search": 0, "hamza": 0, "multi_ayah": 0}
    diag_miss_printed = 0
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        sentence = str(row.get("sentence") or "")
        tashkeel = str(row.get("tashkeel_sentence") or "")
        if diag_n < 200:
            scores = iqra_match_scores(db, sentence, tashkeel)
            flags = iqra_row_keep_flags(scores)
            diag_n += 1
            for k in ("baseline", "search", "hamza"):
                if flags[k]:
                    diag_keep[k] += 1
            if not flags["baseline"] and diag_miss_printed < 8:
                words = len(sentence.split())
                print(
                    f"[iqra] miss#{diag_miss_printed} words={words} "
                    f"baseline={flags['baseline_score']:.3f} search={flags['search_score']:.3f} "
                    f"hamza={flags['hamza_score']:.3f} tashkeel_len={len(tashkeel)} "
                    f"sentence={sentence[:80]!r}"
                )
                diag_miss_printed += 1
            if diag_n == 200:
                def _rate(k: str) -> str:
                    return f"{diag_keep[k]}/{diag_n} ({100.0 * diag_keep[k] / diag_n:.1f}%)"

                print(
                    f"[iqra] diag n={diag_n} keep@0.95 "
                    f"baseline(match_verse)={_rate('baseline')} "
                    f"search(top_k=1)={_rate('search')} "
                    f"hamza-strip={_rate('hamza')}"
                )
        hit = match_iqra_row(db, sentence, tashkeel)
        if hit is None:
            _bump_skip(stats, "low_match")
            _finish_row("iqra", state, stats, crash_after, limit, kept=False)
            continue
        ayah_end = hit.get("ayah_end") or int(hit["ayah"])
        if ayah_end != int(hit["ayah"]):
            diag_keep["multi_ayah"] += 1
        wav, dur = _audio_for_clip(
            source="iqra",
            idx=idx,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            audio_obj=row.get("audio"),
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("iqra", state, stats, crash_after, limit, kept=False)
            continue
        kept = _ingest_clip(
            source="iqra",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            ayah_end=ayah_end,
            speaker=str(row.get("id") or "iqra"),
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            state=state,
        )
        _finish_row("iqra", state, stats, crash_after, limit, kept=kept)
    if diag_n and diag_n < 200:
        def _rate_partial(k: str) -> str:
            return f"{diag_keep[k]}/{diag_n} ({100.0 * diag_keep[k] / max(diag_n, 1):.1f}%)"

        print(
            f"[iqra] diag n={diag_n} keep@0.95 "
            f"baseline(match_verse)={_rate_partial('baseline')} "
            f"search(top_k=1)={_rate_partial('search')} "
            f"hamza-strip={_rate_partial('hamza')}"
        )
    stats["iqra_match_diag"] = {
        "rows": diag_n,
        "keep_baseline": diag_keep["baseline"],
        "keep_search": diag_keep["search"],
        "keep_hamza": diag_keep["hamza"],
        "multi_ayah_kept": diag_keep["multi_ayah"],
    }
    return _finalize_source("iqra", state, stats)


def _prepare_retasy(limit: int, force: bool, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("retasy")
    stats["hf_repo"] = HF_RETASY
    state, skipped = _begin_or_skip("retasy", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_RETASY)
    _print_features("retasy", HF_RETASY, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "train"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_RETASY, "train"), already, "retasy")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        idx = consume_row(state)
        label = row.get("final_label")
        if not retasy_keep(label):
            reason = "bad_label" if label in BAD_RETASY_LABELS else "not_correct"
            _bump_skip(stats, reason)
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        hit = _match_ayah(db, row.get("Aya") or "")
        if hit is None:
            _bump_skip(stats, "low_match")
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        wav, dur = _audio_for_clip(
            source="retasy",
            idx=idx,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            audio_obj=row.get("audio"),
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("retasy", state, stats, crash_after, limit, kept=False)
            continue
        kept = _ingest_clip(
            source="retasy",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=int(hit["surah"]),
            ayah=int(hit["ayah"]),
            ayah_end=hit.get("ayah_end") or int(hit["ayah"]),
            speaker=str(row.get("reciter_id") or "retasy"),
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            state=state,
            extra_custom={"final_label": label},
        )
        _finish_row("retasy", state, stats, crash_after, limit, kept=kept)
    return _finalize_source("retasy", state, stats)


def _prepare_tlog(limit: int, force: bool, tlog_max_hours: float, crash_after: int = 0) -> dict:
    from datasets import load_dataset, load_dataset_builder

    stats = _empty_stats("tlog")
    stats["hf_repo"] = HF_TLOG
    stats["tlog_max_hours"] = tlog_max_hours
    excl = load_qlab_exclusions()
    holdout = excl["tlog_ids"]
    sample = sorted(holdout)[:3]
    print(f"[tlog] q-lab tlog_holdout exclusion: {len(holdout)} ids sample={sample}")
    state, skipped = _begin_or_skip("tlog", force, stats)
    if skipped is not None:
        return skipped
    corpus, tokenizer, _db, OOVError = _load_labelers()
    builder = load_dataset_builder(HF_TLOG)
    _print_features("tlog", HF_TLOG, builder.info.features, builder.info.splits)
    stats["features"] = str(builder.info.features)
    stats["split"] = "clean"
    already = int(state.get("rows_seen") or 0)
    ds = skip_hf_stream(_stream_ds(HF_TLOG, "clean"), already, "tlog")
    for row in ds:
        if limit and state["clips_kept"] >= limit:
            break
        if tlog_max_hours > 0 and float(state.get("hours_kept") or 0.0) >= tlog_max_hours:
            _bump_skip(stats, "hours_cap")
            break
        idx = consume_row(state)
        if not row.get("is_clean", True):
            _bump_skip(stats, "unclean")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        audio = row.get("audio") or {}
        path = ""
        if isinstance(audio, dict):
            path = str(audio.get("path") or "")
        for cand in (path, row.get("file_name"), row.get("id"), row.get("label")):
            if cand and parse_surah_ayah_filename(str(cand)):
                path = str(cand)
                break
        parsed = parse_surah_ayah_filename(path)
        if parsed is None:
            if int((stats.get("skipped") or {}).get("unmapped") or 0) < 3:
                print(f"[tlog] unmapped path={path!r} audio_keys={list(audio) if isinstance(audio, dict) else type(audio)}")
            _bump_skip(stats, "unmapped")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        if tlog_holdout_key(path) in holdout:
            _bump_skip(stats, "qlab_holdout")
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        surah, ayah = parsed
        wav, dur = _audio_for_clip(
            source="tlog",
            idx=idx,
            surah=surah,
            ayah=ayah,
            audio_obj=audio,
            force=force,
            stats=stats,
        )
        if dur is None:
            _finish_row("tlog", state, stats, crash_after, limit, kept=False, commit_every=50)
            continue
        kept = _ingest_clip(
            source="tlog",
            idx=idx,
            wav=wav,
            duration=dur,
            surah=surah,
            ayah=ayah,
            ayah_end=ayah,
            speaker="tlog",
            condition="crowd",
            stats=stats,
            corpus=corpus,
            tokenizer=tokenizer,
            OOVError=OOVError,
            state=state,
        )
        _finish_row("tlog", state, stats, crash_after, limit, kept=kept, commit_every=50)
    return _finalize_source("tlog", state, stats)


@app.function(**_FN_KW)
def prepare_everyayah(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_everyayah(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_qua(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_qua(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_qurantts(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_qurantts(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_iqra(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_iqra(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_retasy(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_retasy(limit, force, crash_after=crash_after)


@app.function(**_FN_KW)
def prepare_tlog(limit: int = 0, force: bool = False, tlog_max_hours: float = 100.0, crash_after: int = 0):
    _boot_remote()
    return _prepare_tlog(limit, force, tlog_max_hours, crash_after=crash_after)


@app.function(**_FN_KW)
def compute_fbank(source: str, no_speed_perturb: bool = False, force: bool = False):
    """Extract lhotse fbanks (+ optional 0.9/1.1 speed copies) for one source."""
    _boot_remote()
    import torch
    from lhotse import CutSet, Fbank, FbankConfig
    from lhotse.features.io import LilcomChunkyWriter
    from shared.fbank import LHOTSE_FBANK_CONFIG

    torch.set_num_threads(1)

    cuts_path = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    if not cuts_path.is_file():
        raise FileNotFoundError(f"missing {cuts_path}; run prepare first")
    out_path = Path(f"/vol/manifests/{source}_cuts_fbank.jsonl.gz")
    if out_path.is_file() and not force:
        print(f"[fbank/{source}] {out_path} exists; skip (pass --force to redo)")
        return {"source": source, "skipped_existing": True, "path": str(out_path)}
    print(f"[fbank/{source}] load {cuts_path}")
    cuts = CutSet.from_file(str(cuts_path))
    if not no_speed_perturb:
        print(f"[fbank/{source}] speed perturb 0.9/1.1")
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    storage = Path(f"/vol/fbank/{source}")
    storage.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(**LHOTSE_FBANK_CONFIG))
    print(f"[fbank/{source}] extract → {storage} ({len(cuts)} cuts, num_jobs={FBANK_NUM_JOBS})")
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage),
        storage_type=LilcomChunkyWriter,
        num_jobs=FBANK_NUM_JOBS,
    )
    cuts.to_file(str(out_path))
    print(f"[fbank/{source}] wrote {out_path}")
    vol.commit()
    return {"source": source, "cuts": len(cuts), "path": str(out_path)}


@app.function(**_FN_KW)
def compute_fbank_shard(
    source: str, shard: int, n_shards: int, no_speed_perturb: bool = False
):
    """Extract fbanks for one exact-partition shard (every n-th cut after perturb)."""
    _boot_remote()
    import time

    import torch
    from lhotse import CutSet, Fbank, FbankConfig
    from lhotse.features.io import LilcomChunkyWriter
    from shared.fbank import LHOTSE_FBANK_CONFIG

    torch.set_num_threads(1)

    cuts_path = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    if not cuts_path.is_file():
        raise FileNotFoundError(f"missing {cuts_path}; run prepare first")
    out_path = fbank_shard_manifest_path(source, shard)
    if out_path.is_file():
        print(
            f"[fbank/{source} shard {shard}/{n_shards}] {out_path} exists; skip"
        )
        return {
            "source": source,
            "shard": shard,
            "skipped_existing": True,
            "path": str(out_path),
        }
    print(f"[fbank/{source} shard {shard}/{n_shards}] load {cuts_path}")
    cuts = CutSet.from_file(str(cuts_path))
    n_raw = len(cuts)
    if not no_speed_perturb:
        print(f"[fbank/{source} shard {shard}/{n_shards}] speed perturb 0.9/1.1")
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    ids = [c.id for c in cuts]
    shard_ids = shard_items(ids, shard, n_shards)
    cuts = cuts.subset(cut_ids=shard_ids)
    storage = fbank_shard_storage_path(source, shard)
    storage.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(**LHOTSE_FBANK_CONFIG))
    print(
        f"[fbank/{source} shard {shard}/{n_shards}] extract → {storage} "
        f"({len(cuts)} cuts of {len(ids)} perturbed, raw={n_raw}, "
        f"num_jobs={FBANK_NUM_JOBS})"
    )
    t0 = time.time()
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage),
        storage_type=LilcomChunkyWriter,
        num_jobs=FBANK_NUM_JOBS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(str(out_path))
    elapsed = time.time() - t0
    rate = (len(cuts) / elapsed) if elapsed else 0.0
    print(
        f"[fbank/{source} shard {shard}/{n_shards}] wrote {out_path} "
        f"cuts={len(cuts)} {elapsed:.1f}s {rate:.1f} cuts/s"
    )
    vol.commit()
    return {
        "source": source,
        "shard": shard,
        "n_shards": n_shards,
        "cuts": len(cuts),
        "n_raw": n_raw,
        "seconds": elapsed,
        "cuts_per_s": rate,
        "path": str(out_path),
    }


@app.function(**_FN_KW)
def merge_fbank_shards(
    source: str, n_shards: int, no_speed_perturb: bool = False
):
    """Concatenate shard manifests in order into the training fbank CutSet path."""
    _boot_remote()
    from lhotse import CutSet

    parts = []
    for i in range(n_shards):
        p = fbank_shard_manifest_path(source, i)
        if not p.is_file():
            raise FileNotFoundError(f"missing shard manifest {p}")
        print(f"[fbank/{source}] load shard {i}/{n_shards} {p}")
        parts.append(CutSet.from_file(str(p)))
    merged_ids = merge_shard_items([[c.id for c in cs] for cs in parts])
    cuts = parts[0]
    for extra in parts[1:]:
        cuts = cuts + extra
    if [c.id for c in cuts] != merged_ids:
        raise RuntimeError(f"[fbank/{source}] merge order mismatch vs shard concat")
    out_path = Path(f"/vol/manifests/{source}_cuts_fbank.jsonl.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cuts.to_file(str(out_path))
    raw_path = Path(f"/vol/manifests/{source}_cuts.jsonl.gz")
    n_raw = len(CutSet.from_file(str(raw_path))) if raw_path.is_file() else None
    expected = (
        None if n_raw is None else expected_fbank_cut_count(n_raw, no_speed_perturb)
    )
    print(
        f"[fbank/{source}] merged {n_shards} shards → {out_path} "
        f"cuts={len(cuts)} expected={expected} (3× raw when perturb on)"
    )
    if expected is not None and len(cuts) != expected:
        raise RuntimeError(
            f"[fbank/{source}] merged cuts={len(cuts)} != expected={expected} "
            f"(raw={n_raw}, no_speed_perturb={no_speed_perturb})"
        )
    vol.commit()
    return {
        "source": source,
        "cuts": len(cuts),
        "n_raw": n_raw,
        "expected": expected,
        "n_shards": n_shards,
        "path": str(out_path),
    }


@app.function(**_FN_KW)
def flatten_multi_ayah_fbank():
    """Rewrite everyayah_multi fbank cuts as MonoCuts with whole-cut text.

    Does not re-extract features. Fixes icefall crash
    ``ap-8UG5vomxNqwtHYeXNz7epn`` (supervision duration = first ayah).
    """
    _boot_remote()
    from lhotse import CutSet, MonoCut

    src = MULTI_SOURCE
    path = Path(f"/vol/manifests/{src}_cuts_fbank.jsonl.gz")
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run --multi-windows fbank first")
    cuts = CutSet.from_file(str(path))
    n = len(cuts)
    types: dict[str, int] = {}
    n_short = 0
    n_no_feat = 0
    n_ns = 0
    samples = []
    for i, cut in enumerate(cuts):
        tname = type(cut).__name__
        types[tname] = types.get(tname, 0) + 1
        n_sup = len(cut.supervisions)
        n_ns += n_sup
        sup_dur = cut.supervisions[0].duration if n_sup else 0.0
        if not cut.supervisions or not supervision_covers_window(
            cut.duration, cut.supervisions[0].start, cut.supervisions[0].duration
        ):
            n_short += 1
        if not getattr(cut, "has_features", False):
            n_no_feat += 1
        if i < 8:
            samples.append(
                {
                    "id": cut.id,
                    "type": tname,
                    "duration": round(float(cut.duration), 3),
                    "n_sup": n_sup,
                    "sup_dur": round(float(sup_dur), 3),
                    "has_features": bool(getattr(cut, "has_features", False)),
                    "num_frames": getattr(cut, "num_frames", None),
                    "text_len": len(cut.supervisions[0].text) if n_sup else 0,
                }
            )
    print(
        f"[{src}] inspect n={n} types={types} short_sup={n_short} "
        f"no_features={n_no_feat} mean_n_sup={n_ns / n if n else 0:.2f}"
    )
    for s in samples:
        print(f"  sample {s}")
    if n_short == 0 and all(t == "MonoCut" for t in types):
        print(f"[{src}] already flattened; skip rewrite")
        return {
            "source": src,
            "cuts": n,
            "types": types,
            "short_sup": 0,
            "rewritten": False,
            "samples": samples,
        }
    flat = [_flatten_fbank_window_cut(cut) for cut in cuts]
    n_still_short = sum(
        1
        for c in flat
        if not supervision_covers_window(
            c.duration, c.supervisions[0].start, c.supervisions[0].duration
        )
    )
    if n_still_short:
        raise RuntimeError(f"flatten left {n_still_short} short supervisions")
    if any(not c.has_features for c in flat[: min(32, len(flat))]):
        raise RuntimeError("flatten dropped features on a sample cut")
    if any(not isinstance(c, MonoCut) for c in flat[: min(8, len(flat))]):
        raise RuntimeError("flatten did not yield MonoCut")
    tmp = path.parent / path.name.replace(".jsonl.gz", ".writing.jsonl.gz")
    CutSet.from_cuts(flat).to_file(str(tmp))
    os.replace(tmp, path)
    vol.commit()
    print(
        f"[{src}] flattened {n} cuts → {path} short_before={n_short} "
        f"types_before={types} all MonoCut with whole-cut text"
    )
    return {
        "source": src,
        "cuts": n,
        "types_before": types,
        "short_sup_before": n_short,
        "no_features_before": n_no_feat,
        "rewritten": True,
        "samples": samples,
        "path": str(path),
    }


def _wipe_multi_ayah_outputs() -> None:
    src = MULTI_SOURCE
    man = Path("/vol/manifests")
    for p in (
        man / f"{src}_cuts.jsonl.gz",
        man / f"{src}_cuts_fbank.jsonl.gz",
        man / f"{src}_stats.json",
        man / f"{src}_progress.json",
        man / f"{src}_cuts.partial.jsonl",
    ):
        if p.is_file():
            p.unlink()
    for p in man.glob(f"{src}_cuts_fbank.shard-*.jsonl.gz"):
        if p.is_file():
            p.unlink()
    for d in (Path("/vol/fbank") / src, Path("/vol/fbank_sharded") / src):
        if d.is_dir():
            shutil.rmtree(d)


def _chain_window_cuts(cuts: list, gaps: tuple[float, ...]):
    """append() + optional right-pad silence. No new audio files."""
    out = cuts[0]
    for nxt, gap in zip(cuts[1:], gaps):
        gap = float(gap)
        if gap > 1e-4:
            out = out.pad(duration=out.duration + gap, direction="right")
        out = out.append(nxt)
    return out


def _assign_window_supervision(cut, *, wid: str, text: str, speaker: str, custom: dict):
    """Attach one window-level supervision.

    MixedCut.supervisions is derived from tracks — appending to the property
    is a no-op (smoke ap-eOa3fVLZJuYotDOl7IxYPt IndexError). Drop track
    supervisions, then put the concatenated phoneme text on the first
    non-padding track (duration must fit that track; icefall reads ``.text``).
    """
    from lhotse import SupervisionSegment

    try:
        from lhotse.utils import fastcopy
    except ImportError:
        from dataclasses import replace as fastcopy

    if hasattr(cut, "drop_supervisions"):
        cut = cut.drop_supervisions()
    tracks = getattr(cut, "tracks", None)
    if tracks:
        idx = None
        for i, track in enumerate(tracks):
            inner = track.cut
            if type(inner).__name__ == "PaddingCut":
                continue
            idx = i
            break
        if idx is None:
            raise RuntimeError(f"no non-padding track to attach supervision for {wid}")
        track = tracks[idx]
        inner = track.cut
        rec_id = getattr(inner, "recording_id", None) or wid
        sup = SupervisionSegment(
            id=wid,
            recording_id=str(rec_id),
            start=0.0,
            duration=inner.duration,
            channel=0,
            text=text,
            language="quran-phonemes",
            speaker=speaker,
            custom=custom,
        )
        inner = fastcopy(inner, supervisions=[sup])
        new_tracks = list(tracks)
        new_tracks[idx] = fastcopy(track, cut=inner)
        cut = fastcopy(cut, tracks=new_tracks)
        if hasattr(cut, "with_id"):
            cut = cut.with_id(wid)
        else:
            cut = fastcopy(cut, id=wid)
        if not cut.supervisions or cut.supervisions[0].text != text:
            raise RuntimeError(
                f"MixedCut supervision did not stick for {wid} "
                f"n={len(cut.supervisions)}"
            )
        return cut
    rec_id = getattr(cut, "recording_id", None) or wid
    sup = SupervisionSegment(
        id=wid,
        recording_id=str(rec_id),
        start=0.0,
        duration=cut.duration,
        channel=0,
        text=text,
        language="quran-phonemes",
        speaker=speaker,
        custom=custom,
    )
    cut = fastcopy(cut, supervisions=[sup])
    if hasattr(cut, "with_id"):
        cut = cut.with_id(wid)
    else:
        cut = fastcopy(cut, id=wid)
    return cut


def _flatten_fbank_window_cut(cut):
    """MonoCut + one supervision covering the full window.

    Icefall reads ``cut.supervisions[0].text`` against ``cut.num_frames``.
    MixedCut keeps the window text on the first ayah track, so
    ``sup.duration`` is ayah-1 only and DataLoader workers die (ft-multi
    ``ap-8UG5vomxNqwtHYeXNz7epn``).
    """
    from lhotse import MonoCut, SupervisionSegment

    try:
        from lhotse.utils import fastcopy
    except ImportError:
        from dataclasses import replace as fastcopy

    if not cut.supervisions:
        raise RuntimeError(f"{cut.id} has no supervisions")
    src_sup = cut.supervisions[0]
    text = src_sup.text or ""
    speaker = src_sup.speaker
    custom = dict(src_sup.custom or {})
    duration = float(cut.duration)
    rec_id = getattr(cut, "recording_id", None) or cut.id
    fields = whole_cut_supervision_fields(
        cut_id=str(cut.id),
        duration=duration,
        recording_id=str(rec_id),
        text=text,
        speaker=speaker,
        custom=custom,
    )
    sup = SupervisionSegment(
        id=fields["id"],
        recording_id=fields["recording_id"],
        start=fields["start"],
        duration=fields["duration"],
        channel=0,
        text=fields["text"],
        language="quran-phonemes",
        speaker=fields["speaker"],
        custom=fields["custom"],
    )
    feats = None
    if getattr(cut, "has_features", False):
        feats = getattr(cut, "features", None)
    else:
        feats = getattr(cut, "features", None)
    if type(cut).__name__ == "MonoCut":
        out = fastcopy(cut, supervisions=[sup])
        if not supervision_covers_window(out.duration, out.supervisions[0].start, out.supervisions[0].duration):
            raise RuntimeError(
                f"{cut.id} MonoCut supervision still short "
                f"sup={out.supervisions[0].duration:.3f} cut={out.duration:.3f}"
            )
        return out
    if feats is None:
        raise RuntimeError(
            f"{cut.id} type={type(cut).__name__} has no features to flatten"
        )
    kwargs = dict(
        id=cut.id,
        start=0.0,
        duration=duration,
        channel=0,
        features=feats,
        supervisions=[sup],
    )
    rec = getattr(cut, "recording", None)
    if rec is not None:
        kwargs["recording"] = rec
    out = MonoCut(**kwargs)
    if not supervision_covers_window(out.duration, out.supervisions[0].start, out.supervisions[0].duration):
        raise RuntimeError(f"{cut.id} flatten did not cover window")
    if not out.has_features:
        raise RuntimeError(f"{cut.id} flatten dropped features")
    return out


@app.function(**_FN_KW)
def build_multi_ayah_windows(n_windows: int, seed: int = 0, force: bool = False):
    """Synthesize MixedCut windows of 2–4 consecutive EveryAyah ayahs (B1).

    Reads raw ``/vol/manifests/everyayah_cuts.jsonl.gz`` (not perturbed).
    Writes ``/vol/manifests/everyayah_multi_cuts.jsonl.gz`` via lhotse
    ``append`` + silence pad (0–800 ms). Does not re-decode audio. Noise
    mix is skipped (in-memory recordings would need new files; inode cap).
    """
    _boot_remote()
    from lhotse import CutSet

    n_windows = int(n_windows)
    if n_windows <= 0:
        raise ValueError(f"n_windows must be > 0, got {n_windows}")
    src = MULTI_SOURCE
    raw_path = Path("/vol/manifests/everyayah_cuts.jsonl.gz")
    out_path = Path(f"/vol/manifests/{src}_cuts.jsonl.gz")
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing {raw_path}; stage everyayah first")
    if force:
        _wipe_multi_ayah_outputs()
        vol.commit()
        print(f"[{src}] --force: wiped prior cuts/fbank")
    elif out_path.is_file():
        print(f"[{src}] {out_path} exists; skip (pass --force to redo)")
        existing = Path(f"/vol/manifests/{src}_stats.json")
        if existing.is_file():
            return json.loads(existing.read_text(encoding="utf-8"))
        return {"source": src, "skipped_existing": True, "clips": 0, "hours": 0.0}

    print(f"[{src}] load {raw_path}")
    cuts = CutSet.from_file(str(raw_path))
    records = []
    skipped_span = 0
    by_id = {}
    for cut in cuts:
        by_id[cut.id] = cut
        if not cut.supervisions:
            skipped_span += 1
            continue
        sup = cut.supervisions[0]
        rec = cut_dict_window_record(
            {
                "id": cut.id,
                "duration": cut.duration,
                "supervisions": [
                    {
                        "speaker": sup.speaker,
                        "text": sup.text or "",
                        "custom": sup.custom or {},
                    }
                ],
            }
        )
        if rec is None:
            skipped_span += 1
            continue
        records.append(rec)
    print(
        f"[{src}] everyayah cuts={len(cuts)} usable={len(records)} "
        f"skipped_span_or_bad={skipped_span}"
    )
    groups = group_window_records(records)
    candidates = candidate_multi_ayah_windows(groups)
    print(f"[{src}] candidates={len(candidates)} groups={len(groups)}")
    if len(candidates) < n_windows:
        print(
            f"[{src}] WARNING requested {n_windows} windows but only "
            f"{len(candidates)} duration-eligible candidates"
        )
    picked = select_windows_stratified(candidates, n_windows, seed)
    if not picked:
        raise RuntimeError(f"[{src}] no windows selected")

    corpus, tokenizer, _db, OOVError = _load_labelers()
    rng = random.Random(int(seed) ^ 0xA5A5)
    built = []
    built_picks: list[MultiAyahWindowPick] = []
    dropped_long = 0
    dropped_missing = 0
    phoneme_rates: list[float] = []
    hours = 0.0
    for i, pick in enumerate(picked):
        try:
            text = multi_ayah_window_text(corpus, pick.surah, pick.ayah, pick.ayah_end)
            tokenizer.encode(text)
        except OOVError:
            raise RuntimeError(
                f"[{src}] OOV on {pick.surah}:{pick.ayah}-{pick.ayah_end} "
                f"speaker={pick.speaker}"
            ) from None
        except ValueError:
            dropped_missing += 1
            continue
        window_cuts = []
        missing = False
        for cid in pick.cut_ids:
            c = by_id.get(cid)
            if c is None:
                missing = True
                break
            window_cuts.append(c)
        if missing:
            dropped_missing += 1
            continue
        gaps = sample_window_gaps(pick.n_ayahs, rng)
        chained = _chain_window_cuts(window_cuts, gaps)
        if chained.duration > MULTI_MAX_DURATION_S:
            dropped_long += 1
            continue
        wid = make_clip_id(src, i, pick.surah, pick.ayah)
        custom = {
            "surah": pick.surah,
            "ayah": pick.ayah,
            "ayah_end": pick.ayah_end,
            "source": src,
            "n_ayahs": pick.n_ayahs,
        }
        chained = _assign_window_supervision(
            chained,
            wid=wid,
            text=text,
            speaker=pick.speaker,
            custom=custom,
        )
        built.append(chained)
        built_picks.append(pick)
        hours += chained.duration / 3600.0
        if chained.duration > 0:
            phoneme_rates.append(len(text) / chained.duration)
        if len(built) % 5000 == 0:
            vol.commit()
            print(f"[{src}] built {len(built)} windows")

    if not built:
        raise RuntimeError(f"[{src}] built 0 windows")
    try:
        out = CutSet.from_cuts(built)
    except AttributeError:
        out = CutSet(built)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(str(out_path))
    hist = n_ayahs_histogram(built_picks)
    mean_pps = sum(phoneme_rates) / len(phoneme_rates) if phoneme_rates else 0.0
    sample_n = min(8, len(built))
    print(f"[{src}] sample text-vs-duration (phonemes/sec, expect ~8–25):")
    for c in built[:sample_n]:
        sup = c.supervisions[0]
        pps = (len(sup.text) / c.duration) if c.duration else 0.0
        custom = sup.custom or {}
        print(
            f"  {c.id} {custom.get('surah')}:{custom.get('ayah')}-"
            f"{custom.get('ayah_end')} dur={c.duration:.2f}s "
            f"chars={len(sup.text)} pps={pps:.1f} n={custom.get('n_ayahs')}"
        )
    stats = {
        "source": src,
        "clips": len(built),
        "hours": hours,
        "oov": 0,
        "skipped": {
            "dropped_long": dropped_long,
            "dropped_missing": dropped_missing,
            "skipped_span_or_bad": skipped_span,
        },
        "n_ayahs_hist": hist,
        "n_requested": n_windows,
        "n_candidates": len(candidates),
        "seed": int(seed),
        "mean_phonemes_per_sec": mean_pps,
        "noise_mix": False,
        "noise_skip_reason": (
            "in-memory noise recordings do not round-trip through CutSet "
            "JSONL without new files; volume near inode cap"
        ),
        "path": str(out_path),
        "license_ok": True,
        "reused_flac": len(built),
    }
    _write_stats(stats)
    vol.commit()
    print(
        f"[{src}] wrote {out_path} clips={len(built)} hours={hours:.3f} "
        f"hist={hist} mean_pps={mean_pps:.2f} oov=0"
    )
    return stats


def _run_fbank_shards_local(
    src: str, n_shards: int, no_speed_perturb: bool
) -> dict:
    print(f"[fbank/{src}] spawning {n_shards} shards", flush=True)
    pending = list(range(n_shards))
    results: list = [None] * n_shards
    attempts = {i: 0 for i in pending}
    max_attempts = 3
    while pending:
        handles = {
            i: compute_fbank_shard.spawn(src, i, n_shards, no_speed_perturb)
            for i in pending
        }
        next_pending: list[int] = []
        for i, handle in handles.items():
            try:
                results[i] = handle.get()
                print(f"  shard done: {results[i]}", flush=True)
            except Exception as e:
                attempts[i] += 1
                print(
                    f"  shard {i} FAILED attempt {attempts[i]}/{max_attempts}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                if attempts[i] < max_attempts:
                    next_pending.append(i)
                else:
                    raise
        pending = next_pending
    merged = merge_fbank_shards.remote(src, n_shards, no_speed_perturb)
    print(f"FBANK MERGED {src}: {merged}", flush=True)
    return merged


@app.function(**_FN_KW)
def summarize():
    _boot_remote()
    rows = []
    for src in ALL_SOURCES:
        p = Path(f"/vol/manifests/{src}_stats.json")
        if p.is_file():
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    summary = {r["source"]: r for r in rows}
    Path("/vol/manifests/summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    vol.commit()
    print()
    print(f"{'source':<12} {'clips':>8} {'hours':>10} {'oov':>6} {'reused':>8} {'license_ok':>12}  skipped")
    print("-" * 88)
    for src in ALL_SOURCES:
        r = summary.get(src)
        if not r:
            print(f"{src:<12} {'—':>8}")
            continue
        skipped = r.get("skipped") or {}
        skip_s = ",".join(f"{k}={v}" for k, v in skipped.items()) or "—"
        print(
            f"{src:<12} {r.get('clips', 0):8d} {float(r.get('hours') or 0):10.3f} "
            f"{r.get('oov', 0):6d} {int(r.get('reused_flac') or 0):8d} "
            f"{str(r.get('license_ok', True)):>12}  {skip_s}"
        )
    print()
    print("wrote /vol/manifests/summary.json")
    return summary


PREPARE_FNS = {
    "everyayah": prepare_everyayah,
    "qua": prepare_qua,
    "qurantts": prepare_qurantts,
    "iqra": prepare_iqra,
    "retasy": prepare_retasy,
    "tlog": prepare_tlog,
}


@app.local_entrypoint()
def main(
    sources: str = "everyayah,qua,iqra,retasy,tlog",
    limit: int = 0,
    force: bool = False,
    skip_fbank: bool = False,
    tlog_max_hours: float = 100.0,
    summary_only: bool = False,
    no_speed_perturb: bool = False,
    crash_after: int = 0,
    fbank_shards: int = 0,
    multi_windows: int = 0,
    seed: int = 0,
    flatten_multi: bool = False,
):
    if summary_only:
        summarize.remote()
        return
    if flatten_multi:
        print(f"FLATTEN {MULTI_SOURCE}: {flatten_multi_ayah_fbank.remote()}", flush=True)
        return
    if multi_windows > 0:
        print(
            f"multi-ayah windows n={multi_windows} seed={seed} force={force} "
            f"skip_fbank={skip_fbank} fbank_shards={fbank_shards}",
            flush=True,
        )
        stats = build_multi_ayah_windows.remote(multi_windows, seed, force)
        print(f"DONE {MULTI_SOURCE}: {stats}", flush=True)
        if skip_fbank:
            return
        src = MULTI_SOURCE
        n_raw = int(stats.get("clips") or 0)
        n_shards = int(fbank_shards)
        n_feat = expected_fbank_cut_count(n_raw, no_speed_perturb)
        if n_shards <= 0 and n_feat > FBANK_SHARD_CUT_THRESHOLD:
            n_shards = 4
            print(
                f"[fbank/{src}] auto n_shards={n_shards} "
                f"(perturbed={n_feat} > {FBANK_SHARD_CUT_THRESHOLD})",
                flush=True,
            )
        if n_shards > 0:
            _run_fbank_shards_local(src, n_shards, no_speed_perturb)
        else:
            print(f"FBANK {src}: {compute_fbank.remote(src, no_speed_perturb, force)}")
        print(f"FLATTEN {src}: {flatten_multi_ayah_fbank.remote()}", flush=True)
        return
    selected = parse_sources(sources)
    if fbank_shards < 0:
        raise ValueError(f"fbank_shards must be >= 0, got {fbank_shards}")
    if fbank_shards > 0:
        print(
            f"sharded fbank sources={selected} n_shards={fbank_shards} "
            f"no_speed_perturb={no_speed_perturb}"
        )
        for src in selected:
            print(f"[fbank/{src}] spawning {fbank_shards} shards", flush=True)
            pending = list(range(fbank_shards))
            results: list = [None] * fbank_shards
            attempts = {i: 0 for i in pending}
            max_attempts = 3
            while pending:
                handles = {
                    i: compute_fbank_shard.spawn(src, i, fbank_shards, no_speed_perturb)
                    for i in pending
                }
                next_pending: list[int] = []
                for i, handle in handles.items():
                    try:
                        results[i] = handle.get()
                        print(f"  shard done: {results[i]}", flush=True)
                    except Exception as e:
                        attempts[i] += 1
                        print(
                            f"  shard {i} FAILED attempt {attempts[i]}/{max_attempts}: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )
                        if attempts[i] < max_attempts:
                            next_pending.append(i)
                        else:
                            raise
                pending = next_pending
            merged = merge_fbank_shards.remote(src, fbank_shards, no_speed_perturb)
            print(f"FBANK MERGED {src}: {merged}", flush=True)
        return
    print(
        f"staging sources={selected} limit={limit} force={force} "
        f"skip_fbank={skip_fbank} crash_after={crash_after}"
    )
    handles = {
        src: PREPARE_FNS[src].spawn(limit, force, tlog_max_hours, crash_after)
        for src in selected
    }
    for src, handle in handles.items():
        try:
            stats = handle.get()
            print(
                f"DONE {src}: clips={stats.get('clips')} hours={stats.get('hours')} "
                f"reused_flac={stats.get('reused_flac')}"
            )
        except Exception as e:
            print(f"FAILED {src}: {type(e).__name__}: {e}")
    if not skip_fbank:
        fbank_handles = {
            src: compute_fbank.spawn(src, no_speed_perturb, force) for src in selected
        }
        for src, handle in fbank_handles.items():
            try:
                print(f"FBANK {src}: {handle.get()}")
            except Exception as e:
                print(f"FBANK FAILED {src}: {type(e).__name__}: {e}")
    summarize.remote()
