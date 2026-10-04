"""Pull crowd recitations from tilawa.dev/help into private storage.

Reads HELP_ADMIN_TOKEN from the environment. Rows with kind != "none" are
acted mistakes. By default they are dropped as soon as each export line is
parsed and counted only as one excluded total. With --include-acted they are
kept as use="acted" with their mistake label, for private correction eval and
tuning only: acted audio, ids, labels and per-clip outputs are never
published or committed.

Audio and manifest.json go to --out (default /tmp/help, outside the repo)
and, with --upload, to the Modal volume zipformer-ctc-training at /help/.
One 16 kHz mono PCM16 wav per clip, plus the manifest. Nothing is committed.

--prior-manifest keeps every speaker of an earlier manifest in its earlier
split; only speakers new to the export are assigned, by the same hash order.

Usage (cwd = lab/):
    ../.venv/bin/python scripts/pull_help_recordings.py --out /tmp/help --upload
    ../.venv/bin/python scripts/pull_help_recordings.py --out /tmp/help --include-acted \
        --prior-manifest /tmp/help_old/manifest.json --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from shared.phoneme_labels import PhonemeCorpus  # noqa: E402
from shared.quran_db import QuranDB  # noqa: E402

API_BASE = "https://tilawa.dev/help/api"
VOLUME_NAME = "zipformer-ctc-training"
VOLUME_DIR = "/help"
MAX_PARALLEL = 8
USER_AGENT = "tilawa-lab-help-pull/1.0"
_REF_SINGLE = re.compile(r"^(\d+):(\d+)$")
_REF_SPAN = re.compile(r"^(\d+):(\d+)-(\d+):(\d+)$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class RowError(ValueError):
    """A kind==none row that cannot become a corpus sample."""


def take_clean_row(row: Mapping) -> dict | None:
    """Copy a kind=="none" row. Acted mistakes return None and are not copied."""
    if row.get("kind") != "none":
        return None
    return dict(row)


def filter_kind_none(rows: Iterable[Mapping]) -> tuple[list[dict], int]:
    """Keep kind=="none". Return (kept, excluded_count).

    Excluded rows are not copied into the result. The count is the only
    record of acted mistakes.
    """
    kept: list[dict] = []
    excluded = 0
    for row in rows:
        clean = take_clean_row(row)
        if clean is None:
            excluded += 1
            continue
        kept.append(clean)
    return kept, excluded


def parse_ref(ref: str) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Parse "s:a" or "s:a-s:b" into start and optional end endpoints.

    The end is None for a single ayah. Endpoints are not expanded; call
    VerseIndex.expand so a span follows corpus order, including across surahs.
    """
    text = ref.strip()
    single = _REF_SINGLE.match(text)
    if single:
        return (int(single.group(1)), int(single.group(2))), None
    span = _REF_SPAN.match(text)
    if span:
        start = (int(span.group(1)), int(span.group(2)))
        end = (int(span.group(3)), int(span.group(4)))
        return start, end
    raise ValueError(f"bad ref: {ref!r}")


class VerseIndex:
    """Mushaf order of (surah, ayah) keys. Lookup is by those keys, never text."""

    def __init__(self, order: list[tuple[int, int]]):
        self.order = list(order)
        self.pos = {ref: i for i, ref in enumerate(self.order)}

    @classmethod
    def load(cls) -> VerseIndex:
        db = QuranDB()
        corpus = PhonemeCorpus()
        order = sorted(
            ((int(v["surah"]), int(v["ayah"])) for v in db.verses),
            key=lambda sa: (sa[0], sa[1]),
        )
        missing = 0
        for surah, ayah in order:
            try:
                corpus.word_phonemes(surah, ayah)
            except ValueError:
                missing += 1
        if missing:
            raise RuntimeError(f"phoneme corpus is missing {missing} quran.json ayahs")
        return cls(order)

    def expand(
        self, start: tuple[int, int], end: tuple[int, int] | None
    ) -> list[dict]:
        if start not in self.pos:
            raise ValueError(f"unknown ayah {start[0]}:{start[1]}")
        if end is None:
            return [{"surah": start[0], "ayah": start[1]}]
        if end not in self.pos:
            raise ValueError(f"unknown ayah {end[0]}:{end[1]}")
        i0, i1 = self.pos[start], self.pos[end]
        if i1 < i0:
            raise ValueError(f"reversed span {start[0]}:{start[1]}-{end[0]}:{end[1]}")
        return [{"surah": s, "ayah": a} for s, a in self.order[i0 : i1 + 1]]


def expected_verses(row: Mapping, index: VerseIndex) -> list[dict]:
    """Ordered {surah, ayah} list from ayahs (v4) or ref.

    When both are present they must name the same ayahs in the same order.
    """
    ayahs = row.get("ayahs")
    from_list: list[dict] | None = None
    if isinstance(ayahs, list) and ayahs:
        from_list = []
        for item in ayahs:
            if not isinstance(item, str):
                raise ValueError("ayahs entry is not a ref string")
            start, end = parse_ref(item)
            from_list.extend(index.expand(start, end))
    ref = row.get("ref")
    from_ref: list[dict] | None = None
    if isinstance(ref, str) and ref.strip():
        start, end = parse_ref(ref)
        from_ref = index.expand(start, end)
    if from_list is not None and from_ref is not None and from_list != from_ref:
        raise ValueError("ref and ayahs disagree")
    verses = from_list if from_list is not None else from_ref
    if not verses:
        raise ValueError("missing ref")
    return verses


def _speaker_hash(speaker: str) -> str:
    return hashlib.sha256(speaker.encode("utf-8")).hexdigest()


def assign_speaker_splits(
    speaker_seconds: Mapping[str, float],
    dev_fraction: float = 0.4,
) -> dict[str, str]:
    """Speaker-disjoint dev/test.

    Order is sha256(speaker), so the cut is deterministic and does not depend
    on row order. Each speaker is entirely dev or entirely test. Dev is filled
    toward dev_fraction of clip seconds. A speaker who fits under the target
    goes to dev. A speaker who would overshoot goes to dev only when that
    overshoot is closer to the target than the speakers still left in hash order.
    """
    if not speaker_seconds:
        return {}
    ordered = sorted(speaker_seconds, key=_speaker_hash)
    total = float(sum(speaker_seconds.values()))
    if total <= 0:
        return {sp: ("dev" if i == 0 else "test") for i, sp in enumerate(ordered)}
    target = dev_fraction * total
    dev = 0.0
    out: dict[str, str] = {}
    for index, speaker in enumerate(ordered):
        seconds = float(speaker_seconds[speaker])
        if dev >= target:
            out[speaker] = "test"
            continue
        if dev + seconds <= target:
            out[speaker] = "dev"
            dev += seconds
            continue
        # Oversized for the remaining budget. Take it only when that lands
        # closer to the target than filling the rest of the hash order can.
        rest = sum(float(speaker_seconds[later]) for later in ordered[index + 1 :])
        take_err = abs((dev + seconds) - target)
        skip_err = abs(min(target, dev + rest) - target)
        if take_err < skip_err:
            out[speaker] = "dev"
            dev += seconds
        else:
            out[speaker] = "test"
    if not any(split == "dev" for split in out.values()):
        out[ordered[0]] = "dev"
    return out


def merge_speaker_splits(
    prior: Mapping[str, str],
    speaker_seconds: Mapping[str, float],
    dev_fraction: float = 0.4,
) -> dict[str, str]:
    """Keep each prior speaker's split; assign only new speakers.

    New speakers are split among themselves toward dev_fraction of their own
    seconds, so a speaker never moves between dev and test across pulls.
    """
    out = {sp: prior[sp] for sp in speaker_seconds if sp in prior}
    new = {sp: s for sp, s in speaker_seconds.items() if sp not in prior}
    out.update(assign_speaker_splits(new, dev_fraction) if new else {})
    return out


def load_prior_splits(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"] if isinstance(data, dict) else data
    splits: dict[str, str] = {}
    for sample in samples:
        speaker, split = sample.get("speaker"), sample.get("split")
        if not isinstance(speaker, str) or split not in {"dev", "test"}:
            continue
        if splits.setdefault(speaker, split) != split:
            raise ValueError("prior manifest puts one speaker in both splits")
    return splits


def acted_label(row: Mapping, verses: list[dict]) -> dict:
    """The single labelled mistake of an acted row, as manifest fields.

    ``word`` is the export's index into the ayah's words; ``skip_ayah`` has
    none. The ayah must be one of the clip's expected verses.
    """
    kind = row.get("kind")
    mistakes = row.get("mistakes")
    if not isinstance(kind, str) or not isinstance(mistakes, list) or len(mistakes) != 1:
        raise RowError("mistakes")
    mistake = mistakes[0]
    if not isinstance(mistake, Mapping) or mistake.get("kind") != kind:
        raise RowError("mistake kind")
    try:
        (surah, ayah), end = parse_ref(str(mistake.get("ayah")))
    except ValueError as exc:
        raise RowError("mistake ayah") from exc
    if end is not None or {"surah": surah, "ayah": ayah} not in verses:
        raise RowError("mistake ayah not in clip")
    word = mistake.get("word")
    if kind == "skip_ayah":
        word = None
    elif isinstance(word, bool) or not isinstance(word, int) or word < 0:
        raise RowError("mistake word")
    keep = ("expected", "said", "said_uthmani", "said_from", "rule", "letter", "from", "to",
            "from_letter", "to_letter", "continuous_reading")
    return {
        "kind": kind,
        "surah": surah,
        "ayah": ayah,
        "word": word,
        **{key: mistake[key] for key in keep if key in mistake},
    }


def _optional_str(row: Mapping, key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RowError(f"{key} is not a string")
    return value


def audio_object_name(file_field: str) -> str:
    """Export `file` is `audio/<name>`. <name> is the /audio path segment."""
    if not isinstance(file_field, str) or not file_field.startswith("audio/"):
        raise RowError("file")
    name = file_field[len("audio/") :]
    if (
        not name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or ".." in name
    ):
        raise RowError("file")
    return name


def ayah_end_for(verses: list[dict]) -> int | None:
    """Same-surah contiguous span end, or None for a single ayah or a broken span."""
    if len(verses) <= 1:
        return None
    surah = verses[0]["surah"]
    if any(v["surah"] != surah for v in verses):
        return None
    ayahs = [v["ayah"] for v in verses]
    if ayahs != list(range(ayahs[0], ayahs[-1] + 1)):
        return None
    return ayahs[-1]


def prepare_row(row: Mapping, index: VerseIndex) -> dict:
    """Validate a row into manifest fields. No network.

    kind=="none" rows are clean takes (use "clean", or "slip" with
    extra_mistake). Any other kind is an acted take (use "acted") and carries
    its mistake label.
    """
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not _SAFE_ID.match(sample_id):
        raise RowError("id")
    name = audio_object_name(row.get("file"))
    speaker = row.get("speaker")
    if not isinstance(speaker, str) or not speaker.strip():
        raise RowError("speaker")
    prompt_version = row.get("prompt_version")
    if isinstance(prompt_version, bool) or not isinstance(prompt_version, int):
        raise RowError("prompt_version")
    extra = row.get("extra_mistake")
    if not isinstance(extra, bool):
        raise RowError("extra_mistake")
    passage = row.get("passage")
    if passage is not None and passage not in {"single", "run"}:
        raise RowError("passage")
    try:
        verses = expected_verses(row, index)
    except ValueError as exc:
        raise RowError(str(exc)) from exc
    n_ayahs = len(verses)
    if passage == "single" and n_ayahs != 1:
        raise RowError("passage does not match ayahs")
    if passage == "run" and n_ayahs < 2:
        raise RowError("passage does not match ayahs")
    acted = row.get("kind") != "none"
    mistake = acted_label(row, verses) if acted else None
    return {
        "id": sample_id,
        "file": f"{sample_id}.wav",
        "surah": verses[0]["surah"],
        "ayah": verses[0]["ayah"],
        "ayah_end": ayah_end_for(verses),
        "expected_verses": verses,
        "n_ayahs": n_ayahs,
        "speaker": speaker,
        "device": _optional_str(row, "device"),
        "gender": _optional_str(row, "gender"),
        "level": _optional_str(row, "level"),
        "prompt_version": prompt_version,
        "passage": passage,
        "extra_mistake": extra,
        "source": "help",
        "use": "acted" if acted else ("slip" if extra else "clean"),
        "mistake": mistake,
        "mechanism": _optional_str(row, "mechanism") if acted else None,
        "elicitation": _optional_str(row, "elicitation") if acted else None,
        "_audio_name": name,
        "_mime": row.get("mime") if isinstance(row.get("mime"), str) else "unknown",
    }


def pcm_duration_s(path: Path) -> float:
    """Duration from decoded PCM samples. Ignores container duration metadata."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000 or handle.getsampwidth() != 2:
            raise ValueError("wav is not 16 kHz mono pcm16")
        nframes = handle.getnframes()
        frames = handle.readframes(nframes)
        width = handle.getsampwidth()
        if nframes <= 0 or len(frames) != nframes * width:
            raise ValueError("wav has no samples")
        return nframes / 16000.0


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/x-ndjson, */*",
    }


def _redact(message: str, token: str) -> str:
    if token and token in message:
        message = message.replace(token, "<redacted>")
    return message.replace("\n", " ")[:240]


def _read_bytes(url: str, token: str, timeout: int = 120) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(url, headers=_headers(token))
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"download failed: {last_error}")


def fetch_export(token: str, include_acted: bool = False) -> tuple[list[dict], int, int]:
    """Stream /export. Unless include_acted, drop kind != none before the row is retained."""
    url = f"{API_BASE}/export"
    body = _read_bytes(url, token, timeout=180)
    kept: list[dict] = []
    total = 0
    excluded = 0
    for raw in body.splitlines():
        line = raw.decode("utf-8").strip()
        if not line:
            continue
        total += 1
        row = json.loads(line)
        clean = dict(row) if include_acted else take_clean_row(row)
        if clean is None:
            excluded += 1
            continue
        kept.append(clean)
    del body
    return kept, total, excluded


def _suffix_for(audio_name: str) -> str:
    suffix = Path(audio_name).suffix.lower()
    if suffix in {".webm", ".wav", ".m4a", ".mp4", ".ogg", ".opus", ".mp3"}:
        return suffix
    return ".bin"


def normalize_to_wav(src: Path, dest: Path) -> float:
    """Decode the whole file to 16 kHz mono PCM16 and measure samples."""
    partial = dest.with_name(f".{dest.stem}.partial.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(partial),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0 or not partial.is_file():
            err = result.stderr.decode("utf-8", "replace")[-180:]
            raise RuntimeError(f"ffmpeg failed ({result.returncode}): {err}")
        duration = pcm_duration_s(partial)
        if duration <= 0:
            raise RuntimeError("decoded audio is empty")
        os.replace(partial, dest)
        return duration
    finally:
        partial.unlink(missing_ok=True)


def _download_one(spec: dict, out_dir: Path, token: str) -> dict:
    dest = out_dir / spec["file"]
    if dest.is_file():
        try:
            duration = pcm_duration_s(dest)
            return {**spec, "duration_s": round(duration, 3), "_resumed": True}
        except ValueError:
            dest.unlink(missing_ok=True)
    audio_name = spec["_audio_name"]
    url = f"{API_BASE}/audio/{urllib.parse.quote(audio_name)}"
    raw_path = out_dir / f".{spec['id']}.src{_suffix_for(audio_name)}"
    try:
        payload = _read_bytes(url, token)
        if not payload:
            raise RuntimeError("empty audio response")
        raw_path.write_bytes(payload)
        duration = normalize_to_wav(raw_path, dest)
        return {**spec, "duration_s": round(duration, 3), "_resumed": False}
    finally:
        raw_path.unlink(missing_ok=True)


def _public_sample(spec: dict, split: str) -> dict:
    return {
        "id": spec["id"],
        "file": spec["file"],
        "surah": spec["surah"],
        "ayah": spec["ayah"],
        "ayah_end": spec["ayah_end"],
        "expected_verses": spec["expected_verses"],
        "n_ayahs": spec["n_ayahs"],
        "speaker": spec["speaker"],
        "device": spec["device"],
        "gender": spec["gender"],
        "level": spec["level"],
        "prompt_version": spec["prompt_version"],
        "passage": spec["passage"],
        "extra_mistake": spec["extra_mistake"],
        "duration_s": spec["duration_s"],
        "source": "help",
        "split": split,
        "use": spec["use"],
        **(
            {"mistake": spec["mistake"], "mechanism": spec["mechanism"], "elicitation": spec["elicitation"]}
            if spec["use"] == "acted"
            else {}
        ),
    }


def _repo_roots() -> set[Path]:
    roots: set[Path] = set()
    here = Path(__file__).resolve()
    for parent in here.parents:
        git = parent / ".git"
        if not git.exists():
            continue
        roots.add(parent)
        if git.is_file():
            for line in git.read_text(encoding="utf-8").splitlines():
                if not line.startswith("gitdir:"):
                    continue
                gitdir = Path(line.split(":", 1)[1].strip()).resolve()
                if gitdir.parent.name == "worktrees":
                    roots.add(gitdir.parent.parent.parent)
                else:
                    roots.add(gitdir.parent)
        break
    return roots


def ensure_private_out(out: Path) -> Path:
    resolved = out.expanduser().resolve()
    for root in _repo_roots():
        if resolved == root or root in resolved.parents:
            raise SystemExit(f"--out must be outside the repo ({root})")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _slice_minutes(samples: list[dict], key: str) -> list[tuple[str, float]]:
    totals: dict[str, float] = {}
    for sample in samples:
        if key == "span":
            label = "single" if sample["n_ayahs"] == 1 else "multi"
        else:
            value = sample.get(key)
            label = value if isinstance(value, str) and value else "unknown"
        totals[label] = totals.get(label, 0.0) + float(sample["duration_s"])
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def format_summary(
    total: int,
    excluded: int,
    samples: list[dict],
    failures: list[tuple[str, str]],
    formats: Counter,
) -> str:
    lines = [
        f"rows_total {total}",
        f"excluded {excluded}",
        f"allowed {total - excluded}",
        f"stored {len(samples)}",
        f"failures {len(failures)}",
        "use_split",
    ]
    for use in ("clean", "slip", "acted"):
        for split in ("dev", "test"):
            chosen = [s for s in samples if s["use"] == use and s["split"] == split]
            minutes = sum(float(s["duration_s"]) for s in chosen) / 60.0
            lines.append(f"  {use} {split}: {len(chosen)} clips, {minutes:.2f} min")
    acted_kinds = Counter((s["mistake"]["kind"], s["split"]) for s in samples if s["use"] == "acted")
    if acted_kinds:
        lines.append("acted_kind_split")
        for (kind, split), count in sorted(acted_kinds.items()):
            lines.append(f"  {kind} {split}: {count}")
    stored_min = sum(float(s["duration_s"]) for s in samples) / 60.0
    lines.append(f"stored_min {stored_min:.2f}")
    for key in ("device", "gender", "level", "span"):
        parts = [f"{label}={minutes / 60.0:.2f}" for label, minutes in _slice_minutes(samples, key)]
        lines.append(f"{key}_min " + " ".join(parts))
    lines.append("formats")
    for mime, count in sorted(formats.items()):
        lines.append(f"  {mime}: {count}")
    if failures:
        lines.append("failure_list")
        for sample_id, reason in failures:
            lines.append(f"  {sample_id}: {reason}")
    else:
        lines.append("failure_list none")
    return "\n".join(lines)


def _upload(out: Path, filenames: list[str]) -> tuple[int, int]:
    import modal
    from modal.exception import NotFoundError

    volume = modal.Volume.from_name(VOLUME_NAME)
    existing: dict[str, int] = {}
    try:
        for entry in volume.listdir(VOLUME_DIR):
            existing[Path(entry.path).name] = int(entry.size)
    except NotFoundError:
        existing = {}
    wavs = [name for name in filenames if name != "manifest.json"]
    to_send = []
    skipped = 0
    for name in wavs:
        local = out / name
        if existing.get(name) == local.stat().st_size:
            skipped += 1
            continue
        to_send.append(name)
    manifest_exists = "manifest.json" in existing
    overwrite = manifest_exists or any(name in existing for name in to_send)
    with volume.batch_upload(force=overwrite) as batch:
        for name in to_send:
            batch.put_file(str(out / name), f"{VOLUME_DIR}/{name}")
        batch.put_file(str(out / "manifest.json"), f"{VOLUME_DIR}/manifest.json")
    listed = volume.listdir(VOLUME_DIR)
    names = {Path(entry.path).name for entry in listed}
    if "manifest.json" not in names:
        raise RuntimeError("upload failed: /help/manifest.json missing")
    wavs_on_volume = sum(1 for name in names if name.endswith(".wav"))
    if wavs_on_volume < len(wavs):
        raise RuntimeError(
            f"upload failed: volume has {wavs_on_volume} wavs, expected at least {len(wavs)}"
        )
    print(f"volume {VOLUME_NAME}:{VOLUME_DIR}/manifest.json")
    print(f"volume_dir {VOLUME_NAME}:{VOLUME_DIR}/")
    print(f"volume_files {len(names)}")
    print(f"volume_wavs {wavs_on_volume}")
    return len(to_send) + 1, skipped


def _require_token() -> str:
    token = os.environ.get("HELP_ADMIN_TOKEN", "")
    if not token:
        raise SystemExit("HELP_ADMIN_TOKEN is not set")
    return token


def run(out: Path, upload: bool, include_acted: bool = False, prior_manifest: Path | None = None) -> str:
    token = _require_token()
    out = ensure_private_out(out)
    prior = load_prior_splits(prior_manifest) if prior_manifest else {}
    kept, total, excluded = fetch_export(token, include_acted)
    index = VerseIndex.load()
    specs: list[dict] = []
    failures: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in kept:
        sample_id = row.get("id")
        label = sample_id if isinstance(sample_id, str) else "(no id)"
        try:
            spec = prepare_row(row, index)
        except RowError as exc:
            failures.append((label, f"malformed: {exc}"))
            continue
        if spec["id"] in seen:
            failures.append((spec["id"], "malformed: duplicate id"))
            continue
        seen.add(spec["id"])
        specs.append(spec)

    stored_specs: list[dict] = []
    if specs:
        workers = min(MAX_PARALLEL, len(specs))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_download_one, spec, out, token): spec for spec in specs
            }
            done = 0
            for future in as_completed(futures):
                spec = futures[future]
                done += 1
                try:
                    stored_specs.append(future.result())
                except Exception as exc:
                    failures.append((spec["id"], f"download: {_redact(str(exc), token)}"))
                if done % 25 == 0 or done == len(specs):
                    print(f"progress {done}/{len(specs)}", file=sys.stderr)
    by_id = {spec["id"]: spec for spec in stored_specs}
    ordered = [by_id[spec["id"]] for spec in specs if spec["id"] in by_id]
    speaker_seconds: dict[str, float] = {}
    for spec in ordered:
        speaker_seconds[spec["speaker"]] = speaker_seconds.get(spec["speaker"], 0.0) + float(
            spec["duration_s"]
        )
    splits = merge_speaker_splits(prior, speaker_seconds)
    print(f"speakers {len(splits)} prior_kept {sum(1 for sp in splits if sp in prior)}")
    samples = [_public_sample(spec, splits[spec["speaker"]]) for spec in ordered]
    formats = Counter(spec["_mime"] for spec in ordered)
    manifest = {"samples": samples}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = format_summary(total, excluded, samples, failures, formats)
    print(summary)
    if upload and samples:
        uploaded, skipped = _upload(out, [sample["file"] for sample in samples])
        print(f"uploaded {uploaded}")
        print(f"upload_skipped_existing {skipped}")
    elif upload:
        print("upload skipped: no stored clips")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/help"),
        help="private scratch directory outside the repo (default: /tmp/help)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"upload wavs and manifest.json to {VOLUME_NAME}:{VOLUME_DIR}/",
    )
    parser.add_argument(
        "--include-acted",
        action="store_true",
        help='keep kind != "none" rows as use="acted" (private eval only; never publish)',
    )
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        default=None,
        help="earlier manifest.json whose speaker splits are kept",
    )
    args = parser.parse_args(argv)
    run(args.out, args.upload, args.include_acted, args.prior_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
