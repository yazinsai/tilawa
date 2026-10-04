"""Reciter-level held-out exclusion for every training manifest.

q-lab (`Quran-Lab/quranic-asr-benchmark`) holds out three EveryAyah reciters,
one QUL reciter and 200 tlog clips. Excluding by *split* is not enough: the
`greentechapps/everyayah_curated_1s_20s` repo was re-split after the benchmark
was cut, and the three held-out reciters now live in its train/validation
splits — so a split-only exclusion ingests them (and the exact benchmark clips).

Everything here is pure (no lhotse, no Modal) so tests and the train-time
assertion share one implementation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

# canonical id -> alternatives; each alternative is a tuple of fragments that
# must all appear in the squashed (lowercase, alnum-only) field text.
HELDOUT_RECITERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "sahl_yassin": (("sahl", "yas"), ("sahl", "yaseen"), ("سهل", "ياسين")),
    "akram_alalaqimy": (("akram", "alaq"), ("akram", "alalq"), ("اكرم", "علاقمي")),
    "muhsin_al_qasim": (
        ("muhsin", "qasim"),
        ("mohsin", "qasim"),
        ("mohsen", "qasim"),
        ("muhsen", "qasim"),
        ("محسن", "قاسم"),
    ),
    "alnufais": (("nufais",), ("nufays",), ("nefais",), ("نفيس",)),
}

# How far reciter-disjointness can be enforced per training source.
SOURCE_POLICY: dict[str, str] = {
    "everyayah": "enforced: speaker = EveryAyah qari dir",
    "everyayah_multi": "enforced: derived from everyayah cuts (speaker kept)",
    "qua": "enforced by mushaf name (transliteration variants listed in HELDOUT_RECITERS)",
    "iqra": "unenforceable: no speaker id; not a studio-reciter source",
    "retasy": "unenforceable: anonymous crowd reciter_id; held-out reciters are studio",
    "tlog": "unenforceable: no speaker id (filename suffix is per-recording); "
    "holdout clips checked by (surah, ayah, duration) twins",
    "qurantts": "n/a: synthetic TTS voices",
    "synthetic": "n/a",
}

_AR_NORM = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def squash(text: object) -> str:
    s = unicodedata.normalize("NFKC", str(text or "")).lower().translate(_AR_NORM)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", "", s)


def match_heldout_reciter(*values: object) -> str | None:
    """Return the canonical held-out reciter id any value names, else None."""
    blobs = [squash(v) for v in values if v not in (None, "")]
    for rid, alts in HELDOUT_RECITERS.items():
        for frags in alts:
            if any(all(squash(f) in b for f in frags) for b in blobs):
                return rid
    return None


_SP_SUFFIX = re.compile(r"_sp\d+(?:\.\d+)?$")


def base_cut_id(cut_id: str) -> str:
    """Strip lhotse speed-perturb suffixes (`_sp0.9`, `_sp1.1`)."""
    return _SP_SUFFIX.sub("", str(cut_id))


def cut_identity_fields(cut: Mapping) -> list[object]:
    """Every field of a lhotse cut dict that can carry a reciter name."""
    out: list[object] = []
    for sup in cut.get("supervisions") or []:
        out.append(sup.get("speaker"))
        custom = sup.get("custom") or {}
        for key in ("slug", "reciter", "reciter_id", "name_en", "name_ar", "qari", "source_speaker"):
            if key in custom:
                out.append(custom[key])
    rec = cut.get("recording") or {}
    for src in rec.get("sources") or []:
        out.append(src.get("source"))
    custom = cut.get("custom") or {}
    for key in ("speaker", "slug", "reciter"):
        if key in custom:
            out.append(custom[key])
    return out


def cut_surah_ayah(cut: Mapping) -> tuple[int, int, int] | None:
    for sup in cut.get("supervisions") or []:
        c = sup.get("custom") or {}
        if "surah" in c and "ayah" in c:
            return int(c["surah"]), int(c["ayah"]), int(c.get("ayah_end") or c["ayah"])
    return None


@dataclass(frozen=True)
class HeldoutClip:
    source: str
    clip_id: str
    surah: int
    ayah: int
    duration: float


@dataclass
class LeakReport:
    source: str
    n_cuts: int = 0
    reciter_hits: Counter = field(default_factory=Counter)
    twin_hits: Counter = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    flagged_ids: set[str] = field(default_factory=set)
    _bases: set[str] | None = field(default=None, repr=False)
    _bases_n: int = field(default=-1, repr=False)

    @property
    def n_flagged(self) -> int:
        return len(self.flagged_ids)

    def is_flagged(self, cut_id: str) -> bool:
        """True for a flagged cut and every speed-perturbed copy of it."""
        if self._bases is None or self._bases_n != len(self.flagged_ids):
            self._bases = {base_cut_id(i) for i in self.flagged_ids}
            self._bases_n = len(self.flagged_ids)
        return base_cut_id(cut_id) in self._bases

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "policy": next((v for k, v in sorted(SOURCE_POLICY.items(), key=lambda kv: -len(kv[0]))
                            if self.source.startswith(k)), "unknown source"),
            "n_cuts": self.n_cuts,
            "n_flagged": self.n_flagged,
            "reciter_hits": dict(self.reciter_hits),
            "twin_hits": dict(self.twin_hits),
            "examples": {k: v[:5] for k, v in self.examples.items()},
        }


def build_twin_index(clips: Iterable[HeldoutClip]) -> dict[tuple[int, int], list[HeldoutClip]]:
    idx: dict[tuple[int, int], list[HeldoutClip]] = defaultdict(list)
    for c in clips:
        idx[(c.surah, c.ayah)].append(c)
    return idx


def scan_cuts(
    source: str,
    cuts: Iterable[Mapping],
    twins: Mapping[tuple[int, int], list[HeldoutClip]] | None = None,
    twin_tol_s: float = 0.06,
) -> LeakReport:
    """Flag cuts naming a held-out reciter, or single-ayah cuts whose
    (surah, ayah) matches a held-out clip within ``twin_tol_s`` seconds.

    Twin tolerance: curated EveryAyah clips re-encoded from the same mp3 land
    within ~0.05 s of the benchmark wav; distinct recitations of one ayah
    rarely agree to 60 ms, so false positives stay rare and are reported.
    """
    rep = LeakReport(source=source)
    for cut in cuts:
        rep.n_cuts += 1
        cid = str(cut.get("id") or "")
        rid = match_heldout_reciter(*cut_identity_fields(cut))
        if rid is not None:
            rep.reciter_hits[rid] += 1
            rep.flagged_ids.add(cid)
            if len(rep.examples[rid]) < 5:
                rep.examples[rid].append(cid)
        if twins:
            sa = cut_surah_ayah(cut)
            if sa is None or sa[2] != sa[1]:
                continue
            dur = float(cut.get("duration") or 0.0)
            for h in twins.get((sa[0], sa[1]), ()):
                # perturbed copies are caught via is_flagged(base id)
                if cid != base_cut_id(cid):
                    break
                if abs(dur - h.duration) <= twin_tol_s:
                    key = f"twin:{h.source}"
                    rep.twin_hits[h.source] += 1
                    rep.flagged_ids.add(cid)
                    if len(rep.examples[key]) < 5:
                        rep.examples[key].append(cid)
                    break
    return rep


# Training source prefix -> held-out slice its twins are enforced against.
ENFORCED_TWINS: dict[str, str] = {"everyayah": "everyayah_heldout", "tlog": "tlog_holdout"}


def load_heldout_clips(path) -> list[HeldoutClip]:
    import json
    from pathlib import Path

    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [HeldoutClip(r["source"], r["id"], int(r["surah"]), int(r["ayah"]), float(r["duration"])) for r in rows]


def check_manifests(sources: Iterable[str], manifest_dir, clips: Iterable[HeldoutClip] = ()) -> list[LeakReport]:
    """Scan `<src>_cuts_fbank.jsonl.gz` for every training source."""
    import gzip
    import json
    from pathlib import Path

    clips = list(clips)
    reports = []
    for src in sources:
        slice_ = next((v for k, v in ENFORCED_TWINS.items() if src.startswith(k)), None)
        twins = build_twin_index(c for c in clips if c.source == slice_) if slice_ else None
        path = Path(manifest_dir) / f"{src}_cuts_fbank.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            cuts = (json.loads(line) for line in f if line.strip())
            reports.append(scan_cuts(src, cuts, twins=twins))
    return reports


class LeakError(RuntimeError):
    pass


def assert_no_leaks(reports: Iterable[LeakReport], *, allow_twins: bool = False) -> None:
    bad = []
    for r in reports:
        n_rec = sum(r.reciter_hits.values())
        n_twin = 0 if allow_twins else sum(r.twin_hits.values())
        if n_rec or n_twin:
            bad.append(f"{r.source}: reciter={dict(r.reciter_hits)} twins={dict(r.twin_hits)}")
    if bad:
        raise LeakError("held-out leak in training manifests:\n  " + "\n  ".join(bad))
