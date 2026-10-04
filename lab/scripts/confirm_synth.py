"""Synthetic confirmatory set for the two structural correction checks (private; aggregates out).

Built from EveryAyah dev reciters only (``build_heldout_multi.DEV_RECITERS``), on passages disjoint
from every help prompt (acted, slip, clean; dev and test) and v1 passage:

- ``skip``: 3-4 ayah windows, the per-ayah files joined with a short crossfade at their natural
  pauses, with one interior ayah cut out; ``skip-ctl`` is the same window uncut.
- ``sister``: for look-alike families (connected components of the similar-verse index) that hold
  no prompt ayah, ayah E with one 1-2 word ``replace`` region spliced from the same reciter's
  audio of look-alike M at word boundaries (crossfade); ``sister-ctl`` is E uncut.

Audio, clip ids and per-clip outputs stay under /tmp and the Modal volume
(``synthetic/confirmatory/``). Never publish them.

    ../.venv/bin/python scripts/confirm_synth.py select --out /tmp/conf/syn
    ../.venv/bin/python scripts/tracker_correction_eval.py --sets x --extra align=/tmp/conf/syn/align.jsonl ...
    ../.venv/bin/python scripts/confirm_synth.py build --out /tmp/conf/syn --align-run /tmp/conf/align
    ../.venv/bin/python scripts/confirm_synth.py score --out /tmp/conf/syn --run /tmp/conf/run/a0w-nopass ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB / "scripts"))


def _heldout_multi():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_heldout_multi", LAB / "benchmark" / "build_heldout_multi.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_hm = _heldout_multi()
DEV_RECITERS, SR, fetch_pcm = _hm.DEV_RECITERS, _hm.SR, _hm.fetch_pcm

from ayah_order import ident_guarded  # noqa: E402
from similar_verse import COST, Corpus, align_path, encode  # noqa: E402

XFADE_S = 0.02
MAX_WORDS = 25
MAX_WINDOW_S = 40.0
MAX_WINDOW_WORDS = 30
MAX_AYAH_S = 25.0


# ---------------------------------------------------------------- exclusions


def prompt_passages(help_manifest: Path, v1_manifest: Path, labels: Path) -> list[tuple[int, int, int]]:
    out = []
    for man in (help_manifest, v1_manifest):
        data = json.loads(man.read_text(encoding="utf-8"))
        rows = data["samples"] if isinstance(data, dict) else data
        rows = list(rows.values()) if isinstance(rows, dict) else rows
        for r in rows:
            vs = r.get("expected_verses") or []
            for s in {int(v["surah"]) for v in vs}:
                ay = [int(v["ayah"]) for v in vs if int(v["surah"]) == s]
                out.append((s, min(ay), max(ay)))
            for alt in r.get("also_accept") or []:
                for v in alt:
                    out.append((int(v["surah"]), int(v["ayah"]), int(v["ayah"])))
    if labels.is_file():
        for line in labels.read_text(encoding="utf-8").splitlines():
            lab = json.loads(line)
            if lab.get("surah") and lab.get("ayah"):
                out.append((int(lab["surah"]), int(lab["ayah"]), int(lab["ayah"])))
    return out


def overlaps(s: int, a0: int, a1: int, blocked: dict[int, list[tuple[int, int]]]) -> bool:
    """Detector windows [first-2 .. last+4] overlap (the k-fold grouping)."""
    lo, hi = a0 - 2, a1 + 4
    return any(not (hi < f - 2 or l + 4 < lo) for f, l in blocked.get(s, []))


def components(index: dict) -> list[list[tuple[int, int]]]:
    adj: dict[tuple[int, int], set] = defaultdict(set)
    for k, cands in index.items():
        a = tuple(int(x) for x in k.split(":"))
        for c in cands:
            b = tuple(c["m"])
            adj[a].add(b)
            adj[b].add(a)
    seen, comps = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


# ---------------------------------------------------------------- select


def select(out: Path, index_path: Path, help_manifest: Path, v1_manifest: Path, labels: Path,
           n_skip: int, n_fam: int, cache: Path) -> None:
    corpus = Corpus()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    prompts = prompt_passages(help_manifest, v1_manifest, labels)
    prompt_ayahs = {(s, a) for s, a0, a1 in prompts for a in range(a0, a1 + 1)}
    blocked: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s, a0, a1 in prompts:
        blocked[s].append((a0, a1))
    nwords = {k: len(v) for k, v in corpus.ph.items()}
    rng = random.Random(0)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"prompt_passages": len(set(prompts))}

    # (b) families first; their ayahs are then blocked for (a)
    comps = components(index)
    stats["components"] = len(comps)
    clean = [c for c in comps if not any(k in prompt_ayahs for k in c)]
    stats["components_without_prompt_ayah"] = len(clean)
    rng.shuffle(clean)
    families = []
    for comp in clean:
        if len(families) >= n_fam:
            break
        opts = []
        for e in comp:
            for c in index.get(f"{e[0]}:{e[1]}", []):
                m = tuple(c["m"])
                if not (2 <= nwords[e] <= MAX_WORDS and 2 <= nwords[m] <= MAX_WORDS):
                    continue
                if corpus.keys[e] == corpus.keys[m]:
                    continue
                for tag, i1, i2, j1, j2 in c["regions"]:
                    if tag == "replace" and 1 <= i2 - i1 <= 2 and 1 <= j2 - j1 <= 2:
                        opts.append({"e": list(e), "m": list(m), "region": [i1, i2, j1, j2]})
        if not opts:
            continue
        rng.shuffle(opts)
        k = len(families)
        recs = [DEV_RECITERS[(2 * k) % 6], DEV_RECITERS[(2 * k + 1) % 6]]
        pick = None
        for o in opts:
            ok = True
            for rec in recs:
                for key in (o["e"], o["m"]):
                    pcm = fetch_pcm(rec, key[0], key[1], cache)
                    if pcm is None or len(pcm) / SR > MAX_AYAH_S:
                        ok = False
            if ok:
                pick = o
                break
        if pick is None:
            continue
        families.append({"family": f"{comp[0][0]}:{comp[0][1]}", "comp_size": len(comp), **pick, "reciters": recs})
    for f in families:
        for key in (f["e"], f["m"]):
            blocked[key[0]].append((key[1], key[1]))
    stats["families"] = len(families)

    # (a) skip windows: 3-4 ayahs, an interior ayah cut, windows mutually disjoint
    cands = []
    for s in sorted({k[0] for k in corpus.ph}):
        n = max(a for (x, a) in corpus.ph if x == s)
        for a0 in range(2, n + 1):
            for size in (3, 4):
                a1 = a0 + size - 1
                if a1 > n:
                    continue
                if any(not (2 <= nwords[(s, a)] <= MAX_WORDS) for a in range(a0, a1 + 1)):
                    continue
                cands.append((s, a0, a1))
    rng.shuffle(cands)
    windows: list[dict] = []
    for s, a0, a1 in cands:
        if len(windows) >= n_skip:
            break
        if overlaps(s, a0, a1, blocked):
            continue
        cut = a0 + 1 if a1 - a0 == 2 else rng.choice((a0 + 1, a0 + 2))
        if ident_guarded(corpus, s, cut):
            stats["skip_ident_excluded"] = stats.get("skip_ident_excluded", 0) + 1
            continue
        if sum(nwords[(s, a)] for a in range(a0, a1 + 1)) > MAX_WINDOW_WORDS:
            continue
        rec = None
        for r in range(6):  # rotate; a slow reciter falls through to the next one
            cand = DEV_RECITERS[(len(windows) + r) % 6]
            pcms = [fetch_pcm(cand, s, a, cache) for a in range(a0, a1 + 1)]
            if all(p is not None for p in pcms) and sum(len(p) for p in pcms) / SR <= MAX_WINDOW_S:
                rec = cand
                break
        if rec is None:
            continue
        windows.append({"surah": s, "a0": a0, "a1": a1, "cut": cut, "reciter": rec})
        blocked[s].append((a0, a1))
    stats["skip_windows"] = len(windows)
    plan = {"families": families, "windows": windows, "stats": stats}
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    # clean single-ayah takes of E and M, decoded once for word boundaries
    aud = out / "audio"
    aud.mkdir(exist_ok=True)
    import soundfile as sf

    with (out / "align.jsonl").open("w", encoding="utf-8") as fh:
        for f in families:
            for rec in f["reciters"]:
                for key in (f["e"], f["m"]):
                    cid = f"al_{rec.split('_')[0].lower()}_{key[0]:03d}_{key[1]:03d}"
                    path = aud / f"{cid}.wav"
                    if not path.is_file():
                        sf.write(path, fetch_pcm(rec, key[0], key[1], cache), SR, subtype="PCM_16")
                    fh.write(json.dumps({"id": cid, "audio": str(path), "split": None, "speaker": rec,
                                         "expected_verses": [{"surah": key[0], "ayah": key[1]}]}) + "\n")
    print(json.dumps(stats))


# ---------------------------------------------------------------- build


def join(parts: list[np.ndarray]) -> np.ndarray:
    n = int(XFADE_S * SR)
    out = parts[0].astype(np.float32)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for p in parts[1:]:
        p = p.astype(np.float32)
        if len(out) < n or len(p) < n:
            out = np.concatenate([out, p])
            continue
        mid = out[-n:] * (1 - ramp) + p[:n] * ramp
        out = np.concatenate([out[:-n], mid, p[n:]])
    return out


def rms_curve(x: np.ndarray, hop: int = 160, win: int = 400) -> np.ndarray:
    if len(x) < win:
        return np.array([float(np.sqrt(np.mean(x ** 2) + 1e-12))])
    idx = np.arange(0, len(x) - win, hop)
    return np.array([float(np.sqrt(np.mean(x[i:i + win] ** 2) + 1e-12)) for i in idx])


def onset_s(x: np.ndarray) -> float:
    r = rms_curve(x)
    thr = 0.1 * float(np.percentile(r, 95))
    above = np.nonzero(r > thr)[0]
    return float(above[0]) * 0.01 if len(above) else 0.0


def word_frames(corpus: Corpus, key: tuple[int, int], tokens: list) -> list[tuple[float, float]] | None:
    """(first, last) aligned char time per word, from the free-decode token frames."""
    toks = sorted(tokens, key=lambda t: t[1])
    chars, frames = [], []
    for sym, frame, _t in toks:
        for c in sym:
            chars.append(c)
            frames.append(int(frame))
    if len(chars) < 4:
        return None
    words = corpus.ph[key]
    ref = "".join(words)
    last, _rc = align_path(encode("".join(chars)), encode(ref), COST, 0.5)
    out, pos = [], 0
    for w in words:
        idx = [int(last[j]) for j in range(pos, pos + len(w)) if int(last[j]) >= 0]
        pos += len(w)
        if not idx:
            out.append((float("nan"), float("nan")))
            continue
        out.append((frames[min(idx)] * 0.04, frames[max(idx)] * 0.04))
    return out


def cut_point(x: np.ndarray, t_end_prev: float, t_start_next: float) -> int:
    """Lowest-energy point between the previous word's last char and the next word's first char."""
    lo, hi = min(t_end_prev, t_start_next) - 0.04, max(t_end_prev, t_start_next) + 0.04
    if hi - lo < 0.12:
        c = (lo + hi) / 2
        lo, hi = c - 0.06, c + 0.06
    a, b = max(0, int(lo * SR)), min(len(x) - 1, int(hi * SR))
    if b - a < 400:
        return max(0, min(len(x) - 1, int((lo + hi) / 2 * SR)))
    r = rms_curve(x[a:b], hop=80, win=240)
    return a + int(np.argmin(r)) * 80 + 120


def boundary(x: np.ndarray, wf: list[tuple[float, float]], i: int, delay: float) -> int | None:
    """Sample index of the boundary before word ``i`` (0 = start, len = end)."""
    if i <= 0:
        return 0
    if i >= len(wf):
        return len(x)
    prev, nxt = wf[i - 1][1], wf[i][0]
    if np.isnan(prev) or np.isnan(nxt):
        return None
    return cut_point(x, prev - delay, nxt - delay)


def build(out: Path, align_run: Path, cache: Path, upload: bool) -> None:
    import soundfile as sf

    corpus = Corpus()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    rows = {}
    for line in (align_run / "align.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("error"):
            rows[r["id"]] = r
    aud = out / "audio"
    pcm = {}
    delays = []
    for cid, r in rows.items():
        x, _ = sf.read(str(aud / f"{cid}.wav"), dtype="float32")
        pcm[cid] = x
        if r.get("tokens"):
            delays.append(min(t[1] for t in r["tokens"]) * 0.04 - onset_s(x))
    delay = float(np.median(delays)) if delays else 0.0
    sets: dict[str, list[dict]] = {k: [] for k in ("skip", "skip-ctl", "sister", "sister-ctl")}
    meta: dict[str, dict] = {}
    syn = out / "clips"
    syn.mkdir(exist_ok=True)
    n = 0

    def emit(name: str, x: np.ndarray, expected: list[tuple[int, int]], speaker: str, info: dict) -> None:
        nonlocal n
        cid = f"cf_{n:04d}"
        n += 1
        path = syn / f"{cid}.wav"
        sf.write(path, x, SR, subtype="PCM_16")
        sets[name].append({"id": cid, "audio": str(path), "split": None, "speaker": speaker,
                           "duration_s": round(len(x) / SR, 3),
                           "expected_verses": [{"surah": s, "ayah": a} for s, a in expected]})
        meta[cid] = {"set": name, **info}

    skipped_b = 0
    for f in plan["families"]:
        e, m = tuple(f["e"]), tuple(f["m"])
        i1, i2, j1, j2 = f["region"]
        for rec in f["reciters"]:
            tag = rec.split("_")[0].lower()
            ce, cm = f"al_{tag}_{e[0]:03d}_{e[1]:03d}", f"al_{tag}_{m[0]:03d}_{m[1]:03d}"
            if ce not in rows or cm not in rows:
                skipped_b += 1
                continue
            we = word_frames(corpus, e, rows[ce].get("tokens") or [])
            wm = word_frames(corpus, m, rows[cm].get("tokens") or [])
            if we is None or wm is None:
                skipped_b += 1
                continue
            xe, xm = pcm[ce], pcm[cm]
            b = [boundary(xe, we, i1, delay), boundary(xe, we, i2, delay),
                 boundary(xm, wm, j1, delay), boundary(xm, wm, j2, delay)]
            if any(v is None for v in b) or b[1] <= b[0] or b[3] <= b[2]:
                skipped_b += 1
                continue
            parts = [p for p in (xe[: b[0]], xm[b[2]: b[3]], xe[b[1]:]) if len(p)]
            info = {"family": f["family"], "surah": e[0], "ayah": e[1], "word": i1, "span": i2 - i1,
                    "lookalike": list(m), "spliced_s": round((b[3] - b[2]) / SR, 2)}
            emit("sister", join(parts), [e], rec, info)
            emit("sister-ctl", xe, [e], rec, {"family": f["family"], "surah": e[0], "ayah": e[1]})
    for w in plan["windows"]:
        s, a0, a1, cut, rec = w["surah"], w["a0"], w["a1"], w["cut"], w["reciter"]
        parts = {a: fetch_pcm(rec, s, a, cache) for a in range(a0, a1 + 1)}
        exp = [(s, a) for a in range(a0, a1 + 1)]
        grp = f"{s}:{a0}"
        emit("skip", join([parts[a] for a in range(a0, a1 + 1) if a != cut]), exp, rec,
             {"group": grp, "surah": s, "ayah": cut, "n_ayahs": a1 - a0 + 1})
        emit("skip-ctl", join([parts[a] for a in range(a0, a1 + 1)]), exp, rec,
             {"group": grp, "surah": s, "a0": a0, "a1": a1})
    for name, clips in sets.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for c in clips:
                fh.write(json.dumps(c) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    spliced = [v["spliced_s"] for v in meta.values() if v["set"] == "sister"]
    print(json.dumps({"delay_s": round(delay, 3), **{k: len(v) for k, v in sets.items()},
                      "sister_skipped": skipped_b,
                      "sister_spliced_median_s": round(float(np.median(spliced)), 2) if spliced else None,
                      "minutes": {k: round(sum(c["duration_s"] for c in v) / 60, 1) for k, v in sets.items()}}))
    if upload:
        import subprocess

        subprocess.run([str(LAB / ".venv" / "bin" / "modal"), "volume", "put", "--force", "zipformer-ctc-training",
                        str(out) + "/", "synthetic/confirmatory/"], check=True)


# ---------------------------------------------------------------- score


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def cluster_ci(hits_by_group: dict[str, list[int]], reps: int = 2000) -> tuple[float, float]:
    g = list(hits_by_group.values())
    if not g:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(g), len(g))
        k = sum(sum(g[i]) for i in pick)
        n = sum(len(g[i]) for i in pick)
        vals.append(k / max(1, n))
    return (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3))


ERR = {"possible_omission", "possible_substitution", "possible_vowel", "possible_skipped_ayah", "unclear_ayah"}


def stack(run: Path, sv: Path, ao: Path, name: str, sv_all: bool, ao_rule: dict) -> list[dict]:
    from ayah_order_stress import stack_rows

    return stack_rows(run, sv, ao, name, sv_all, ao_rule)


def score(out: Path, run: Path, sv: Path, ao: Path, sv_all: bool, ao_rule: dict) -> dict:
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    res: dict = {}
    for fam, pos, ctl, src in (("skip", "skip", "skip-ctl", "ayah_order"), ("sister", "sister", "sister-ctl",
                                                                             "similar_verse")):
        rows = stack(run, sv, ao, pos, sv_all, ao_rule)
        groups: dict[str, list[int]] = defaultdict(list)
        eng_groups: dict[str, list[int]] = defaultdict(list)
        span: dict[int, list[int]] = defaultdict(list)
        new_flags = new_ok = 0
        for r in rows:
            m = meta[str(r["id"])]
            g = m.get("group") or m.get("family")
            if fam == "skip":
                ok = lambda i: int(i.get("surah", 0)) == m["surah"] and int(i.get("ayah", 0)) == m["ayah"]  # noqa
            else:
                ok = lambda i: (int(i.get("surah", 0)) == m["surah"] and int(i.get("ayah", 0)) == m["ayah"]  # noqa
                                and m["word"] - 1 <= int(i.get("word", -9)) <= m["word"] + m["span"])
            errs = [i for i in r["issues"] if i.get("kind") in ERR]
            hit = any(ok(i) for i in errs)
            eng = any(ok(i) for i in errs if i.get("source") not in ("ayah_order", "similar_verse"))
            groups[g].append(int(hit))
            eng_groups[g].append(int(eng))
            if fam == "sister":
                span[m["span"]].append(int(hit))
            for i in errs:
                if i.get("source") in ("ayah_order", "similar_verse"):
                    new_flags += 1
                    new_ok += int(ok(i))
        k = sum(sum(v) for v in groups.values())
        n = sum(len(v) for v in groups.values())
        ke = sum(sum(v) for v in eng_groups.values())
        crow = stack(run, sv, ao, ctl, sv_all, ao_rule)
        c_rule = sum(1 for r in crow for i in r["issues"] if i.get("kind") in ERR
                     and i.get("source") in ("ayah_order", "similar_verse"))
        c_eng = sum(1 for r in crow for i in r["issues"] if i.get("kind") in ERR
                    and i.get("source") not in ("ayah_order", "similar_verse"))
        c_by = defaultdict(int)
        for r in crow:
            for i in r["issues"]:
                if i.get("kind") in ERR:
                    c_by[i.get("source") or "engine"] += 1
        res[fam] = {"recall": f"{k}/{n}", "rate": round(k / max(1, n), 3), "wilson": wilson(k, n),
                    "cluster_ci": cluster_ci(groups), "groups": len(groups), "engine_only": f"{ke}/{n}",
                    "new_flags_correct": f"{new_ok}/{new_flags}",
                    "controls": len(crow), "ctl_rule_flags": c_rule, "ctl_engine_flags": c_eng,
                    "ctl_rule_per100": round(100 * c_rule / max(1, len(crow)), 2),
                    "ctl_rule_per100_ci": [round(100 * x, 2) for x in wilson(c_rule, len(crow))],
                    "ctl_by_source": dict(c_by),
                    "errors": sum(1 for _ in rows) - n}
        if span:
            res[fam]["by_span"] = {str(s): f"{sum(v)}/{len(v)}" for s, v in sorted(span.items())}
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "build", "score"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--index", type=Path, default=Path("/tmp/conf/sv_index.json"))
    ap.add_argument("--help-manifest", type=Path, default=Path("/tmp/help/manifest.json"))
    ap.add_argument("--v1-manifest", type=Path, default=LAB / "benchmark" / "test_corpus" / "manifest.json")
    ap.add_argument("--labels", type=Path, default=Path("/tmp/correction_eval/acted_located.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/conf/mp3"))
    ap.add_argument("--n-skip", type=int, default=120)
    ap.add_argument("--n-fam", type=int, default=50)
    ap.add_argument("--align-run", type=Path)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--run", type=Path)
    ap.add_argument("--sv", type=Path)
    ap.add_argument("--ao", type=Path)
    ap.add_argument("--sv-all", action="store_true")
    ap.add_argument("--ao-rule", help="JSON rule; default ayah_order.RULE")
    args = ap.parse_args(argv)
    if str(args.out.resolve()).startswith(str(LAB.parent.resolve())):
        raise SystemExit("--out must be outside the repo")
    if args.cmd == "select":
        select(args.out, args.index, args.help_manifest, args.v1_manifest, args.labels, args.n_skip, args.n_fam,
               args.cache)
    elif args.cmd == "build":
        build(args.out, args.align_run, args.cache, args.upload)
    else:
        from ayah_order import RULE

        rule = json.loads(args.ao_rule) if args.ao_rule else RULE
        print(json.dumps(score(args.out, args.run, args.sv, args.ao, args.sv_all, rule), indent=1))


if __name__ == "__main__":
    main()
