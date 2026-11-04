"""What builders may run on Modal and see of the test data (.scratch/autolab/spec.md decision 14). Pure functions:
modal/autolab_exec.py and tools/autolab.py do the side effects.

    request(req, tier, cfg)        a builder's modal-run request, checked and filled in (ValueError: refused, and why)
    cost(q, seconds) / reserve(q)  GPU dollars of a run, and the most it can cost (its timeout)
    model_path(doc, exp)           .autolab/model.json's path, relative to the experiment's scratch area
    copy_practice(src, stage, spec)   the practice-only copy in tilawa-lab-exec (practice_manifest, practice_files,
                                      only_practice; exam_pattern checks the result)
"""

import json
import posixpath
import re
import shutil
from pathlib import Path

# Modal's list prices per GPU-hour (modal.com/pricing); config exec.gpuUsdPerHour overrides.
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100": 2.10, "A100-80GB": 2.50, "H100": 3.95,
                    "H200": 4.54, "B200": 6.25}
IMAGES = ("node", "train")
LIMITS = {"concurrent": 2, "maxShards": 20, "maxGpuShards": 4, "maxMinutes": 60, "betMaxMinutes": 24 * 60}
DROP = ("summary", "totals")  # aggregates over every session, exam included


def limits(cfg: dict) -> dict:
    return {**LIMITS, **cfg.get("exec", {})}


def request(req: dict, tier: str, cfg: dict) -> dict:
    lim = limits(cfg)
    rates = {**GPU_USD_PER_HOUR, **lim.get("gpuUsdPerHour", {})}
    cmd = req.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("no command")
    gpu = str(req.get("gpu") or "").upper() or None
    if gpu and tier != "bet":
        raise ValueError(f"GPUs are for bet cards only; this card is {tier}")
    if gpu and gpu not in rates:
        raise ValueError(f"unknown GPU {gpu}; one of {', '.join(rates)}")
    shards, most = int(req.get("shards") or 1), lim["maxGpuShards"] if gpu else lim["maxShards"]
    if not 1 <= shards <= most:
        raise ValueError(f"--shards must be 1 to {most}")
    minutes, top = int(req.get("timeout") or 30), lim["betMaxMinutes"] if tier == "bet" else lim["maxMinutes"]
    if not 1 <= minutes <= top:
        raise ValueError(f"--timeout must be 1 to {top} minutes")
    image = req.get("image") or ("train" if gpu else "node")
    if image not in IMAGES:
        raise ValueError(f"--image must be one of {', '.join(IMAGES)}")
    return {"cmd": cmd, "gpu": gpu, "shards": shards, "timeout": minutes, "image": image, "usdPerHour": rates[gpu] if gpu else 0.0}


def cost(q: dict, seconds: float) -> float:
    return round(q["shards"] * q["usdPerHour"] * seconds / 3600, 4)


def reserve(q: dict) -> float:
    return cost(q, q["timeout"] * 60)


def model_path(doc: dict, exp: str) -> str:
    p = re.sub(rf"^/scratch/{re.escape(exp)}(/|$)", "", str(doc.get("path") or "").strip())
    norm = posixpath.normpath(p) if p else ""
    if not norm or norm == "." or norm.startswith(("/", "..")):
        raise ValueError(f".autolab/model.json path {doc.get('path')!r} must be a directory under /scratch/{exp}")
    return norm


def practice_manifest(m: dict, practice: list[str]) -> dict:
    keep = set(practice)
    sessions = [s for s in m["sessions"] if s["id"] in keep]
    if bad := [s["id"] for s in sessions if s["split"] != "practice"]:
        raise ValueError(f"the ledger's practice sessions are not practice in the manifest: {bad[:5]}")
    if missing := sorted(keep - {s["id"] for s in sessions}):
        raise ValueError(f"practice sessions missing from the manifest: {missing[:5]}")
    return {**m, "sessions": sessions}


def practice_files(m: dict, listeners: list[str]) -> list[str]:
    """What practice needs, relative to the test set's directory: labels, audio, per-session checks, and the evidence of
    `listeners`."""
    out = set()
    for s in m["sessions"]:
        out.update(p for p in (s.get("label"), s.get("session") if isinstance(s.get("session"), str) else None,
                               *(s.get("audio") or {}).values()) if p)
        out.add(f"sessions/{s['id']}.checks.json")
        for li in listeners:
            for v in m.get("variants") or ("phone", "studio"):
                out.add(m["evidencePattern"].replace("{id}", s["id"]).replace("{listener}", li).replace("{variant}", v))
    return sorted(out)


def only_practice(x, practice: set, sessions: set):
    """A test set document (flags, checks, decode record) without anything about a session outside `practice`: list
    items naming one (`session` or `id`), keys `<session>.<variant>`, and the all-session aggregates."""
    def other(sid):
        return sid in sessions and sid not in practice

    if isinstance(x, list):
        return [only_practice(i, practice, sessions) for i in x
                if not (isinstance(i, dict) and other(i.get("session") if isinstance(i.get("session"), str) else i.get("id")))]
    if isinstance(x, dict):
        return {k: only_practice(v, practice, sessions) for k, v in x.items()
                if k not in DROP and not other(k) and not other(k.rsplit(".", 1)[0])}
    return x


def exam_pattern(exam: list[str]) -> re.Pattern:
    """Matches any exam session id (not as the prefix of a longer id)."""
    return re.compile("|".join(re.escape(i) + r"(?![0-9])" for i in sorted(exam, key=len, reverse=True)) or r"(?!x)x")


HELP_DIR = "help-train"


def copy_help_train(src: Path, stage: Path) -> dict:
    """The train-speaker half of the help-page recordings (tilawa-help: labels.jsonl + wav/, real learners reading with
    scripted mistakes) into `stage`/help-train, for training only. The eval half is the release benchmark's: raises,
    leaving `stage` to be thrown away, if any eval clip or eval speaker's clip would be copied."""
    rows = [json.loads(x) for x in (src / "labels.jsonl").read_text().splitlines() if x.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    held = {r["speaker"] for r in rows if r.get("split") != "train"}
    if leak := sorted({r["speaker"] for r in train} & held):
        raise ValueError(f"help speakers in both splits: {leak[:5]}")
    out, nbytes = stage / HELP_DIR, 0
    (out / "wav").mkdir(parents=True, exist_ok=True)
    for r in train:
        wav = Path(posixpath.normpath(r["wav"]))
        if wav.is_absolute() or wav.parts[0] != "wav":
            raise ValueError(f"{r['id']}: wav {r['wav']} is outside wav/")
        shutil.copyfile(src / wav, out / wav)
        nbytes += (out / wav).stat().st_size
    (out / "labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train))
    if stray := {r["id"] for r in rows if r.get("split") != "train"} & {f.stem for f in (out / "wav").iterdir()}:
        raise ValueError(f"eval help clips in the copy: {sorted(stray)[:5]}")
    return {"clips": len(train), "speakers": len({r["speaker"] for r in train}), "bytes": nbytes}


def copy_practice(src: Path, stage: Path, spec: dict) -> dict:
    """Copies from `src` (a /lab/data) into `stage` what practice needs of one test set: {manifest, practice, listeners,
    models}. Raises, leaving `stage` to be thrown away, when any file name or JSON text names an exam session."""
    full = json.loads((src / spec["manifest"]).read_text())
    sessions = {s["id"] for s in full["sessions"]}
    m = practice_manifest(full, spec["practice"])
    practice = {s["id"] for s in m["sessions"]}
    base = Path(spec["manifest"]).parent
    n, nbytes, missing = 0, 0, []

    def put(rel: Path, data: bytes | None = None, frm: Path | None = None):
        nonlocal n, nbytes
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data) if data is not None else shutil.copyfile(frm, p)
        n, nbytes = n + 1, nbytes + p.stat().st_size

    for rel in practice_files(m, spec["listeners"]):
        p = Path(posixpath.normpath(base / rel))
        if p.is_absolute() or p.parts[0] == "..":
            raise ValueError(f"{rel} is outside the data directory")
        put(p, frm=src / p) if (src / p).is_file() else missing.append(str(p))
    put(Path(spec["manifest"]), json.dumps(m, ensure_ascii=False, indent=1).encode())
    for f in sorted((src / base).glob("*.json")):
        if f.name.startswith(("flags-", "checks-", "decode-")):
            put(base / f.name, json.dumps(only_practice(json.loads(f.read_text()), practice, sessions), ensure_ascii=False, indent=1).encode())
    for model in spec["models"]:
        for f in sorted((src / model).rglob("*")):
            if f.is_file() and not f.is_symlink():
                put(f.relative_to(src), frm=f)
    exam = exam_pattern(sorted(sessions - practice))
    for f in stage.rglob("*"):
        if exam.search(f.name) or (f.suffix == ".json" and exam.search(f.read_text(errors="ignore"))):
            raise ValueError(f"exam session data in the practice copy: {f.relative_to(stage)}")
    return {"testSet": full["testSet"], "sessions": len(practice), "files": n, "bytes": nbytes, "missing": missing[:50]}
