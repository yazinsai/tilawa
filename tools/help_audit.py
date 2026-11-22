# /// script
# requires-python = ">=3.12"
# dependencies = ["rapidfuzz==3.14.1"]
# ///
"""Human spot-check of the tilawa.dev help-page clips before they become eval labels.

    uv run tools/help_audit.py pull      # rsync submissions + audio, convert to 16 kHz WAV
    modal run modal/help_pass.py         # optional: teacher decode -> .scratch/help/teacher.json
    uv run tools/help_audit.py check     # teacher decode vs expected passage -> passage.json
    uv run tools/help_audit.py sample    # risky strata in full-ish, plus random per kind; appends on re-run
    uv run tools/help_audit.py serve     # http://localhost:8765, answers -> .scratch/help/answers.jsonl
    uv run tools/help_audit.py report    # docs/benchmarks/help-audit-v1.md

Keys on the page: 1 label right, 2 mistake not made, 3 other or extra mistake, 4 wrong or incomplete
passage, 5 audio unusable, 0 can't tell; j/k next/prev, space play, n note.
"""

import http.server
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / ".scratch/help"
RAW, WAV = DIR / "raw", DIR / "wav"
REMOTE = "root@dokku-server:/var/lib/dokku/data/storage/offline-tarteel/help/"
QURAN = ROOT / "web/frontend/public/quran.json"
REPORT = ROOT / "docs/benchmarks/help-audit-v1.md"
INDEX = ROOT / "packages/core/assets/ref-v2.json"
# Flag a clip when another passage beats the expected one by this much phoneme error rate.
PASSAGE_MARGIN = 0.15
# ...and only when that other passage really fits; otherwise the decode is just garbled.
BEST_MAX_PER = 0.25
PORT = 8765
SEED = 12
RANDOM_PER_KIND = 8
RISKY_CAP = 25
ARABIC_LETTER = re.compile(r"[\u0621-\u064a\u0671]")

VERDICTS = {
    "1": "label right",
    "2": "mistake not made",
    "3": "other or extra mistake",
    "4": "wrong or incomplete passage",
    "5": "audio unusable",
    "0": "can't tell",
}


def records() -> list[dict]:
    return [json.loads(line) for line in (RAW / "submissions.jsonl").open()]


def probe(path: Path) -> float | None:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def pull() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    WAV.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", f"{REMOTE}submissions.jsonl", f"{REMOTE}audio", str(RAW)], check=True)
    durations = {}
    for r in records():
        src, dst = RAW / r["file"], WAV / f"{r['id']}.wav"
        if not dst.exists():
            res = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)], capture_output=True)
            if res.returncode:
                print(f"convert failed: {r['id']}: {res.stderr.decode()[:200]}")
                continue
        durations[r["id"]] = probe(dst)
    (DIR / "durations.json").write_text(json.dumps(durations))
    print(f"{len(durations)} clips converted under {WAV}")


def quran() -> dict[str, dict]:
    return {f"{a['surah']}:{a['ayah']}": a for a in json.loads(QURAN.read_text(encoding="utf-8-sig"))}


def ayahs_of(r: dict) -> list[str]:
    if r.get("ayahs"):
        return r["ayahs"]
    ref = r["ref"]
    if "-" not in ref:
        return [ref]
    (s1, a1), (s2, a2) = (map(int, x.split(":")) for x in ref.split("-"))
    assert s1 == s2, ref
    return [f"{s1}:{a}" for a in range(a1, a2 + 1)]


def words(text: str) -> list[str]:
    """Words with standalone waqf marks dropped, matching the help page's 1-based indices."""
    return [w for w in text.replace("\ufeff", "").split() if ARABIC_LETTER.search(w)]


def check() -> None:
    """Score each clip's teacher decode against its expected passage and against every same-length run of ayahs."""
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    hyps = json.loads((DIR / "teacher.json").read_text())
    ayahs = json.loads(INDEX.read_text())["ayahs"]
    refs = [f"{a['s']}:{a['a']}" for a in ayahs]
    phon = {r: a["p"] for r, a in zip(refs, ayahs)}
    windows: dict[int, tuple[list[str], list[str]]] = {}
    out = {}
    for r in records():
        hyp = hyps.get(r["id"], "")
        exp = ayahs_of(r)
        if r["kind"] == "skip_ayah":
            exp = [a for a in exp if a not in {m["ayah"] for m in r["mistakes"]}]
        n = len(exp)
        if n not in windows:
            keys = [refs[i : i + n] for i in range(len(refs) - n + 1)]
            windows[n] = (["-".join(k) for k in keys], ["".join(phon[x] for x in k) for k in keys])
        names, texts = windows[n]
        per_exp = Levenshtein.normalized_distance(hyp, "".join(phon[a] for a in exp))
        _, d, i = process.extractOne(hyp, texts, scorer=Levenshtein.normalized_distance)
        best = names[i]
        borrowed = {m["said_from"].rsplit(":", 1)[0] for m in r["mistakes"] if m.get("said_from")}
        wrong = per_exp - d > PASSAGE_MARGIN and d <= BEST_MAX_PER and best not in borrowed
        out[r["id"]] = {"hyp": hyp, "per_expected": round(per_exp, 3), "best": best, "per_best": round(d, 3), "wrong_passage": wrong}
    (DIR / "passage.json").write_text(json.dumps(out, ensure_ascii=False))
    bad = [k for k, v in out.items() if v["wrong_passage"]]
    print(f"{len(bad)}/{len(out)} clips decode closer to another passage")
    by_v = Counter(next(r["prompt_version"] for r in records() if r["id"] == k) for k in bad)
    print("by prompt version:", dict(by_v))


def risks(r: dict, q: dict, dur: float | None, passage: dict) -> list[str]:
    out = []
    p = passage.get(r["id"])
    if p and p["wrong_passage"]:
        out.append("teacher: other passage")
    elif p and p["per_expected"] > 0.5:
        out.append("teacher: poor match")
    if r.get("extra_mistake"):
        out.append("extra_mistake flagged")
    if r["prompt_version"] == 1 and r["kind"] != "none":
        out.append("v1 random mistake")
    for m in r.get("mistakes") or []:
        if m.get("rule") in ("letter_confusion", "light_emphatic", "no_shadda", "no_ghunna", "madd_short"):
            out.append(f"rare: {m['rule']}")
    if r["mime"] != "audio/wav":
        out.append("browser codec")
    n_words = sum(len(words(q[a]["text_clean"])) for a in ayahs_of(r) if a in q)
    if dur is None:
        out.append("convert failed")
    else:
        if r.get("duration_s") and abs(dur - r["duration_s"]) > 1.0:
            out.append("duration differs from client")
        if n_words and not 0.25 <= dur / n_words <= 4.0:
            out.append("seconds per word out of range")
    return out


def sample() -> None:
    q = quran()
    durations = json.loads((DIR / "durations.json").read_text())
    passage_path = DIR / "passage.json"
    passage = json.loads(passage_path.read_text()) if passage_path.exists() else {}
    rng = random.Random(SEED)
    rows = records()
    picked: dict[str, list[str]] = {}
    by_risk = defaultdict(list)
    for r in rows:
        for k in risks(r, q, durations.get(r["id"]), passage):
            by_risk[k].append(r["id"])
    for k, ids in sorted(by_risk.items()):
        cap = len(ids) if k.startswith("teacher") else RISKY_CAP
        for i in rng.sample(ids, min(cap, len(ids))):
            picked.setdefault(i, []).append(k)
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["id"])
    for k, ids in by_kind.items():
        for i in rng.sample(ids, min(RANDOM_PER_KIND, len(ids))):
            picked.setdefault(i, []).append(f"random: {k}")
    swap_path = DIR / "version_swap.json"
    swaps = json.loads(swap_path.read_text()) if swap_path.exists() else {}
    for i in swaps:
        picked.setdefault(i, []).append("v1 stored, v2 shown")
    index = {r["id"]: r for r in rows}
    items = []
    for i, why in picked.items():
        r = index[i]
        stored = None
        if i in swaps:
            stored = {"ref": r["ref"], "kind": r["kind"], "mistakes": r.get("mistakes") or []}
            s = swaps[i]
            r = {**r, "ref": s["ref"], "kind": s["kind"], "mistakes": s.get("mistakes") or [], "scenario": s.get("scenario"), "prompt_version": 2}
        items.append(
            {
                "id": i,
                "why": why,
                "kind": r["kind"],
                "scenario": r.get("scenario"),
                "prompt_version": r["prompt_version"],
                "level": r.get("level"),
                "extra_mistake": r.get("extra_mistake"),
                "duration": durations.get(i),
                "mistakes": r.get("mistakes") or [],
                "ayahs": [{"ref": a, "words": words(q[a]["text_uthmani"])} for a in ayahs_of(r) if a in q],
                "teacher": passage.get(i),
                "stored": stored,
            }
        )
    rng.shuffle(items)
    # Keep an existing sample's order so answers in progress stay where they were; new picks go on the end.
    path = DIR / "sample.json"
    if path.exists():
        old = {it["id"]: n for n, it in enumerate(json.loads(path.read_text()))}
        items.sort(key=lambda it: old.get(it["id"], len(old)))
    path.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"{len(items)} clips sampled")
    for k, v in sorted(Counter(w for it in items for w in it["why"]).items()):
        print(f"  {v:4} {k}")


PAGE = r"""<!doctype html><meta charset="utf-8"><title>help audit</title>
<style>
body{font:15px system-ui;margin:2rem auto;max-width:900px;color:#222}
.text{font:28px/2 "Amiri","Scheherazade New",serif;direction:rtl;background:#fafafa;padding:1rem;border-radius:8px}
.ref{font:12px system-ui;color:#999;margin-inline-start:.5rem}
.hit{background:#ffe08a;border-radius:4px;padding:0 .2rem}
.skip{background:#ffc9c9;text-decoration:line-through}
.task{font-size:18px;margin:1rem 0;padding:.6rem 1rem;border-left:4px solid #e8a400;background:#fff8e6}
.task b{font:22px "Amiri",serif}
.meta{color:#777;font-size:13px}
.v{display:inline-block;margin:.2rem;padding:.3rem .6rem;border:1px solid #ccc;border-radius:6px;cursor:pointer}
.v.on{background:#222;color:#fff}
audio{width:100%;margin:1rem 0}
input{width:100%;padding:.4rem;font-size:15px}
</style>
<div class="meta" id="pos"></div>
<div class="task" id="task"></div>
<div class="text" id="text"></div>
<audio id="au" controls></audio>
<div id="vs"></div>
<p><input id="note" placeholder="note (n to focus, enter to save)"></p>
<div class="meta" id="meta"></div>
<script>
const V=__VERDICTS__;
let items=[],answers={},i=0;
const $=id=>document.getElementById(id);
function task(it){
  if(it.kind==="none")return "Correct recitation expected.";
  return it.mistakes.map(m=>{
    const w=`<b>${m.expected||""}</b>`;
    if(m.kind==="vowel")return `Vowel: ${w} with ${m.from} → ${m.to}${m.said_uthmani?` (said <b>${m.said_uthmani}</b>)`:""}`;
    if(m.kind==="substitution")return `Substitution: ${w} → <b>${m.said_uthmani||m.said}</b>${m.rule?` (${m.from_letter}→${m.to_letter})`:""}`;
    if(m.kind==="skip_word")return `Skip word ${w}`;
    if(m.kind==="repeat")return `Repeat word ${w}`;
    if(m.kind==="skip_ayah")return `Skip ayah ${m.ayah}`;
    if(m.kind==="tajweed")return `Tajweed ${m.rule}: ${w}${m.said_uthmani?` (said <b>${m.said_uthmani}</b>)`:""}`;
    return JSON.stringify(m);
  }).join("<br>");
}
function render(){
  const it=items[i],a=answers[it.id]||{};
  $("pos").textContent=`${i+1}/${items.length} · ${Object.keys(answers).length} answered · ${it.id}`;
  $("task").innerHTML=task(it)+(it.stored?`<div class="meta">Shown on screen (v2 from the same seed). Stored label was ${it.stored.ref}: ${task(it.stored)}</div>`:"");
  $("text").innerHTML=it.ayahs.map(ay=>{
    const ms=it.mistakes.filter(m=>m.ayah===ay.ref);
    if(ms.some(m=>m.kind==="skip_ayah"))return `<span class="skip">${ay.words.join(" ")}</span><span class="ref">${ay.ref}</span>`;
    return ay.words.map((w,k)=>{
      const m=ms.find(m=>m.word===k+1);
      return m?`<span class="hit ${m.kind==="skip_word"?"skip":""}">${w}</span>`:w;
    }).join(" ")+`<span class="ref">${ay.ref}</span>`;
  }).join(" ");
  $("au").src=`/wav/${it.id}.wav`;
  $("vs").innerHTML=Object.entries(V).map(([k,t])=>`<span class="v ${a.verdict===k?"on":""}" onclick="pick('${k}')">${k} ${t}</span>`).join("");
  $("note").value=a.note||"";
  const t=it.teacher;
  $("meta").textContent=`why: ${it.why.join(", ")} · v${it.prompt_version} · ${it.scenario||"-"} · level ${it.level||"-"} · ${it.duration?.toFixed(1)}s`
    +(t?` · teacher: expected PER ${t.per_expected}, best ${t.best} PER ${t.per_best}`:"");
}
async function save(patch){
  const it=items[i];answers[it.id]={...(answers[it.id]||{}),...patch,id:it.id,why:it.why,kind:it.kind};
  await fetch("/api/answer",{method:"POST",body:JSON.stringify(answers[it.id])});
}
async function pick(k){await save({verdict:k});if(i<items.length-1)i++;render();$("au").play()}
document.addEventListener("keydown",e=>{
  if(e.target===$("note")){if(e.key==="Enter"){save({note:$("note").value});$("note").blur()}return}
  if(V[e.key])pick(e.key);
  else if(e.key==="j"&&i<items.length-1){i++;render()}
  else if(e.key==="k"&&i>0){i--;render()}
  else if(e.key===" "){e.preventDefault();$("au").paused?$("au").play():$("au").pause()}
  else if(e.key==="n"){e.preventDefault();$("note").focus()}
});
(async()=>{
  items=await (await fetch("/sample.json")).json();
  answers=await (await fetch("/api/answers")).json();
  i=Math.max(0,items.findIndex(it=>!answers[it.id]));render();
})();
</script>"""


def load_answers() -> dict[str, dict]:
    path = DIR / "answers.jsonl"
    out = {}
    if path.exists():
        for line in path.open():
            a = json.loads(line)
            out[a["id"]] = a
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DIR), **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        # sample.json is rewritten by `sample` while a page may be open.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, body: str, ctype="application/json") -> None:
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self.send_json(PAGE.replace("__VERDICTS__", json.dumps(VERDICTS)), "text/html")
        if self.path == "/api/answers":
            return self.send_json(json.dumps(load_answers()))
        return super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with (DIR / "answers.jsonl").open("a") as f:
            f.write(json.dumps(json.loads(body), ensure_ascii=False) + "\n")
        self.send_json("{}")


def serve() -> None:
    print(f"http://localhost:{PORT}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


def report() -> None:
    answers = list(load_answers().values())
    strata = defaultdict(Counter)
    for a in answers:
        if "verdict" not in a:
            continue
        for w in a["why"]:
            strata[w][a["verdict"]] += 1
    head = "| stratum | n | " + " | ".join(VERDICTS.values()) + " | label error rate |"
    lines = [
        "# Help-page clip audit v1",
        "",
        f"{len(answers)} clips reviewed by ear. The label error rate counts verdicts 2–5 over answered clips, excluding \"can't tell\". A clip appears in every stratum it was sampled for.",
        "",
        head,
        "|" + "---|" * (len(VERDICTS) + 3),
    ]
    for k in sorted(strata):
        c = strata[k]
        n = sum(c.values())
        bad, known = sum(c[v] for v in "2345"), n - c["0"]
        rate = f"{bad}/{known} {100 * bad / known:.0f}%" if known else "—"
        lines.append(f"| {k} | {n} | " + " | ".join(str(c[v]) for v in VERDICTS) + f" | {rate} |")
    notes = [a for a in answers if a.get("note")]
    if notes:
        lines += ["", "## Notes", "", "| clip | kind | verdict | note |", "|---|---|---|---|"]
        lines += [f"| {a['id']} | {a['kind']} | {VERDICTS.get(a.get('verdict'), '-')} | {a['note']} |" for a in notes]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT.read_text())


if __name__ == "__main__":
    {"pull": pull, "check": check, "sample": sample, "serve": serve, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "serve"]()
