"""Blind spot-listen for the E1b pack (ticket 11).

Clips are shuffled and shown without spliced/control, so the question is just
"do you hear the error?". pass = heard on a splice, or not heard on a control.

    uv run tools/e1b_listen.py          # http://localhost:8766
    uv run tools/e1b_listen.py report   # per-class table, also written to README.md pass column

Keys: y / 1 heard the error, n / 2 sounds clean, 0 unsure, space replay, j/k move.
"""

import http.server
import json
import random
import re
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent.parent / ".scratch/e1b/listen"
README = PACK / "README.md"
ANSWERS = PACK / "answers.jsonl"
PORT = 8766
ROW = re.compile(r"^\| `([^`]+)` \(([^)]*)\) \|([^|]*)\|([^|]*)\|([^|]*)\|$")


def rows() -> list[dict]:
    out = []
    for line in README.read_text().splitlines():
        if m := ROW.match(line):
            file, src, canon, said, _ = m.groups()
            cls, name = file.split("/")
            out.append({"file": file, "src": src, "cls": cls, "canon": canon.strip(), "kind": name.split("-")[0]})
    return out


def answers() -> dict[str, str]:
    if not ANSWERS.exists():
        return {}
    return {a["file"]: a["heard"] for a in map(json.loads, ANSWERS.read_text().splitlines()) if a}


def verdict(kind: str, heard: str) -> str:
    if heard == "?":
        return "unsure"
    return "pass" if (heard == "y") == (kind == "spliced") else "fail"


def write_readme() -> None:
    got = answers()
    kinds = {r["file"]: r["kind"] for r in rows()}

    def fill(line: str) -> str:
        m = ROW.match(line)
        if not m or m[1] not in got:
            return line
        return line[: line.rstrip("|").rstrip().rfind("|") + 1] + f" {verdict(kinds[m[1]], got[m[1]])} |"

    README.write_text("\n".join(map(fill, README.read_text().splitlines())) + "\n")


def report() -> None:
    write_readme()
    got = answers()
    table: dict[str, dict[str, list[int]]] = {}
    for r in rows():
        if r["file"] in got and got[r["file"]] != "?":
            t = table.setdefault(r["cls"], {"spliced": [0, 0], "control": [0, 0]})[r["kind"]]
            t[0] += verdict(r["kind"], got[r["file"]]) == "pass"
            t[1] += 1
    print("| class | splices heard as the error | controls heard as clean |\n|---|---|---|")
    for c, t in sorted(table.items()):
        print(f"| {c} | " + " | ".join(f"{k}/{n}" for k, n in (t["spliced"], t["control"])) + " |")


PAGE = """<!doctype html><meta charset=utf-8><title>E1b listen</title>
<style>body{font:18px system-ui;max-width:760px;margin:40px auto;background:#111;color:#eee}
#cls{font-size:64px;text-align:center}#canon{font-size:30px;direction:rtl;text-align:center;margin:12px 0 28px}
.b{display:inline-block;padding:10px 18px;margin:4px;border:1px solid #555;border-radius:8px;cursor:pointer}
.on{background:#2a6;border-color:#2a6}#pos,#keys{color:#999;font-size:14px}audio{width:100%}</style>
<div id=pos></div><div id=sp style='float:right;color:#999'></div><div id=cls></div><div id=canon></div><audio id=au controls></audio>
<div><span class=b data-k=y>y · heard the error</span><span class=b data-k=n>n · sounds clean</span><span class=b data-k=?>0 · unsure</span></div>
<p id=keys>y/1 error · n/2 clean · 0 unsure · space replay · [/] slower/faster · j/k next/prev · answering auto-advances</p>
<script>
let items=[],ans={},i=0;const $=id=>document.getElementById(id);
const SP=[1,1.25,1.5,2,2.5,3];let sp=Number(localStorage.getItem('speed'))||1;
function setSp(s){sp=s;localStorage.setItem('speed',s);$('au').defaultPlaybackRate=$('au').playbackRate=s;$('sp').textContent=s+'×'}
async function load(){const s=await (await fetch('/api/state')).json();items=s.items;ans=s.answers;
 i=items.findIndex(x=>!(x.file in ans));if(i<0)i=0;render()}
function render(){const it=items[i];const done=Object.keys(ans).length;
 $('pos').textContent=`${i+1}/${items.length} · ${done} answered`+(done===items.length?' · ALL DONE, run report':'');
 $('cls').textContent=it.cls;$('canon').textContent=it.canon;$('au').src='/audio/'+encodeURIComponent(it.file);$('au').play();
 document.querySelectorAll('.b').forEach(b=>b.classList.toggle('on',ans[it.file]===b.dataset.k))}
async function pick(k){const f=items[i].file;ans[f]=k;await fetch('/api/answer',{method:'POST',body:JSON.stringify({file:f,heard:k})});
 const next=items.findIndex((x,j)=>j>i&&!(x.file in ans));i=next>=0?next:Math.min(i+1,items.length-1);render()}
document.querySelectorAll('.b').forEach(b=>b.onclick=()=>pick(b.dataset.k));
const K={y:'y','1':'y',n:'n','2':'n','0':'?'};
document.addEventListener('keydown',e=>{if(K[e.key])pick(K[e.key]);
 else if(e.key===' '){e.preventDefault();$('au').currentTime=0;$('au').play()}
 else if(e.key==='['||e.key===']'){const k=SP.indexOf(sp)+(e.key===']'?1:-1);setSp(SP[Math.max(0,Math.min(SP.length-1,k))])}
 else if(e.key==='j'&&i<items.length-1){i++;render()}else if(e.key==='k'&&i>0){i--;render()}});
setSp(sp);load()</script>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            items = [{k: r[k] for k in ("file", "cls", "canon")} for r in rows()]
            random.Random(11).shuffle(items)
            self.send(json.dumps({"items": items, "answers": answers()}).encode(), "application/json")
        elif self.path.startswith("/audio/"):
            from urllib.parse import unquote

            f = (PACK / unquote(self.path[7:])).resolve()
            if PACK in f.parents and f.exists():
                self.send(f.read_bytes(), "audio/wav")
            else:
                self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["content-length"]))
        with ANSWERS.open("a") as fh:
            fh.write(json.dumps(json.loads(body), ensure_ascii=False) + "\n")
        write_readme()
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        print(f"http://localhost:{PORT}  ({len(rows())} clips, answers -> {ANSWERS})")
        http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
