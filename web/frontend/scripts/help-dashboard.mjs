#!/usr/bin/env node
// Local dashboard for tilawa.dev/help submissions.
//   node web/frontend/scripts/help-dashboard.mjs   ->  http://localhost:8790
// The admin token stays in this process; the browser only talks to localhost.
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';

const BASE = process.env.HELP_BASE ?? 'https://tilawa.dev/help/api';
const PORT = Number(process.env.PORT ?? 8790);
const TOKEN =
  process.env.HELP_ADMIN_TOKEN ??
  execFileSync('ssh', ['dokku-server', 'dokku config:get tilawa HELP_ADMIN_TOKEN'], { encoding: 'utf8' }).trim();
const auth = { authorization: `Bearer ${TOKEN}` };

const jsonl = text => text.split('\n').filter(Boolean).map(l => JSON.parse(l));
async function data() {
  const [subs, emails] = await Promise.all(
    ['export', 'subscribers'].map(p => fetch(`${BASE}/${p}`, { headers: auth }).then(r => (r.ok ? r.text() : Promise.reject(new Error(`${p}: ${r.status}`))))),
  );
  return { submissions: jsonl(subs), subscribers: jsonl(emails).length, fetchedAt: new Date().toISOString() };
}

createServer(async (req, res) => {
  try {
    if (req.url === '/data') {
      res.writeHead(200, { 'content-type': 'application/json' });
      return res.end(JSON.stringify(await data()));
    }
    const audio = req.url.match(/^\/audio\/([a-z0-9_]+\.(webm|ogg|m4a|mp3|wav))$/);
    if (audio) {
      const r = await fetch(`${BASE}/audio/${audio[1]}`, { headers: auth });
      res.writeHead(r.status, { 'content-type': audio[2] === 'm4a' ? 'audio/mp4' : `audio/${audio[2]}` });
      return res.end(Buffer.from(await r.arrayBuffer()));
    }
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(PAGE);
  } catch (err) {
    res.writeHead(502, { 'content-type': 'text/plain' });
    res.end(String(err));
  }
}).listen(PORT, () => console.log(`help dashboard: http://localhost:${PORT}  (source ${BASE})`));

const PAGE = /* html */ `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>/help dashboard</title>
<style>
  :root { --ink:#1f1b16; --muted:#6b6258; --line:#e4dccd; --bg:#f6f1e7; --card:#fffdf8; --accent:#1d5c4a; --warn:#b3261e; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 system-ui, sans-serif; color: var(--ink); background: var(--bg); }
  main { max-width: 1180px; margin: 0 auto; padding: 24px 18px 48px; }
  header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; }
  h1 { font-size: 22px; margin: 0; }
  .muted { color: var(--muted); }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }
  .stat b { display: block; font-size: 28px; font-variant-numeric: tabular-nums; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; margin-bottom: 14px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; overflow-x: auto; }
  h2 { font-size: 14px; margin: 0 0 12px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
  .bar { display: grid; grid-template-columns: 120px 1fr 64px; align-items: center; gap: 8px; margin: 5px 0; }
  .track { position: relative; height: 16px; background: #efe8da; border-radius: 4px; }
  .fill { height: 100%; background: var(--accent); border-radius: 4px; }
  .target { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--warn); }
  .num { text-align: right; font-variant-numeric: tabular-nums; }
  .timeline { display: flex; align-items: flex-end; gap: 2px; height: 120px; }
  .timeline div { flex: 1; background: var(--accent); min-height: 1px; border-radius: 2px 2px 0 0; }
  .tl-axis { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; }
  td.n { text-align: right; font-variant-numeric: tabular-nums; }
  code { font-size: 12px; }
  .flag { color: var(--warn); font-weight: 600; }
  audio { height: 28px; vertical-align: middle; }
  .err { color: var(--warn); }
</style></head>
<body><main>
<header><h1>tilawa.dev/help</h1><span class="muted" id="updated">loading…</span></header>
<p class="err" id="err"></p>
<section class="stats" id="stats"></section>
<section class="grid">
  <div class="card"><h2>Mistake type vs target <span class="muted">(red = target share)</span></h2><div id="kinds"></div></div>
  <div class="card"><h2>Submissions per hour (local)</h2><div class="timeline" id="timeline"></div><div class="tl-axis" id="tl-axis"></div></div>
</section>
<section class="grid">
  <div class="card"><h2>Device</h2><div id="device"></div></div>
  <div class="card"><h2>Platform</h2><div id="platform"></div></div>
  <div class="card"><h2>Voice</h2><div id="gender"></div></div>
  <div class="card"><h2>Level</h2><div id="level"></div></div>
</section>
<section class="card" style="margin-bottom:14px"><h2>Contributors</h2><table id="contributors"></table></section>
<section class="card"><h2>Latest 25</h2><table id="latest"></table></section>
</main>
<script>
const TARGET = { none: 20, skip_word: 15, substitution: 15, repeat: 10, vowel: 15, tajweed: 10, skip_ayah: 15 };
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const count = (xs, f) => xs.reduce((m, x) => ((m[f(x)] = (m[f(x)] ?? 0) + 1), m), {});
function platform(ua) {
  const os = /iPhone/.test(ua) ? 'iPhone' : /iPad/.test(ua) ? 'iPad' : /Android/.test(ua) ? 'Android' : /Macintosh/.test(ua) ? 'Mac' : /Windows/.test(ua) ? 'Windows' : /Linux/.test(ua) ? 'Linux' : 'other';
  const br = /CriOS|Chrome/.test(ua) ? 'Chrome' : /FxiOS|Firefox/.test(ua) ? 'Firefox' : /Safari/.test(ua) ? 'Safari' : 'other';
  return os + ' / ' + br;
}
function bars(el, counts, total, targets) {
  const keys = targets ? Object.keys(targets) : Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  const max = Math.max(1, ...keys.map(k => counts[k] ?? 0), ...(targets ? keys.map(k => (targets[k] / 100) * total) : []));
  el.innerHTML = keys.map(k => {
    const n = counts[k] ?? 0;
    const t = targets ? '<span class="target" style="left:' + ((targets[k] / 100) * total / max) * 100 + '%"></span>' : '';
    return '<div class="bar"><span>' + esc(k) + '</span><span class="track"><span class="fill" style="display:block;width:' + (n / max) * 100 + '%"></span>' + t + '</span><span class="num">' + n + ' · ' + Math.round((n / (total || 1)) * 100) + '%</span></div>';
  }).join('') || '<span class="muted">no data</span>';
}
function timeline(subs) {
  if (!subs.length) return ($('timeline').innerHTML = '');
  const hour = 3600e3;
  const t0 = Math.floor(new Date(subs[0].ts) / hour) * hour;
  const t1 = Math.floor(Date.now() / hour) * hour;
  const n = Math.min(168, (t1 - t0) / hour + 1);
  const start = t1 - (n - 1) * hour;
  const buckets = Array(n).fill(0);
  for (const s of subs) { const i = Math.floor((new Date(s.ts) - start) / hour); if (i >= 0 && i < n) buckets[i]++; }
  const max = Math.max(1, ...buckets);
  const fmt = t => new Date(t).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' });
  $('timeline').innerHTML = buckets.map((b, i) => '<div title="' + fmt(start + i * hour) + ': ' + b + '" style="height:' + (b / max) * 100 + '%"></div>').join('');
  $('tl-axis').innerHTML = '<span>' + fmt(start) + '</span><span>' + fmt(t1) + '</span>';
}
async function load() {
  try {
    const { submissions: s, subscribers, fetchedAt } = await (await fetch('/data')).json();
    $('err').textContent = '';
    const speakers = count(s, x => x.speaker);
    const minutes = s.reduce((a, x) => a + x.duration_s, 0) / 60;
    const flagged = s.filter(x => x.extra_mistake).length;
    const last = s.at(-1);
    $('stats').innerHTML = [
      ['Submissions', s.length], ['Contributors', Object.keys(speakers).length], ['Audio (min)', minutes.toFixed(1)],
      ['Email signups', subscribers], ['Extra-mistake flags', flagged],
      ['Last submission', last ? new Date(last.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—'],
    ].map(([k, v]) => '<div class="stat"><span class="muted">' + k + '</span><b>' + v + '</b></div>').join('');
    bars($('kinds'), count(s, x => x.kind), s.length, TARGET);
    bars($('device'), count(s, x => x.device), s.length);
    bars($('platform'), count(s, x => platform(x.user_agent)), s.length);
    bars($('gender'), count(s, x => x.gender ?? 'not given'), s.length);
    bars($('level'), count(s, x => x.level ?? 'not given'), s.length);
    timeline(s);

    const byS = {};
    for (const x of s) (byS[x.speaker] ??= []).push(x);
    const rows = Object.entries(byS).sort((a, b) => b[1].length - a[1].length);
    $('contributors').innerHTML = '<tr><th>ID</th><th>n</th><th>device</th><th>platform</th><th>voice</th><th>level</th><th>min</th><th>flags</th><th>first → last</th><th>kinds</th></tr>' +
      rows.map(([id, xs]) => {
        const t = x => new Date(x.ts).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        const kinds = Object.entries(count(xs, x => x.kind)).map(([k, n]) => k + ' ' + n).join(', ');
        const f = xs.filter(x => x.extra_mistake).length;
        return '<tr><td><code>' + esc(id) + '</code></td><td class="n">' + xs.length + '</td><td>' + esc(xs[0].device) + '</td><td>' + esc(platform(xs[0].user_agent)) + '</td><td>' + esc(xs[0].gender ?? '—') + '</td><td>' + esc(xs[0].level ?? '—') + '</td><td class="n">' + (xs.reduce((a, x) => a + x.duration_s, 0) / 60).toFixed(1) + '</td><td class="n' + (f ? ' flag' : '') + '">' + f + '</td><td>' + t(xs[0]) + ' → ' + t(xs.at(-1)) + '</td><td>' + esc(kinds) + '</td></tr>';
      }).join('');

    $('latest').innerHTML = '<tr><th>time</th><th>contributor</th><th>kind</th><th>ref</th><th>mistake</th><th>s</th><th>audio</th></tr>' +
      s.slice(-25).reverse().map(x => {
        const m = x.mistakes[0];
        const what = !m ? '—' : m.kind === 'skip_ayah' ? 'skip ' + m.ayah : 'w' + m.word + ' ' + (m.said_uthmani ? m.expected + ' → ' + m.said_uthmani : m.rule ?? m.expected);
        return '<tr><td>' + new Date(x.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) + '</td><td><code>' + esc(x.speaker.slice(0, 8)) + '</code></td><td>' + esc(x.kind) + (x.extra_mistake ? ' <span class="flag">+extra</span>' : '') + '</td><td>' + esc(x.ref) + '</td><td dir="auto">' + esc(what) + '</td><td class="n">' + x.duration_s.toFixed(1) + '</td><td><audio controls preload="none" src="/audio/' + esc(x.file.slice(6)) + '"></audio></td></tr>';
      }).join('');
    $('updated').textContent = 'updated ' + new Date(fetchedAt).toLocaleTimeString() + ' · auto-refresh 30s';
  } catch (e) {
    $('err').textContent = 'Fetch failed: ' + e;
  }
}
load();
setInterval(load, 30000);
</script></body></html>`;
