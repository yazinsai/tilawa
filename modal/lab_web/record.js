import { icon, style } from './item.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const ar = (s) => `<bdi class="li-ar" lang="ar" dir="rtl">${esc(s)}</bdi>`;
const url = new URL(location.href);
const scriptId = url.searchParams.get('script');
const KEY = 'tilawa.recorder';
const NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const RESERVED = ['scripts', 'sessions', 'review'];
const CLASS = { wrong_word: 'Wrong word', added_word: 'Added word', omitted_word: 'Word left out', self_correction: 'Self-correction' };
const VERDICT = {
  ok: ['ok', 'Follows the script'],
  listen: ['wait', 'Needs a listen'],
  redo: ['bad', 'Record again'],
  error: ['bad', 'Could not process'],
  refused: ['bad', 'Could not join'],
  processing: ['', 'Checking…'],
};
const PATCH = { ok: 'follow the script', listen: 'need a listen', redo: "don't follow the script" };
const STATUS = { performed: 'heard', unsure: 'not sure', not_performed: 'not heard' };
const MIMES = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm'];
const AR_DIGITS = (n) => String(n).replace(/\d/g, (d) => '٠١٢٣٤٥٦٧٨٩'[d]);
const mins = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const when = (iso) => (iso ? new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');

const saved = JSON.parse(localStorage.getItem(KEY) || '{}');
const st = { speaker: url.searchParams.get('speaker') || saved.speaker || '', device: saved.device || '', script: null, takes: [], poll: null, fix: null };

const save = () => localStorage.setItem(KEY, JSON.stringify({ speaker: st.speaker, device: st.device }));

async function api(path, init) {
  const r = await fetch(path, init);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${r.status}`);
  return body;
}

function topbar() {
  $('head').innerHTML = `<a class="brand" href="/">${icon('mark')}Tilawa lab</a>
    <nav class="tabs"><a href="/">Score</a><a href="/#/release">Release</a><a href="/review">Listening check</a><a href="/record" aria-current="page">Record</a></nav>
    ${st.speaker ? `<span class="who">Recording as <b>${esc(st.speaker)}</b> · <button id="who">change</button></span>` : ''}`;
  if ($('who')) $('who').onclick = () => askName(true);
}

function nameError(n) {
  if (!NAME.test(n)) return 'Use letters, digits, ".", "_" or "-" (up to 64), starting with a letter or digit.';
  if (RESERVED.includes(n.toLowerCase())) return `"${n}" is reserved; pick another name.`;
  return '';
}

function askName(change = false) {
  const n = prompt('Your name for the recordings (letters, digits, - or _). It is stored with every take.', change ? st.speaker : '');
  if (n === null) return;
  const err = nameError(n.trim());
  if (err) return alert(err);
  st.speaker = n.trim();
  save();
  location.reload();
}

function nameForm() {
  return `<form class="rec-name" id="name-form">
    <label for="name-in"><b>Your name</b><span class="muted small">Stored with every take, so takes from different people stay apart.</span></label>
    <div><input id="name-in" autocomplete="nickname" placeholder="e.g. yazin" required /><button class="btn">Continue</button></div>
    <p class="small rec-name-err" id="name-err"></p>
  </form>`;
}

function bindName() {
  const f = $('name-form');
  if (!f) return;
  f.onsubmit = (e) => {
    e.preventDefault();
    const n = $('name-in').value.trim();
    const err = nameError(n);
    if (err) return ($('name-err').textContent = err);
    st.speaker = n;
    save();
    location.reload();
  };
}

const pill = (verdict) => {
  const [k, t] = VERDICT[verdict] || ['', verdict];
  return `<span class="rec-pill ${k}">${t}</span>`;
};

// ---- script list --------------------------------------------------------------------------------------------------

async function list() {
  const d = await api(`/api/record/scripts${st.speaker ? `?speaker=${encodeURIComponent(st.speaker)}` : ''}`);
  const rows = d.scripts
    .map((s) => {
      const last = s.takes[0];
      const n = Object.entries(s.counts || {}).reduce((a, [k, v]) => a + (k === 'self_correction' ? 0 : v), 0);
      const what = s.clean ? 'No planted mistakes' : `${n} mistakes${s.counts?.self_correction ? ` · ${s.counts.self_correction} self-correction${s.counts.self_correction > 1 ? 's' : ''}` : ''}`;
      return `<li><a href="/record?script=${encodeURIComponent(s.id)}">
        <span class="t"><b>${esc(s.title)}</b><span>${esc(s.range)} · ${esc(s.id)}</span></span>
        <span class="num">${Math.round(s.estimatedS / 60)} min<span>${what}</span></span>
        <span>${last ? `${pill(last.verdict)}<span>take ${last.take}</span>` : '<span>not recorded</span>'}</span>
        ${icon('right')}</a></li>`;
    })
    .join('');
  const done = d.scripts.filter((s) => s.takes.some((t) => t.verdict === 'ok' || t.verdict === 'listen')).length;
  $('main').innerHTML = `<div class="rv-head">
      <h1>Record a script</h1>
      <p class="rec-lede">Each script is a passage to recite in one take, with a few mistakes planted on purpose. The recording becomes a test case whose mistakes are known exactly.</p>
      ${st.speaker ? `<p class="rec-facts small muted"><span><b class="num">${done}</b> of ${d.scripts.length} recorded by ${esc(st.speaker)}</span></p>` : nameForm()}
    </div>
    ${d.scripts.length ? `<ul class="queues rec-list">${rows}</ul>` : '<div class="notice">No scripts are published yet.</div>'}`;
  bindName();
}

// ---- one script ---------------------------------------------------------------------------------------------------

function tokenHtml(t, plants) {
  const n = t.plant ? plants[t.plant]?.n : null;
  const sup = n ? `<sup class="rec-n" title="Mistake ${n}">${AR_DIGITS(n)}</sup>` : '';
  switch (t.mark) {
    case 'wrong':
    case 'slip':
    case 'added':
      return `<mark class="li-mark" data-plant="${esc(t.plant)}" tabindex="0">${esc(t.text)}</mark>${sup}`;
    case 'omitted':
      return `<s class="rec-omit" data-plant="${esc(t.plant)}" tabindex="0">${esc(t.text)}</s>${sup}`;
    case 'back':
      return t.plant ? `<span class="rec-back" data-plant="${esc(t.plant)}" tabindex="0">↺</span>` : `<span class="rec-back" title="Stop and go back">↺</span>`;
    case 'retake':
      return `<span class="rec-retake">${esc(t.text)}</span>`;
    default:
      return esc(t.text);
  }
}

/** The planted mistake's instruction beside its word in the text, on hover, keyboard focus or tap. */
function bindTips(s) {
  const plants = Object.fromEntries(s.plants.map((p) => [p.id, p]));
  const tip = document.querySelector('.rec-tip') || document.body.appendChild(Object.assign(document.createElement('div'), { className: 'rec-tip', role: 'tooltip' }));
  tip.hidden = true;
  let pinned = null;
  const show = (el) => {
    const p = plants[el.dataset.plant];
    if (!p) return;
    tip.innerHTML = `<span class="rec-step num">${p.n}</span><span>${esc(p.instruction).replace(/«([^»]+)»/g, (_, w) => ar(w))}</span>`;
    tip.hidden = false;
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;
    tip.style.top = `${r.top - h - 10 < 8 ? r.bottom + 10 : r.top - h - 10}px`;
  };
  const hide = () => { tip.hidden = true; pinned = null; };
  const text = document.querySelector('.rec-text');
  if (!text) return;
  text.addEventListener('pointerover', (e) => { const el = e.target.closest('[data-plant]'); if (el && !pinned) show(el); });
  text.addEventListener('pointerout', (e) => { if (!pinned && e.target.closest('[data-plant]')) tip.hidden = true; });
  text.addEventListener('focusin', (e) => { const el = e.target.closest('[data-plant]'); if (el) show(el); });
  text.addEventListener('focusout', () => { if (!pinned) tip.hidden = true; });
  text.addEventListener('click', (e) => {
    const el = e.target.closest('[data-plant]');
    if (!el) return hide();
    if (pinned === el) return hide();
    pinned = el;
    show(el);
  });
  addEventListener('scroll', () => { if (!tip.hidden) hide(); }, { passive: true });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
}

function scriptHtml(s) {
  const plants = Object.fromEntries(s.plants.map((p) => [p.id, p]));
  const text = s.ayahs
    .map((a) =>
      a.basmala
        ? `<span class="rec-basmala">${a.tokens.map((t) => esc(t.text)).join(' ')}</span>`
        : `<span class="rec-ayah" id="a-${a.surah}-${a.ayah}">${a.tokens.map((t) => tokenHtml(t, plants)).join(' ')} <span class="rec-end">﴿${AR_DIGITS(a.ayah)}﴾</span></span>`,
    )
    .join(' ');
  const steps = s.plants
    .map(
      (p) => `<li><span class="rec-step num">${p.n}</span><div>
        <p>${esc(p.instruction).replace(/«([^»]+)»/g, (_, w) => ar(w))}</p>
        <p class="small muted">${esc(CLASS[p.class])} · ${esc(p.ayah)}${p.why ? ` · ${esc(p.why.replace(/^«[^»]+»: /, ''))}` : ''}</p>
      </div><a class="small" href="#a-${p.location.surah}-${p.location.ayah}">Find</a></li>`,
    )
    .join('');
  return `<div class="rv-head">
      <p class="small"><a href="/record">${icon('back')} All scripts</a></p>
      <h1>${esc(s.title)} <span class="muted" style="font-weight:400">${esc(s.range)}</span></h1>
      <p class="rec-facts small muted"><span>About <b class="num">${Math.round(s.estimatedS / 60)}</b> min at a normal pace</span><span>${s.ayahCount} ayahs</span><span>${esc(s.id)} · v${s.version}</span></p>
    </div>
    <div class="notice rec-how">Recite the whole passage in <b>one take</b> at your normal pace; pauses are fine.
      ${s.clean ? 'This script has <b>no planted mistakes</b>: recite it as written.' : `When you reach a <span class="rec-key">marked word</span>, do what its step says, then carry on.`}
      Saying the isti'adha first and «صدق الله العظيم» at the end is fine. If you slip somewhere the script doesn't ask for, stop and record again.</div>
    ${steps ? `<h2 class="rec-h">The mistakes to make</h2><ol class="rec-steps">${steps}</ol>` : ''}
    <h2 class="rec-h">What to recite</h2>
    <div class="li-page"><div class="li-text rec-text" lang="ar" dir="rtl">${text}</div></div>
    <h2 class="rec-h">Your takes</h2>
    <div id="takes"></div>`;
}

// ---- fixing a passage: record only the ayahs that failed, joined into the take at the pauses either side ----------

const ayahKeys = () => st.script.ayahs.filter((a) => !a.basmala).map((a) => a.key);
const span = (a, b) => (a === b ? `ayah ${a}` : `ayahs ${a}–${b}`);
const clip = (path, a, b) => `/audio?path=${encodeURIComponent(path)}&start=${Math.max(0, a).toFixed(2)}&end=${b.toFixed(2)}`;
const chips = (plants) =>
  plants.length
    ? `<span class="rec-checks">${plants.map((p) => `<span class="rec-chk ${p.status}" title="${esc(p.reasons?.[0] || '')}">${p.n} ${STATUS[p.status] || p.status}</span>`).join('')}</span>`
    : '';

/** What a fix take recorded again, its own check, and where it was joined. */
function fixHtml(t) {
  const f = t.fix;
  const pc = f.patchCheck;
  const patch = `<p><span class="muted">${esc(span(f.from, f.to)[0].toUpperCase() + span(f.from, f.to).slice(1))}, recorded again:</span>
      ${pc ? `${esc(PATCH[pc.verdict] || pc.verdict)}${pc.flagged?.length ? ` · ${esc(pc.flagged.join(', '))} sound different` : ''} ${chips(pc.plants || [])}` : 'checking…'}
      ${f.patchS ? `<a href="${clip(f.patch, 0, Math.min(600, f.patchS))}" target="_blank">Listen</a>` : ''}</p>`;
  const joins = f.joins?.length
    ? `<p><span class="muted">Joined into take ${f.base}:</span> ${f.joins
        .map((j) => `${j.before ? `before ${esc(j.before)}` : `after ${esc(j.after)}`} at <span class="num">${mins(j.atS)}</span> (${j.pauseS.toFixed(1)} s pause)${t.audioS ? ` <a href="${clip(t.upload, j.atS - 3, j.atS + 3)}" target="_blank">Hear</a>` : ''}`)
        .join(' · ')} · level ${f.gainDb === 0 ? 'matched' : `${f.gainDb > 0 ? '+' : ''}${f.gainDb.toFixed(1)} dB to match`}</p>`
    : '';
  return `<div class="rec-fixinfo small">${patch}${joins}</div>`;
}

/** The passages a failed take could fix, with a range to widen and one button to start. */
function offerHtml(t) {
  if (st.fix) return `<div class="rec-fix small"><p><b>Recording a fix of take ${st.fix.base}: ${esc(span(st.fix.from, st.fix.to))}.</b> Use the recorder below. <button class="rec-fix-pick" data-unfix>Cancel</button></p></div>`;
  const r = t.fixable[0];
  const opts = (sel) => ayahKeys().map((k) => `<option ${k === sel ? 'selected' : ''}>${esc(k)}</option>`).join('');
  const list = t.fixable
    .map((x) => {
      const why = Object.entries(x.ayahs).map(([k, w]) => `${x.from === x.to ? '' : `${esc(k)}: `}${esc(w.join(', '))}`).join(' · ');
      return `<li><button class="rec-fix-pick" data-pick="${esc(x.from)}|${esc(x.to)}">${esc(span(x.from, x.to))}</button> <span class="muted">${why}</span></li>`;
    })
    .join('');
  return `<div class="rec-fix small" data-take="${t.take}">
      <p><b>Re-record just the ayahs that failed</b>${t.fixable.length > 1 ? ', one passage at a time' : ''}</p>
      <ul class="rec-fix-list">${list}</ul>
      <div class="rec-fix-row">
        <label>From <select data-end="from">${opts(r.from)}</select></label><label>to <select data-end="to">${opts(r.to)}</select></label>
        <button class="btn rec-up" data-fix-go>Re-record ${esc(span(r.from, r.to))}</button>
      </div>
      <p class="muted">Whole ayahs only. Leave a moment of silence before the first and after the last. They are joined into take ${t.take} at its pauses either side, and the joined take is checked again as a new take; take ${t.take} is kept.</p>
    </div>`;
}

function bindOffer() {
  const box = document.querySelector('.rec-fix[data-take]');
  document.querySelector('[data-unfix]')?.addEventListener('click', () => setFix(null));
  if (!box) return;
  const [from, to] = ['from', 'to'].map((k) => box.querySelector(`[data-end="${k}"]`));
  const go = box.querySelector('[data-fix-go]');
  const sync = (changed) => {
    const ks = ayahKeys();
    if (ks.indexOf(from.value) > ks.indexOf(to.value)) (changed === from ? to : from).value = changed.value;
    go.textContent = `Re-record ${span(from.value, to.value)}`;
  };
  from.onchange = () => sync(from);
  to.onchange = () => sync(to);
  for (const b of box.querySelectorAll('[data-pick]'))
    b.onclick = () => {
      [from.value, to.value] = b.dataset.pick.split('|');
      sync(from);
    };
  go.onclick = () => setFix({ base: +box.dataset.take, from: from.value, to: to.value });
}

/** Enter (or leave, with null) fix mode: the recorder records only these ayahs, marked in the text. */
function setFix(fix) {
  if (live()) return;
  if (rec.blob && !confirm('Discard the recording you have not uploaded?')) return;
  if (rec.blob) discard();
  st.fix = fix;
  const ks = ayahKeys();
  const on = fix ? new Set(ks.slice(ks.indexOf(fix.from), ks.indexOf(fix.to) + 1)) : new Set();
  for (const el of document.querySelectorAll('.rec-ayah')) el.classList.toggle('rec-fixing', on.has(el.id.slice(2).replace('-', ':')));
  renderTakes();
  ready();
  if (fix) document.getElementById(`a-${fix.from.replace(':', '-')}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function ready() {
  $('rec-unfix').hidden = !st.fix || live() || !!rec.blob;
  $('rec-up').textContent = st.fix ? 'Upload fix' : 'Upload take';
  if (st.fix) setState(`Fix for take ${st.fix.base}: ${span(st.fix.from, st.fix.to)}`, 'start, wait a moment, recite only these ayahs, pause, then stop');
  else setState('Ready to record');
}

function takesHtml() {
  if (!st.takes.length) return '<p class="muted small">None yet. Record below; the take is checked a minute or two after it uploads.</p>';
  const top = st.takes[0];
  return `<ul class="rec-takes">${st.takes
    .map((t) => {
      const stale = t.scriptSha !== st.script.sha ? '<span class="muted"> · recorded from an older version of this script</span>' : '';
      const plants = t.plants.length ? `<p class="small">${chips(t.plants)}</p>` : '';
      const listen = t.audioS ? `<a class="small" href="/audio?path=${encodeURIComponent(t.upload)}&start=0&end=${Math.min(600, t.audioS)}" target="_blank">Listen</a>` : '';
      const retry = t.verdict === 'error' ? `<button class="btn" data-retry="${t.take}">Try again</button>` : '';
      const s = t.fix?.suggest;
      const again =
        t === top && t.verdict === 'refused' && !st.fix
          ? s
            ? `<p class="small"><button class="btn rec-up" data-refix="${t.fix.base}|${esc(s.from)}|${esc(s.to)}">Re-record ${esc(span(s.from, s.to))}</button></p>`
            : '<p class="small muted">Record the whole take again.</p>'
          : '';
      const what = t.fix ? ` · take ${t.fix.base} with ${esc(span(t.fix.from, t.fix.to))} recorded again` : '';
      return `<li><div class="rec-take-h"><b>Take ${t.take}</b>${pill(t.verdict)}<span class="muted small">${when(t.uploadedAt)}${t.audioS ? ` · ${mins(t.audioS)}` : ''}${what}${t.device ? ` · ${esc(t.device)}` : ''}${stale}</span>${listen}${retry}</div>
        ${t.why?.length ? `<ul class="rec-why small">${t.why.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>` : ''}${plants}${t.fix ? fixHtml(t) : ''}${again}
        ${t === top && t.fixable?.length ? offerHtml(t) : ''}</li>`;
    })
    .join('')}</ul>`;
}

function renderTakes() {
  $('takes').innerHTML = takesHtml();
  bindOffer();
  for (const b of document.querySelectorAll('[data-refix]'))
    b.onclick = () => {
      const [base, from, to] = b.dataset.refix.split('|');
      setFix({ base: +base, from, to });
    };
  for (const b of document.querySelectorAll('[data-retry]'))
    b.onclick = async () => {
      b.disabled = true;
      try {
        await api('/api/record/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ speaker: st.speaker, script: st.script.id, take: +b.dataset.retry }) });
        await refresh();
      } catch (e) {
        alert(e.message);
        b.disabled = false;
      }
    };
  clearTimeout(st.poll);
  if (st.takes.some((t) => t.verdict === 'processing')) st.poll = setTimeout(refresh, 5000);
}

async function refresh() {
  const d = await api(`/api/record/script?id=${encodeURIComponent(scriptId)}&speaker=${encodeURIComponent(st.speaker)}`);
  st.takes = d.takes;
  renderTakes();
}

async function scriptView() {
  const d = await api(`/api/record/script?id=${encodeURIComponent(scriptId)}${st.speaker ? `&speaker=${encodeURIComponent(st.speaker)}` : ''}`);
  st.script = d.script;
  st.takes = d.takes;
  document.title = `${d.script.title} · Record · Tilawa lab`;
  $('main').innerHTML = scriptHtml(d.script) + (st.speaker ? '' : nameForm());
  bindTips(d.script);
  bindName();
  if (!st.speaker) {
    $('takes').innerHTML = '<p class="muted small">Enter your name below to record.</p>';
    return;
  }
  renderTakes();
  recorder();
}

// ---- recorder -----------------------------------------------------------------------------------------------------

const rec = { stream: null, mr: null, chunks: [], blob: null, t0: 0, acc: 0, resumes: [], dur: 0, timer: null, ctx: null, raf: null, lock: null, peak: 0, quiet: 0 };
/** Seconds recorded so far, paused time excluded. */
const elapsed = () => rec.acc + (rec.mr?.state === 'recording' ? (performance.now() - rec.t0) / 1000 : 0);
const live = () => rec.mr && rec.mr.state !== 'inactive';

function dockHtml() {
  return `<div class="rec-dock-in">
    <button class="rec-disc" id="rec-go" aria-label="Start recording">${micIcon()}</button>
    <button class="btn rec-pause" id="rec-pause" hidden>Pause</button>
    <button class="btn" id="rec-unfix" hidden>Cancel fix</button>
    <div class="rec-mid">
      <p class="rec-state" id="rec-state">Ready to record</p>
      <div class="rec-meter" aria-hidden="true"><i id="rec-level"></i></div>
      <p class="small muted rec-sub"><span class="num" id="rec-time">0:00</span><span id="rec-hint"></span></p>
    </div>
    <div class="rec-after" id="rec-after" hidden>
      <audio id="rec-play" controls preload="metadata"></audio>
      <button class="btn" id="rec-again">Record again</button>
      <button class="btn rec-up" id="rec-up">Upload take</button>
    </div>
    <label class="rec-dev small"><span class="muted">Microphone</span><select id="rec-dev"></select></label>
  </div>
  <div class="rec-progress" id="rec-progress" hidden><i></i></div>`;
}

const micIcon = () =>
  '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/></svg>';
const stopIcon = () => '<svg class="icon fill" viewBox="0 0 24 24" aria-hidden="true"><rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/></svg>';

function setState(text, hint = '') {
  $('rec-state').textContent = text;
  $('rec-hint').textContent = hint ? ` · ${hint}` : '';
}

function recorder() {
  const dock = $('dock');
  dock.innerHTML = dockHtml();
  dock.hidden = false;
  document.body.classList.add('has-dock');
  $('rec-go').onclick = () => (live() ? stop() : start());
  $('rec-pause').onclick = () => (rec.mr?.state === 'paused' ? resume() : pause());
  $('rec-again').onclick = () => {
    if (rec.blob && !confirm('Discard this take and record again?')) return;
    discard();
  };
  $('rec-up').onclick = upload;
  $('rec-unfix').onclick = () => setFix(null);
  $('rec-dev').onchange = (e) => {
    st.device = e.target.value;
    save();
    openMic().catch(micError);
  };
  window.addEventListener('beforeunload', (e) => {
    if (rec.blob || live()) e.preventDefault();
  });
  devices();
}

async function devices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const ds = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  const sel = $('rec-dev');
  sel.innerHTML = ds.length
    ? ds.map((d, i) => `<option value="${esc(d.deviceId)}" ${d.deviceId === st.device ? 'selected' : ''}>${esc(d.label || `Microphone ${i + 1}`)}</option>`).join('')
    : '<option value="">Default microphone</option>';
}

function micError(e) {
  setState('No microphone', e.name === 'NotAllowedError' ? 'allow microphone access for this site and try again' : e.message);
}

async function openMic() {
  rec.stream?.getTracks().forEach((t) => t.stop());
  const audio = { channelCount: 1, sampleRate: 16000, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  if (st.device) audio.deviceId = { exact: st.device };
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio });
  } catch (e) {
    if (e.name !== 'OverconstrainedError' && e.name !== 'NotFoundError') throw e;
    st.device = '';
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { ...audio, deviceId: undefined } });
  }
  await devices();
  meter();
  return rec.stream;
}

function meter() {
  cancelAnimationFrame(rec.raf);
  rec.ctx?.close();
  rec.ctx = new AudioContext();
  const an = rec.ctx.createAnalyser();
  an.fftSize = 2048;
  rec.ctx.createMediaStreamSource(rec.stream).connect(an);
  const buf = new Float32Array(an.fftSize);
  const tick = () => {
    an.getFloatTimeDomainData(buf);
    let sum = 0;
    let peak = 0;
    for (const v of buf) {
      sum += v * v;
      peak = Math.max(peak, Math.abs(v));
    }
    const db = 10 * Math.log10(sum / buf.length + 1e-10);
    const el = $('rec-level');
    if (el) el.style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;
    if (rec.mr?.state === 'recording') {
      rec.peak = Math.max(rec.peak * 0.995, peak);
      rec.quiet = db < -55 ? rec.quiet + 1 : 0;
      const hint = rec.peak > 0.98 ? 'too loud: move back from the microphone' : rec.quiet > 240 ? "can't hear you: check the microphone" : '';
      $('rec-hint').textContent = hint ? ` · ${hint}` : '';
    }
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}

async function start() {
  if (!window.MediaRecorder) return setState('This browser cannot record', 'use a recent Chrome, Safari or Firefox');
  try {
    if (!rec.stream || rec.stream.getTracks().every((t) => t.readyState === 'ended')) await openMic();
  } catch (e) {
    return micError(e);
  }
  const mimeType = MIMES.find((m) => MediaRecorder.isTypeSupported(m));
  rec.mr = new MediaRecorder(rec.stream, { ...(mimeType ? { mimeType } : {}), audioBitsPerSecond: 96000 });
  rec.chunks = [];
  rec.mr.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
  rec.mr.onstop = stopped;
  rec.mr.start(1000);
  rec.t0 = performance.now();
  rec.acc = 0;
  rec.resumes = [];
  rec.peak = 0;
  rec.quiet = 0;
  rec.timer = setInterval(() => ($('rec-time').textContent = mins(elapsed())), 250);
  rec.lock = await navigator.wakeLock?.request('screen').catch(() => null);
  $('rec-go').innerHTML = stopIcon();
  $('rec-go').setAttribute('aria-label', 'Stop recording');
  $('rec-go').classList.add('on');
  $('rec-after').hidden = true;
  $('rec-pause').hidden = false;
  $('rec-pause').textContent = 'Pause';
  $('rec-unfix').hidden = true;
  setState(st.fix ? `Recording the fix: ${span(st.fix.from, st.fix.to)}` : 'Recording');
}

function pause() {
  rec.acc = elapsed();
  rec.mr.pause();
  $('rec-pause').textContent = 'Resume';
  setState('Paused', 'resume at the start of an ayah, so the join falls where a reciter would pause');
}

function resume() {
  rec.resumes.push(+rec.acc.toFixed(2));
  rec.t0 = performance.now();
  rec.mr.resume();
  $('rec-pause').textContent = 'Pause';
  setState(st.fix ? `Recording the fix: ${span(st.fix.from, st.fix.to)}` : 'Recording');
}

function stop() {
  rec.dur = elapsed();
  rec.mr.stop();
  $('rec-pause').hidden = true;
}

function stopped() {
  clearInterval(rec.timer);
  rec.lock?.release().catch(() => {});
  rec.blob = new Blob(rec.chunks, { type: rec.mr.mimeType || 'audio/webm' });
  $('rec-play').src = URL.createObjectURL(rec.blob);
  $('rec-go').innerHTML = micIcon();
  $('rec-go').setAttribute('aria-label', 'Start recording');
  $('rec-go').classList.remove('on');
  $('rec-go').hidden = true;
  $('rec-after').hidden = false;
  $('rec-up').disabled = false;
  setState(`${st.fix ? 'Fix' : 'Take'} recorded · ${mins(rec.dur)}`, 'listen back, then upload it or record again');
}

function discard() {
  if ($('rec-play').src) URL.revokeObjectURL($('rec-play').src);
  $('rec-play').removeAttribute('src');
  rec.blob = null;
  $('rec-after').hidden = true;
  $('rec-go').hidden = false;
  $('rec-time').textContent = '0:00';
  ready();
}

function upload() {
  const s = st.script;
  const track = rec.stream?.getAudioTracks()[0];
  const fd = new FormData();
  const ext = { 'audio/ogg': 'ogg', 'audio/mp4': 'mp4' }[rec.blob.type.split(';')[0]] || 'webm';
  fd.append('speaker', st.speaker);
  fd.append('script', s.id);
  fd.append('scriptVersion', String(s.version));
  fd.append('scriptSha', s.sha);
  fd.append('batch', s.batch || '');
  fd.append('mime', rec.blob.type);
  fd.append('device', track?.label || '');
  fd.append('trackSettings', JSON.stringify(track?.getSettings?.() || {}));
  fd.append('userAgent', navigator.userAgent);
  fd.append('recordedAt', new Date(Date.now() - rec.dur * 1000).toISOString());
  fd.append('durationS', rec.dur.toFixed(2));
  fd.append('resumesS', JSON.stringify(rec.resumes));
  const fix = st.fix;
  if (fix) {
    fd.append('fixOf', String(fix.base));
    fd.append('fixFrom', fix.from);
    fd.append('fixTo', fix.to);
  }
  fd.append('audio', rec.blob, `take.${ext}`);
  const bar = $('rec-progress');
  bar.hidden = false;
  $('rec-up').disabled = true;
  $('rec-again').disabled = true;
  setState('Uploading…');
  const x = new XMLHttpRequest();
  x.open('POST', '/api/record/upload');
  x.upload.onprogress = (e) => e.lengthComputable && (bar.firstElementChild.style.width = `${(e.loaded / e.total) * 100}%`);
  x.onload = async () => {
    bar.hidden = true;
    $('rec-again').disabled = false;
    let body = {};
    try {
      body = JSON.parse(x.responseText);
    } catch {}
    if (x.status !== 200) {
      $('rec-up').disabled = false;
      return setState('Upload failed', typeof body.detail === 'string' ? body.detail : `HTTP ${x.status}; try again`);
    }
    rec.blob = null;
    setFix(null);
    discard();
    setState(`Uploaded take ${body.take}`, body.processing || (fix ? `joining it into take ${fix.base}, then checking the whole take; a minute or two` : 'checking it now; this takes a minute or two'));
    await refresh();
  };
  x.onerror = () => {
    bar.hidden = true;
    $('rec-up').disabled = false;
    $('rec-again').disabled = false;
    setState('Upload failed', 'check the connection and try again; the take is still here');
  };
  x.send(fd);
}

// ---- boot ---------------------------------------------------------------------------------------------------------

style();
topbar();
(scriptId ? scriptView() : list()).catch((e) => ($('main').innerHTML = `<div class="notice err">${esc(e.message)}</div>`));
