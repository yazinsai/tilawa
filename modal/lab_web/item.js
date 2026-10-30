// One review item: play controls, the label in plain words, the Arabic said, grouped verdict buttons and a note.
// Used by the review page and the lab dashboard drill-down:
//   import { mountItem, VERDICTS, FAULTS } from '/static/item.js';
//   const view = mountItem(el, item, { review, verdicts: [...VERDICTS, ...FAULTS], aside: html, models: '/api/models?queue=…&id=…',
//                                      onVerdict: (verdict, note, how) => save() });
//   view.play('audio' | 'before' | 'after'); view.choose(0); view.toggleModels(); view.destroy();
// Styles live in /static/lab.css.

export const VERDICTS = [
  { key: 'ok', text: 'Sounds right', hint: 'The audio matches the label.' },
  { key: 'bad_label', text: 'Bad label', hint: 'The words or the mistake shown don’t match what you hear.' },
  { key: 'bad_audio', text: 'Bad audio', hint: 'Noise, clipping, silence or a cut-off makes it unusable.' },
  { key: 'bad_splice', text: 'Bad splice', hint: 'You can hear the edit: a click, jump or odd join.' },
];
export const FAULTS = [
  { key: 'listener', text: 'Listener fault', hint: 'Misheard the sounds.' },
  { key: 'tracker', text: 'Tracker fault', hint: 'Lost its place in the Quran.' },
  { key: 'checker', text: 'Checker fault', hint: 'Judged the mistake wrongly.' },
];
const GROUPS = [
  { test: (k) => k === 'ok', title: 'It matches', sub: '' },
  { test: (k) => k.startsWith('bad_'), title: 'Bad data', sub: 'The test case itself is wrong' },
  { test: () => true, title: 'Engine fault', sub: 'The data is fine; the engine got it wrong' },
];
export const verdictKind = (k) => (!k ? 'todo' : k === 'ok' ? 'ok' : k.startsWith('bad_') ? 'data' : 'fault');
const CLIPS = [
  ['audio', 'Play'],
  ['before', 'Original before'],
  ['after', 'Original after'],
];
const KEYS = { audio: 'space', before: 'b', after: 'a' };

const PATHS = {
  play: '<path d="M7 4.5v15l12.5-7.5z"/>',
  pause: '<path d="M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z"/>',
  left: '<path d="M15 18l-6-6 6-6"/>',
  right: '<path d="M9 18l6-6-6-6"/>',
  back: '<path d="M19 12H5M11 18l-6-6 6-6"/>',
  copy: '<rect x="8.5" y="8.5" width="12" height="12" rx="2"/><path d="M15.5 8.5V5.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"/>',
  check: '<path d="M4.5 12.5l5 5 10-11"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  down: '<path d="M12 5v14M6 13l6 6 6-6"/>',
  up: '<path d="M12 19V5M6 11l6-6 6 6"/>',
  minus: '<path d="M5 12h14"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6.5 10h.01M10 10h.01M14 10h.01M17.5 10h.01M7.5 14h9"/>',
  mark: '<circle cx="12" cy="12" r="9.5"/><circle cx="12" cy="12" r="5.5"/><path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/>',
};
export const icon = (name, cls = '') =>
  `<svg class="icon ${name === 'play' || name === 'pause' ? 'fill' : ''} ${cls}" viewBox="0 0 24 24" aria-hidden="true">${PATHS[name]}</svg>`;

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const ar = (s) => `<bdi class="li-ar" lang="ar" dir="rtl">${esc(s)}</bdi>`;

export function labelHtml(label = {}) {
  const { type, expected, said } = label;
  switch (type) {
    case 'wrong_word': return `Planted mistake: said ${ar(said)} instead of ${ar(expected)}`;
    case 'omitted_word': return `Planted mistake: left out ${ar(expected)}`;
    case 'added_word': return `Planted mistake: added ${ar(said)}`;
    case 'skipped_passage': return `Planted mistake: skipped ${ar(expected)}`;
    case 'repeat': return `Not a mistake: repeated ${ar(said || expected)}`;
    case 'restart': return `Not a mistake: went back and started again at ${ar(said || expected)}`;
    case 'self_correction': return `Not a mistake: fixed it to ${ar(expected)}`;
    case 'clean': return 'No mistake here';
    default: return `Label: ${esc(type)}${expected ? ` · expected ${ar(expected)}` : ''}${said ? ` · said ${ar(said)}` : ''}`;
  }
}

// The labelled word inside the recited text, so the eye lands on it.
function textHtml(text, label = {}) {
  const word = label.said || (['repeat', 'restart', 'self_correction'].includes(label.type) ? label.expected : null);
  const words = String(text).split(/\s+/);
  const at = word ? words.indexOf(word) : -1;
  return words.map((w, i) => (i === at ? `<mark class="li-mark">${esc(w)}</mark>` : esc(w))).join(' ');
}

export const clipUrl = (clip, base = '') =>
  `${base}/audio?path=${encodeURIComponent(clip.path)}&start=${clip.startS}&end=${clip.endS}`;

const secs = (s) => `${Math.max(0, s).toFixed(1)} s`;

// "What the models heard": the triage models' transcripts and scores for this item, from GET /api/models. Only its
// open state is kept in localStorage.
const MODELS_OPEN = 'tilawa.lab.modelsOpen';
const MODEL_NAMES = { nemotron: 'Nemotron', soniox: 'Soniox', fastconformer: 'FastConformer' };
const DECIDED = { match: 'fits', mismatch: 'doesn’t fit', unsure: 'unsure' };
const chip = (k) => (k === 'match' ? 'better' : k === 'mismatch' ? 'worse' : '');
const heardCache = new Map();
const modelsOpen = () => {
  try {
    return localStorage.getItem(MODELS_OPEN) === '1';
  } catch {
    return false;
  }
};

function modelsSummary(item) {
  const t = item.triage || {};
  return Object.keys(MODEL_NAMES)
    .filter((m) => t[m]?.decision)
    .map((m) => `${MODEL_NAMES[m]} ${DECIDED[t[m].decision] || t[m].decision}`)
    .join(' · ');
}

const diffHtml = (segs) =>
  segs
    .map((s) => {
      const edge = s.edge ? ' md-edge' : '';
      if (s.op === 'same') return esc(s.ref);
      if (s.op === 'del') return `<del class="md-del${edge}" title="In the label, not heard">${esc(s.ref)}</del>`;
      if (s.op === 'ins') return `<ins class="md-ins${edge}" title="Heard, not in the label">${esc(s.heard)}</ins>`;
      return `<del class="md-del" title="In the label">${esc(s.ref)}</del> <ins class="md-ins" title="Heard instead">${esc(s.heard)}</ins>`;
    })
    .join(' ');

function modelsHtml(d) {
  if (!d.found) return `<p class="md-none">${esc(d.message)}.</p>`;
  const rows = d.models
    .map((m) => {
      const differs = m.diff.segments.some((s) => s.op !== 'same');
      return `<section class="md-m">
      <h4><b>${esc(m.short)}</b><span>${esc(m.role)}</span><span class="verdict ${chip(m.decision.key)}">${esc(m.decision.text)}</span>${m.inRule ? '' : '<span class="md-off">not used for this queue</span>'}</h4>
      <dl class="md-dl">
        <dt>Heard</dt><dd>${m.transcript ? `<p class="md-ar" lang="ar" dir="rtl">${esc(m.transcript)}</p>` : '<span class="md-q">No transcript</span>'}${m.transcript && !differs ? `<small>${esc(m.diff.summary)}</small>` : ''}${m.heardNote ? `<small>${esc(m.heardNote)}</small>` : ''}</dd>
        ${differs ? `<dt>Against the label</dt><dd><p class="md-ar md-diff" lang="ar" dir="rtl">${diffHtml(m.diff.segments)}</p><small>${esc(m.diff.summary)}</small></dd>` : ''}
        <dt>Score</dt><dd class="md-q"><span title="${esc(m.how)}">${esc(m.score)}</span></dd>
      </dl></section>`;
    })
    .join('');
  return `${d.why ? `<p class="md-why">${esc(d.why)}.</p>` : ''}
    ${d.alternative ? `<p class="md-alt">Each model weighs the label against ${esc(d.alternative.name)}: <bdi lang="ar" dir="rtl">${esc(d.alternative.text)}</bdi></p>` : ''}
    ${rows}
    <p class="md-key"><del class="md-del">Struck</del> is in the label but wasn’t heard; <ins class="md-ins">shaded</ins> was heard but isn’t in the label. Words are compared without vowels.</p>`;
}

const CSS = `
.li{display:flex;flex-direction:column;margin-top:var(--s6)}
.li-title{font-size:var(--t-3xl);letter-spacing:-.015em;line-height:1.2}
.li-context{margin-top:var(--s2);color:var(--g-600);font-size:var(--t-sm)}
.li-play{display:flex;align-items:center;gap:var(--s4);flex-wrap:wrap;margin-top:var(--s5)}
.li-main{display:inline-flex;align-items:center;gap:var(--s3);border:0;background:none;padding:0;cursor:pointer;color:var(--g-900);text-align:left}
.li-main .li-disc{width:64px;height:64px;border-radius:50%;background:var(--green-500);color:#fff;display:grid;place-items:center;box-shadow:var(--shadow-1);transition:background .15s ease-out,transform .15s ease-out,box-shadow .15s ease-out}
.li-main .li-disc .icon{width:26px;height:26px}
.li-main:hover .li-disc{background:var(--green-600);box-shadow:var(--shadow-2)}
.li-main:active .li-disc{transform:scale(.96)}
.li-main b{display:block;font-size:var(--t-lg);font-weight:600;color:var(--g-900)}
.li-main b+span{display:flex;align-items:center;gap:6px;font-size:var(--t-sm);color:var(--g-600)}
.li-alt{display:inline-flex;align-items:center;gap:var(--s2);height:40px;padding:0 var(--s4) 0 var(--s3);border-radius:999px;border:1px solid var(--g-300);background:#fff;cursor:pointer;font-size:var(--t-sm);color:var(--g-900);transition:border-color .15s ease-out}
.li-alt:hover{border-color:var(--green-500)}
.li-alt .icon{width:14px;height:14px;color:var(--green-500)}
.li-alt[aria-pressed=true]{border-color:var(--green-500);background:var(--green-50)}
.li-alts{display:flex;gap:var(--s2);flex-wrap:wrap}
.li-time{height:3px;margin-top:var(--s4);background:var(--g-200);border-radius:2px;overflow:hidden}
.li-time i{display:block;height:100%;width:0;background:var(--g-700);transition:width .1s linear}
.li-label{margin-top:var(--s6);font-size:var(--t-xl);line-height:1.6}
.li-ar{font-family:var(--ar);font-size:1.4em;line-height:1;padding:0 2px}
.li-page{margin-top:var(--s4);padding:3px;border:1px solid var(--g-300);border-radius:var(--radius);background:#fff}
.li-text{font-family:var(--ar);font-size:var(--t-4xl);line-height:2.1;text-align:center;padding:var(--s5) var(--s6);border:1px solid var(--g-200);border-radius:6px;color:var(--g-900)}
.li-mark{background:none;color:inherit;border-bottom:2px solid var(--gold-400);padding:0 2px}
.li-ask{margin-top:var(--s6);font-size:var(--t-xl);font-weight:600}
.li-groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:var(--s5);margin-top:var(--s4)}
.li-group h3{font-size:var(--t-sm);font-weight:600;color:var(--g-900)}
.li-group h3 span{font-weight:400;color:var(--g-600)}
.li-opts{display:flex;flex-direction:column;gap:var(--s2);margin-top:var(--s2)}
.li-opts button{display:grid;grid-template-columns:auto 1fr auto;gap:2px var(--s3);align-items:center;text-align:left;padding:var(--s3) var(--s4);border-radius:var(--radius);border:1px solid var(--g-300);background:#fff;cursor:pointer;transition:border-color .15s ease-out,background .15s ease-out}
.li-opts button:hover{border-color:var(--g-500)}
.li-opts button b{font-weight:600}
.li-opts button small{grid-column:2;font-size:var(--t-sm);color:var(--g-600)}
.li-opts button kbd{grid-row:1/3;align-self:center}
.li-opts button .icon{grid-row:1/3;grid-column:3;opacity:0;width:18px;height:18px;stroke-width:2.25}
.li-opts button[aria-pressed=true]{border-color:var(--g-900);background:var(--g-100);box-shadow:inset 0 0 0 1px var(--g-900)}
.li-opts button[aria-pressed=true] .icon{opacity:1}
.li-opts button[data-kind=ok][aria-pressed=true]{border-color:var(--better);background:var(--better-50);box-shadow:inset 0 0 0 1px var(--better);color:var(--better)}
.li-opts button[data-kind=data][aria-pressed=true]{border-color:var(--worse);background:var(--worse-50);box-shadow:inset 0 0 0 1px var(--worse)}
.li-opts button[data-kind=data][aria-pressed=true] b,.li-opts button[data-kind=data][aria-pressed=true] .icon{color:var(--worse)}
.li-foot{display:flex;flex-wrap:wrap;align-items:flex-start;gap:var(--s3) var(--s5);margin-top:var(--s4)}
.li-note{flex:1 1 320px;font:inherit;font-size:var(--t-sm);padding:var(--s2) var(--s3);border-radius:var(--radius);border:1px solid var(--g-300);background:#fff;min-height:40px;resize:vertical;color:var(--g-900)}
.li-note::placeholder{color:var(--g-600)}
.li-note:focus{outline:2px solid var(--green-500);outline-offset:1px;border-color:transparent}
.li-status{display:flex;align-items:center;gap:6px;font-size:var(--t-sm);color:var(--g-600);min-height:40px}
.li-status.li-ok{color:var(--better)}
.li-status.li-err{color:var(--worse)}
.li-status .icon{width:14px;height:14px;stroke-width:2.25}
.md{margin-top:var(--s4)}
.md>summary{align-items:center}
.md>summary span{color:var(--g-600)}
.md>summary kbd{margin-left:auto}
.md>summary kbd+.icon{margin-left:0}
.md-body{padding:0 var(--s4) var(--s4);font-size:var(--t-sm);color:var(--g-700)}
.md-why,.md-none{color:var(--g-700)}
.md-alt{margin-top:var(--s1);color:var(--g-600)}
.md-alt bdi{font-family:var(--ar);font-size:var(--t-lg);color:var(--g-700);padding:0 2px}
.md-m{margin-top:var(--s4);padding-top:var(--s4);border-top:1px solid var(--g-200)}
.md-m h4{margin:0;display:flex;align-items:center;flex-wrap:wrap;gap:var(--s2);font-size:var(--t-sm);font-weight:400;color:var(--g-600)}
.md-m h4 b{font-size:var(--t-base);font-weight:600;color:var(--g-900)}
.md-off{font-size:var(--t-xs)}
.md-dl{display:grid;grid-template-columns:128px minmax(0,1fr);gap:var(--s2) var(--s4);margin:var(--s3) 0 0}
.md-dl dt{color:var(--g-600);padding-top:6px}
.md-dl dd{margin:0;min-width:0}
.md-dl dd.md-q{padding-top:6px}
.md-dl small{display:block;margin-top:2px;font-size:var(--t-xs);color:var(--g-600)}
.md-q{color:var(--g-600)}
.md-q span[title]{border-bottom:1px dotted var(--g-400);cursor:help}
.md-ar{font-family:var(--ar);font-size:var(--t-2xl);line-height:1.9;color:var(--g-900)}
.md-diff{color:var(--g-700)}
.md-del{text-decoration:line-through;text-decoration-color:var(--g-500);text-decoration-thickness:1px;color:var(--g-500)}
.md-ins{text-decoration:none;background:var(--g-200);color:var(--g-900);border-radius:4px;padding:0 4px}
.md-edge{opacity:.6}
.md-key{margin-top:var(--s4);font-size:var(--t-xs);color:var(--g-600)}
@media (max-width:600px){.md-dl{grid-template-columns:1fr;gap:2px}.md-dl dt{padding-top:var(--s2)}.md-dl dd.md-q{padding-top:0}}
`;

export function style() {
  if (document.getElementById('li-style')) return;
  const s = document.createElement('style');
  s.id = 'li-style';
  s.textContent = CSS;
  document.head.append(s);
}

export function mountItem(el, item, { base = '', review = null, verdicts = VERDICTS, aside = '', models = '', onVerdict } = {}) {
  style();
  const alts = CLIPS.filter(([k]) => k !== 'audio' && item[k]);
  const name = (k) => item[k].title || CLIPS.find(([c]) => c === k)[1];
  const index = new Map(verdicts.map((v, i) => [v.key, i]));
  const groups = GROUPS.map((g) => ({ ...g, items: [] }));
  for (const v of verdicts) groups.find((g) => g.test(v.key)).items.push(v);
  const n = verdicts.length;
  el.innerHTML = `<div class="li">
    <h1 class="li-title">${esc(item.title || (item.kind === 'splice' ? 'Planted edit' : 'Case'))}</h1>
    ${item.context ? `<p class="li-context">${esc(item.context)}</p>` : ''}
    <div class="li-play">
      <button class="li-main" data-clip="audio" aria-pressed="false" title="Key: space">
        <span class="li-disc">${icon('play')}</span>
        <span><b>${esc(name('audio'))}</b><span>${secs(item.audio.endS - item.audio.startS)} · <kbd>space</kbd></span></span>
      </button>
      ${alts.length ? `<span class="li-alts">${alts
        .map(([k]) => `<button class="li-alt" data-clip="${k}" aria-pressed="false" title="Key: ${KEYS[k]}">${icon('play')}${esc(name(k))} <kbd>${KEYS[k]}</kbd></button>`)
        .join('')}</span>` : ''}
    </div>
    <div class="li-time" aria-hidden="true"><i></i></div>
    <p class="li-label">${labelHtml(item.label)}</p>
    ${item.text ? `<div class="li-page"><p class="li-text" lang="ar" dir="rtl">${textHtml(item.text, item.label)}</p></div>` : ''}
    ${models ? `<details class="why md"${modelsOpen() ? ' open' : ''}>
      <summary><b>What the models heard</b><span>${esc(modelsSummary(item))}</span><kbd>m</kbd>${icon('right')}</summary>
      <div class="md-body" aria-live="polite"></div>
    </details>` : ''}
    ${aside}
    <h2 class="li-ask">What do you hear?</h2>
    <div class="li-groups">${groups
      .filter((g) => g.items.length)
      .map((g) => `<div class="li-group" role="group" aria-label="${esc(g.title)}"><h3>${esc(g.title)}${g.sub ? ` <span>· ${esc(g.sub)}</span>` : ''}</h3>
        <div class="li-opts">${g.items
          .map((v) => `<button data-v="${v.key}" data-kind="${verdictKind(v.key)}" aria-pressed="false" title="${esc(v.hint)}">
            <kbd>${index.get(v.key) + 1}</kbd><b>${esc(v.text)}</b><small>${esc(v.hint)}</small>${icon('check')}</button>`)
          .join('')}</div></div>`)
      .join('')}</div>
    <div class="li-foot">
      <textarea class="li-note" rows="1" placeholder="Add a note (optional, saved exactly as typed)"></textarea>
      <p class="li-status" role="status"></p>
    </div>
    <p class="li-context">Press <kbd>1</kbd>–<kbd>${n}</kbd> to choose; it saves and moves on. <button class="keys-link" data-keys>${icon('keyboard')}All shortcuts <kbd>?</kbd></button></p>
  </div>`;
  const root = el.firstElementChild;
  const $ = (s) => root.querySelector(s);
  const note = $('.li-note');
  const status = $('.li-status');
  const bar = $('.li-time i');
  const player = new Audio(clipUrl(item.audio, base));
  player.preload = 'auto';
  let current = 'audio';
  let saved = review;

  const press = (key) => {
    for (const b of root.querySelectorAll('[data-v]')) b.setAttribute('aria-pressed', String(key === b.dataset.v));
  };
  const setStatus = (html, cls = '') => {
    status.className = `li-status ${cls}`;
    status.innerHTML = html;
  };
  const show = () => {
    press(saved?.verdict);
    if (saved) setStatus(`${icon('check')}Saved ${new Date(saved.at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`, 'li-ok');
    else setStatus('Not reviewed yet');
  };
  const mark = () => {
    for (const b of root.querySelectorAll('[data-clip]')) {
      const on = b.dataset.clip === current && !player.paused;
      b.setAttribute('aria-pressed', String(on));
      const svg = b.querySelector('svg');
      if (svg) svg.outerHTML = icon(on ? 'pause' : 'play');
    }
  };
  player.addEventListener('play', mark);
  player.addEventListener('pause', mark);
  player.addEventListener('ended', mark);
  player.addEventListener('timeupdate', () => {
    bar.style.width = player.duration ? `${(100 * player.currentTime) / player.duration}%` : '0';
  });

  const play = (which = 'audio') => {
    if (!item[which]) return;
    if (current === which && !player.paused) return player.pause();
    if (current !== which) {
      current = which;
      player.src = clipUrl(item[which], base);
    } else if (player.ended) player.currentTime = 0;
    player.play().catch((e) => setStatus(`Could not play: ${esc(e.message)}`, 'li-err'));
  };

  const save = async (key, how) => {
    if (!onVerdict) return;
    press(key);
    setStatus('Saving…');
    try {
      saved = (await onVerdict(key, note.value, how)) ?? saved;
      show();
      return saved;
    } catch (e) {
      press(saved?.verdict);
      setStatus(`${icon('x')}Not saved: ${esc(e.message)}. Try again.`, 'li-err');
      throw e;
    }
  };

  $('.li-play').addEventListener('click', (e) => {
    const b = e.target.closest('[data-clip]');
    if (!b) return;
    b.blur();
    play(b.dataset.clip);
  });
  $('.li-groups').addEventListener('click', (e) => {
    const b = e.target.closest('[data-v]');
    if (!b) return;
    b.blur();
    save(b.dataset.v, 'button').catch(() => {});
  });
  note.value = review?.note ?? '';
  note.addEventListener('blur', () => {
    if (saved && note.value !== saved.note) save(saved.verdict, 'note').catch(() => {});
  });
  show();

  const panel = models ? $('.md') : null;
  let heard = false;
  const loadHeard = () => {
    if (heard) return;
    heard = true;
    const body = $('.md-body');
    body.innerHTML = '<p class="md-none">Loading…</p>';
    let req = heardCache.get(models);
    if (!req) {
      req = fetch(`${base}${models}`).then(async (r) => {
        const b = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(typeof b.detail === 'string' ? b.detail : `HTTP ${r.status}`);
        return b;
      });
      heardCache.set(models, req);
    }
    req.then(
      (d) => (body.innerHTML = modelsHtml(d)),
      (e) => {
        heardCache.delete(models);
        heard = false;
        body.innerHTML = `<p class="md-none">Couldn’t load the models’ output: ${esc(e.message)}</p>`;
      },
    );
  };
  if (panel) {
    panel.querySelector('summary').addEventListener('click', (e) => e.currentTarget.blur());
    panel.addEventListener('toggle', () => {
      try {
        localStorage.setItem(MODELS_OPEN, panel.open ? '1' : '0');
      } catch {}
      if (panel.open) loadHeard();
    });
    if (panel.open) loadHeard();
  }

  return {
    el,
    play,
    choose: (i) => (verdicts[i] ? save(verdicts[i].key, 'key') : Promise.resolve()),
    note: () => note.value,
    toggleModels: () => {
      if (panel) panel.open = !panel.open;
    },
    setReview: (r) => {
      saved = r;
      show();
    },
    destroy: () => {
      player.pause();
      player.removeAttribute('src');
      el.innerHTML = '';
    },
  };
}

// The keyboard shortcut sheet, toggled by `?` or any [data-keys] button.
export function keySheet(rows) {
  let sheet = null;
  const close = () => {
    sheet?.remove();
    sheet = null;
  };
  const toggle = () => {
    if (sheet) return close();
    sheet = document.createElement('div');
    sheet.className = 'sheet';
    sheet.setAttribute('role', 'dialog');
    sheet.setAttribute('aria-label', 'Keyboard shortcuts');
    sheet.innerHTML = `<h2>Keyboard shortcuts <button aria-label="Close">${icon('x')}</button></h2>
      <dl>${rows.map(([k, t]) => `<dt>${k.split(' ').map((x) => `<kbd>${esc(x)}</kbd>`).join('')}</dt><dd>${esc(t)}</dd>`).join('')}</dl>`;
    sheet.querySelector('button').onclick = close;
    document.body.append(sheet);
  };
  document.addEventListener('click', (e) => {
    if (e.target.closest('[data-keys]')) toggle();
  });
  return { toggle, close, open: () => !!sheet };
}

// The ayah-marker rail: one round marker per item, filled by its verdict, the current one ringed. Long lists show the
// first and last markers and a window around the current one.
export function railHtml(items, current, verdictOf, label = (i) => `Item ${i + 1}`) {
  const W = 12;
  const show = (i) => items.length <= 48 || i === 0 || i === items.length - 1 || Math.abs(i - current) <= W;
  let out = '';
  items.forEach((it, i) => {
    if (!show(i)) {
      if (show(i - 1)) out += '<span class="rail-gap" aria-hidden="true">…</span>';
      return;
    }
    const v = verdictOf(it);
    out += `<button data-go="${i}" class="${verdictKind(v)}" ${i === current ? 'aria-current="true"' : ''} aria-label="${esc(label(i, v))}" title="${esc(label(i, v))}">${i + 1}</button>`;
  });
  return `<nav class="rail" aria-label="Items">${out}</nav>`;
}
