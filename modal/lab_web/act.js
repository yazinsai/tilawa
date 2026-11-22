import { icon } from './item.js';
import { $, esc, qs, span, sw, NAMES, plainLines, sentence } from './al.js';

// Your commands from the page: approve, retry, drop, pause, resume and agree/disagree. Each one goes to the
// supervisor's inbox (POST /api/autolab/command) and is applied at the start of its next tick, about 2 minutes.
// Commands need the shared access code; this browser keeps it once it has worked, until you forget it.

const KEY = 'autolab-access-code';
const sent = new Map(); // experiment id, or 'lab' for pause/resume -> {op, at (ms), label}
const noteOpen = new Map(); // experiment id -> text typed so far in its "retry with a note" box
const RECENT_MS = 30_000;

export const hasCode = () => !!localStorage.getItem(KEY);

// A modal dialog: resolves to the field's text ('' without a field) or null when cancelled.
export function ask({ title, text = '', confirm, field = null }) {
  let d = $('al-dlg');
  if (!d) {
    d = document.createElement('dialog');
    d.id = 'al-dlg';
    d.className = 'al-dlg';
    document.body.append(d);
  }
  const f = field;
  const input = !f
    ? ''
    : f.multiline
      ? `<textarea name="v" rows="4" maxlength="${f.max}" ${f.required ? 'required' : ''} placeholder="${esc(f.placeholder || '')}"></textarea>`
      : `<input name="v" type="${f.type || 'text'}" maxlength="${f.max || 300}" ${f.required ? 'required' : ''} autocomplete="${f.type === 'password' ? 'current-password' : 'off'}" placeholder="${esc(f.placeholder || '')}">`;
  d.innerHTML = `<form method="dialog"><h2>${esc(title)}</h2>${text ? `<p>${text}</p>` : ''}
    ${f ? `<label class="al-field"><span>${esc(f.label)}</span>${input}</label>` : ''}
    <div class="al-dlg-b"><button class="btn go" value="ok">${esc(confirm)}</button><button class="btn" value="cancel" formnovalidate>Cancel</button></div></form>`;
  return new Promise((res) => {
    d.returnValue = '';
    d.onclose = () => res(d.returnValue === 'ok' ? (f ? d.querySelector('[name=v]').value.trim() : '') : null);
    d.showModal();
    (d.querySelector('[name=v]') || d.querySelector('.go')).focus();
  });
}

function toast(html, kind = '') {
  let t = $('al-toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'al-toast';
    t.setAttribute('role', 'status');
    t.setAttribute('aria-live', 'polite');
    document.body.append(t);
  }
  t.className = `al-toast ${kind}`;
  t.innerHTML = html;
  t.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => (t.hidden = true), kind === 'err' ? 9000 : 5000);
}

async function send(body, root) {
  for (let tries = 0; tries < 3; tries++) {
    let code = localStorage.getItem(KEY);
    if (!code) {
      code = await ask({
        title: 'Access code',
        text: tries
          ? '<b>That code wasn’t right.</b> Check it and try again.'
          : 'Commands change what autolab does, and approving a big bet spends money, so they need the shared access code. This browser keeps it until you choose to forget it.',
        confirm: 'Continue',
        field: { label: 'Access code', type: 'password', required: true, max: 200 },
      });
      if (code == null) return null;
    }
    const r = await fetch(`/api/autolab/command${root ? `?${qs({ root })}` : ''}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-autolab-token': code },
      body: JSON.stringify(body),
    });
    const out = await r.json().catch(() => ({}));
    if (r.status === 401) {
      localStorage.removeItem(KEY);
      continue;
    }
    if (!r.ok) throw new Error(typeof out.detail === 'string' ? out.detail : `The server said ${r.status}`);
    localStorage.setItem(KEY, code);
    return out;
  }
  throw new Error('The access code wasn’t right three times; nothing was sent.');
}

// What a sent command looks like until the supervisor has applied it.

function inFlight(id, pending) {
  const server = (pending || []).find((p) => (id === 'lab' ? p.op === 'pause' : p.id === id && p.op !== 'match'));
  const mine = sent.get(id);
  if (server) return { op: server.resume ? 'resume' : server.op, label: mine?.label || LABEL[server.resume ? 'resume' : server.op] };
  return mine && Date.now() - mine.at < RECENT_MS ? mine : null;
}

const LABEL = { approve: 'approve', retry: 'retry', drop: 'drop', pause: 'pause', resume: 'resume', match: 'your verdict' };

function turnedDown(id, events) {
  const mine = sent.get(id);
  if (!mine) return null;
  return (events || []).find((e) => e.to === 'rejected' && (id === 'lab' ? !e.id : e.id === id) && Date.parse(e.at) >= mine.at - 5000) || null;
}

const sentLine = (s, tick) => `<p class="al-sent" role="status">${icon('check')}<span><b>Sent: ${esc(s.label)}.</b> Autolab applies it within about ${span(tick || 120)}.</span></p>`;
const rejLine = (e) => `<p class="al-rej" role="status">${icon('x')}<span><b>Turned down:</b> ${esc((e.note || '').replace(/^your \w+: /, ''))}</span></p>`;

// The panel: everything waiting for you, each with what it is, why it waits and what you can do.

const BTN = {
  approve: (w) => `<button class="btn go" data-act="approve" data-id="${esc(w.id)}" data-usd="${esc(w.usd ?? '')}">Approve${w.usd != null ? ` ($${esc(w.usd)})` : ''}</button>`,
  retry: (w) => `<button class="btn" data-act="retry" data-id="${esc(w.id)}" title="Run its checks, tests and scoring again">Retry</button>`,
  retryNote: (w) => `<button class="btn" data-act="note" data-id="${esc(w.id)}" aria-expanded="${noteOpen.has(w.id)}">Retry with a note…</button>`,
  drop: (w) => `<button class="btn" data-act="drop" data-id="${esc(w.id)}" data-kind="${esc(w.kind)}">${w.kind === 'approval' ? 'Drop' : 'Drop…'}</button>`,
};

function noteForm(w) {
  return `<form class="al-note" data-note="${esc(w.id)}"><label class="al-field"><span>What should the builder change? Your note goes back to it as feedback.</span>
      <textarea name="note" rows="3" maxlength="1000" required>${esc(noteOpen.get(w.id) || '')}</textarea></label>
    <div class="al-note-b"><button class="btn go">Send back to the builder</button><button class="btn" type="button" data-act="note" data-id="${esc(w.id)}">Cancel</button></div></form>`;
}

function item(w, v, events) {
  const tick = v.heartbeat?.tickSeconds;
  const s = inFlight(w.id, v.pending);
  const rej = !s && turnedDown(w.id, events);
  const who = [NAMES[w.component] || w.component, w.tierText, `waiting ${span(w.ageSeconds)}`].filter(Boolean).join(' · ');
  const acts = s ? sentLine(s, tick) : `${rej ? rejLine(rej) : ''}<div class="al-acts">${w.actions.map((a) => BTN[a](w)).join('')}</div>`;
  return `<article class="al-wait${s ? ' is-sent' : ''}" data-wait="${esc(w.id)}"><div class="al-wait-t">
    <h3 class="exp-id">${sw(w.component)}${esc(w.id)} <small>${esc(who)}</small></h3>
    ${plainLines({ eli5: w.eli5, hypothesis: w.hypothesis, expectText: w.expectText })}
    <p class="al-why" title="${esc(w.what)}"><b>Why it’s waiting:</b> ${esc(w.whyText)}</p></div>
    <div class="al-wait-a">${acts}</div>${!s && noteOpen.has(w.id) ? noteForm(w) : ''}</article>`;
}

export function waitingPanel(items, v, events) {
  if (!items?.length) return '';
  const n = items.length;
  return `<section class="al-waiting" aria-labelledby="al-w-h"><div class="al-w-h"><h2 id="al-w-h">Waiting for you <span>${n}</span></h2>
      <p>${n === 1 ? 'Autolab has stopped on this one until you decide.' : 'Autolab has stopped on these until you decide.'} Your answer reaches it within about ${span(v.heartbeat?.tickSeconds || 120)}.</p></div>
    ${items.map((w) => item(w, v, events)).join('')}</section>`;
}

// Keep the panel as it is while you're typing a note, so a refresh doesn't eat it.
export const typing = () => !!document.activeElement?.closest?.('.al-note') || $('al-dlg')?.open;

// Pause and resume, for the top bar and the paused notice.

export function pauseControl(v) {
  if (!v?.started) return '';
  const s = inFlight('lab', v.pending);
  if (s) return `<span class="al-pz sent">${icon('check')}${s.op === 'resume' ? 'Resume' : 'Pause'} sent</span>`;
  return v.heartbeat?.paused
    ? `<button class="btn sm go" data-act="resume">Resume</button>`
    : `<button class="btn sm" data-act="pause" title="Running work finishes; nothing new starts until you resume">Pause</button>`;
}

export function pausedNotice(v, events) {
  const why = v.heartbeat?.paused;
  if (!why) return '';
  const s = inFlight('lab', v.pending);
  const rej = !s && turnedDown('lab', events);
  return `<div class="notice warn al-paused"><span><b>Autolab is paused:</b> ${esc(sentence(why))}. Work already running finishes; nothing new starts until you resume it.
      ${/test data/i.test(why) ? ' Resuming also accepts the test data as it is now.' : ''}</span>
    ${s ? sentLine(s, v.heartbeat?.tickSeconds) : `${rej ? rejLine(rej) : ''}<button class="btn sm go" data-act="resume">Resume</button>`}</div>`;
}

export const forgetLink = () => (hasCode() ? `<button class="al-forget" data-act="forget">Forget the access code on this device</button>` : '');

export function dropButton(id) {
  return `<button class="btn sm" data-act="drop" data-id="${esc(id)}" data-kind="running">Drop…</button>`;
}

export function verdictAsk(r, matches, pending) {
  if (!['folded', 'discarded'].includes(r.state)) return '';
  const mine = matches?.[r.id];
  const s = (pending || []).find((p) => p.id === r.id && p.op === 'match') || (sent.get(`m:${r.id}`) && Date.now() - sent.get(`m:${r.id}`).at < RECENT_MS);
  if (s) return `<p class="al-match">${icon('check')}Your verdict is on its way</p>`;
  const b = (v, t) => `<button class="btn sm${mine === v ? ' on' : ''}" data-act="match" data-id="${esc(r.id)}" data-verdict="${v}" aria-pressed="${mine === v}">${t}</button>`;
  return `<p class="al-match"><span>${mine ? `You ${mine === 'agree' ? 'agreed' : 'disagreed'}.` : `Would you have ${r.state === 'folded' ? 'kept' : 'dropped'} it too?`}</span>${b('agree', 'Agree')}${b('disagree', 'Disagree')}</p>`;
}

// One handler for both pages. `refresh` reloads the page's data; `rerender` redraws from what it already has.

export function wire({ root, refresh, rerender }) {
  const go = async (btn, body, key, label) => {
    btn && (btn.disabled = true);
    try {
      const out = await send(body, root);
      if (!out) return;
      sent.set(key, { op: body.op, at: Date.now(), label });
      noteOpen.delete(body.id);
      toast(`${icon('check')}<span><b>Sent.</b> Autolab applies it within about ${span(out.appliesWithin)}.</span>`);
      refresh();
    } catch (e) {
      toast(`${icon('x')}<span><b>Not sent:</b> ${esc(e.message)}</span>`, 'err');
    } finally {
      btn && (btn.disabled = false);
    }
  };
  document.addEventListener('click', async (e) => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const { act, id } = b.dataset;
    if (act === 'forget') {
      localStorage.removeItem(KEY);
      toast(`${icon('check')}<span>This device no longer has the access code.</span>`);
      return rerender();
    }
    if (act === 'note') {
      if (noteOpen.has(id)) noteOpen.delete(id);
      else noteOpen.set(id, '');
      rerender();
      document.querySelector(`[data-note="${CSS.escape(id)}"] textarea`)?.focus();
      return;
    }
    if (act === 'retry') return go(b, { op: 'retry', id, note: '' }, id, 'retry its checks, tests and scoring');
    if (act === 'approve') {
      const usd = b.dataset.usd;
      const ok = await ask({
        title: `Approve ${id}?`,
        text: `This big bet starts as soon as autolab picks it up${usd ? ` and can spend about <b>$${esc(usd)}</b> of GPU time on Modal` : ''}. You can still drop it later.`,
        confirm: usd ? `Approve $${usd}` : 'Approve',
      });
      return ok == null ? null : go(b, { op: 'approve', id }, id, `approve${usd ? ` ($${usd})` : ''}`);
    }
    if (act === 'drop') {
      const bet = b.dataset.kind === 'approval';
      const why = await ask({
        title: `Drop ${id}?`,
        text: `It ends here${b.dataset.kind === 'running' ? ': any agent still writing it is stopped' : ''} and can’t be resumed; a new card can try the idea again later.`,
        confirm: 'Drop it',
        field: { label: bet ? 'Why (optional)' : 'Why, in a few words', required: !bet, max: 300, placeholder: bet ? 'too expensive for now' : 'e.g. the approach is wrong' },
      });
      return why == null ? null : go(b, { op: 'drop', id, why: why || (bet ? 'too expensive for now' : '') }, id, 'drop');
    }
    if (act === 'pause') {
      const why = await ask({
        title: 'Pause autolab?',
        text: 'Work already running finishes; nothing new starts until you resume.',
        confirm: 'Pause',
        field: { label: 'Why, in a few words', required: true, max: 300, placeholder: 'e.g. checking the new test data' },
      });
      return why == null ? null : go(b, { op: 'pause', why }, 'lab', 'pause');
    }
    if (act === 'resume') return go(b, { op: 'resume' }, 'lab', 'resume');
    if (act === 'match') return go(b, { op: 'match', id, verdict: b.dataset.verdict }, `m:${id}`, 'your verdict');
  });
  document.addEventListener('input', (e) => {
    const f = e.target.closest?.('.al-note');
    if (f) noteOpen.set(f.dataset.note, e.target.value);
  });
  document.addEventListener('submit', (e) => {
    const f = e.target.closest('.al-note');
    if (!f) return;
    e.preventDefault();
    const note = f.querySelector('textarea').value.trim();
    if (note) go(f.querySelector('.go'), { op: 'retry', id: f.dataset.note, note }, f.dataset.note, 'retry with your note');
  });
}
