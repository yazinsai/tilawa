// One review item: play buttons, labelled Arabic words, the label in plain words, verdict buttons and a note.
// Used by the review page and the lab dashboard drill-down:
//   import { mountItem, VERDICTS, FAULTS } from '/static/item.js';
//   const view = mountItem(el, item, { review, verdicts: [...VERDICTS, ...FAULTS], onVerdict: (verdict, note, how) => save() });
//   view.play('audio' | 'before' | 'after'); view.choose(0); view.destroy();

export const VERDICTS = [
  { key: 'ok', text: 'Sounds right', hint: 'The audio matches the label.' },
  { key: 'bad_label', text: 'Bad label', hint: 'The words or the mistake shown do not match what you hear.' },
  { key: 'bad_audio', text: 'Bad audio', hint: 'Noise, clipping, silence or a cut-off makes it unusable.' },
  { key: 'bad_splice', text: 'Bad splice', hint: 'You can hear the edit: a click, jump or odd join.' },
];
export const FAULTS = [
  { key: 'listener', text: 'Listener fault', hint: 'The data is fine; the engine misheard the sounds.' },
  { key: 'tracker', text: 'Tracker fault', hint: 'The data is fine; the engine lost its place.' },
  { key: 'checker', text: 'Checker fault', hint: 'The data is fine; the engine judged the mistake wrongly.' },
];
const CLIPS = [
  ['audio', 'Play'],
  ['before', 'Original before'],
  ['after', 'Original after'],
];

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const ar = (s) => `“<bdi class="li-ar" lang="ar" dir="rtl">${esc(s)}</bdi>”`;

export function labelHtml(label = {}) {
  const { type, expected, said } = label;
  switch (type) {
    case 'wrong_word': return `Planted: said ${ar(said)} instead of ${ar(expected)}`;
    case 'omitted_word': return `Planted: left out ${ar(expected)}`;
    case 'added_word': return `Planted: added ${ar(said)}`;
    case 'skipped_passage': return `Planted: skipped ${ar(expected)}`;
    case 'repeat': return `Not a mistake: repeated ${ar(said || expected)}`;
    case 'restart': return `Not a mistake: went back and started again at ${ar(said || expected)}`;
    case 'self_correction': return `Not a mistake: fixed it to ${ar(expected)}`;
    case 'clean': return 'No mistake here';
    default: return `Label: ${esc(type)}${expected ? ` · expected ${ar(expected)}` : ''}${said ? ` · said ${ar(said)}` : ''}`;
  }
}

export const clipUrl = (clip, base = '') =>
  `${base}/audio?path=${encodeURIComponent(clip.path)}&start=${clip.startS}&end=${clip.endS}`;

const CSS = `
.li{display:flex;flex-direction:column;gap:18px;max-width:760px;margin:0 auto}
.li-title{font-size:22px;font-weight:600;margin:0}
.li-context{color:#666;font-size:14px;margin-top:-12px}
.li-label{font-size:20px;line-height:1.6}
.li-ar{font-family:"Amiri","Noto Naskh Arabic","Geeza Pro",serif;font-size:1.15em}
.li-text{font-family:"Amiri","Noto Naskh Arabic","Geeza Pro",serif;font-size:34px;line-height:1.9;text-align:right;background:#faf8f3;border-radius:12px;padding:14px 20px}
.li-play{display:flex;gap:12px;flex-wrap:wrap}
.li-play button{font-size:20px;padding:16px 26px;border-radius:14px;border:0;background:#1f5f4a;color:#fff;cursor:pointer;min-width:150px}
.li-play button.li-alt{background:#e7efe9;color:#1f5f4a}
.li-play button[aria-pressed=true]{outline:3px solid #f2b84b}
.li-ask{font-size:15px;color:#444}
.li-verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.li-verdicts button{font-size:17px;padding:14px 10px;border-radius:12px;border:2px solid #ccc;background:#fff;cursor:pointer;text-align:left}
.li-verdicts button kbd{float:right;color:#999;font-family:inherit}
.li-verdicts button[data-v=ok][aria-pressed=true]{border-color:#1f8a4c;background:#e6f5ec}
.li-verdicts button[aria-pressed=true]{border-color:#c0392b;background:#fbeaea}
.li-note{font-size:16px;padding:10px;border-radius:10px;border:1px solid #ccc;min-height:60px;font-family:inherit;resize:vertical}
.li-status{font-size:14px;color:#666;min-height:1.2em}
.li-status.li-err{color:#c0392b}
`;

function style() {
  if (document.getElementById('li-style')) return;
  const s = document.createElement('style');
  s.id = 'li-style';
  s.textContent = CSS;
  document.head.append(s);
}

export function mountItem(el, item, { base = '', review = null, verdicts = VERDICTS, onVerdict } = {}) {
  style();
  const clips = CLIPS.filter(([k]) => item[k]);
  const keys = { audio: 'space', before: 'b', after: 'a' };
  el.innerHTML = `<div class="li">
    <h2 class="li-title">${esc(item.title || (item.kind === 'splice' ? 'Planted edit' : 'Case'))}</h2>
    ${item.context ? `<div class="li-context">${esc(item.context)}</div>` : ''}
    <div class="li-play">${clips
      .map(([k, t]) => `<button data-clip="${k}" class="${k === 'audio' ? '' : 'li-alt'}" title="Key: ${keys[k]}">▶ ${esc(item[k].title || t)}</button>`)
      .join('')}</div>
    <div class="li-label">${labelHtml(item.label)}</div>
    ${item.text ? `<div class="li-text" lang="ar" dir="rtl">${esc(item.text)}</div>` : ''}
    <div class="li-ask">Does the audio match? If not, what is wrong?</div>
    <div class="li-verdicts">${verdicts
      .map((v, i) => `<button data-v="${v.key}" title="${esc(v.hint)}" aria-pressed="false">${esc(v.text)}<kbd>${i + 1}</kbd></button>`)
      .join('')}</div>
    <textarea class="li-note" placeholder="Note (optional, saved exactly as typed)"></textarea>
    <div class="li-status"></div>
  </div>`;
  const $ = (s) => el.querySelector(s);
  const note = $('.li-note');
  const status = $('.li-status');
  const player = new Audio(clipUrl(item.audio, base));
  player.preload = 'auto';
  let current = 'audio';
  let saved = review;

  const press = (key) => {
    for (const b of el.querySelectorAll('[data-v]')) b.setAttribute('aria-pressed', String(key === b.dataset.v));
  };
  const show = () => {
    press(saved?.verdict);
    status.classList.remove('li-err');
    status.textContent = saved ? `Saved ${new Date(saved.at).toLocaleString()}` : 'Not reviewed yet';
  };
  const buttons = () => el.querySelectorAll('[data-clip]');
  const mark = () => {
    for (const b of buttons()) {
      const on = b.dataset.clip === current && !player.paused;
      b.setAttribute('aria-pressed', String(on));
      b.textContent = `${on ? '❚❚' : '▶'} ${item[b.dataset.clip].title || CLIPS.find(([k]) => k === b.dataset.clip)[1]}`;
    }
  };
  player.addEventListener('play', mark);
  player.addEventListener('pause', mark);
  player.addEventListener('ended', mark);

  const play = (which = 'audio') => {
    if (!item[which]) return;
    if (current === which && !player.paused) return player.pause();
    if (current !== which) {
      current = which;
      player.src = clipUrl(item[which], base);
    } else if (player.ended) player.currentTime = 0;
    player.play().catch((e) => {
      status.classList.add('li-err');
      status.textContent = `Could not play: ${e.message}`;
    });
  };

  const save = async (key, how) => {
    if (!onVerdict) return;
    press(key);
    status.classList.remove('li-err');
    status.textContent = 'Saving…';
    try {
      saved = (await onVerdict(key, note.value, how)) ?? saved;
      show();
      return saved;
    } catch (e) {
      press(saved?.verdict);
      status.classList.add('li-err');
      status.textContent = `Not saved: ${e.message}. Try again.`;
      throw e;
    }
  };

  el.querySelector('.li-play').addEventListener('click', (e) => {
    const b = e.target.closest('[data-clip]');
    if (!b) return;
    b.blur();
    play(b.dataset.clip);
  });
  el.querySelector('.li-verdicts').addEventListener('click', (e) => {
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

  return {
    el,
    play,
    choose: (i) => (verdicts[i] ? save(verdicts[i].key, 'key') : Promise.resolve()),
    note: () => note.value,
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
