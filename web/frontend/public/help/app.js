import { STRINGS } from './i18n.js';

const $ = id => document.getElementById(id);
const MAX_SECONDS = 60;
let lang = localStorage.lang ?? (navigator.language?.startsWith('ar') ? 'ar' : 'en');
const t = (key, ...args) => {
  const v = STRINGS[lang][key] ?? STRINGS.en[key];
  return typeof v === 'function' ? v(...args) : v;
};

const speaker = localStorage.speaker ?? (localStorage.speaker = crypto.randomUUID().replaceAll('-', '').slice(0, 20));
let count = Number(localStorage.count ?? 0);
let consentVersion = '';
let prompt = null;
let recorder = null;
let stream = null;
let chunks = [];
let blob = null;
let startedAt = 0;
let duration = 0;
let tick = null;
let recState = 'recIdle';
let errorKey = '';

function applyLang() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.title = t('title');
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  $('rec').setAttribute('aria-label', t('recAria'));
  $('rec-status').textContent = t(recState);
  $('error').textContent = errorKey ? t(errorKey) : '';
  renderCount();
  if (prompt) renderPrompt();
}

$('lang').addEventListener('click', () => {
  lang = lang === 'ar' ? 'en' : 'ar';
  localStorage.lang = lang;
  applyLang();
});

const saved = JSON.parse(localStorage.profile ?? '{}');
for (const k of ['device', 'gender', 'level']) if (saved[k] !== undefined) $(k).value = saved[k];
if (!saved.device) $('device').value = guessDevice();

// iPadOS reports a Mac UA, so touch support is the tell; Android tablets omit "Mobile".
function guessDevice() {
  const ua = navigator.userAgent;
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 0)) return 'tablet';
  if (/Android/.test(ua) && !/Mobile/.test(ua)) return 'tablet';
  if (navigator.userAgentData?.mobile || /iPhone|iPod|Android.*Mobile|Mobi/.test(ua)) return 'phone';
  if (matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 600) return 'phone';
  return 'laptop';
}
applyLang();

fetch('api/config').then(r => r.json()).then(c => {
  consentVersion = c.consent;
  if (localStorage.consent === consentVersion) $('consent').checked = true;
  $('start').disabled = !$('consent').checked;
});
$('consent').addEventListener('change', () => { $('start').disabled = !$('consent').checked || !consentVersion; });

$('start').addEventListener('click', async () => {
  localStorage.consent = consentVersion;
  localStorage.profile = JSON.stringify({ device: $('device').value, gender: $('gender').value, level: $('level').value });
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true, channelCount: 1 } });
  } catch {
    return alert(t('errMic'));
  }
  $('intro').hidden = true;
  document.body.classList.add('recording');
  $('task').hidden = false;
  await nextPrompt();
});

function showError(key) {
  errorKey = key;
  $('error').textContent = key ? t(key) : '';
}

function setRecState(key) {
  recState = key;
  $('rec-status').textContent = t(key);
}

function renderCount() {
  $('count').textContent = count ? t('count', count) : '';
  $('finish').hidden = !count;
  $('done-title').textContent = t('doneTitle', count);
}

$('finish').addEventListener('click', () => {
  if (recorder?.state === 'recording') recorder.stop();
  $('task').hidden = true;
  $('done').hidden = false;
  if (localStorage.subscribed) {
    $('subscribe').hidden = true;
    $('subscribed').hidden = false;
  }
  scrollTo({ top: 0 });
});

$('more').addEventListener('click', () => {
  $('done').hidden = true;
  $('task').hidden = false;
  nextPrompt();
});

$('subscribe').addEventListener('submit', async e => {
  e.preventDefault();
  $('sub-error').textContent = '';
  const res = await fetch('api/subscribe', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ email: $('email').value, lang }),
  }).catch(() => null);
  if (!res?.ok) {
    $('sub-error').textContent = t(res?.status === 400 ? 'errEmail' : res?.status === 429 ? 'errRate' : 'errUpload');
    return;
  }
  localStorage.subscribed = '1';
  $('subscribe').hidden = true;
  $('subscribed').hidden = false;
});

async function nextPrompt() {
  resetRecording();
  showError('');
  const kind = new URLSearchParams(location.search).get('kind');
  prompt = await (await fetch(`api/prompt${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`)).json();
  renderPrompt();
  scrollTo({ top: 0 });
}

function renderPrompt() {
  $('task').dataset.kind = prompt.kind;
  $('kind-badge').textContent = t(`kind.${prompt.kind}`);
  const first = prompt.display[0];
  $('ref').textContent = `${prompt.display.map(a => a.ref).join(' · ')} — ${lang === 'ar' ? first.surahNameAr : first.surahName}`;
  $('instruction').textContent = prompt.kind === 'none' ? t('noneInstruction') : prompt.instruction[lang];
  $('followed-label').textContent = t(`confirm.${prompt.kind}`);
  $('extra-label').textContent = t(prompt.kind === 'none' ? 'extra.none' : 'extra.other');

  const text = $('text');
  text.replaceChildren();
  for (const ayah of prompt.display) {
    const div = document.createElement('div');
    div.className = `ayah ${ayah.mark ?? ''}`;
    if (ayah.mark) div.dataset.kind = prompt.kind;
    ayah.words.forEach((w, i) => {
      if (i) div.append(' ');
      const span = document.createElement('span');
      span.className = 'w';
      span.textContent = w.text;
      if (w.mark) {
        span.classList.add('mark', w.mark);
        span.dataset.kind = w.mark;
      }
      div.append(span);
    });
    const no = document.createElement('span');
    no.className = 'ayah-no';
    no.textContent = `(${ayah.ref.split(':')[1]})`;
    div.append(' ', no);
    text.append(div);
  }
}

function resetRecording() {
  blob = null;
  chunks = [];
  duration = 0;
  $('review').hidden = true;
  $('followed').checked = false;
  $('extra').checked = false;
  $('submit').disabled = true;
  $('timer').textContent = '0:00';
  setRecState('recIdle');
  $('rec').classList.remove('on');
  $('rec').disabled = false;
}

function pickMime() {
  for (const m of ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus', 'audio/webm']) {
    if (window.MediaRecorder?.isTypeSupported?.(m)) return m;
  }
  return '';
}

function fmt(s) {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

$('rec').addEventListener('click', () => {
  if (recorder?.state === 'recording') return recorder.stop();
  resetRecording();
  const mime = pickMime();
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    clearInterval(tick);
    duration = (performance.now() - startedAt) / 1000;
    blob = new Blob(chunks, { type: recorder.mimeType || mime || 'audio/webm' });
    $('rec').classList.remove('on');
    setRecState('recDone');
    $('playback').src = URL.createObjectURL(blob);
    $('review').hidden = false;
  };
  recorder.start();
  startedAt = performance.now();
  $('rec').classList.add('on');
  setRecState('recOn');
  tick = setInterval(() => {
    const s = (performance.now() - startedAt) / 1000;
    $('timer').textContent = fmt(s);
    if (s >= MAX_SECONDS) recorder.stop();
  }, 200);
});

$('followed').addEventListener('change', () => { $('submit').disabled = !$('followed').checked; });
$('redo').addEventListener('click', resetRecording);
$('skip').addEventListener('click', () => { if (recorder?.state !== 'recording') nextPrompt(); });

$('submit').addEventListener('click', async () => {
  if (!blob || !prompt) return;
  if (duration < 1) return showError('errShort');
  $('submit').disabled = true;
  showError('');
  const q = new URLSearchParams({
    seed: String(prompt.seed),
    speaker,
    consent: consentVersion,
    followed: '1',
    extra: $('extra').checked ? '1' : '0',
    duration: duration.toFixed(2),
    device: $('device').value,
    gender: $('gender').value,
    level: $('level').value,
  });
  try {
    const res = await fetch(`api/submit?${q}`, { method: 'POST', headers: { 'content-type': blob.type.split(';')[0] }, body: blob });
    if (res.status === 429) throw 'errRate';
    if (res.status === 409) throw 'errDup';
    if (!res.ok) throw 'errUpload';
    localStorage.count = ++count;
    renderCount();
    await nextPrompt();
  } catch (err) {
    $('submit').disabled = false;
    showError(typeof err === 'string' ? err : 'errUpload');
  }
});
