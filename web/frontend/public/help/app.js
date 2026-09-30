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
let introErrorKey = '';
let busy = false;

function applyLang() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'ar' ? 'rtl' : 'ltr';
  document.title = t('title');
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of document.querySelectorAll('.sid')) el.textContent = speaker;
  $('lang').lang = lang === 'ar' ? 'en' : 'ar';
  $('rec').setAttribute('aria-label', t(recState === 'recOn' ? 'recStop' : 'recAria'));
  $('rec').setAttribute('aria-pressed', String(recState === 'recOn'));
  $('intro-error').textContent = introErrorKey ? t(introErrorKey) : '';
  $('rec-status').textContent = t(recState);
  if (errorKey === 'errLoad') $('skip').textContent = t('retry');
  if (busy && blob) $('submit').querySelector('[data-i18n]').textContent = t('sending');
  $('error').textContent = errorKey ? t(errorKey) : '';
  $('share-message').value = t('shareMessage');
  $('share-whatsapp').href = `https://wa.me/?text=${encodeURIComponent(t('shareMessage'))}`;
  $('share-status').textContent = '';
  renderCount();
  if (prompt) renderPrompt();
}

$('lang').addEventListener('click', () => {
  lang = lang === 'ar' ? 'en' : 'ar';
  localStorage.lang = lang;
  applyLang();
});

function profileValue(key) {
  return key === 'device' ? $('device').value : document.querySelector(`input[name="${key}"]:checked`)?.value ?? '';
}

function saveProfile() {
  localStorage.profile = JSON.stringify(Object.fromEntries(['device', 'gender', 'level'].map(key => [key, profileValue(key)])));
}

const saved = JSON.parse(localStorage.profile ?? '{}');
$('device').value = saved.device || guessDevice();
for (const key of ['gender', 'level']) {
  for (const input of document.querySelectorAll(`input[name="${key}"]`)) {
    input.checked = input.value === (saved[key] ?? '');
  }
}
document.querySelector('.profile').addEventListener('change', saveProfile);

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

fetch('api/config').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(c => {
  consentVersion = c.consent;
  if (localStorage.consent === consentVersion) $('consent').checked = true;
  $('start').disabled = !$('consent').checked;
}).catch(() => { introErrorKey = 'errConfig'; $('intro-error').textContent = t(introErrorKey); });
$('consent').addEventListener('change', () => { $('start').disabled = !$('consent').checked || !consentVersion; });

$('start').addEventListener('click', async () => {
  if (busy) return;
  busy = true;
  $('start').disabled = true;
  introErrorKey = '';
  $('intro-error').textContent = '';
  localStorage.consent = consentVersion;
  saveProfile();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true, channelCount: 1 } });
  } catch {
    introErrorKey = 'errMic';
    $('intro-error').textContent = t(introErrorKey);
    $('start').disabled = false;
    busy = false;
    return;
  }
  $('intro').hidden = true;
  document.body.classList.add('recording');
  $('task').hidden = false;
  busy = false;
  await nextPrompt();
});

function showError(key) {
  errorKey = key;
  $('error').textContent = key ? t(key) : '';
}

function setRecState(key) {
  recState = key;
  $('rec-status').textContent = t(key);
  $('rec').setAttribute('aria-label', t(key === 'recOn' ? 'recStop' : 'recAria'));
  $('rec').setAttribute('aria-pressed', String(key === 'recOn'));
  $('skip').disabled = key === 'recOn';
  $('finish').disabled = key === 'recOn';
}

function renderCount() {
  $('count').textContent = count ? t('count', count) : '';
  $('finish').hidden = !count;
  $('done-title').textContent = t('doneTitle', count);
}

$('finish').addEventListener('click', () => {
  if (recorder?.state === 'recording') recorder.stop();
  stream?.getTracks().forEach(track => track.stop());
  stream = null;
  $('playback').pause();
  $('task').hidden = true;
  $('done').hidden = false;
  if (localStorage.subscribed) {
    $('subscribe').hidden = true;
    $('subscribed').hidden = false;
  }
  scrollTo({ top: 0 });
  $('done-title').focus({ preventScroll: true });
});

$('more').addEventListener('click', () => {
  $('done').hidden = true;
  $('task').hidden = false;
  nextPrompt();
});

$('share-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(t('shareMessage'));
    $('share-status').textContent = t('shareCopied');
  } catch {
    $('share-message').focus();
    $('share-message').select();
    $('share-status').textContent = t('shareCopyFallback');
  }
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

function setBusy(value) {
  busy = value;
  for (const id of ['rec', 'redo', 'skip', 'finish', 'followed', 'extra']) $(id).disabled = value;
  $('submit').disabled = value || !$('followed').checked;
  $('task').setAttribute('aria-busy', String(value));
}

async function nextPrompt() {
  resetRecording();
  showError('');
  setBusy(true);
  prompt = null;
  $('text').replaceChildren();
  $('instruction').textContent = '';
  $('ref').textContent = '';
  $('kind-badge').textContent = '';
  const kind = new URLSearchParams(location.search).get('kind');
  try {
    const res = await fetch(`api/prompt${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`);
    if (!res.ok) throw new Error();
    prompt = await res.json();
    renderPrompt();
    scrollTo({ top: 0 });
    $('instruction').focus({ preventScroll: true });
  } catch {
    showError('errLoad');
  } finally {
    setBusy(false);
    $('rec').disabled = !prompt;
    $('skip').textContent = t(prompt ? 'skip' : 'retry');
  }
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
  $('task').classList.remove('reviewing');
  $('playback').pause();
  if ($('playback').src) URL.revokeObjectURL($('playback').src);
  $('playback').removeAttribute('src');
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

$('rec').addEventListener('click', async () => {
  if (busy || !prompt) return;
  if (recorder?.state === 'recording') return recorder.stop();
  if (!stream?.active) {
    setBusy(true);
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true, channelCount: 1 } });
    } catch {
      showError('errMic');
      return;
    } finally {
      setBusy(false);
    }
  }
  showError('');
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
    $('task').classList.add('reviewing');
    $('followed').focus({ preventScroll: true });
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
$('redo').addEventListener('click', () => { resetRecording(); $('rec').focus(); });
$('skip').addEventListener('click', () => { if (recorder?.state !== 'recording') nextPrompt(); });

$('submit').addEventListener('click', async () => {
  if (busy || !blob || !prompt || !$('followed').checked) return;
  if (duration < 1) return showError('errShort');
  setBusy(true);
  $('submit').querySelector('[data-i18n]').textContent = t('sending');
  showError('');
  const q = new URLSearchParams({
    seed: String(prompt.seed),
    speaker,
    consent: consentVersion,
    followed: '1',
    extra: $('extra').checked ? '1' : '0',
    duration: duration.toFixed(2),
    device: $('device').value,
    gender: profileValue('gender'),
    level: profileValue('level'),
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
    showError(typeof err === 'string' ? err : 'errUpload');
  } finally {
    setBusy(false);
    $('rec').disabled = !prompt;
    $('submit').querySelector('[data-i18n]').textContent = t('submit');
  }
});
