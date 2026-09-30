const $ = id => document.getElementById(id);
const MAX_SECONDS = 60;
const KIND_LABEL = {
  none: 'No mistake',
  skip_word: 'Skip a word',
  skip_ayah: 'Skip an ayah',
  substitution: 'Change a word',
  repeat: 'Repeat a word',
  vowel: 'Wrong haraka',
  tajweed: 'Tajweed slip',
};
const CONFIRM = {
  none: 'I read it correctly',
  skip_word: 'I skipped the highlighted word',
  skip_ayah: 'I skipped the crossed-out ayah',
  substitution: 'I said the other word instead',
  repeat: 'I said the highlighted word twice',
  vowel: 'I read the word with the changed haraka',
  tajweed: 'I made the tajweed slip',
};
const EXTRA = {
  none: 'I slipped up by accident somewhere',
  other: 'I also made a different mistake by accident',
};
const TAG = { repeat: '×2' };

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

const saved = JSON.parse(localStorage.profile ?? '{}');
for (const k of ['device', 'gender', 'level']) if (saved[k] !== undefined) $(k).value = saved[k];
if (!saved.device && /Mobi|Android|iPhone/i.test(navigator.userAgent)) $('device').value = 'phone';
if (!saved.device && !/Mobi|Android|iPhone/i.test(navigator.userAgent)) $('device').value = 'laptop';

fetch('/api/config').then(r => r.json()).then(c => {
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
    return showError('Microphone access is needed to record. Allow it in your browser settings and try again.');
  }
  $('intro').hidden = true;
  document.body.classList.add('recording');
  $('task').hidden = false;
  renderCount();
  await nextPrompt();
});

function showError(msg) {
  $('error').textContent = msg;
  if (!$('intro').hidden) alert(msg);
}

function renderCount() {
  $('count').textContent = count ? `${count} submitted — thank you` : '';
}

async function nextPrompt() {
  resetRecording();
  $('error').textContent = '';
  const kind = new URLSearchParams(location.search).get('kind');
  prompt = await (await fetch(`/api/prompt${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`)).json();
  const task = $('task');
  task.dataset.kind = prompt.kind;
  $('kind-badge').textContent = KIND_LABEL[prompt.kind];
  $('ref').textContent = prompt.display.map(a => a.ref).join(' · ') + ` — ${prompt.display[0].surahName}`;
  $('instruction').textContent = prompt.kind === 'none'
    ? 'No mistake this time. Read it normally — we need correct readings to compare against.'
    : prompt.instruction.en;
  $('followed-label').textContent = CONFIRM[prompt.kind];
  $('extra-label').textContent = prompt.kind === 'none' ? EXTRA.none : EXTRA.other;
  $('instruction-ar').textContent = prompt.instruction.ar;

  const text = $('text');
  text.replaceChildren();
  let say = '';
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
        if (TAG[w.mark]) {
          const t = document.createElement('span');
          t.className = 'tag';
          t.textContent = TAG[w.mark];
          span.append(t);
        }
        if (w.alt) say = w.alt;
      }
      div.append(span);
    });
    const no = document.createElement('span');
    no.className = 'ayah-no';
    no.textContent = `(${ayah.ref.split(':')[1]})`;
    div.append(' ', no);
    text.append(div);
  }
  $('say').hidden = !say;
  $('say').dataset.kind = prompt.kind;
  $('say').querySelector('.say-label').textContent = prompt.kind === 'vowel' ? 'Read it as' : 'Say instead';
  $('say-word').textContent = say;
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
  $('rec-status').textContent = 'Tap to record';
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
    $('rec-status').textContent = 'Listen back, then submit';
    $('playback').src = URL.createObjectURL(blob);
    $('review').hidden = false;
  };
  recorder.start();
  startedAt = performance.now();
  $('rec').classList.add('on');
  $('rec-status').textContent = 'Recording — tap to stop';
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
  if (duration < 1) return showError('That recording is too short. Record the whole ayah.');
  $('submit').disabled = true;
  $('error').textContent = '';
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
    const res = await fetch(`/api/submit?${q}`, { method: 'POST', headers: { 'content-type': blob.type.split(';')[0] }, body: blob });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error ?? 'Upload failed');
    localStorage.count = ++count;
    renderCount();
    await nextPrompt();
  } catch (err) {
    $('submit').disabled = false;
    showError(`${err.message}. Try again.`);
  }
});
