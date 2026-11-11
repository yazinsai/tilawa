// tilawa.dev/help: consent → Start → natural reading → a few questions → send → receipt. One task per screen.
import "./style.css";
import { decodeReport, encodeReport, ReportSendError, sendReport, type ReportConsent, type ReportReceipt, type TilawaReport } from "@tilawa/core/report";
import { buildIncidents, type Answer, type MomentAnswer } from "./claims";
import { sliceWav, wavUrl } from "./clip";
import { HelpEngineError, loadEngine, prepareMic, TAKE_MAX_MS, type HelpEngine, type LivePosition, type LiveWarning, type Take, type TakeRecording } from "./engine";
import { initialLang, numberFormat, STRINGS, type Key, type Lang } from "./i18n";
import { dropTake, keepTake, keptTakes, type KeptTake } from "./keep";
import { markedMoment, selectMoments, type Moment } from "./moments";
import { loadQuran, type QuranText } from "./quran";
import type { WordRef } from "./types";
import { progressFetch } from "./upload";

export const CONSENT_POLICY = "tilawa-help-report";
export const CONSENT_VERSION = "2026-11-report-1";
const ENDPOINT = "/help/api/v2/reports";
const LIVE_THROTTLE_MS = 5_000;
const MIN_TAKE_MS = 2_000;

type Screen = "consent" | "ready" | "reading" | "review" | "send" | "done";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

let lang: Lang = initialLang();
let fmt = numberFormat(lang);
const t = (key: Key, ...args: string[]): string => {
  const v = (STRINGS[lang][key] ?? STRINGS.en[key]) as unknown;
  return typeof v === "function" ? (v as (...a: string[]) => string)(...args) : Array.isArray(v) ? v.join(" ") : String(v);
};
const list = (key: Key): string[] => (STRINGS[lang][key] as unknown as string[]) ?? [];

function contributorId(): string | undefined {
  try {
    let id = localStorage.getItem("speaker");
    if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) {
      id = crypto.randomUUID().replaceAll("-", "").slice(0, 20);
      localStorage.setItem("speaker", id);
    }
    return id;
  } catch {
    return undefined;
  }
}

interface ReviewMoment extends Moment {
  text?: string;
  note?: string;
}

const state = {
  screen: "consent" as Screen,
  consent: null as ReportConsent | null,
  quran: null as QuranText | null,
  engine: null as HelpEngine | null,
  engineLoading: null as Promise<HelpEngine> | null,
  starting: false,
  readyProblem: null as null | { kind: HelpEngineError["kind"] | "tooShort" },
  take: null as Take | null,
  recording: null as TakeRecording | null,
  shown: null as WordRef | null,
  warnings: new Map<string, LiveWarning>(),
  posState: "acquiring" as LivePosition["state"] | "waiting",
  stumbles: 0,
  lastLive: 0,
  liveTimer: 0 as ReturnType<typeof setTimeout> | 0,
  tick: 0 as ReturnType<typeof setInterval> | 0,
  stopping: false,
  limitHit: false,
  moments: [] as ReviewMoment[],
  index: 0,
  answers: {} as Record<string, MomentAnswer>,
  wrongAyah: {} as Record<string, { sessionMs: number; wallMs: number }>,
  marks: 0,
  clipUrl: "",
  takeUrl: "",
  report: null as TilawaReport | null,
  restored: null as KeptTake | null,
  kept: false,
  keptUntil: 0,
  sending: false,
  sendPct: 0,
  sendError: null as Key | null,
  receipt: null as ReportReceipt | null,
  kept0: null as KeptTake | null,
};

/* ---------------------------------------------------------------- language */

function applyLang() {
  fmt = numberFormat(lang);
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.title = t("title");
  for (const el of document.querySelectorAll<HTMLElement>("[data-i18n]")) el.textContent = t(el.dataset.i18n as Key);
  for (const el of document.querySelectorAll<HTMLElement>("[data-i18n-html]")) el.innerHTML = t(el.dataset.i18nHtml as Key);
  $("lang").lang = lang === "ar" ? "en" : "ar";
  renderSurahOptions();
  renderKept();
  renderReady();
  renderReading();
  if (state.screen === "review") renderMoment();
  renderSend();
  renderDone();
}

$("lang").addEventListener("click", () => {
  lang = lang === "ar" ? "en" : "ar";
  try {
    localStorage.setItem("lang", lang);
  } catch {
    /* private mode */
  }
  applyLang();
});

function show(screen: Screen) {
  state.screen = screen;
  for (const s of document.querySelectorAll<HTMLElement>(".screen")) s.hidden = s.id !== `screen-${screen}`;
  document.body.dataset.screen = screen;
  $("limit-note").hidden = !(state.limitHit && !!state.recording && (screen === "review" || screen === "send"));
  window.scrollTo(0, 0);
  const focus = screen === "reading" ? $("reading-status") : screen === "review" ? $("moment-count") : $(`${screen}-h1`);
  focus?.focus({ preventScroll: true });
}

const surahName = (surah: number) => {
  const s = state.quran?.surahs[surah - 1];
  return s ? (lang === "ar" ? s.nameAr : s.nameEn) : fmt(surah);
};

function fmtTime(ms: number) {
  const total = Math.max(0, Math.round(ms / 1000));
  return t("minutes", fmt(Math.floor(total / 60)), fmt(total % 60).padStart(2, fmt(0)));
}

function fmtDate(ms: number) {
  return new Intl.DateTimeFormat(lang === "ar" ? "ar-EG" : "en-GB", { dateStyle: "long" }).format(new Date(ms));
}

/* ---------------------------------------------------------------- ayah text */

function renderAyah(el: HTMLElement, at: WordRef | null, opts: { current?: number; flags?: Set<number>; progress?: boolean } = {}) {
  el.replaceChildren();
  if (!at || !state.quran) return;
  const ayah = state.quran.ayah(at.surah, at.ayah);
  if (!ayah) return;
  ayah.words.forEach((w, i) => {
    const n = i + 1;
    const span = document.createElement("span");
    span.className = "w";
    span.textContent = w;
    if (opts.progress && opts.current && n < opts.current) span.classList.add("read");
    if (opts.current === n) {
      span.classList.add("now");
      span.setAttribute("aria-current", "true");
    }
    if (opts.flags?.has(n)) {
      span.classList.add("flag");
      const sr = document.createElement("span");
      sr.className = "visually-hidden";
      sr.lang = lang;
      sr.textContent = ` (${t("possibleMistake")})`;
      span.append(sr);
    }
    el.append(span, " ");
  });
  const no = document.createElement("span");
  no.className = "ayah-no";
  no.textContent = `\u06DD${numberFormat("ar")(at.ayah)}`;
  no.setAttribute("aria-hidden", "true");
  el.append(no);
}

/* ---------------------------------------------------------------- 1. consent */

$("consent-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const send = $<HTMLInputElement>("consent-send");
  if (!send.checked) {
    $("consent-error").textContent = t("needConsent");
    send.focus();
    return;
  }
  $("consent-error").textContent = "";
  state.consent = {
    policy: CONSENT_POLICY,
    version: CONSENT_VERSION,
    send: true,
    training: $<HTMLInputElement>("consent-training").checked,
    publicDataset: $<HTMLInputElement>("consent-public").checked,
  };
  show("ready");
});
$("consent-send").addEventListener("change", () => {
  if ($<HTMLInputElement>("consent-send").checked) $("consent-error").textContent = "";
});

function renderKept() {
  const k = state.kept0;
  $("kept").hidden = !k || state.screen !== "consent";
  if (k) $("kept-body").textContent = t("keptBody", fmtDate(k.expiresAt));
}

async function checkKept() {
  try {
    const kept = await keptTakes();
    state.kept0 = kept[kept.length - 1] ?? null;
  } catch {
    state.kept0 = null;
  }
  renderKept();
}

$("kept-send").addEventListener("click", async () => {
  const k = state.kept0;
  if (!k) return;
  try {
    state.report = await decodeReport(k.json, new Uint8Array(k.wav));
  } catch {
    await dropTake(k.reportId).catch(() => undefined);
    state.kept0 = null;
    renderKept();
    return;
  }
  state.restored = k;
  state.kept = true;
  state.keptUntil = k.expiresAt;
  state.sendError = null;
  show("send");
  renderSend();
});

$("kept-delete").addEventListener("click", async () => {
  const k = state.kept0;
  if (!k || !(await confirmDiscard())) return;
  await dropTake(k.reportId).catch(() => undefined);
  state.kept0 = null;
  renderKept();
  $("consent-h1").focus();
});

/* ---------------------------------------------------------------- 2. ready / Start */

function renderSurahOptions() {
  const sel = $<HTMLSelectElement>("surah");
  const value = sel.value;
  sel.replaceChildren(new Option(t("surahAny"), ""));
  for (const s of state.quran?.surahs ?? []) sel.append(new Option(`${fmt(s.number)}. ${lang === "ar" ? s.nameAr : s.nameEn}`, String(s.number)));
  sel.value = value;
}

function platform(): "ios" | "android" | "other" {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) return "ios";
  if (/Android/.test(ua)) return "android";
  return "other";
}

function steps(titleKey: Key, stepsKey: Key, moreKey?: Key) {
  const wrap = document.createElement("div");
  const h = document.createElement("h3");
  h.textContent = t(titleKey);
  const ol = document.createElement("ol");
  for (const s of list(stepsKey)) {
    const li = document.createElement("li");
    li.textContent = s;
    ol.append(li);
  }
  wrap.append(h, ol);
  if (moreKey) {
    const p = document.createElement("p");
    p.className = "note quiet";
    p.textContent = t(moreKey);
    wrap.append(p);
  }
  return wrap;
}

function renderReady() {
  const box = $("ready-error");
  const p = state.readyProblem;
  box.hidden = !p;
  if (!p) return box.replaceChildren();
  const para = (text: string) => {
    const el = document.createElement("p");
    el.textContent = text;
    return el;
  };
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "primary";
  retry.textContent = t("tryAgain");
  retry.addEventListener("click", () => void start());
  if (p.kind === "mic-denied") {
    const h = document.createElement("h2");
    h.textContent = t("micDeniedTitle");
    const blocks = { ios: steps("micIos", "micIosSteps", "micIosMore"), android: steps("micAndroid", "micAndroidSteps", "micAndroidMore"), other: steps("micOther", "micOtherSteps") };
    const mine = platform();
    const rest = document.createElement("details");
    const sum = document.createElement("summary");
    sum.textContent = t("micOtherDevices");
    rest.append(sum, ...(Object.keys(blocks) as (keyof typeof blocks)[]).filter((k) => k !== mine).map((k) => blocks[k]));
    box.replaceChildren(h, para(t("micDeniedBody")), blocks[mine], rest, retry);
  } else {
    const key: Key =
      p.kind === "load" ? "loadFailed" : p.kind === "mic-unavailable" ? "micUnavailable" : p.kind === "insecure" ? "micInsecure" : p.kind === "tooShort" || p.kind === "empty" ? "tooShort" : "micFailed";
    box.replaceChildren(para(t(key)), ...(p.kind === "tooShort" || p.kind === "empty" ? [] : [retry]));
  }
}

function setLoading(on: boolean, fraction: number | null = null) {
  $("loading").hidden = !on;
  $("start").toggleAttribute("disabled", on);
  const bar = $<HTMLProgressElement>("loading-bar");
  if (fraction === null) bar.removeAttribute("value");
  else bar.value = fraction;
  $("loading-text").textContent = fraction === null || fraction >= 1 ? t("preparing") : t("preparingPct", `${fmt(Math.round(fraction * 100))}%`);
}

async function start() {
  if (state.starting || !state.consent) return;
  state.starting = true;
  // Before any await: keeps iOS Safari's audio unlocked by this tap through the first download.
  const prepared = prepareMic();
  state.readyProblem = null;
  renderReady();
  setLoading(true);
  try {
    state.quran ??= await loadQuran();
    renderSurahOptions();
    state.engine ??= await (state.engineLoading ??= loadEngine((p) => setLoading(true, p.fraction)));
  } catch (e) {
    state.engineLoading = null;
    console.error(e);
    state.readyProblem = { kind: "load" };
    prepared?.release();
    finishStart();
    return;
  }
  try {
    resetTakeView();
    state.take = await state.engine.startTake({
      prepared,
      consent: state.consent,
      contributor: contributorId(),
      maxMs: TAKE_MAX_MS,
      onPosition,
      onChecker,
      onLimit: () => {
        state.limitHit = true;
        void stopTake();
      },
    });
  } catch (e) {
    console.error(e);
    state.readyProblem = { kind: e instanceof HelpEngineError ? e.kind : "failed" };
    finishStart();
    $("ready-error").querySelector("button")?.focus();
    return;
  }
  finishStart();
  startReading();
}

function finishStart() {
  state.starting = false;
  setLoading(false);
  renderReady();
}

$("start").addEventListener("click", () => void start());

/* ---------------------------------------------------------------- 3. reading */

function resetTakeView() {
  state.shown = null;
  state.warnings.clear();
  state.posState = "waiting";
  state.stumbles = 0;
  state.lastLive = 0;
  state.limitHit = false;
  state.stopping = false;
  $("stumble-count").textContent = "";
  $("live").textContent = "";
}

function startReading() {
  const chosen = Number($<HTMLSelectElement>("surah").value) || 0;
  state.shown = chosen ? { surah: chosen, ayah: 1, word: 0 } : null;
  show("reading");
  renderReading();
  state.tick = setInterval(renderTimer, 250);
  renderTimer();
}

function renderTimer() {
  const take = state.take;
  if (!take) return;
  const now = take.now();
  $("timer").textContent = fmtTime(now);
  const left = TAKE_MAX_MS - now;
  $("time-left").textContent = left <= 30_000 ? `· ${t("timeLeft", fmtTime(left))}` : "";
}

function statusKey(): Key {
  if (state.stopping) return "stopping";
  switch (state.posState) {
    case "waiting":
      return "listening";
    case "acquiring":
      return "finding";
    case "lost":
      return "lost";
    default:
      return state.shown && state.shown.word > 0 ? "following" : "finding";
  }
}

function renderReading() {
  $("reading-status").textContent = t(statusKey());
  const at = state.shown;
  $("ayah-ref").textContent = at ? t("ayahRef", surahName(at.surah), fmt(at.ayah)) : "";
  const flags = new Set<number>();
  for (const w of state.warnings.values()) if (w.state === "confirmed" && at && w.location.surah === at.surah && w.location.ayah === at.ayah) flags.add(w.location.word);
  renderAyah($("ayah"), at, { current: at?.word, flags, progress: true });
  $("stumble").toggleAttribute("disabled", state.stopping);
  $("stop").toggleAttribute("disabled", state.stopping);
}

/** Polite, throttled: only ayah changes, at most one announcement per LIVE_THROTTLE_MS. */
function announceAyah() {
  const at = state.shown;
  if (!at || state.liveTimer) return;
  const wait = Math.max(0, state.lastLive + LIVE_THROTTLE_MS - Date.now());
  state.liveTimer = setTimeout(() => {
    state.liveTimer = 0;
    const now = state.shown;
    if (!now || state.screen !== "reading") return;
    state.lastLive = Date.now();
    $("live").textContent = t("liveAyah", surahName(now.surah), fmt(now.ayah));
  }, wait);
}

function onPosition(p: LivePosition) {
  state.posState = p.state;
  const prev = state.shown;
  if (p.word) {
    state.shown = p.word;
    if (!prev || prev.surah !== p.word.surah || prev.ayah !== p.word.ayah) announceAyah();
  }
  if (state.screen === "reading") renderReading();
}

function onChecker(c: LiveWarning) {
  state.warnings.set(c.id, c);
  if (state.screen === "reading") renderReading();
}

$("stumble").addEventListener("click", () => {
  const take = state.take;
  if (!take || state.stopping) return;
  try {
    take.bookmark();
  } catch (e) {
    console.error(e);
    return;
  }
  state.stumbles++;
  $("stumble-count").textContent = `${t("stumbleSaid")} · ${t("stumbleCount", fmt(state.stumbles))}`;
  const b = $("stumble");
  b.classList.remove("pulse");
  void b.offsetWidth;
  b.classList.add("pulse");
});

$("stop").addEventListener("click", () => void stopTake());

async function stopTake() {
  const take = state.take;
  if (!take || state.stopping) return;
  state.stopping = true;
  renderReading();
  clearInterval(state.tick);
  clearTimeout(state.liveTimer);
  state.liveTimer = 0;
  let rec: TakeRecording;
  try {
    rec = await take.stop();
  } catch (e) {
    console.error(e);
    state.take = null;
    state.readyProblem = { kind: e instanceof HelpEngineError && e.kind === "empty" ? "empty" : "failed" };
    show("ready");
    renderReady();
    return;
  }
  state.take = null;
  if (rec.durationMs < MIN_TAKE_MS) {
    rec.dispose();
    state.readyProblem = { kind: "tooShort" };
    show("ready");
    renderReady();
    return;
  }
  state.recording = rec;
  state.moments = selectMoments(rec.events, rec.bookmarks, { durationMs: rec.durationMs });
  state.index = 0;
  state.answers = {};
  state.wrongAyah = {};
  state.report = null;
  state.kept = false;
  state.sendError = null;
  if (state.moments.length) {
    show("review");
    renderMoment();
  } else {
    show("send");
    renderSend();
  }
}

/* ---------------------------------------------------------------- 4. review */

const audio = $<HTMLAudioElement>("moment-audio");

function setClip(m: Moment) {
  if (state.clipUrl) URL.revokeObjectURL(state.clipUrl);
  state.clipUrl = state.recording ? wavUrl(sliceWav(state.recording.wav, m.clip.startMs, m.clip.endMs)) : "";
  audio.src = state.clipUrl;
}

function renderMoment() {
  const m = state.moments[state.index];
  if (!m) return;
  $("moment-count").textContent = t("momentOf", fmt(state.index + 1), fmt(state.moments.length));
  const why: Key =
    m.kind === "warning" ? (m.confirmed ? "why.warning" : "why.warningSuspected") : m.kind === "stumble" ? "why.stumble" : m.kind === "position" ? "why.position" : m.kind === "sample" ? "why.sample" : "why.marked";
  $("moment-why").textContent = t(why);
  $("moment-ref").textContent = m.location ? t("ayahRef", surahName(m.location.surah), fmt(m.location.ayah)) : t("noPlace");
  renderAyah($("moment-ayah"), m.location, { current: m.location?.word });
  if (audio.dataset.moment !== m.id) {
    audio.pause();
    setClip(m);
    audio.dataset.moment = m.id;
    $("moment-mark").hidden = true;
    $("moment-mark-status").textContent = "";
    $<HTMLTextAreaElement>("moment-text").value = m.text ?? "";
    $<HTMLTextAreaElement>("moment-note").value = m.note ?? "";
    $<HTMLDetailsElement>("moment-details").open = !!(m.text || m.note);
  }
  renderPlay();
  $("moment-question").textContent = t(`q.${m.question}` as Key);
  const chosen = state.answers[m.id]?.answer;
  for (const b of document.querySelectorAll<HTMLButtonElement>(".answer")) b.setAttribute("aria-pressed", String(b.dataset.answer === chosen));
  const wrong = $("moment-wrong-ayah");
  wrong.hidden = m.question === "wrong_ayah" || !m.location;
  const on = !!state.wrongAyah[m.id];
  wrong.setAttribute("aria-pressed", String(on));
  wrong.textContent = t(on ? "wrongAyahOn" : "wrongAyah");
  $("moment-back").hidden = state.index === 0;
}

function renderPlay() {
  const playing = !audio.paused && !audio.ended;
  $("moment-play").classList.toggle("playing", playing);
  $("moment-play-label").textContent = t(playing ? "pause" : "play");
}

$("moment-play").addEventListener("click", () => {
  if (audio.paused || audio.ended) void audio.play().catch((e) => console.error(e));
  else audio.pause();
});
for (const ev of ["play", "pause", "ended"]) audio.addEventListener(ev, renderPlay);
audio.addEventListener("play", () => ($("moment-mark").hidden = false));

function addMarked(atMs: number): number {
  const rec = state.recording;
  if (!rec) return -1;
  state.marks++;
  const m: ReviewMoment = markedMoment(`x${state.marks}`, Math.round(atMs), rec.events, rec.durationMs);
  const at = Math.min(state.moments.length, state.index + 1);
  state.moments.splice(at, 0, m);
  return at;
}

$("moment-mark").addEventListener("click", () => {
  const m = state.moments[state.index];
  if (!m) return;
  addMarked(m.clip.startMs + audio.currentTime * 1000);
  $("moment-mark-status").textContent = t("markAdded");
  $("moment-count").textContent = t("momentOf", fmt(state.index + 1), fmt(state.moments.length));
});

$("moment-text").addEventListener("input", (e) => {
  const m = state.moments[state.index];
  if (m) m.text = (e.target as HTMLTextAreaElement).value;
});
$("moment-note").addEventListener("input", (e) => {
  const m = state.moments[state.index];
  if (m) m.note = (e.target as HTMLTextAreaElement).value;
});

$("moment-wrong-ayah").addEventListener("click", () => {
  const m = state.moments[state.index];
  if (!m || !state.recording) return;
  if (state.wrongAyah[m.id]) delete state.wrongAyah[m.id];
  else state.wrongAyah[m.id] = { sessionMs: state.recording.durationMs, wallMs: Date.now() };
  renderMoment();
});

for (const b of document.querySelectorAll<HTMLButtonElement>(".answer")) {
  b.addEventListener("click", () => {
    const m = state.moments[state.index];
    if (!m || !state.recording) return;
    state.answers[m.id] = {
      question: m.question,
      answer: b.dataset.answer as Answer,
      answeredAt: { sessionMs: state.recording.durationMs, wallMs: Date.now() },
      text: m.text,
      note: m.note,
    };
    audio.pause();
    if (state.index + 1 < state.moments.length) {
      state.index++;
      renderMoment();
      $("moment-count").focus();
    } else {
      show("send");
      renderSend();
    }
  });
}

$("moment-back").addEventListener("click", () => {
  if (state.index === 0) return;
  state.index--;
  renderMoment();
  $("moment-count").focus();
});

/* ---------------------------------------------------------------- 5. send */

function answeredCount() {
  if (state.restored && state.report) return state.report.incidents.filter((i) => i.claim.answer !== null).length;
  return Object.keys(state.answers).length;
}

function renderSend() {
  const durationMs = state.report?.audio.durationMs ?? state.recording?.durationMs ?? 0;
  const n = answeredCount();
  $("send-summary").textContent = n ? t("sendSummary", fmtTime(durationMs), fmt(n)) : t("sendSummaryNone", fmtTime(durationMs));
  // Once the report is built it is final: retries resend the exact same bytes.
  const canListen = !!state.recording && !state.report;
  $("listen-all").hidden = !canListen;
  if (canListen && !state.takeUrl && state.recording) {
    state.takeUrl = wavUrl(state.recording.wav);
    $<HTMLAudioElement>("take-audio").src = state.takeUrl;
  }
  $("send").toggleAttribute("disabled", state.sending);
  $("send-label").textContent = t(state.sendError ? "tryAgain" : "send");
  $("send-progress").hidden = !state.sending;
  $<HTMLProgressElement>("send-bar").value = state.sendPct;
  $("send-progress-text").textContent = t("sending", `${fmt(Math.round(state.sendPct * 100))}%`);
  $("send-error").hidden = !state.sendError;
  $("send-error-text").textContent = state.sendError ? t(state.sendError) : "";
  $<HTMLInputElement>("keep").checked = state.kept;
  $("keep-status").textContent = state.kept && state.keptUntil ? t("keptUntil", fmtDate(state.keptUntil)) : "";
  $("discard").toggleAttribute("disabled", state.sending);
}

$("take-mark").addEventListener("click", () => {
  const a = $<HTMLAudioElement>("take-audio");
  a.pause();
  const at = addMarked(a.currentTime * 1000);
  if (at < 0) return;
  state.index = at;
  show("review");
  renderMoment();
});

async function buildReport(): Promise<TilawaReport> {
  if (state.report) return state.report;
  const rec = state.recording;
  if (!rec || !state.consent) throw new Error("nothing to send");
  const incidents = buildIncidents({ moments: state.moments, answers: state.answers, wrongAyah: state.wrongAyah, bookmarks: rec.bookmarks, durationMs: rec.durationMs });
  state.report = await rec.report(incidents, state.consent);
  return state.report;
}

$("send").addEventListener("click", async () => {
  if (state.sending) return;
  state.sending = true;
  state.sendPct = 0;
  state.sendError = null;
  renderSend();
  let receipt: ReportReceipt;
  try {
    const report = await buildReport();
    receipt = await sendReport(report, {
      endpoint: ENDPOINT,
      fetch: progressFetch((f) => {
        state.sendPct = f;
        renderSend();
      }),
      retries: 1,
      backoffMs: 800,
    });
  } catch (e) {
    console.error(e);
    const status = e instanceof ReportSendError ? e.status : null;
    state.sendError = status === 429 ? "sendLimited" : status !== null && status >= 400 && status < 500 ? "sendRejected" : "sendFailed";
    state.sending = false;
    renderSend();
    $("send-error-text").focus();
    return;
  }
  state.sending = false;
  state.receipt = receipt;
  if (state.kept && state.report) await dropTake(state.report.reportId).catch(() => undefined);
  clearTake();
  void checkKept();
  show("done");
  renderDone();
});

$("keep").addEventListener("change", async () => {
  const report = state.report;
  if (!report) return;
  const on = $<HTMLInputElement>("keep").checked;
  try {
    if (on) {
      const { json, wav } = encodeReport(report);
      await keepTake(report.reportId, json, wav);
      state.keptUntil = Date.now() + 7 * 86_400_000;
    } else await dropTake(report.reportId);
    state.kept = on;
    renderSend();
  } catch (e) {
    console.error(e);
    state.kept = false;
    renderSend();
    $("keep-status").textContent = t("keepFailed");
  }
});

$("discard").addEventListener("click", async () => {
  if (!(await confirmDiscard())) return;
  if (state.kept && state.report) await dropTake(state.report.reportId).catch(() => undefined);
  clearTake();
  state.kept0 = null;
  void checkKept();
  show(state.consent ? "ready" : "consent");
});

function clearTake() {
  audio.pause();
  if (state.clipUrl) URL.revokeObjectURL(state.clipUrl);
  if (state.takeUrl) URL.revokeObjectURL(state.takeUrl);
  audio.removeAttribute("src");
  delete audio.dataset.moment;
  $<HTMLAudioElement>("take-audio").removeAttribute("src");
  state.clipUrl = state.takeUrl = "";
  state.recording?.dispose();
  state.recording = null;
  state.moments = [];
  state.answers = {};
  state.wrongAyah = {};
  state.report = null;
  state.restored = null;
  state.kept = false;
  state.keptUntil = 0;
  state.sendError = null;
  state.sendPct = 0;
  state.limitHit = false;
}

function confirmDiscard(): Promise<boolean> {
  const dialog = $<HTMLDialogElement>("confirm");
  $("confirm-title").textContent = t("discardTitle");
  $("confirm-body").textContent = t("discardBody");
  $("confirm-yes").textContent = t("discardYes");
  $("confirm-no").textContent = t("discardNo");
  if (typeof dialog.showModal !== "function") return Promise.resolve(window.confirm(`${t("discardTitle")}\n${t("discardBody")}`));
  return new Promise((resolve) => {
    const done = (v: boolean) => {
      dialog.close();
      $("confirm-yes").removeEventListener("click", yes);
      $("confirm-no").removeEventListener("click", no);
      dialog.removeEventListener("cancel", no);
      resolve(v);
    };
    const yes = () => done(true);
    const no = () => done(false);
    $("confirm-yes").addEventListener("click", yes);
    $("confirm-no").addEventListener("click", no);
    dialog.addEventListener("cancel", no);
    dialog.showModal();
    $("confirm-no").focus();
  });
}

/* ---------------------------------------------------------------- 6. done */

function renderDone() {
  const r = state.receipt;
  if (!r) return;
  $<HTMLInputElement>("code").value = r.deleteToken ? `${r.reportId}.${r.deleteToken}` : r.reportId;
  $("delete-now").hidden = !r.deleteToken;
}

$("copy").addEventListener("click", async () => {
  const input = $<HTMLInputElement>("code");
  try {
    await navigator.clipboard.writeText(input.value);
    $("code-status").textContent = t("copied");
  } catch {
    input.select();
    $("code-status").textContent = t("copyFallback");
  }
});

$("delete-now").addEventListener("click", async () => {
  const r = state.receipt;
  if (!r?.deleteToken || !window.confirm(t("deleteConfirm"))) return;
  try {
    const res = await fetch(`${ENDPOINT}/${encodeURIComponent(r.reportId)}`, { method: "DELETE", headers: { "X-Delete-Token": r.deleteToken } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    $("code-status").textContent = t("deleted");
    $("delete-now").hidden = true;
  } catch (e) {
    console.error(e);
    $("code-status").textContent = t("deleteFailed");
  }
});

$("again").addEventListener("click", () => {
  state.receipt = null;
  $<HTMLInputElement>("code").value = "";
  $("code-status").textContent = "";
  state.readyProblem = null;
  renderReady();
  show(state.consent ? "ready" : "consent");
});

/* ---------------------------------------------------------------- page lifecycle */

window.addEventListener("beforeunload", (e) => {
  const unsent = !!state.take || ((!!state.recording || !!state.report) && !state.kept && !state.receipt);
  if (!unsent) return;
  e.preventDefault();
  e.returnValue = "";
});

applyLang();
show("consent");
void checkKept();
void loadQuran()
  .then((q) => {
    state.quran = q;
    renderSurahOptions();
  })
  .catch(() => undefined);
