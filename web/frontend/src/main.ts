import "@fontsource/amiri/400.css";
import "@fontsource/amiri/700.css";
import { addCollection } from "iconify-icon";
import "./style.css";

import { initSurahDropdown, openReportDialog } from "./report-dialog";

import type {
  VerseMatchMessage,
  VerseCandidateMessage,
  FinalSequenceMessage,
  RawTranscriptMessage,
  WordProgressMessage,
  WorkerOutbound,
  QuranVerse,
  DebugMessage,
} from "./lib/types";
import { DEFAULT_STREAMING_CONFIG } from "./lib/types";

addCollection({
  prefix: "solar",
  width: 24,
  height: 24,
  icons: {
    "shield-check-broken": {
      body: '<g fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.5"><path stroke-linejoin="round" d="m9.5 12.4l1.429 1.6l3.571-4"/><path d="M3 10.417c0-3.198 0-4.797.378-5.335c.377-.537 1.88-1.052 4.887-2.081l.573-.196C10.405 2.268 11.188 2 12 2s1.595.268 3.162.805l.573.196c3.007 1.029 4.51 1.544 4.887 2.081C21 5.62 21 7.22 21 10.417v1.574c0 2.505-.837 4.437-2 5.913M3.193 14c.857 4.298 4.383 6.513 6.706 7.527c.721.315 1.082.473 2.101.473c1.02 0 1.38-.158 2.101-.473c.579-.252 1.231-.58 1.899-.994"/></g>',
    },
    "cloud-cross-broken": {
      body: '<path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.5" d="M22 13.353c0 2.707-1.927 4.97-4.5 5.52M6.286 19C3.919 19 2 17.104 2 14.765s1.919-4.236 4.286-4.236q.427.001.83.08m7.265-2.582a5.8 5.8 0 0 1 1.905-.321c.654 0 1.283.109 1.87.309m-11.04 2.594a5.6 5.6 0 0 1-.354-1.962C6.762 5.528 9.32 3 12.476 3c2.94 0 5.361 2.194 5.68 5.015m-11.04 2.594a4.3 4.3 0 0 1 1.55.634m9.49-3.228A5.7 5.7 0 0 1 20 9.061M13.5 17.5L12 19m0 0l-1.5 1.5M12 19l-1.5-1.5M12 19l1.5 1.5"/>',
    },
    "map-point-wave-broken": {
      body: '<g fill="none" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" d="M5.875 12.573C5.308 11.25 5 9.84 5 8.515C5 4.917 8.134 2 12 2s7 2.917 7 6.515c0 3.57-2.234 7.735-5.72 9.225a3.28 3.28 0 0 1-2.56 0c-1.113-.476-2.099-1.225-2.925-2.14"/><path d="M14 9a2 2 0 1 1-4 0a2 2 0 0 1 4 0Z"/><path stroke-linecap="round" d="M20.96 15.5c.666.602 1.04 1.282 1.04 2c0 .925-.62 1.785-1.684 2.5M3.04 15.5c-.666.602-1.04 1.282-1.04 2C2 19.985 6.477 22 12 22c1.653 0 3.212-.18 4.586-.5"/></g>',
    },
    "document-text-broken": {
      body: '<path fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.5" d="M8 12h1m7 0h-4m4-4h-1m-3 0H8m0 8h5M3 14v-4c0-3.771 0-5.657 1.172-6.828S7.229 2 11 2h2c3.771 0 5.657 0 6.828 1.172c.654.653.943 1.528 1.07 2.828M21 10v4c0 3.771 0 5.657-1.172 6.828S16.771 22 13 22h-2c-3.771 0-5.657 0-6.828-1.172c-.654-.653-.943-1.528-1.07-2.828"/>',
    },
  },
});

// ---------------------------------------------------------------------------
// Types (UI-only)
// ---------------------------------------------------------------------------
interface SurahVerse {
  ayah: number;
  text_uthmani: string;
}

interface SurahData {
  surah: number;
  surah_name: string;
  surah_name_en: string;
  verses: SurahVerse[];
}

interface VerseGroup {
  surah: number;
  surahName: string;
  surahNameEn: string;
  currentAyah: number;
  verses: SurahVerse[];
  element: HTMLElement;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
interface DiagnosticEvent {
  timestamp: number;
  type: string;
  data: Record<string, unknown>;
}

const MAX_DIAGNOSTIC_EVENTS = 50;
const MAX_DEBUG_EVENTS = 80;
const DIAGNOSTIC_COOLDOWN_MS = 30_000;
const DEBUG_VIEW_ENABLED = Boolean(import.meta.env.VITE_DEBUG_MODE);

const state = {
  groups: [] as VerseGroup[],
  worker: null as Worker | null,
  audioCtx: null as AudioContext | null,
  stream: null as MediaStream | null,
  isActive: false,
  hasFirstMatch: false,
  modelReady: false,
  surahCache: new Map<number, SurahData>(),
  quranData: null as QuranVerse[] | null,
  sessionAudioChunks: [] as Float32Array[],
  lastModelPrediction: null as { surah: number; ayah: number; confidence: number } | null,
  diagnosticEvents: [] as DiagnosticEvent[],
  debugEvents: [] as DebugMessage[],
  lastDiagnosticSentAt: 0,
  recentVerseMatches: [] as { surah: number; ayah: number; timestamp: number }[],
  finalSequence: [] as { surah: number; ayah: number; confidence: number }[],
  streamingConfig: DEFAULT_STREAMING_CONFIG,
  audioProcessor: null as AudioWorkletNode | null,
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $verses = document.getElementById("verses")!;
const $rawTranscript = document.getElementById("raw-transcript")!;
const $indicator = document.getElementById("listening-indicator")!;
const $permissionPrompt = document.getElementById("permission-prompt")!;
const $listeningStatus = document.getElementById("listening-status")!;
const $modelStatus = document.getElementById("model-status")!;
const $loadingStatus = document.getElementById("loading-status")!;
const $loadingProgress = document.getElementById("loading-progress")!;
const $loadingDetail = document.getElementById("loading-detail")!;
const $introScreen = document.getElementById("intro-screen")!;
const $readyState = document.getElementById("ready-state")!;
const $recordingState = document.getElementById("recording-state")!;
const $postRecording = document.getElementById("post-recording")!;
const $btnBeginTest = document.getElementById("btn-begin-test") as HTMLButtonElement;
const $btnStart = document.getElementById("btn-start")!;
const $btnStop = document.getElementById("btn-stop")!;
const $btnReport = document.getElementById("btn-report")!;
const $btnRestart = document.getElementById("btn-restart")!;
const $candidateStatus = document.getElementById("candidate-status")!;
const $debugPanel = document.getElementById("debug-panel") as HTMLDetailsElement;
const $debugSummary = document.getElementById("debug-summary")!;
const $debugContent = document.getElementById("debug-content")!;
const $debugCopy = document.getElementById("debug-copy") as HTMLButtonElement;
const $debugCopyStatus = document.getElementById("debug-copy-status")!;
const $waveform = document.getElementById("listening-waveform")!;
const $waveformBars = Array.from($waveform.querySelectorAll<HTMLElement>(".waveform-bar"));

const WAVEFORM_BAR_PHASES = [0.34, 0.72, 0.48, 0.95, 0.58, 1, 0.68, 0.86, 0.42, 0.76, 0.52];

function updateListeningWaveform(rms: number): void {
  const strength = Math.min(1, Math.max(0, (rms - 0.004) * 24));
  const drift = performance.now() / 180;
  $waveform.style.setProperty("--waveform-strength", strength.toFixed(3));

  for (let i = 0; i < $waveformBars.length; i++) {
    const phase = WAVEFORM_BAR_PHASES[i % WAVEFORM_BAR_PHASES.length];
    const motion = 0.55 + 0.45 * Math.sin(drift + i * 0.78);
    const level = 0.18 + strength * (phase * 0.62 + motion * 0.34);
    $waveformBars[i].style.setProperty("--bar-level", Math.min(1, level).toFixed(3));
  }
}

function resetListeningWaveform(): void {
  $waveform.style.setProperty("--waveform-strength", "0");
  for (let i = 0; i < $waveformBars.length; i++) {
    const idleLevel = 0.18 + (i % 3) * 0.035;
    $waveformBars[i].style.setProperty("--bar-level", idleLevel.toFixed(3));
  }
}

function pushStreamingConfig(): void {
  state.worker?.postMessage({ type: "set_config", config: state.streamingConfig });
  state.audioProcessor?.port.postMessage({
    type: "set_config",
    audioChunkMs: state.streamingConfig.audioChunkMs,
  });
}

// ---------------------------------------------------------------------------
// Arabic numeral converter
// ---------------------------------------------------------------------------
const arabicNumerals = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"];
function toArabicNum(n: number): string {
  return String(n)
    .split("")
    .map((d) => arabicNumerals[parseInt(d)])
    .join("");
}

// ---------------------------------------------------------------------------
// Surah data (loaded from quran.json, no server needed)
// ---------------------------------------------------------------------------
async function loadQuranData(): Promise<void> {
  if (state.quranData) return;
  const res = await fetch("/quran.json");
  if (!res.ok) throw new Error(`quran.json fetch failed: ${res.status}`);
  state.quranData = await res.json();
  initSurahDropdown(state.quranData);
}

async function fetchSurah(surahNum: number): Promise<SurahData> {
  const cached = state.surahCache.get(surahNum);
  if (cached) return cached;

  await loadQuranData();
  const verses = state.quranData!.filter((v) => v.surah === surahNum);
  if (!verses.length) throw new Error(`Surah ${surahNum} not found`);

  const data: SurahData = {
    surah: surahNum,
    surah_name: verses[0].surah_name,
    surah_name_en: verses[0].surah_name_en,
    verses: verses.map((v) => ({
      ayah: v.ayah,
      text_uthmani: v.text_uthmani,
    })),
  };
  state.surahCache.set(surahNum, data);
  return data;
}

// ---------------------------------------------------------------------------
// Verse rendering
// ---------------------------------------------------------------------------
const WAQF_MARKS = new Set([
  "\u06D6", "\u06D7", "\u06D8", "\u06D9", "\u06DA", "\u06DB", "\u06DC",
]);

function isWaqfToken(token: string): boolean {
  return token.length <= 2 && [...token].every((c) => WAQF_MARKS.has(c));
}

interface WordToken {
  text: string;
  isRealWord: boolean;
}

function splitUthmaniWords(text: string): WordToken[] {
  const raw = text.split(/\s+/).filter((w) => w.length > 0);
  const result: WordToken[] = [];

  for (const token of raw) {
    if (isWaqfToken(token) && result.length > 0) {
      result[result.length - 1].text += " " + token;
    } else {
      result.push({ text: token, isRealWord: true });
    }
  }

  return result;
}

const BISMILLAH_WORD_COUNT = 4;
const BISMILLAH_BASE = "بسم الله الرحمن الرحيم";

function stripDiacritics(s: string): string {
  return s.replace(/[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]/g, "");
}

function startsWithBismillah(text: string): boolean {
  const stripped = stripDiacritics(text);
  return stripped.startsWith(BISMILLAH_BASE) || stripped.startsWith(stripDiacritics(BISMILLAH_BASE));
}

function createVerseGroupElement(group: VerseGroup): HTMLElement {
  const el = document.createElement("div");
  el.className = "verse-group";
  el.setAttribute("data-surah", String(group.surah));

  const header = document.createElement("div");
  header.className = "surah-header";
  header.textContent = group.surahNameEn;
  el.appendChild(header);

  const hasBismillah =
    group.surah !== 1 &&
    group.surah !== 9 &&
    startsWithBismillah(group.verses[0]?.text_uthmani ?? "");
  if (hasBismillah) {
    const words = group.verses[0].text_uthmani.split(/\s+/);
    const bsmText = words.slice(0, BISMILLAH_WORD_COUNT).join(" ");
    const bsmEl = document.createElement("div");
    bsmEl.className = "bismillah";
    bsmEl.dir = "rtl";
    bsmEl.lang = "ar";
    bsmEl.textContent = bsmText;
    el.appendChild(bsmEl);
  }

  const body = document.createElement("div");
  body.className = "verse-body";
  body.dir = "rtl";
  body.lang = "ar";

  for (const v of group.verses) {
    const verseEl = document.createElement("span");
    verseEl.className = "verse verse--upcoming";
    verseEl.setAttribute("data-ayah", String(v.ayah));

    const allWords = splitUthmaniWords(v.text_uthmani);
    const skipBsm = hasBismillah && v.ayah === 1;
    const startIdx = skipBsm ? BISMILLAH_WORD_COUNT : 0;

    const textEl = document.createElement("span");
    textEl.className = "verse-text";
    for (let i = startIdx; i < allWords.length; i++) {
      const wordEl = document.createElement("span");
      wordEl.className = "word";
      wordEl.setAttribute("data-word-idx", String(i));
      wordEl.textContent = allWords[i].text;
      textEl.appendChild(wordEl);
      if (i < allWords.length - 1) {
        textEl.appendChild(document.createTextNode(" "));
      }
    }
    verseEl.appendChild(textEl);

    const markerEl = document.createElement("span");
    markerEl.className = "verse-marker";
    markerEl.textContent = ` \u06DD${toArabicNum(v.ayah)} `;
    verseEl.appendChild(markerEl);

    body.appendChild(verseEl);
  }

  el.appendChild(body);
  return el;
}

function updateVerseHighlight(group: VerseGroup, newAyah: number): void {
  const el = group.element;
  const oldAyah = group.currentAyah;

  const verses = el.querySelectorAll<HTMLElement>(".verse");
  for (const verseEl of verses) {
    const ayah = parseInt(verseEl.getAttribute("data-ayah") || "0");
    if (ayah === newAyah) {
      verseEl.className = "verse verse--active";
    } else if (ayah <= newAyah && (ayah >= oldAyah || ayah < oldAyah)) {
      if (
        verseEl.classList.contains("verse--active") ||
        (ayah > oldAyah && ayah < newAyah) ||
        ayah <= oldAyah
      ) {
        verseEl.className = "verse verse--recited";
      }
    }
  }

  group.currentAyah = newAyah;
  scrollToActiveVerse();
}

function scrollToActiveVerse(): void {
  const active = document.querySelector(".verse--active");
  if (active) {
    active.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------
async function handleVerseMatch(msg: VerseMatchMessage): Promise<void> {
  $rawTranscript.textContent = "";
  $rawTranscript.classList.remove("visible");
  $candidateStatus.hidden = true;

  state.lastModelPrediction = { surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence };

  if (!state.hasFirstMatch) {
    state.hasFirstMatch = true;
    $listeningStatus.hidden = true;
    $indicator.classList.add("has-verses");
  }

  const lastGroup = state.groups[state.groups.length - 1];

  if (lastGroup && lastGroup.surah === msg.surah) {
    updateVerseHighlight(lastGroup, msg.ayah);
    return;
  }

  if (lastGroup) {
    lastGroup.element.classList.add("verse-group--exiting");
    const oldEl = lastGroup.element;
    setTimeout(() => oldEl.remove(), 400);
  }

  const surahData = await fetchSurah(msg.surah);

  const group: VerseGroup = {
    surah: msg.surah,
    surahName: surahData.surah_name,
    surahNameEn: surahData.surah_name_en,
    currentAyah: 0,
    verses: surahData.verses,
    element: document.createElement("div"),
  };
  group.element = createVerseGroupElement(group);
  state.groups.push(group);
  $verses.appendChild(group.element);

  updateVerseHighlight(group, msg.ayah);
}

let _matchedWordIndices = new Set<number>();
let _trackingKey = "";

function handleWordProgress(msg: WordProgressMessage): void {
  const lastGroup = state.groups[state.groups.length - 1];
  if (!lastGroup || lastGroup.surah !== msg.surah) return;

  const verseEl = lastGroup.element.querySelector<HTMLElement>(
    `.verse[data-ayah="${msg.ayah}"]`,
  );
  if (!verseEl) return;

  if (!verseEl.classList.contains("verse--active")) {
    updateVerseHighlight(lastGroup, msg.ayah);
  }

  const key = `${msg.surah}:${msg.ayah}`;
  if (key !== _trackingKey) {
    _matchedWordIndices = new Set<number>();
    _trackingKey = key;
  }

  for (const idx of msg.matched_indices) {
    _matchedWordIndices.add(idx);
  }

  let contiguousMax = -1;
  for (let i = 0; i <= msg.total_words; i++) {
    if (_matchedWordIndices.has(i)) {
      contiguousMax = i;
    } else {
      break;
    }
  }

  const wordEls = verseEl.querySelectorAll<HTMLElement>(".word");
  for (const wordEl of wordEls) {
    const idx = parseInt(wordEl.getAttribute("data-word-idx") || "-1");
    if (idx <= contiguousMax) {
      wordEl.classList.add("word--spoken");
    }
  }
}

function handleRawTranscript(msg: RawTranscriptMessage): void {
  $rawTranscript.textContent = msg.text;
  $rawTranscript.classList.add("visible");
}

async function handleVerseCandidate(msg: VerseCandidateMessage): Promise<void> {
  const best = msg.candidates[0];
  if (!best) {
    return;
  }
  if (state.hasFirstMatch && best.source !== "tracking") return;

  const surah = await fetchSurah(best.surah);
  const range =
    best.ayah_end && best.ayah_end > best.ayah
      ? `${best.ayah}-${best.ayah_end}`
      : String(best.ayah);
  const label = best.source === "tracking"
    ? "Pending next"
    : msg.stable ? "Likely" : "Listening near";

  $candidateStatus.textContent =
    `${label}: ${surah.surah_name_en} ${range} (${Math.round(best.confidence * 100)}%)`;
  $candidateStatus.classList.toggle("candidate-status--stable", msg.stable);
  $candidateStatus.classList.toggle("candidate-status--pending", best.source === "tracking");
  $candidateStatus.hidden = false;
  $listeningStatus.hidden = true;
}

async function handleFinalSequence(msg: FinalSequenceMessage): Promise<void> {
  state.finalSequence = msg.verses;
  if (!msg.verses.length) return;

  const first = msg.verses[0];
  const last = msg.verses[msg.verses.length - 1];
  const surah = await fetchSurah(first.surah);
  const range =
    first.surah === last.surah && first.ayah !== last.ayah
      ? `${first.ayah}-${last.ayah}`
      : String(first.ayah);

  $candidateStatus.textContent =
    `Final from streaming evidence: ${surah.surah_name_en} ${range} (${Math.round(msg.confidence * 100)}%)`;
  $candidateStatus.classList.add("candidate-status--stable");
  $candidateStatus.hidden = false;
}

function handleDebugMessage(msg: DebugMessage): void {
  state.debugEvents.push(msg);
  if (state.debugEvents.length > MAX_DEBUG_EVENTS) {
    state.debugEvents.shift();
  }
  renderDebugPanel();
}

function syncDebugEnabled(): void {
  state.worker?.postMessage({ type: "set_debug", enabled: $debugPanel.open });
  renderDebugPanel();
}

function buildDebugBundle() {
  const totalSamples = state.sessionAudioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const activeGroup = state.groups[state.groups.length - 1] ?? null;
  return {
    schema: "tilawa-debug-bundle/v1",
    createdAt: new Date().toISOString(),
    pageUrl: location.href,
    userAgent: navigator.userAgent,
    modelReady: state.modelReady,
    isActive: state.isActive,
    streamingConfig: state.streamingConfig,
    audio: {
      sampleRate: 16000,
      chunkCount: state.sessionAudioChunks.length,
      totalSamples,
      durationSec: Math.round((totalSamples / 16000) * 1000) / 1000,
    },
    ui: {
      hasFirstMatch: state.hasFirstMatch,
      lastModelPrediction: state.lastModelPrediction,
      activeGroup: activeGroup
        ? {
            surah: activeGroup.surah,
            surahNameEn: activeGroup.surahNameEn,
            currentAyah: activeGroup.currentAyah,
          }
        : null,
      candidateStatus: {
        text: $candidateStatus.textContent ?? "",
        hidden: $candidateStatus.hidden,
      },
      rawTranscript: {
        text: $rawTranscript.textContent ?? "",
        visible: $rawTranscript.classList.contains("visible"),
      },
      finalSequence: state.finalSequence,
      recentVerseMatches: state.recentVerseMatches,
    },
    diagnostics: state.diagnosticEvents,
    debugEvents: state.debugEvents,
  };
}

async function copyDebugBundle(): Promise<void> {
  const json = JSON.stringify(buildDebugBundle(), null, 2);
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(json);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = json;
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    $debugCopyStatus.textContent = "Copied";
  } catch (err) {
    console.error("Failed to copy debug bundle:", err);
    $debugCopyStatus.textContent = "Copy failed";
  }

  setTimeout(() => {
    $debugCopyStatus.textContent = "";
  }, 1800);
}

function formatDebugValue(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "null";
  return JSON.stringify(value);
}

function truncateDebugText(value: unknown, max = 70): string {
  const text = formatDebugValue(value).replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function debugChip(label: string, value: unknown, variant = ""): HTMLElement {
  const chip = document.createElement("span");
  chip.className = `debug-chip ${variant}`.trim();
  chip.textContent = `${label}: ${truncateDebugText(value)}`;
  return chip;
}

function refsFromDebugList(value: unknown, limit = 4): string {
  if (!Array.isArray(value)) return "";
  return value
    .slice(0, limit)
    .map((entry) => {
      if (!entry || typeof entry !== "object") return formatDebugValue(entry);
      const obj = entry as Record<string, unknown>;
      const score = typeof obj.score === "number" ? ` ${obj.score.toFixed(2)}` : "";
      const fusion = typeof obj.fusion === "number" ? ` ${obj.fusion.toFixed(2)}` : "";
      const ref = obj.ref ?? obj.top_ref ?? "?";
      return `${ref}${score}${fusion}`;
    })
    .join("  ");
}

function summarizeDebugEvent(event: DebugMessage): { label: string; chips: HTMLElement[] } {
  const data = event.data;
  const chips: HTMLElement[] = [];

  if (event.event === "transcribe") {
    chips.push(debugChip("sec", data.audioSec));
    chips.push(debugChip("text", data.text, "debug-chip--wide"));
    chips.push(debugChip("phon", data.rawPhonemes, "debug-chip--wide"));
    const beam = refsFromDebugList(data.beam);
    if (beam) chips.push(debugChip("beam", beam, "debug-chip--wide"));
    return { label: "asr", chips };
  }

  const trackerType = typeof data.type === "string" ? data.type : "tracker";
  if (trackerType === "discovery_cycle") {
    chips.push(debugChip("text", data.text, "debug-chip--wide"));
    chips.push(debugChip("cands", refsFromDebugList(data.candidates), "debug-chip--wide"));
    return { label: "discover", chips };
  }
  if (trackerType === "tracking_cycle") {
    chips.push(debugChip("ref", data.ref));
    chips.push(debugChip("words", `${data.word_position}/${data.total_words}`));
    chips.push(debugChip("cov", data.coverage));
    chips.push(debugChip("primary", data.word_matches));
    chips.push(debugChip("advanced", data.advanced));
    chips.push(debugChip("pending", data.pending));
    chips.push(debugChip("final", data.final_flush));
    return { label: "track", chips };
  }
  if (trackerType === "advance_decision") {
    chips.push(debugChip("from", data.from_ref));
    chips.push(debugChip("to", data.to_ref));
    chips.push(debugChip("action", data.action, data.action === "armed" ? "debug-chip--strong" : ""));
    chips.push(debugChip("why", data.reason, "debug-chip--wide"));
    chips.push(debugChip("words", `${data.word_position}/${data.total_words}`));
    chips.push(debugChip("target", data.completion_target));
    chips.push(debugChip("margin", data.margin));
    chips.push(debugChip("strict", data.strict_margin));
    return { label: "advance", chips };
  }
  if (trackerType === "commit") {
    chips.push(debugChip("ref", data.ref, "debug-chip--strong"));
    chips.push(debugChip("why", data.reason));
    chips.push(debugChip("conf", data.confidence));
    chips.push(debugChip("rank", data.selected_rank));
    return { label: "commit", chips };
  }
  if (trackerType === "pending_emission") {
    chips.push(debugChip("action", data.action));
    chips.push(debugChip("ref", data.ref));
    chips.push(debugChip("margin", data.margin));
    chips.push(debugChip("fresh", data.fresh_samples));
    return { label: "pending", chips };
  }
  if (trackerType === "rollback" || trackerType === "stale_exit" || trackerType === "flush") {
    for (const [key, value] of Object.entries(data)) {
      if (key !== "type") chips.push(debugChip(key, value));
    }
    return { label: trackerType, chips };
  }

  for (const [key, value] of Object.entries(data).slice(0, 5)) {
    if (key !== "type") chips.push(debugChip(key, value));
  }
  return { label: trackerType, chips };
}

function renderDebugPanel(): void {
  $debugSummary.textContent = `${state.debugEvents.length} events`;
  if (!$debugPanel.open) return;

  $debugContent.textContent = "";
  for (const event of state.debugEvents.slice().reverse()) {
    const summary = summarizeDebugEvent(event);
    const item = document.createElement("div");
    item.className = `debug-row debug-row--${summary.label}`;

    const time = document.createElement("span");
    time.className = "debug-time";
    time.textContent = new Date(event.at).toLocaleTimeString();

    const label = document.createElement("span");
    label.className = "debug-label";
    label.textContent = summary.label;

    item.append(time, label, ...summary.chips);

    $debugContent.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------
function pushDiagnosticEvent(type: string, data: Record<string, unknown>): void {
  state.diagnosticEvents.push({ timestamp: Date.now(), type, data });
  if (state.diagnosticEvents.length > MAX_DIAGNOSTIC_EVENTS) {
    state.diagnosticEvents.shift();
  }
}

function checkAnomalyAndSend(msg: VerseMatchMessage): void {
  const now = Date.now();

  // Track recent verse matches for rapid switching detection
  state.recentVerseMatches.push({ surah: msg.surah, ayah: msg.ayah, timestamp: now });
  // Keep only last 10 seconds
  state.recentVerseMatches = state.recentVerseMatches.filter(
    (m) => now - m.timestamp < 10_000,
  );

  let trigger: string | null = null;

  // Surah jump: different surah than previous match
  const prev = state.lastModelPrediction;
  if (prev && prev.surah !== msg.surah) {
    trigger = "surah_jump";
  }

  // Rapid switching: 3+ different verses in 10 seconds
  if (!trigger) {
    const unique = new Set(
      state.recentVerseMatches.map((m) => `${m.surah}:${m.ayah}`),
    );
    if (unique.size >= 3) {
      trigger = "rapid_switching";
    }
  }

  if (!trigger) return;

  // Cooldown
  if (now - state.lastDiagnosticSentAt < DIAGNOSTIC_COOLDOWN_MS) return;
  state.lastDiagnosticSentAt = now;

  sendDiagnosticReport(trigger);
}

async function sendDiagnosticReport(trigger: string): Promise<void> {
  try {
    // Build audio WAV from session chunks
    const totalLen = state.sessionAudioChunks.reduce((s, c) => s + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of state.sessionAudioChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    // Only send last 30s of audio max
    const maxSamples = 16000 * 30;
    const audioSlice = merged.length > maxSamples ? merged.slice(-maxSamples) : merged;
    const wavBlob = float32ToWav(audioSlice, 16000);

    const form = new FormData();
    form.append("audio", wavBlob, "diagnostic.wav");
    form.append("events", JSON.stringify(state.diagnosticEvents));
    form.append("trigger", trigger);

    await fetch("/api/diagnostics", { method: "POST", body: form });
  } catch (err) {
    console.error("Failed to send diagnostic report:", err);
  }
}

function float32ToWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeStr(off: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------------------
// Worker message handler
// ---------------------------------------------------------------------------
function handleWorkerMessage(msg: WorkerOutbound): void {
  if (msg.type === "loading") {
    $modelStatus.textContent = `Loading model... ${msg.percent}%`;
    $modelStatus.classList.remove("ready");
    $loadingProgress.style.width = `${msg.percent}%`;
    $loadingDetail.textContent = `Downloading model — ${msg.percent}%`;
  } else if (msg.type === "loading_status") {
    $loadingDetail.textContent = msg.message;
  } else if (msg.type === "error") {
    $loadingDetail.textContent = `Error: ${msg.message}`;
    $modelStatus.textContent = "Error";
    console.error("Worker reported error:", msg.message);
  } else if (msg.type === "ready") {
    $modelStatus.textContent = "Model ready";
    $modelStatus.classList.add("ready");
    state.modelReady = true;
    $loadingStatus.hidden = true;
    $readyState.hidden = false;
  } else if (msg.type === "verse_match") {
    pushDiagnosticEvent("verse_match", {
      surah: msg.surah, ayah: msg.ayah, confidence: msg.confidence,
    });
    checkAnomalyAndSend(msg);
    handleVerseMatch(msg);
  } else if (msg.type === "verse_candidate") {
    pushDiagnosticEvent("verse_candidate", {
      best: msg.candidates[0] ? `${msg.candidates[0].surah}:${msg.candidates[0].ayah}` : null,
      confidence: msg.candidates[0]?.confidence ?? 0,
      stable: msg.stable,
    });
    handleVerseCandidate(msg);
  } else if (msg.type === "final_sequence") {
    pushDiagnosticEvent("final_sequence", {
      verses: msg.verses.map((v) => `${v.surah}:${v.ayah}`),
      confidence: msg.confidence,
    });
    handleFinalSequence(msg);
  } else if (msg.type === "word_progress") {
    pushDiagnosticEvent("word_progress", {
      surah: msg.surah, ayah: msg.ayah,
      word_index: msg.word_index, total_words: msg.total_words,
    });
    handleWordProgress(msg);
  } else if (msg.type === "raw_transcript") {
    pushDiagnosticEvent("raw_transcript", {
      text: msg.text, confidence: msg.confidence,
    });
    handleRawTranscript(msg);
  } else if (msg.type === "debug") {
    handleDebugMessage(msg);
  }
}

// ---------------------------------------------------------------------------
// Audio capture
// ---------------------------------------------------------------------------
async function startAudio(): Promise<boolean> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
    state.stream = stream;
    $permissionPrompt.hidden = true;

    const audioCtx = new AudioContext();
    state.audioCtx = audioCtx;

    await audioCtx.audioWorklet.addModule("/audio-processor.js");
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = new AudioWorkletNode(audioCtx, "audio-stream-processor");
    state.audioProcessor = processor;
    processor.port.postMessage({
      type: "set_config",
      audioChunkMs: state.streamingConfig.audioChunkMs,
    });

    processor.port.onmessage = (e: MessageEvent) => {
      const samples = new Float32Array(e.data as ArrayBuffer);
      // Save copy to session buffer
      state.sessionAudioChunks.push(samples.slice());
      // Send to worker for recognition
      if (state.worker) {
        state.worker.postMessage(
          { type: "audio", samples },
          [samples.buffer],
        );
      }
    };

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    source.connect(processor);

    const levelBuf = new Float32Array(analyser.fftSize);
    state.isActive = true;
    $indicator.classList.add("active");
    resetListeningWaveform();

    const checkLevel = () => {
      if (!state.isActive) return;
      analyser.getFloatTimeDomainData(levelBuf);
      let sum = 0;
      for (let i = 0; i < levelBuf.length; i++) {
        sum += levelBuf[i] * levelBuf[i];
      }
      const rms = Math.sqrt(sum / levelBuf.length);
      updateListeningWaveform(rms);
      if (rms > 0.01) {
        $indicator.classList.add("audio-detected");
        $indicator.classList.remove("silence");
      } else {
        $indicator.classList.remove("audio-detected");
        $indicator.classList.add("silence");
      }
      requestAnimationFrame(checkLevel);
    };
    checkLevel();

    return true;
  } catch (err) {
    console.error("Failed to start audio:", err);
    $permissionPrompt.hidden = false;
    resetListeningWaveform();
    return false;
  }
}

// ---------------------------------------------------------------------------
// Stop audio capture
// ---------------------------------------------------------------------------
function stopAudio(): void {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close();
    state.audioCtx = null;
  }
  state.audioProcessor = null;
  state.isActive = false;
  $indicator.classList.remove("active", "audio-detected", "silence", "has-verses");
  resetListeningWaveform();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
let modelInitStarted = false;

function initializeModel(): void {
  if (modelInitStarted) return;
  modelInitStarted = true;

  $introScreen.hidden = true;
  $loadingStatus.hidden = false;
  $debugPanel.hidden = !DEBUG_VIEW_ENABLED;
  $modelStatus.textContent = "Loading model...";
  $loadingDetail.textContent = "Starting download";

  const worker = new Worker(
    new URL("./worker/inference.ts", import.meta.url),
    { type: "module" },
  );
  state.worker = worker;

  worker.onmessage = (e: MessageEvent<WorkerOutbound>) => {
    handleWorkerMessage(e.data);
  };

  worker.onerror = (e) => {
    console.error("Worker error:", e);
    $loadingDetail.textContent = `Worker error: ${e.message || "unknown"}`;
  };

  worker.postMessage({ type: "init" });
  pushStreamingConfig();
  syncDebugEnabled();
}

document.addEventListener("DOMContentLoaded", () => {
  $debugPanel.addEventListener("toggle", syncDebugEnabled);
  $debugCopy.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    copyDebugBundle();
  });

  $btnBeginTest.addEventListener("click", () => {
    $btnBeginTest.disabled = true;
    initializeModel();
  });

  syncDebugEnabled();

  // Button handlers
  $btnStart.addEventListener("click", async () => {
    $readyState.hidden = true;
    $recordingState.hidden = false;
    $listeningStatus.hidden = false;
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    state.groups = [];
    state.diagnosticEvents = [];
    state.debugEvents = [];
    state.recentVerseMatches = [];
    state.finalSequence = [];
    $verses.innerHTML = "";
    $rawTranscript.textContent = "";
    $rawTranscript.classList.remove("visible");
    $candidateStatus.textContent = "";
    $candidateStatus.hidden = true;
    $candidateStatus.classList.remove("candidate-status--stable", "candidate-status--pending");
    renderDebugPanel();
    // Reset tracker in worker
    state.worker?.postMessage({ type: "reset" });
    pushStreamingConfig();
    const started = await startAudio();
    if (!started) {
      $recordingState.hidden = true;
      $listeningStatus.hidden = true;
      $readyState.hidden = false;
    }
  });

  $btnStop.addEventListener("click", () => {
    stopAudio();
    $recordingState.hidden = true;
    $listeningStatus.hidden = true;
    $postRecording.hidden = false;
  });

  $btnRestart.addEventListener("click", () => {
    state.sessionAudioChunks = [];
    state.lastModelPrediction = null;
    state.hasFirstMatch = false;
    state.groups = [];
    state.debugEvents = [];
    state.finalSequence = [];
    $verses.innerHTML = "";
    $rawTranscript.textContent = "";
    $rawTranscript.classList.remove("visible");
    $candidateStatus.textContent = "";
    $candidateStatus.hidden = true;
    $candidateStatus.classList.remove("candidate-status--stable", "candidate-status--pending");
    renderDebugPanel();
    $postRecording.hidden = true;
    $listeningStatus.hidden = true;
    $readyState.hidden = false;
  });

  $btnReport.addEventListener("click", () => {
    openReportDialog({
      audioChunks: state.sessionAudioChunks,
      modelPrediction: state.lastModelPrediction,
      quranData: state.quranData!,
      debugBundle: buildDebugBundle(),
    });
  });
});
