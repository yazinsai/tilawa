import { loadReferenceIndex } from "@tilawa/core";
import type { EngineEvent, CheckerEvent, PositionEvent, ReferenceAsset, PhraseAsset, WordRef } from "@tilawa/core";
import { createEngine, loadWebModel, MODEL_REGISTRY, startMicrophone, withBase } from "@tilawa/sdk/web";
import type { Engine, Session, Microphone } from "@tilawa/sdk/web";
import reference from "../../core/assets/ref-v2.json";
import phrases from "../../core/assets/phrases-v1.json";
import "./live.css";

declare const __LIVE_BUILD__: { revision: string; branch: string; builtAt: string; engineDirty: boolean };
const build = __LIVE_BUILD__;
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const asset = reference as unknown as ReferenceAsset;
const index = loadReferenceIndex(asset);
const wordKey = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const words = new Map<string, string>();
for (const a of asset.ayahs) a.w.forEach((w, i) => words.set(`${a.s}:${a.a}:${i + 1}`, w));
const clock = (ms: number) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
const status = (s: string) => { $("status").textContent = s; };
const start = $<HTMLButtonElement>("start");
let engine: Engine | null = null, session: Session | null = null, mic: Microphone | null = null;
let busy = false, saving = false, chunks: Float32Array[] = [], samples = 0, events: EngineEvent[] = [];
let latest: PositionEvent | null = null, audio: Blob | null = null, audioUrl = "", sessionId = "", beganAt = "";
let flagTime = 0, flagSnapshot: PositionEvent | null = null, stopped = false, active = false;
type Flag = { timeMs: number; component: string; note: string; position: PositionEvent | null };
let flags: Flag[] = [], warnings = new Map<string, CheckerEvent>(), lastSounds = 0, savedFlags = 0;
// Arrival time keeps fresh sounds visible even when inference trails the microphone.
let soundTrail: { text: string; heardAt: number }[] = [];
$("version").textContent = `${build.branch} / ${build.revision.slice(0, 8)}${build.engineDirty ? " (modified engine)" : ""}`;

function controls() {
  start.disabled = busy || saving;
  start.textContent = mic ? "Stop recording" : stopped ? "Start new recording" : "Start microphone";
  document.querySelectorAll<HTMLButtonElement>("[data-flag]").forEach(b => { b.disabled = !session || busy; });
  $<HTMLButtonElement>("save").disabled = !stopped || !flags.length || savedFlags === flags.length || saving;
  $<HTMLButtonElement>("download").disabled = !session;
}
function report() {
  return { schema: 1, id: sessionId, beganAt, build, model: session?.capabilities, durationMs: samples / 16,
    userAgent: navigator.userAgent, flags, events, evidence: session?.evidence(), savedAt: new Date().toISOString() };
}
function wav(): Blob {
  const buffer = new ArrayBuffer(44 + samples * 2), v = new DataView(buffer);
  const str = (at: number, s: string) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples * 2, true); str(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples * 2, true);
  let at = 44;
  for (const chunk of chunks) for (const s of chunk) { v.setInt16(at, Math.max(-1, Math.min(1, s)) * (s < 0 ? 32768 : 32767), true); at += 2; }
  return new Blob([buffer], { type: "audio/wav" });
}
function render() {
  $("clock").textContent = clock(samples / 16);
  if (!session) return;
  const evidence = session.evidence();
  const now = performance.now();
  if (evidence.length !== lastSounds) {
    const incoming = evidence.slice(lastSounds).flatMap(e => e.phonemes);
    lastSounds = evidence.length;
    if (incoming.length) {
      soundTrail.push(...incoming.map(p => ({ text: p.p, heardAt: now })));
      soundTrail = soundTrail.slice(-64);
      const sounds = $("sounds");
      sounds.replaceChildren(...soundTrail.map(p => {
        const span = document.createElement("span");
        span.textContent = p.text;
        return span;
      }));
      // The trail reads right to left; its newest sounds are at the left edge.
      sounds.scrollLeft = -sounds.scrollWidth;
    }
  }
  const spans = $("sounds").children;
  soundTrail.forEach((p, i) => spans[i]?.classList.toggle("is-fresh", now - p.heardAt < 1800));
  $("heard-count").textContent = soundTrail.length
    ? "Latest sounds · highlighted as they arrive"
    : "Waiting for sounds";
  if (latest) {
    const best = latest.observed[0];
    $("position").textContent = best ? `${latest.state} · Surah ${best.word.surah}, ayah ${best.word.ayah}, word ${best.word.word}` : latest.state;
    const verse = $("verse"); verse.replaceChildren();
    if (best) {
      const ayah = asset.ayahs.find(a => a.s === best.word.surah && a.a === best.word.ayah);
      ayah?.w.forEach((w, i) => { const span = document.createElement(i + 1 === best.word.word ? "mark" : "span"); span.textContent = w + " "; verse.append(span); });
    }
    $("candidates").textContent = latest.observed.slice(0, 3).map(c => `${wordKey(c.word)} (${Math.round(c.confidence * 100)}%)`).join(" / ") || "No position yet";
  }
  $("checker-status").textContent = !latest || latest.state !== "locked" ? "Waiting for a confident position" : warnings.size ? `${warnings.size} checks raised` : "No warnings so far";
  $("warnings").replaceChildren(...Array.from(warnings.values()).map(e => {
    const li = document.createElement("li"); li.className = e.state;
    li.textContent = `${e.state}: ${e.errorClass.replaceAll("_", " ")} at ${wordKey(e.location)} ${words.get(wordKey(e.location)) ?? ""} (${Math.round(e.confidence * 100)}%)`;
    return li;
  }));
  $("events").textContent = events.slice(-12).map(e => JSON.stringify(e)).join("\n");
}
function renderFlags() {
  $("flag-count").textContent = String(flags.length);
  $("flags").replaceChildren(...flags.map(f => { const li = document.createElement("li"); li.textContent = `${clock(f.timeMs)} · ${f.component}: ${f.note}`; return li; }));
  controls();
}
async function save() {
  if (!audio || !flags.length || saving) return;
  const submittedFlags = flags.length;
  saving = true; controls(); $("save-status").textContent = "Saving audio and diagnostic trace…";
  try {
    const form = new FormData(); form.append("report", new Blob([JSON.stringify(report())], { type: "application/json" }), "report.json"); form.append("audio", audio, "audio.wav");
    const res = await fetch("/api/live/sessions", { method: "POST", body: form });
    if (!res.ok) throw new Error(`Save failed (${res.status}): ${(await res.text()).slice(0, 300)}`);
    const out = await res.json(); savedFlags = submittedFlags;
    $("save-status").textContent = `Saved ${submittedFlags} flags with audio. `;
    for (const [label, file] of [["Audio", "audio.wav"], ["Diagnostic trace", "report.json"]]) {
      const link = document.createElement("a"); link.textContent = label;
      link.href = `/api/live/sessions/${encodeURIComponent(out.id)}/${file}`;
      link.target = "_blank"; link.rel = "noopener";
      $("save-status").append(link, " ");
    }
    $("save-status").append(`Reference: ${out.id}`);
  } catch (e) { $("save-status").textContent = `${(e as Error).message}. Your take is still here; retry Save.`; }
  finally { saving = false; controls(); }
}
async function stop() {
  if (!mic || busy) return;
  busy = true; active = false; controls();
  const current = mic; mic = null;
  try { await current.stop(); status("Recording stopped. Listen back or flag a moment."); }
  catch (e) { status(`Recording stopped with an error: ${(e as Error).message}`); }
  finally {
    stopped = true; audio = wav(); chunks = [];
    if (audioUrl) URL.revokeObjectURL(audioUrl); audioUrl = URL.createObjectURL(audio);
    const player = $<HTMLAudioElement>("playback"); player.src = audioUrl; player.hidden = false;
    busy = false; controls(); render();
    if (flags.length) await save();
  }
}
start.onclick = async () => {
  if (mic) { await stop(); return; }
  if (flags.length > savedFlags && !confirm("This take has unsaved flags. Start a new recording and discard them?")) return;
  busy = true; controls();
  try {
    if (!engine) {
      status("Loading the main listener…");
      const model = await loadWebModel({ registry: withBase(MODEL_REGISTRY, new URL("/live/", location.href).href), wasmPaths: "/live/ort/", threads: 1,
        onProgress: (n, total) => status(`Loading model: ${Math.round(n / total * 100)}%`) });
      engine = createEngine({ reference: index, phrases: phrases as PhraseAsset, model, recordEvidence: true });
      $("model-info").textContent = `${model.manifest.id} / ${model.manifest.format} / ${model.backend.name}. Built ${build.builtAt}. Reference ${index.version}.`;
    }
    chunks = []; samples = 0; events = []; flags = []; warnings = new Map(); latest = null; lastSounds = 0; soundTrail = []; $("sounds").textContent = "—"; savedFlags = 0; stopped = false; audio = null;
    sessionId = crypto.randomUUID(); beganAt = new Date().toISOString();
    $("position").textContent = "Finding your place in the Quran"; $("verse").textContent = "—";
    $("candidates").textContent = "Recite a few words to get started.";
    $("save-status").textContent = "Flags will save with the audio when you stop.";
    $<HTMLAudioElement>("playback").pause(); $<HTMLAudioElement>("playback").hidden = true;
    session = engine.startFreeRecitation();
    session.subscribe(e => { events.push(e); if (e.type === "position") latest = e; if (e.type === "checker") warnings.set(e.id, e); });
    const s = session;
    // Capture exactly the PCM passed to the unmodified SDK. No VAD means trace timestamps match the saved audio.
    const recorded: Session = Object.assign(Object.create(s), { pushAudio(pcm: Float32Array, rate: number) {
      if (!active) return Promise.resolve([]);
      const take = pcm.slice(0, Math.max(0, 180 * 16000 - samples));
      if (!take.length) return Promise.resolve([]);
      chunks.push(take); samples += take.length;
      return s.pushAudio(take, rate);
    } });
    active = true;
    mic = await startMicrophone(recorded, { vad: false,
      onLevel: db => { $<HTMLMeterElement>("level").value = db; },
      onError: e => { status(`Inference failed: ${(e as Error).message}`); void stop(); } });
    status("Listening. Recite naturally; first position may take a few words."); renderFlags();
  } catch (e) { active = false; status(`Could not start: ${(e as Error).message}. Check microphone permission and retry.`); }
  finally { busy = false; controls(); }
};
setInterval(() => { render(); if (samples >= 180 * 16000 && mic && !busy) void stop(); }, 250);
const dialog = $<HTMLDialogElement>("flag-dialog");
document.querySelectorAll<HTMLButtonElement>("[data-flag]").forEach(b => b.onclick = () => {
  const player = $<HTMLAudioElement>("playback");
  flagTime = stopped ? player.currentTime * 1000 : samples / 16;
  const atPosition = stopped ? events.filter((e): e is PositionEvent => e.type === "position" && e.timeMs <= flagTime).at(-1) : latest;
  flagSnapshot = atPosition ? structuredClone(atPosition) : null;
  $<HTMLSelectElement>("component").value = b.dataset.flag!;
  $<HTMLTextAreaElement>("note").value = ""; $("flag-at").textContent = `At ${clock(flagTime)} in this recording`;
  dialog.showModal(); $<HTMLTextAreaElement>("note").focus();
});
$("cancel").onclick = () => dialog.close();
$("flag-form").onsubmit = e => {
  e.preventDefault(); const note = $<HTMLTextAreaElement>("note").value.trim(); if (!note) return;
  if (flags.length >= 100) { $("flag-at").textContent = "This take has 100 flags. Stop and save before starting another."; return; }
  flags.push({ timeMs: flagTime, component: $<HTMLSelectElement>("component").value, note, position: flagSnapshot });
  dialog.close(); renderFlags(); $("save-status").textContent = stopped ? "New flag added. Save to keep it with the audio." : "Flag added. It will save with audio when you stop.";
};
$("save").onclick = () => { void save(); };
$("download").onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(report())], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = `tilawa-${sessionId}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); };
window.addEventListener("beforeunload", e => { if (mic || flags.length > savedFlags) { e.preventDefault(); e.returnValue = ""; } });
controls();
