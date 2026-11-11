import { loadReferenceIndex } from "@tilawa/core";
import type { CheckerEvent, EngineEvent, PhraseAsset, ReferenceAsset, WordRef } from "@tilawa/core";
import { MODEL_REGISTRY, createEngine, decodeWav, loadWebModel, startMicrophone, withBase } from "@tilawa/sdk/web";
import type { Engine, Microphone, Session, SessionPolicy } from "@tilawa/sdk/web";
import refAsset from "../../core/assets/ref-v2.json";
import phrases from "../../core/assets/phrases-v1.json";
import "./harness.css";

declare const __SDK_BUILD__: { revision: string; branch: string; builtAt: string; engineDirty: boolean };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const asset = refAsset as unknown as ReferenceAsset;
const index = loadReferenceIndex(asset);
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;
const text = new Map<string, string>();
for (const a of asset.ayahs) a.w.forEach((w, i) => text.set(`${a.s}:${a.a}:${i + 1}`, w));
const status = (s: string) => { $("status").textContent = s; };
const number = (id: string) => {
  const input = $<HTMLInputElement>(id);
  if (!input.checkValidity()) throw new RangeError(`Check ${input.labels?.[0]?.textContent?.trim() ?? id}.`);
  return Number(input.value);
};
let model: Awaited<ReturnType<typeof loadWebModel>> | null = null, engine: Engine | null = null;
let session: Session | null = null, mic: Microphone | null = null;
let busy = false, running = false, stopRequested = false;
let selectedFile: File | null = null, audioUrl = "", audio: Blob | null = null;
let take: { id: string; beganAt: string; source: string; policy: SessionPolicy; vad: boolean; realtime: boolean; tier: string } | null = null;
let chunks: Float32Array[] = [], audioRate = 16000;
let events: { arrivalMs: number; event: EngineEvent }[] = [], began = 0;
let stats = { audioS: 0, computeMs: 0, pushes: [] as number[] };
let warnings = new Map<string, CheckerEvent>();
$("version").textContent = `${__SDK_BUILD__.branch} / ${__SDK_BUILD__.revision.slice(0, 8)}${__SDK_BUILD__.engineDirty ? " (modified engine)" : ""}`;

function controls() {
  $<HTMLFieldSetElement>("session-settings").disabled = busy || running;
  $<HTMLInputElement>("file").disabled = busy || running;
  $<HTMLInputElement>("realtime").disabled = busy || running;
  $<HTMLButtonElement>("mic").disabled = busy;
  $("mic").textContent = running ? "Stop" : "Start recording";
  $<HTMLButtonElement>("replay").disabled = !selectedFile || busy || running;
  for (const id of ["copy", "download"]) $<HTMLButtonElement>(id).disabled = !session || busy || running;
  $<HTMLButtonElement>("audio-download").disabled = !audio || busy || running;
}
function showMode() {
  $("declared-passage").hidden = document.querySelector<HTMLInputElement>('input[name="mode"]:checked')!.value !== "memorization";
}
document.querySelectorAll<HTMLInputElement>('input[name="mode"]').forEach(el => { el.onchange = showMode; });
showMode();
async function prepareEngine() {
  if (engine) return;
  const started = performance.now();
  model = await loadWebModel({
    registry: withBase(MODEL_REGISTRY, new URL(".", location.href).href),
    threads: 1, wasmPaths: new URL("ort/", location.href).href,
    onProgress: (n, total) => status(`Preparing recitation checking: ${Math.round(n / total * 100)}%`),
  });
  engine = createEngine({ reference: index, phrases: phrases as PhraseAsset, model, recordEvidence: true });
  $("model-note").textContent = `${model.manifest.id} (${model.manifest.format}) on ${model.backend.name}. ` +
    `${model.fromCache ? "Cached" : "Downloaded"} in ${((performance.now() - started) / 1000).toFixed(1)} s. Reference ${index.version}. ` +
    `Cross-origin isolated: ${crossOriginIsolated}. ${model.fallbacks.join("; ")}`;
}

function newSession(source: string): Session {
  const confirm = $<HTMLSelectElement>("confirm").value;
  const policy: SessionPolicy = {
    countSelfCorrections: $<HTMLInputElement>("selfcorr").checked,
    confirm: confirm === "next" ? { after: "next-word" } : { after: "delay", ms: number("delay") },
    minConfidence: number("confidence"), evidenceMs: number("evidence"),
  };
  const s = document.querySelector<HTMLInputElement>('input[name="mode"]:checked')!.value === "memorization"
    ? engine!.startMemorization({ surah: number("surah"), fromAyah: number("from"), toAyah: number("to") }, policy)
    : engine!.startFreeRecitation(policy);
  take = { id: crypto.randomUUID(), beganAt: new Date().toISOString(), source, policy,
    vad: source === "microphone" && $<HTMLInputElement>("vad").checked,
    realtime: source !== "microphone" && $<HTMLInputElement>("realtime").checked, tier: "full" };
  events = []; chunks = []; stats = { audioS: 0, computeMs: 0, pushes: [] }; warnings = new Map();
  audio = null; began = performance.now(); stopRequested = false;
  $<HTMLAudioElement>("playback").pause(); $("playback").hidden = true;
  render.reset(s);
  s.subscribe(event => {
    events.push({ arrivalMs: Math.round(performance.now() - began), event });
    render.event(event);
  });
  // Time inference after preceding pushes finish, excluding queue wait from RTF.
  let queue: Promise<unknown> = Promise.resolve();
  return Object.assign(Object.create(s), {
    pushAudio(pcm: Float32Array, rate: number) {
      audioRate = rate; chunks.push(pcm.slice()); stats.audioS += pcm.length / rate;
      const job = queue.then(async () => {
        const started = performance.now();
        const out = await s.pushAudio(pcm, rate);
        const elapsed = performance.now() - started;
        stats.computeMs += elapsed; stats.pushes.push(elapsed);
        return out;
      });
      queue = job.catch(() => undefined);
      return job;
    },
    async end() {
      await queue;
      const started = performance.now();
      const out = await s.end();
      stats.computeMs += performance.now() - started;
      return out;
    },
  });
}
function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function finishAudio() {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  // Float WAV preserves the exact PCM supplied to the session for a repeatable replay.
  const n = chunks.reduce((sum, p) => sum + p.length, 0), b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
  const tag = (at: number, s: string) => [...s].forEach((c, i) => v.setUint8(at + i, c.charCodeAt(0)));
  tag(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); tag(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 3, true); v.setUint16(22, 1, true); v.setUint32(24, audioRate, true);
  v.setUint32(28, audioRate * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 32, true);
  tag(36, "data"); v.setUint32(40, n * 4, true);
  let at = 44;
  for (const chunk of chunks) for (const sample of chunk) { v.setFloat32(at, sample, true); at += 4; }
  audio = new Blob([b], { type: "audio/wav" }); chunks = []; audioUrl = URL.createObjectURL(audio);
  $<HTMLAudioElement>("playback").src = audioUrl; $("playback").hidden = false;
}
async function stopMic() {
  if (!mic || busy) return;
  busy = true; controls();
  const current = mic; mic = null;
  try { await current.stop(); status("Stopped. Review the feedback or download this take."); }
  catch (e) { status(`Stopped with an error: ${(e as Error).message}`); }
  finally { running = false; busy = false; finishAudio(); controls(); refresh(); }
}
$("mic").onclick = async () => {
  if (mic) { await stopMic(); return; }
  if (running) { stopRequested = true; status("Stopping replay…"); return; }
  busy = true; controls();
  try {
    status("Preparing recitation checking…");
    await prepareEngine();
    session = newSession("microphone");
    mic = await startMicrophone(session, {
      vad: take!.vad ? undefined : false,
      onLevel: db => { $<HTMLMeterElement>("meter").value = db; },
      onError: e => { status(`Inference failed: ${(e as Error).message}`); void stopMic(); },
    });
    running = true; status("Listening. Recite a few words to find your place.");
  } catch (e) { status(`Microphone failed: ${(e as Error).message}`); }
  finally { busy = false; controls(); }
};
$<HTMLInputElement>("file").onchange = () => { selectedFile = $<HTMLInputElement>("file").files?.[0] ?? null; controls(); };
$("replay").onclick = async () => {
  if (!selectedFile) return;
  busy = true; controls();
  try {
    const wav = decodeWav(new Uint8Array(await selectedFile.arrayBuffer()));
    if (!Number.isFinite(wav.sampleRate) || wav.sampleRate <= 0 || !wav.samples.length) throw new Error("WAV is empty or has an invalid sample rate.");
    status("Preparing recitation checking…");
    await prepareEngine();
    session = newSession(selectedFile.name); running = true; busy = false; controls();
    const step = Math.max(1, Math.round(wav.sampleRate * 0.02)), started = performance.now();
    for (let at = 0; at < wav.samples.length && !stopRequested; at += step) {
      const end = Math.min(at + step, wav.samples.length);
      if (take!.realtime) {
        const delay = started + end / wav.sampleRate * 1000 - performance.now();
        if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
      }
      if (stopRequested) break;
      await session.pushAudio(wav.samples.subarray(at, end), wav.sampleRate);
      if (at % (step * 25) === 0) { status(`Replaying ${selectedFile.name}: ${(end / wav.sampleRate).toFixed(1)} s`); refresh(); }
    }
    await session.end();
    status(stopRequested ? "Replay stopped." : "Replay complete. Adjust settings and replay to compare.");
  } catch (e) { status(`Replay failed: ${(e as Error).message}`); }
  finally { running = false; busy = false; if (chunks.length) finishAudio(); controls(); refresh(); }
};
function report() {
  return { schema: 1, build: __SDK_BUILD__, take, userAgent: navigator.userAgent, crossOriginIsolated,
    model: model && { id: model.manifest.id, format: model.manifest.format, backend: model.backend.name, fallbacks: model.fallbacks },
    session: session && { mode: session.mode, passage: session.passage, capabilities: session.capabilities },
    stats, mistakes: session?.mistakes(), events, evidence: session?.evidence() };
}
$("copy").onclick = async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(report())); status("Trace copied."); }
  catch { status("Clipboard unavailable. Use Download trace."); }
};
$("download").onclick = () => download(new Blob([JSON.stringify(report())], { type: "application/json" }), `tilawa-${take!.id}.json`);
$("audio-download").onclick = () => { if (audio) download(audio, `tilawa-${take!.id}.wav`); };
function refresh() {
  const p = [...stats.pushes].sort((a, b) => a - b), p95 = p[Math.floor(.95 * (p.length - 1))] ?? 0;
  $("stats").textContent = session ? `${stats.audioS.toFixed(1)} s audio. RTF ${(stats.computeMs / 1000 / Math.max(1e-9, stats.audioS)).toFixed(3)}. Push p95 ${p95.toFixed(0)} ms.` : "";
  const ev = session?.evidence();
  $("sounds").textContent = ev?.slice(-64).flatMap(e => e.phonemes.map(p => p.p)).join("") || "Waiting for sounds.";
  $("mistake-count").textContent = session ? `${session.mistakes().length} ${session.mistakes().length === 1 ? "mistake" : "mistakes"}.` : "Start reciting to see feedback.";
  if (mic && stats.audioS >= 180 && !busy) void stopMic();
}
setInterval(refresh, 500);
window.addEventListener("beforeunload", e => { if (running) e.preventDefault(); });

const render = (() => {
  const spans = new Map<string, HTMLSpanElement>();
  let at: string | null = null, ayah = "";
  const show = (r: WordRef) => {
    const k = key(r);
    let span = spans.get(k);
    if (!span) { span = document.createElement("span"); span.textContent = text.get(k) ?? "?"; spans.set(k, span); $("passage").append(span, " "); }
    return span;
  };
  const repaint = (r: WordRef) => {
    const span = spans.get(key(r)); if (!span) return;
    span.classList.remove("suspected", "confirmed", "resolved", "uncertain");
    // Multiple checks can refer to one word; keep its most severe current state visible.
    const states = [...warnings.values()].filter(e => key(e.location) === key(r)).map(e => e.state);
    const state = (["confirmed", "suspected", "uncertain", "resolved"] as const).find(s => states.includes(s));
    if (state) span.classList.add(state);
  };
  return {
    reset(s: Session) {
      for (const id of ["passage", "mistakes", "log", "candidates", "sounds"]) $(id).textContent = "";
      $("position").textContent = "Finding your place in the Quran.";
      spans.clear(); at = null; ayah = "";
      if (!s.passage) return;
      const ws = index.words.filter(w => w.end > w.start);
      const a = ws.findIndex(w => key(w) === key(s.passage!.from)), b = ws.findIndex(w => key(w) === key(s.passage!.to));
      for (const w of ws.slice(a, b + 1)) show(w).classList.toggle("hidden", $<HTMLInputElement>("hide").checked);
    },
    event(e: EngineEvent) {
      if (e.type === "position") {
        const best = e.observed[0];
        $("position").textContent = best ? `Surah ${best.word.surah}, ayah ${best.word.ayah}` : "Finding your place…";
        $("candidates").textContent = e.observed.slice(0, 3).map(c => `${key(c.word)} (${Math.round(c.confidence * 100)}%)`).join(" / ");
        if (best && e.state === "locked") {
          const r = best.word, nextAyah = `${r.surah}:${r.ayah}`;
          if (session?.mode === "free" && ayah !== nextAyah) {
            $("passage").replaceChildren(); spans.clear(); ayah = nextAyah;
            asset.ayahs.find(a => a.s === r.surah && a.a === r.ayah)?.w.forEach((_, i) => {
              const ref = { ...r, word: i + 1 }; show(ref).classList.toggle("hidden", $<HTMLInputElement>("hide").checked); repaint(ref);
            });
          }
          if (at) spans.get(at)?.classList.remove("at");
          for (const w of index.words) if (w.surah === r.surah && w.ayah === r.ayah && w.word <= r.word) spans.get(key(w))?.classList.remove("hidden");
          at = key(r); show(r).classList.add("at");
        }
      }
      if (e.type === "checker") {
        warnings.set(e.id, e); repaint(e.location);
        let li = $("mistakes").querySelector<HTMLLIElement>(`[data-id="${e.id}"]`);
        if (!li) { li = document.createElement("li"); li.dataset.id = e.id; $("mistakes").append(li); }
        li.className = e.state;
        const stateLabel = { suspected: "Possible mistake", confirmed: "Mistake", resolved: "Corrected", uncertain: "Needs review" }[e.state];
        li.textContent = `${stateLabel}: ${e.errorClass.replaceAll("_", " ")} at ${key(e.location)} «${text.get(key(e.location)) ?? ""}» (${Math.round(e.confidence * 100)}%)`;
      }
      // Full events remain in the download; the visible tail stays small during long takes.
      $("log").textContent = events.slice(-25).map(({ event }) => JSON.stringify(event.type === "position" ? { ...event, observed: event.observed.slice(0, 3) } : event)).join("\n");
    },
  };
})();
controls();
