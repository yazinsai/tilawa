import { loadReferenceIndex } from "@tilawa/core";
import type { CheckerEvent, EngineEvent, PhraseAsset, ReferenceAsset, WordRef } from "@tilawa/core";
import { MODEL_REGISTRY, createEngine, decodeWav, loadWebModel, startMicrophone, withBase } from "@tilawa/sdk/web";
import type { Engine, LoadedModel, Microphone, Session, WebProvider } from "@tilawa/sdk/web";
import refAsset from "../../core/assets/ref-v2.json";
import phrases from "../../core/assets/phrases-v1.json";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const asset = refAsset as unknown as ReferenceAsset;
const index = loadReferenceIndex(asset);
const text = new Map<string, string>();
for (const a of asset.ayahs) a.w.forEach((w, i) => text.set(`${a.s}:${a.a}:${i + 1}`, w));
const key = (r: WordRef) => `${r.surah}:${r.ayah}:${r.word}`;

let model: (LoadedModel & { fromCache: boolean; fallbacks: string[] }) | null = null;
let engine: Engine | null = null;
let session: Session | null = null;
let mic: Microphone | null = null;
const log: { t: number; e: EngineEvent }[] = [];
const stats = { audioS: 0, computeMs: 0, pushes: [] as number[] };
const status = (s: string) => ($("status").textContent = s);

$("load").onclick = async () => {
  ($("load") as HTMLButtonElement).disabled = true;
  try {
    const t0 = performance.now();
    model = await loadWebModel({
      registry: withBase(MODEL_REGISTRY, new URL(".", location.href).href),
      providers: ($<HTMLSelectElement>("provider").value.split(",") as WebProvider[]),
      threads: Number($<HTMLSelectElement>("threads").value),
      wasmPaths: new URL("ort/", location.href).href,
      allowOverBudget: true,
      onProgress: (n, total) => status(`Downloading model ${(n / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`),
    });
    engine = createEngine({ reference: index, phrases: phrases as PhraseAsset, model });
    const tier = MODEL_REGISTRY.tiers[0];
    $("model-note").textContent = `${model.manifest.id} (${model.manifest.format}, ${(tier.bytes / 1e6).toFixed(0)} MB) on ${model.backend.name}, ` +
      `${model.fromCache ? "from cache" : "downloaded"} in ${((performance.now() - t0) / 1000).toFixed(1)} s; crossOriginIsolated=${crossOriginIsolated}. ` +
      `${model.fallbacks.length ? `Fell back: ${model.fallbacks.join("; ")}. ` : ""}${tier.note ?? ""}`;
    ($("mic") as HTMLButtonElement).disabled = false;
    ($("file") as HTMLInputElement).disabled = false;
    status("Ready.");
  } catch (e) {
    status(`Load failed: ${(e as Error).message}`);
    ($("load") as HTMLButtonElement).disabled = false;
  }
};

function newSession(): Session {
  const mode = document.querySelector<HTMLInputElement>('input[name="mode"]:checked')!.value;
  const c = $<HTMLSelectElement>("confirm").value;
  const policy = {
    countSelfCorrections: $<HTMLInputElement>("selfcorr").checked,
    confirm: c === "next" ? ({ after: "next-word" } as const) : ({ after: "delay", ms: Number(c) } as const),
  };
  const s = mode === "memorization"
    ? engine!.startMemorization({ surah: +$<HTMLInputElement>("surah").value, fromAyah: +$<HTMLInputElement>("from").value, toAyah: +$<HTMLInputElement>("to").value }, policy)
    : engine!.startFreeRecitation(policy);
  log.length = 0;
  Object.assign(stats, { audioS: 0, computeMs: 0, pushes: [] });
  render.reset(s);
  s.subscribe((e) => {
    log.push({ t: performance.now(), e });
    render.event(e);
  });
  return s;
}

/** Times every push, so the page shows real-time factor and per-push latency on this device. */
function timed(s: Session): Session {
  const push = s.pushAudio.bind(s);
  return Object.assign(Object.create(s), {
    async pushAudio(samples: Float32Array, rate: number) {
      const t0 = performance.now();
      const out = await push(samples, rate);
      const ms = performance.now() - t0;
      stats.audioS += samples.length / rate;
      stats.computeMs += ms;
      stats.pushes.push(ms);
      return out;
    },
  });
}

$("mic").onclick = async () => {
  const btn = $<HTMLButtonElement>("mic");
  if (mic) {
    btn.disabled = true;
    await mic.stop();
    mic = null;
    btn.textContent = "Start microphone";
    btn.disabled = false;
    status(`Stopped. ${summary()}`);
    return;
  }
  try {
    session = timed(newSession());
    mic = await startMicrophone(session, {
      onLevel: (db) => ($("meter").style.width = `${Math.max(0, Math.min(100, (db + 70) * 1.6))}%`),
      onError: (e) => status(`Error: ${(e as Error).message}`),
    });
    btn.textContent = "Stop";
    status(`Listening at ${mic.sampleRate} Hz…`);
    const tick = setInterval(() => (mic ? status(`Listening (${mic.speaking ? "speech" : "silence"}). ${summary()}`) : clearInterval(tick)), 500);
  } catch (e) {
    status(`Microphone failed: ${(e as Error).message}`);
  }
};

$<HTMLInputElement>("file").onchange = async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  const wav = decodeWav(new Uint8Array(await f.arrayBuffer()));
  session = timed(newSession());
  const step = Math.round(wav.sampleRate * 0.32);
  for (let a = 0; a < wav.samples.length; a += step) {
    await session.pushAudio(wav.samples.subarray(a, a + step), wav.sampleRate);
    status(`File ${(a / wav.sampleRate).toFixed(1)} s. ${summary()}`);
  }
  await session.end();
  status(`File done. ${summary()}`);
};

$("copy").onclick = async () => {
  const body = JSON.stringify({
    userAgent: navigator.userAgent, crossOriginIsolated, model: model && { id: model.manifest.id, backend: model.backend.name },
    session: session && { mode: session.mode, passage: session.passage }, stats: { ...stats, summary: summary() },
    mistakes: session?.mistakes(), events: log.map(({ t, e }) => ({ t: Math.round(t), ...e })),
  });
  await navigator.clipboard.writeText(body).catch(() => ($("log").textContent = body));
  status("Copied.");
};

function summary() {
  const p = [...stats.pushes].sort((a, b) => a - b);
  const p95 = p.length ? p[Math.floor(0.95 * (p.length - 1))] : 0;
  return `audio ${stats.audioS.toFixed(1)} s, RTF ${(stats.computeMs / 1000 / Math.max(1e-9, stats.audioS)).toFixed(3)}, push p95 ${p95.toFixed(0)} ms`;
}

const render = (() => {
  const spans = new Map<string, HTMLSpanElement>();
  const states = new Map<string, string>();
  let at: string | null = null;
  const passage = $("passage");
  const mistakes = $("mistakes");
  const show = (k: string) => {
    let s = spans.get(k);
    if (!s) {
      s = document.createElement("span");
      s.textContent = text.get(k) ?? "?";
      spans.set(k, s);
      passage.append(s, " ");
    }
    return s;
  };
  return {
    reset(s: Session) {
      passage.textContent = "";
      mistakes.textContent = "";
      $("log").textContent = "";
      spans.clear();
      states.clear();
      at = null;
      if (!s.passage) return;
      const words = index.words.filter((w) => w.end > w.start);
      const a = words.findIndex((w) => key(w) === key(s.passage!.from));
      const b = words.findIndex((w) => key(w) === key(s.passage!.to));
      const hide = $<HTMLInputElement>("hide").checked;
      for (const w of words.slice(a, b + 1)) show(key(w)).classList.toggle("hidden", hide);
    },
    event(e: EngineEvent) {
      $("log").textContent += `${JSON.stringify(e.type === "position" ? { ...e, observed: e.observed.slice(0, 1) } : e)}\n`;
      if (e.type === "position" && e.observed[0] && e.state === "locked") {
        if (at) spans.get(at)?.classList.remove("at");
        const ws = index.words;
        const i0 = ws.findIndex((w) => key(w) === key(e.observed[0].span.start));
        const i1 = ws.findIndex((w) => key(w) === key(e.observed[0].word));
        for (let i = i0; i >= 0 && i <= i1; i++) show(key(ws[i])).classList.remove("hidden");
        at = key(e.observed[0].word);
        show(at).classList.add("at");
      }
      if (e.type === "checker") {
        const k = key(e.location);
        const s = show(k);
        const prev = states.get(e.id);
        if (prev) s.classList.remove(prev);
        s.classList.add(e.state);
        states.set(e.id, e.state);
        renderMistake(e);
      }
    },
  };
  function renderMistake(e: CheckerEvent) {
    let li = mistakes.querySelector<HTMLLIElement>(`[data-id="${e.id}"]`);
    if (!li) {
      li = document.createElement("li");
      li.dataset.id = e.id;
      mistakes.append(li);
    }
    li.className = e.state;
    li.textContent = `${e.state} ${e.errorClass} at ${key(e.location)} «${text.get(key(e.location)) ?? ""}» (conf ${e.confidence.toFixed(2)})`;
  }
})();
