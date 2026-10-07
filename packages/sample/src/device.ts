import { loadReferenceIndex } from "@tilawa/core";
import type { PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { MODEL_REGISTRY, createEngine, decodeWav, loadWebModel, measureLatency, percentile, withBase } from "@tilawa/sdk/web";
import type { Engine, LoadedModel, TierName, WebProvider } from "@tilawa/sdk/web";
import refAsset from "../../core/assets/ref-v2.json";
import phrases from "../../core/assets/phrases-v1.json";

/** Device measurements for ticket 10: real-time factor, memory, battery over a real-time run, latency p95. */

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const index = loadReferenceIndex(refAsset as unknown as ReferenceAsset);
const status = (s: string) => ($("status").textContent = s);
/** About what the web microphone delivers per AudioWorklet block. */
const PUSH_MS = 20;

interface Clip { id: string; kind: string; ayahs: string[]; samples: Float32Array; sampleRate: number }
interface Battery { level: number; charging: boolean }

let model: (LoadedModel & { fromCache: boolean; fallbacks: string[] }) | null = null;
let engine: Engine | null = null;
let clips: Clip[] = [];
const results: Record<string, unknown> = {};

const nav = navigator as Navigator & {
  deviceMemory?: number;
  getBattery?: () => Promise<Battery>;
  wakeLock?: { request(t: "screen"): Promise<{ release(): Promise<void> }> };
};
const perf = performance as Performance & {
  memory?: { usedJSHeapSize: number };
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
};

/** Page memory in MB: the UA-specific measurement (Chrome, cross-origin isolated) includes WASM; else the JS heap only. */
async function memoryMB(): Promise<{ mb: number; source: string } | null> {
  if (crossOriginIsolated && perf.measureUserAgentSpecificMemory) {
    try {
      return { mb: (await perf.measureUserAgentSpecificMemory()).bytes / 1e6, source: "measureUserAgentSpecificMemory" };
    } catch {}
  }
  return perf.memory ? { mb: perf.memory.usedJSHeapSize / 1e6, source: "performance.memory (JS heap only)" } : null;
}

async function battery(): Promise<Battery | null> {
  try {
    const b = await nav.getBattery?.();
    return b ? { level: Math.round(b.level * 100), charging: b.charging } : null;
  } catch {
    return null;
  }
}

const stat = (xs: number[]) => ({ n: xs.length, p50: round(percentile(xs, 50)), p95: round(percentile(xs, 95)), max: round(percentile(xs, 100)) });
const round = (x: number | null) => (x === null ? null : Math.round(x));

$("load").onclick = async () => {
  ($("load") as HTMLButtonElement).disabled = true;
  try {
    const tier = $<HTMLSelectElement>("tier").value as TierName;
    const t0 = performance.now();
    model = await loadWebModel({
      tier,
      registry: withBase(MODEL_REGISTRY, new URL(".", location.href).href),
      providers: $<HTMLSelectElement>("provider").value.split(",") as WebProvider[],
      threads: Number($<HTMLSelectElement>("threads").value),
      wasmPaths: new URL("ort/", location.href).href,
      onProgress: (n, total) => status(`Downloading ${(n / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`),
    });
    const loadMs = performance.now() - t0;
    engine = createEngine({ reference: index, phrases: phrases as PhraseAsset, model });
    const t = MODEL_REGISTRY.tiers.find((x) => x.tier === tier)!;
    const list = (await (await fetch("clips/clips.json")).json()) as Omit<Clip, "samples" | "sampleRate">[];
    clips = await Promise.all(list.map(async (c) => ({ ...c, ...decodeWav(new Uint8Array(await (await fetch(`clips/${c.id}.wav`)).arrayBuffer())) })));
    Object.assign(results, {
      schema: 1,
      date: new Date().toISOString(),
      device: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, deviceMemoryGB: nav.deviceMemory ?? null, crossOriginIsolated },
      model: { tier, id: t.id, version: t.version, bytes: t.bytes, backend: model.backend.name, threads: Number($<HTMLSelectElement>("threads").value), fallbacks: model.fallbacks },
      load: { ms: Math.round(loadMs), fromCache: model.fromCache, memory: await memoryMB() },
    });
    $("model-note").textContent = `${t.id}@${t.version} (${(t.bytes / 1e6).toFixed(0)} MB) on ${model.backend.name}, ${model.fromCache ? "cache" : "download"} ${(loadMs / 1000).toFixed(1)} s; ` +
      `crossOriginIsolated=${crossOriginIsolated}; ${clips.length} clips.${model.fallbacks.length ? ` Fell back: ${model.fallbacks.join("; ")}` : ""}`;
    ($("clips") as HTMLButtonElement).disabled = ($("soak") as HTMLButtonElement).disabled = !clips.length;
    status(clips.length ? "Ready." : "No clips: run `pnpm assets` with the help clips present.");
    const b = await battery();
    $("battery-note").textContent = b ? `(read automatically: ${b.level}%${b.charging ? ", charging" : ""})` : "(this browser hides the battery level: type it in)";
  } catch (e) {
    status(`Load failed: ${(e as Error).message}`);
  }
  ($("load") as HTMLButtonElement).disabled = false;
  show();
};

/** One declared-passage session per clip, pushed as fast as the device goes; latency on the real-time clock model. */
async function runClip(c: Clip, realtime: boolean) {
  const [surah, fromAyah] = c.ayahs[0].split(":").map(Number);
  const toAyah = Number(c.ayahs.at(-1)!.split(":")[1]);
  const s = engine!.startMemorization({ surah, fromAyah, toAyah });
  const pushes: number[] = [];
  const timed = Object.assign(Object.create(s), {
    async pushAudio(x: Float32Array, rate: number) {
      const t0 = performance.now();
      const out = await s.pushAudio(x, rate);
      pushes.push(performance.now() - t0);
      return out;
    },
  });
  const r = await measureLatency(timed, c.samples, c.sampleRate, PUSH_MS, realtime);
  return { ...r, pushes };
}

$("clips").onclick = async () => {
  const rows = [];
  const all = { highlight: [] as number[], suspected: [] as number[], pushes: [] as number[] };
  let audio = 0;
  let compute = 0;
  let peak: number | null = null;
  for (const [i, c] of clips.entries()) {
    status(`Clip ${i + 1}/${clips.length}…`);
    const r = await runClip(c, false);
    audio += r.audioMs;
    compute += r.computeMs;
    all.highlight.push(...r.highlight);
    all.suspected.push(...r.suspected);
    all.pushes.push(...r.pushes);
    const m = await memoryMB();
    if (m) peak = Math.max(peak ?? 0, m.mb);
    rows.push({ id: c.id, kind: c.kind, audioS: +(r.audioMs / 1000).toFixed(1), rtf: +(r.computeMs / r.audioMs).toFixed(3), highlightP95: round(percentile(r.highlight, 95)), suspected: r.suspected.map(Math.round) });
  }
  results.clips = {
    pushMs: PUSH_MS, audioS: +(audio / 1000).toFixed(1), rtf: +(compute / audio).toFixed(3), pushComputeMs: stat(all.pushes),
    highlightMs: stat(all.highlight), suspectedMs: stat(all.suspected), peakMemoryMB: peak === null ? null : Math.round(peak), perClip: rows,
  };
  status("Clips done.");
  show();
};

$("soak").onclick = async () => {
  const minutes = Number($<HTMLInputElement>("minutes").value);
  const lock = await nav.wakeLock?.request("screen").catch(() => null);
  const b0 = await battery();
  if (b0) $<HTMLInputElement>("bat0").value = String(b0.level);
  const t0 = performance.now();
  const end = t0 + minutes * 60_000;
  const all = { highlight: [] as number[], pushes: [] as number[] };
  let audio = 0;
  let compute = 0;
  let peak: number | null = null;
  const memory: { min: number; mb: number }[] = [];
  for (let i = 0; performance.now() < end; i++) {
    const r = await runClip(clips[i % clips.length], true);
    audio += r.audioMs;
    compute += r.computeMs;
    all.highlight.push(...r.highlight);
    all.pushes.push(...r.pushes);
    const m = await memoryMB();
    if (m) memory.push({ min: +((performance.now() - t0) / 60_000).toFixed(1), mb: Math.round(m.mb) });
    if (m) peak = Math.max(peak ?? 0, m.mb);
    status(`Battery run: ${((performance.now() - t0) / 60_000).toFixed(1)} / ${minutes} min, RTF ${(compute / audio).toFixed(3)}`);
  }
  await lock?.release();
  const b1 = await battery();
  if (b1) $<HTMLInputElement>("bat1").value = String(b1.level);
  results.soak = {
    minutes: +((performance.now() - t0) / 60_000).toFixed(1), audioS: Math.round(audio / 1000), rtf: +(compute / audio).toFixed(3),
    pushComputeMs: stat(all.pushes), highlightMs: stat(all.highlight), peakMemoryMB: peak === null ? null : Math.round(peak), memory,
    batteryAuto: { start: b0, end: b1 },
  };
  status("Battery run done. Fill in the battery % if it wasn't read automatically, then copy.");
  show();
};

function withBattery() {
  const s = results.soak as Record<string, unknown> | undefined;
  const a = Number($<HTMLInputElement>("bat0").value);
  const b = Number($<HTMLInputElement>("bat1").value);
  if (s && $<HTMLInputElement>("bat0").value && $<HTMLInputElement>("bat1").value) {
    s.batteryPct = { start: a, end: b, drain: a - b, per10min: +(((a - b) * 10) / (s.minutes as number)).toFixed(1) };
  }
  return results;
}

function show() {
  const r = withBattery();
  $("json").textContent = JSON.stringify(r, null, 1);
  const c = r.clips as Record<string, any> | undefined;
  const s = r.soak as Record<string, any> | undefined;
  const row = (k: string, v: unknown) => `<tr><th>${k}</th><td>${v ?? "—"}</td></tr>`;
  $("table").innerHTML = [
    row("RTF (clips)", c?.rtf), row("push compute p95 ms", c?.pushComputeMs.p95), row("word highlight p95 ms", c?.highlightMs.p95),
    row("suspected p95 ms", c?.suspectedMs.p95), row("peak memory MB", c?.peakMemoryMB),
    row("RTF (real-time run)", s?.rtf), row("battery % per 10 min", s?.batteryPct?.per10min), row("highlight p95 ms (real-time run)", s?.highlightMs.p95),
  ].join("");
}

$("bat0").oninput = $("bat1").oninput = show;
$("copy").onclick = async () => {
  const body = JSON.stringify(withBattery());
  await navigator.clipboard.writeText(body).catch(() => ($("json").textContent = body));
  status("Copied.");
};
