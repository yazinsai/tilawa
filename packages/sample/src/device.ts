import { loadReferenceIndex } from "@tilawa/core";
import type { PhraseAsset, ReferenceAsset } from "@tilawa/core";
import { MODEL_REGISTRY, createEngine, decodeWav, loadWebModel, measureLatency, percentile, withBase } from "@tilawa/sdk/web";
import type { Engine, TierName, WebProvider } from "@tilawa/sdk/web";
import refAsset from "../../core/assets/ref-v2.json";
import phrases from "../../core/assets/phrases-v1.json";

/**
 * Phone test for ticket 10: one button runs lite and full over the clips (real-time factor, latency, memory), then a
 * real-time battery run on lite. `?tiers=full&battery=0` limits the run (scripts/device-headless.mjs).
 */

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const index = loadReferenceIndex(refAsset as unknown as ReferenceAsset);
const status = (s: string) => ($("status").textContent = s);
const params = new URLSearchParams(location.search);
const TIERS = (params.get("tiers") ?? "lite,full").split(",") as TierName[];
const BATTERY_MIN = Number(params.get("battery") ?? 10);
/** About what the web microphone delivers per AudioWorklet block. */
const PUSH_MS = 20;

interface Clip { id: string; kind: string; ayahs: string[]; samples: Float32Array; sampleRate: number }
interface Battery { level: number; charging: boolean }

let clips: Clip[] = [];
const results: Record<string, any> = {};

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

async function load(tier: TierName) {
  const t0 = performance.now();
  const model = await loadWebModel({
    tier,
    registry: withBase(MODEL_REGISTRY, new URL(".", location.href).href),
    providers: ["wasm"] as WebProvider[],
    threads: 1,
    wasmPaths: new URL("ort/", location.href).href,
    onProgress: (n, total) => status(`Downloading ${tier} model: ${(n / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`),
  });
  const t = MODEL_REGISTRY.tiers.find((x) => x.tier === tier)!;
  const info = { id: t.id, version: t.version, bytes: t.bytes, backend: model.backend.name, fallbacks: model.fallbacks, loadMs: Math.round(performance.now() - t0), fromCache: model.fromCache };
  return { model, engine: createEngine({ reference: index, phrases: phrases as PhraseAsset, model }), info };
}

/** One declared-passage session per clip; latency on the real-time clock model. */
async function runClip(engine: Engine, c: Clip, realtime: boolean) {
  const [surah, fromAyah] = c.ayahs[0].split(":").map(Number);
  const toAyah = Number(c.ayahs.at(-1)!.split(":")[1]);
  const s = engine.startMemorization({ surah, fromAyah, toAyah });
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

async function speedTest(engine: Engine, tier: TierName) {
  const all = { highlight: [] as number[], suspected: [] as number[], pushes: [] as number[] };
  let audio = 0, compute = 0, peak: number | null = null;
  for (const [i, c] of clips.entries()) {
    status(`Testing ${tier}: clip ${i + 1} of ${clips.length}`);
    const r = await runClip(engine, c, false);
    audio += r.audioMs;
    compute += r.computeMs;
    all.highlight.push(...r.highlight);
    all.suspected.push(...r.suspected);
    all.pushes.push(...r.pushes);
    const m = await memoryMB();
    if (m) peak = Math.max(peak ?? 0, m.mb);
  }
  return { pushMs: PUSH_MS, audioS: +(audio / 1000).toFixed(1), rtf: +(compute / audio).toFixed(3), pushComputeMs: stat(all.pushes), highlightMs: stat(all.highlight), suspectedMs: stat(all.suspected), peakMemoryMB: round(peak) };
}

/** Always asks: some browsers report a fake level (Chrome on Android gave 100% at 41%). */
async function askBattery(when: string): Promise<number | null> {
  const typed = prompt(`What's your battery % ${when}? (Look at the top of the screen.)`);
  return typed && !isNaN(Number(typed)) ? Number(typed) : null;
}

async function batteryTest(engine: Engine, minutes: number) {
  const lock = await nav.wakeLock?.request("screen").catch(() => null);
  const start = await askBattery("right now");
  const t0 = performance.now(), end = t0 + minutes * 60_000;
  let audio = 0, compute = 0;
  for (let i = 0; performance.now() < end; i++) {
    const r = await runClip(engine, clips[i % clips.length], true);
    audio += r.audioMs;
    compute += r.computeMs;
    status(`Battery test: ${Math.ceil((end - performance.now()) / 60_000)} min left. Leave the phone alone.`);
  }
  await lock?.release();
  const endPct = await askBattery("now that the test is done");
  const mins = (performance.now() - t0) / 60_000;
  return { minutes: +mins.toFixed(1), rtf: +(compute / audio).toFixed(3), batteryPct: start !== null && endPct !== null ? { start, end: endPct, per10min: +(((start - endPct) * 10) / mins).toFixed(1) } : null };
}

type Check = { label: string; state: "wait" | "run" | "pass" | "fail" | "info"; detail?: string };
const checks: Check[] = [];
function draw() {
  $("steps").innerHTML = checks.map((c) => `<li class="${c.state}"><span>${{ wait: "○", run: "…", pass: "✓", fail: "✗", info: "•" }[c.state]}</span><b>${c.label}</b><span class="muted">${c.detail ?? ""}</span></li>`).join("");
  $("json").textContent = JSON.stringify(results, null, 1);
}
const verdict = (label: string, ok: boolean, detail: string) => checks.push({ label, state: ok ? "pass" : "fail", detail });

$("start").onclick = async () => {
  ($("start") as HTMLButtonElement).disabled = true;
  checks.length = 0;
  const steps = [...TIERS.map((t) => `Speed test: ${t} model`), ...(BATTERY_MIN ? [`Battery test (${BATTERY_MIN} min)`] : [])];
  steps.forEach((label) => checks.push({ label, state: "wait" }));
  draw();
  try {
    status("Loading test clips…");
    const list = (await (await fetch("clips/clips.json")).json()) as Omit<Clip, "samples" | "sampleRate">[];
    clips = await Promise.all(list.map(async (c) => ({ ...c, ...decodeWav(new Uint8Array(await (await fetch(`clips/${c.id}.wav`)).arrayBuffer())) })));
    if (!clips.length) throw new Error("no test clips on the server (run pnpm assets)");
    Object.assign(results, { schema: 2, date: new Date().toISOString(), device: { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, deviceMemoryGB: nav.deviceMemory ?? null }, tiers: {} });
    let lite: Awaited<ReturnType<typeof load>> | null = null;
    for (const [i, tier] of TIERS.entries()) {
      checks[i].state = "run";
      draw();
      const m = await load(tier);
      const r = await speedTest(m.engine, tier);
      results.tiers[tier] = { model: m.info, ...r };
      checks[i] = { label: checks[i].label, state: r.rtf < 1 ? "pass" : "fail", detail: `${r.rtf < 1 ? "keeps up" : "too slow"} (RTF ${r.rtf})` };
      draw();
      if (tier === "lite") lite = m;
      else await m.model.backend.release();
    }
    if (BATTERY_MIN) {
      const b = checks.length - 1;
      checks[b].state = "run";
      draw();
      lite ??= await load("lite");
      results.battery = await batteryTest(lite.engine, BATTERY_MIN);
      const pct = results.battery.batteryPct;
      checks[b] = { label: checks[b].label, state: "info", detail: pct ? `${pct.per10min}% per 10 min` : "battery % not entered" };
    }
    for (const tier of TIERS) {
      const r = results.tiers[tier];
      verdict(`${tier}: words light up fast enough`, (r.highlightMs.p95 ?? Infinity) <= 600, `${r.highlightMs.p95} ms (need ≤ 600)`);
      verdict(`${tier}: mistakes flagged fast enough`, (r.suspectedMs.p95 ?? Infinity) <= 1100, `${r.suspectedMs.p95} ms (need ≈ 1000)`);
    }
    status(checks.every((c) => c.state !== "fail") ? "Done. Everything passed. Tap Share results." : "Done. Some checks failed. Tap Share results.");
    $("share").hidden = false;
  } catch (e) {
    status(`Something broke: ${(e as Error).message}`);
  }
  draw();
  ($("start") as HTMLButtonElement).disabled = false;
};

$("share").onclick = async () => {
  const text = JSON.stringify(results);
  try {
    if (navigator.share) return await navigator.share({ title: "Tilawa phone test", text });
  } catch {}
  await navigator.clipboard.writeText(text).catch(() => {});
  status("Copied to clipboard.");
};
