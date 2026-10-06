import * as ort from "onnxruntime-web/all";
import { FrameChunker } from "./chunker.js";
import { Fbank } from "./fbank.js";
import { summarise } from "./stats.js";
import { MODELS, Zipformer } from "./zipformer.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const modelSel = $<HTMLSelectElement>("model");
const threadSel = $<HTMLSelectElement>("threads");
const status = (s: string) => ($("status").textContent = s);

for (const m of MODELS) modelSel.add(new Option(m.label, m.id));
const maxThreads = crossOriginIsolated ? navigator.hardwareConcurrency || 1 : 1;
for (let n = 1; n <= maxThreads; n *= 2) threadSel.add(new Option(String(n), String(n)));
if (!crossOriginIsolated) threadSel.title = "Not cross-origin isolated: WASM is single-threaded";

const CHUNK_SAMPLES = 5120;

const heap = () => (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize;
const mb = (b?: number) => (b === undefined ? null : Math.round((b / 2 ** 20) * 10) / 10);

const loadClip = async (url: string) => {
  const ctx = new OfflineAudioContext(1, 1, 16000);
  const buf = await ctx.decodeAudioData(await (await fetch(`${import.meta.env.BASE_URL}${url}`)).arrayBuffer());
  return buf.getChannelData(0);
};

let last = "";

$("run").onclick = async () => {
  const button = $<HTMLButtonElement>("run");
  button.disabled = true;
  try {
    const spec = MODELS.find((m) => m.id === modelSel.value)!;
    const ep = $<HTMLSelectElement>("ep").value as "wasm" | "webgpu";
    ort.env.wasm.numThreads = Number(threadSel.value);
    ort.env.wasm.simd = true;
    let peakHeap = heap() ?? 0;
    const sample = () => (peakHeap = Math.max(peakHeap, heap() ?? 0));

    status("Loading model…");
    const t0 = performance.now();
    const z = await Zipformer.load(spec, ep);
    const loadMs = Math.round(performance.now() - t0);
    sample();

    status("Decoding clip…");
    const audio = await loadClip($<HTMLSelectElement>("clip").value);
    const fbank = new Fbank();
    const chunker = new FrameChunker(z.window, spec.shift);
    const chunkMs: number[] = [];
    for (let off = 0; off < audio.length; off += CHUNK_SAMPLES) {
      const t = performance.now();
      const windows = chunker.push(fbank.push(audio.subarray(off, off + CHUNK_SAMPLES)));
      if (off + CHUNK_SAMPLES >= audio.length) windows.push(...chunker.flush());
      for (const w of windows) await z.step(w);
      chunkMs.push(Math.round((performance.now() - t) * 100) / 100);
      sample();
      if (chunkMs.length % 10 === 0) {
        status(`Streaming… ${((off / 16000) | 0)} s`);
        await new Promise((r) => setTimeout(r));
      }
    }

    let uaMemoryMB: number | null = null;
    const m = performance as unknown as { measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }> };
    if (crossOriginIsolated && m.measureUserAgentSpecificMemory) uaMemoryMB = mb((await m.measureUserAgentSpecificMemory()).bytes);

    const result = {
      model: spec.id,
      backend: ep,
      threads: ort.env.wasm.numThreads,
      crossOriginIsolated,
      clip: $<HTMLSelectElement>("clip").value,
      audioSec: Math.round((audio.length / 16000) * 100) / 100,
      chunkAudioMs: (CHUNK_SAMPLES / 16000) * 1000,
      ...summarise({ audioSec: audio.length / 16000, chunkMs, loadMs }),
      peakJsHeapMB: mb(peakHeap || undefined),
      uaMemoryMB,
      transcript: z.text().slice(0, 300),
      userAgent: navigator.userAgent,
      hardwareConcurrency: navigator.hardwareConcurrency,
      deviceMemoryGB: (navigator as unknown as { deviceMemory?: number }).deviceMemory ?? null,
      at: new Date().toISOString(),
    };
    last = JSON.stringify(result, null, 2);
    $("out").textContent = last;
    $<HTMLButtonElement>("copy").disabled = false;
    status(result.realTime ? `Real time: RTF ${result.rtf}` : `Slower than real time: RTF ${result.rtf}`);
  } catch (e) {
    status(`Error: ${(e as Error).message}`);
    console.error(e);
  } finally {
    button.disabled = false;
  }
};

$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(last);
    status("Copied.");
  } catch {
    const r = document.createRange();
    r.selectNodeContents($("out"));
    getSelection()?.removeAllRanges();
    getSelection()?.addRange(r);
    status("Clipboard blocked: JSON selected, copy it manually.");
  }
};
