// Node harness around the native MIT recitation engine.
//
// Protocol: one JSON request per stdin line, one JSON response per stdout line.
//   {"id": 1, "pcm": "/path/to/float32le-16k.bin", "mode": "tracking" | "correction"}
//   -> {"id": 1, "verses": [{surah, ayah, ok, unsure, words}], "transcript": "...",
//       "events": [...], "decodeMs": n}
// Host loop is @tilawa/core's ZipformerSession + emission (same as the browser worker).

import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_CONFIG,
  displayQuranFromRaw,
  MIN_WORD_FRACTION,
  ZipformerSession,
  type BridgedAyahTally,
  type EngineConfig,
  type WordVerdict,
  type ZipformerIo,
} from "@tilawa/core";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..", "..");
const FRONTEND = path.resolve(ROOT, "web", "frontend");

function camelToZipformerEnv(key: string): string {
  return "ZIPFORMER_" + key.replace(/[A-Z]/g, (c) => "_" + c).toUpperCase();
}

function configFromEnv(): EngineConfig {
  const cfg: EngineConfig = { ...DEFAULT_CONFIG };
  for (const key of Object.keys(DEFAULT_CONFIG) as (keyof EngineConfig)[]) {
    const raw = process.env[camelToZipformerEnv(key)];
    if (raw == null || raw === "") continue;
    const n = Number(raw);
    (cfg as unknown as Record<string, number | string>)[key] = Number.isFinite(n) ? n : raw;
  }
  return cfg;
}

const CONFIG = configFromEnv();

const MODEL = process.env.ZIPFORMER_MODEL ?? path.join(ROOT, "data", "zipformer", "zipformer_interp_gentle_a05.int8.onnx");
const CORPUS = process.env.ZIPFORMER_CORPUS ?? path.join(ROOT, "data", "zipformer", "zipformer_quran.json");
const ORT_DIR = process.env.ZIPFORMER_ORT_DIR ?? path.join(FRONTEND, "node_modules");
const IO_PATH =
  process.env.ZIPFORMER_IO ??
  path.join(HERE, "zipformer-io.json");
const DISPLAY_QURAN =
  process.env.ZIPFORMER_DISPLAY_QURAN ?? path.join(FRONTEND, "public", "quran.json");
const CHUNK = Number(process.env.ZIPFORMER_CHUNK ?? 7680);
const TAIL_SECONDS = Number(process.env.ZIPFORMER_TAIL_SECONDS ?? 2.0);
const MIN_FRAC = Number(process.env.ZIPFORMER_MIN_WORD_FRACTION ?? MIN_WORD_FRACTION);
const ALLOW_GAPS = process.env.ZIPFORMER_ALLOW_GAPS === "1";
const GAP_MAX_WORDS = Number(process.env.ZIPFORMER_GAP_MAX_WORDS ?? 3);
const MODE = process.env.ZIPFORMER_MODE ?? "recognize";
const FALLBACK = process.env.ZIPFORMER_FALLBACK !== "0";
const FALLBACK_MAX_DISTANCE = Number(process.env.ZIPFORMER_FALLBACK_MAX_DISTANCE ?? 0.5);
// ZIPFORMER_TRACE=1: record every correction-controller input and never flag,
// so the decode is the undisturbed verdict stream (replay_correction.ts).
const TRACE = process.env.ZIPFORMER_TRACE === "1";
// JSON CorrectionThresholds overrides for correction mode.
const THRESHOLDS = process.env.ZIPFORMER_CORRECTION_THRESHOLDS
  ? (JSON.parse(process.env.ZIPFORMER_CORRECTION_THRESHOLDS) as Record<string, number | boolean>)
  : null;
const STATES = ["ok", "unsure", "wrong", "skipped", "pending"];
const r3 = (x: number | undefined): number | null => (x === undefined || !Number.isFinite(x) ? null : Math.round(x * 1000) / 1000);
const packVerdict = (v: WordVerdict): unknown[] => [
  v.surah, v.ayah, v.word, v.wordIndex, STATES.indexOf(v.state), r3(v.distance), r3(v.heardRatio), r3(v.margin),
  v.vowelErrors, r3(v.vowelMargin), r3(v.gop), r3(v.gopTwice), r3(v.gopNone), r3(v.repGain), r3(v.pairGop),
];

// ZIPFORMER_LP_CACHE=<dir> + ZIPFORMER_LP_MODE=record|replay: record the
// model's log_probs per clip (keyed by a hash of the PCM), or replay them with
// no ONNX, so tracker variants run on identical acoustic output. Both modes
// keep the streaming encoder running across idle / dismiss resets (the live
// session restarts it), so the stream depends on the audio alone.
const LP_CACHE = process.env.ZIPFORMER_LP_CACHE ?? "";
const LP_MODE = LP_CACHE ? (process.env.ZIPFORMER_LP_MODE ?? "record") : "";
// ZIPFORMER_DIAG=1: per-chunk tracker timeline and timed engine events.
const DIAG = process.env.ZIPFORMER_DIAG === "1";

const require = createRequire(path.join(ORT_DIR, "/"));
const ort = require("onnxruntime-node");
const lpState: { key: string; calls: Float32Array[]; next: number } = { key: "", calls: [], next: 0 };
let encCalls: Float32Array[] = [];
let encReplay: Float32Array[] = [];
let encNext = 0;
if (LP_MODE === "record") {
  const create = ort.InferenceSession.create.bind(ort.InferenceSession);
  ort.InferenceSession.create = async (model: unknown, opts: Record<string, unknown> = {}) => {
    const s = await create(model, opts);
    const run = s.run.bind(s);
    s.run = async (feeds: unknown) => {
      const out = await run(feeds);
      lpState.calls.push(Float32Array.from(out.log_probs.data as Float32Array));
      const enc = out["/Transpose_226_output_0"];
      if (enc) encCalls.push(Float32Array.from(enc.data as Float32Array));
      return out;
    };
    return s;
  };
}
// ZIPFORMER_INTRA_THREADS=1 makes CPU inference bit-reproducible (4 threads is not).
const INTRA = Number(process.env.ZIPFORMER_INTRA_THREADS ?? 0);
if (INTRA > 0) {
  const create = ort.InferenceSession.create.bind(ort.InferenceSession);
  ort.InferenceSession.create = (model: unknown, opts: Record<string, unknown> = {}) =>
    create(model, { ...opts, intraOpNumThreads: INTRA, interOpNumThreads: 1 });
}

const io = JSON.parse(readFileSync(IO_PATH, "utf8")) as ZipformerIo;
const corpusJson = JSON.parse(readFileSync(CORPUS, "utf8"));
const quranDb = existsSync(DISPLAY_QURAN)
  ? displayQuranFromRaw(JSON.parse(readFileSync(DISPLAY_QURAN, "utf8")))
  : displayQuranFromRaw([]);

/** Replays cached log_probs; the states it returns are its inputs (unused). */
function replaySession(): { run(feeds: Record<string, unknown>): Promise<Record<string, unknown>> } {
  return {
    async run(feeds) {
      const data = lpState.calls[lpState.next++];
      if (!data) throw new Error(`lp cache exhausted for ${lpState.key}`);
      const out: Record<string, unknown> = {
        log_probs: new ort.Tensor("float32", data, [1, data.length / io.vocabSize, io.vocabSize]),
      };
      const enc = encReplay[encNext++];
      if (enc && io.encoderFrames) {
        out[io.encoderFrames] = new ort.Tensor("float32", enc, [1, enc.length / 512, 512]);
      }
      for (const [k, v] of Object.entries(feeds)) if (k !== "x") out[`new_${k}`] = v;
      return out;
    },
  };
}

function lpPath(key: string): string {
  return path.join(LP_CACHE, `${key}.f32`);
}

function saveLp(key: string): void {
  const n = lpState.calls.length;
  const per = n ? lpState.calls[0]!.length : 0;
  if (lpState.calls.some((c) => c.length !== per)) throw new Error("variable log_probs size");
  const buf = Buffer.alloc(8 + n * per * 4);
  buf.writeUInt32LE(n, 0);
  buf.writeUInt32LE(per, 4);
  lpState.calls.forEach((c, i) => Buffer.from(c.buffer, c.byteOffset, c.byteLength).copy(buf, 8 + i * per * 4));
  mkdirSync(LP_CACHE, { recursive: true });
  writeFileSync(lpPath(key), buf);
}

function loadLp(key: string): void {
  const buf = readFileSync(lpPath(key));
  const n = buf.readUInt32LE(0);
  const per = buf.readUInt32LE(4);
  const all = new Float32Array(buf.buffer.slice(buf.byteOffset + 8, buf.byteOffset + 8 + n * per * 4));
  lpState.calls = Array.from({ length: n }, (_, i) => all.subarray(i * per, (i + 1) * per));
  lpState.next = 0;
  encReplay = [];
  encNext = 0;
  const encPath = path.join(LP_CACHE, `${key}.enc.f32`);
  if (!existsSync(encPath)) return;
  const eb = readFileSync(encPath);
  const en = eb.readUInt32LE(0);
  const eper = eb.readUInt32LE(4);
  const eraw = new Float32Array(eb.buffer.slice(eb.byteOffset + 8, eb.byteOffset + 8 + en * eper * 4));
  encReplay = Array.from({ length: en }, (_, i) => eraw.subarray(i * eper, (i + 1) * eper));
}

async function createHost(): Promise<ZipformerSession> {
  const acoustic = LP_MODE === "replay"
    ? { session: replaySession(), Tensor: ort.Tensor }
    : { ort, model: new Uint8Array(readFileSync(MODEL)) };
  return ZipformerSession.create({
    ...acoustic,
    io,
    corpus: corpusJson,
    quran: quranDb,
    executionProviders: ["cpu"],
    config: CONFIG,
    tailSeconds: TAIL_SECONDS,
    minWordFraction: MIN_FRAC,
    stayOnSurah: MODE === "stay",
    enableFallback: FALLBACK,
    fallbackMaxDistance: FALLBACK_MAX_DISTANCE,
    allowGaps: ALLOW_GAPS,
    gapMaxWords: GAP_MAX_WORDS,
    debug: true,
  });
}

function ayahWordCount(host: ZipformerSession, surah: number, ayah: number): number {
  try {
    return host.wordCount(surah, ayah);
  } catch {
    return 99;
  }
}

async function recognize(host: ZipformerSession, pcm: Float32Array, mode = "tracking", key = "") {
  const t0 = performance.now();
  const hostInternals = host as unknown as {
    resetDecoder: () => void;
    fbank: { reset(): void };
    runner: { reset(): void };
    decoder: { reset(): void; framesDecoded: number };
    posteriors: { clear(n: number): void };
    engine: { corpus: { wordSurah: Int32Array | number[]; wordAyah: Int32Array | number[]; wordInAyah: Int32Array | number[] }; state: string; tracker: { surah: number; cursorWordIndex: number; lost: boolean; costRate(): number | null } | null };
  };
  delete (hostInternals as unknown as Record<string, unknown>).resetDecoder;
  const readTrack = (): Array<[string, number, number, number, number]> | null => {
    const tr = (hostInternals.engine as unknown as { tracker: { heard: Array<{ ch: string; frame: number }>; trail: number[]; firstWord: number; localWordOfPos: Int32Array; len: number } | null }).tracker;
    if (!tr) return null;
    const c = hostInternals.engine.corpus;
    return tr.heard.map((h, i) => {
      const cell = tr.trail[i]!;
      const w = tr.firstWord + (cell <= 0 ? 0 : tr.localWordOfPos[Math.min(cell, tr.len) - 1]!);
      return [h.ch, h.frame, c.wordSurah[w] as number, c.wordAyah[w] as number, c.wordInAyah[w] as number];
    });
  };
  if (LP_MODE) {
    lpState.key = key;
    lpState.calls = [];
    encCalls = [];
    lpState.next = 0;
    if (LP_MODE === "replay") loadLp(key);
    hostInternals.resetDecoder = () => {
      hostInternals.decoder.reset();
      hostInternals.posteriors.clear(0);
    };
  }
  host.reset();
  if (LP_MODE) {
    hostInternals.fbank.reset();
    hostInternals.runner.reset();
  }
  host.setMode(mode === "correction" ? "correction" : "tracking");
  const diag: unknown[] = [];
  const events: Array<Record<string, unknown>> = [];
  const cursorOrder: string[] = [];
  // Correction issues are dismissed on sight so the rest of the clip is still
  // decoded (the session drops audio while an issue is open).
  const corrections: Array<Record<string, unknown>> = [];
  const notes: Array<Record<string, unknown>> = [];
  let offset = 0;
  const trace: unknown[] = [];
  const ctl = host.correction;
  ctl.thresholds = { ...ctl.thresholds, ...(THRESHOLDS ?? {}) };
  const slipEnv = process.env.ZIPFORMER_SLIP;
  if (slipEnv === "strict" || slipEnv === "high") host.setSlipHead(slipEnv);
  else if (slipEnv === "0" || slipEnv === "off") host.setSlipHead(false);
  const own = ctl as unknown as Record<string, unknown>;
  for (const k of ["observe", "clearEvidence", "raise", "settle"]) delete own[k];
  delete (host as unknown as Record<string, unknown>).dumpTallies;
  // DIAG: every verdict state each word reached in a controller snapshot, as
  // "surah:ayah:word" -> [state bitmask (STATES order), first t, last packed verdict].
  const seen: Record<string, [number, number, unknown[]]> = {};
  if (DIAG && !TRACE) {
    const note = (vs: readonly WordVerdict[]) => {
      for (const v of vs) {
        const k = `${v.surah}:${v.ayah}:${v.word}`;
        const cur = seen[k] ?? (seen[k] = [0, offset / 16000, []]);
        cur[0] |= 1 << STATES.indexOf(v.state);
        cur[2] = packVerdict(v);
      }
    };
    const proto = Object.getPrototypeOf(ctl) as Record<string, (...a: unknown[]) => unknown>;
    own.observe = (...a: unknown[]) => { note(a[0] as WordVerdict[]); return proto.observe!.apply(ctl, a); };
    own.settle = (...a: unknown[]) => { note(a[0] as WordVerdict[]); return proto.settle!.apply(ctl, a); };
  }
  if (TRACE) {
    // Session internals, read only: the tracker trail gives backward jumps,
    // and every tally dump is a settled snapshot of the tracker being dropped.
    const internals = host as unknown as {
      engine: { tracker: { trail: number[]; heard: Array<{ frame: number }>; firstWord: number; localWordOfPos: Int32Array; len: number } | null; tracer: { verdicts(settled: boolean): WordVerdict[] } | null };
      dumpTallies: () => void;
      decoder: { framesDecoded: number };
    };
    const jumps = (): number[][] => {
      const tr = internals.engine.tracker;
      if (!tr) return [];
      const out: number[][] = [];
      const word = (cell: number) => tr.firstWord + (cell <= 0 ? 0 : tr.localWordOfPos[Math.min(cell, tr.len) - 1]!);
      for (let g = 1; g < tr.trail.length; g++) {
        if (tr.trail[g]! < tr.trail[g - 1]!) out.push([tr.heard[g]!.frame, word(tr.trail[g - 1]!), word(tr.trail[g]! + 1)]);
      }
      return out;
    };
    own.observe = (verdicts: WordVerdict[], cursor: object, frame: number) => {
      trace.push({ op: "observe", t: offset / 16000, frame, cursor: { ...cursor }, v: verdicts.map(packVerdict), j: jumps() });
      return false;
    };
    own.clearEvidence = () => { trace.push({ op: "clear", t: offset / 16000 }); };
    own.raise = (issue: object) => { trace.push({ op: "raise", t: offset / 16000, issue: { ...issue } }); return false; };
    own.settle = () => false;
    const dump = internals.dumpTallies;
    internals.dumpTallies = function () {
      const tracer = internals.engine.tracer;
      if (tracer) {
        trace.push({ op: "settle", t: offset / 16000, frame: internals.decoder.framesDecoded,
          v: tracer.verdicts(true).map(packVerdict), j: jumps() });
      }
      return dump.call(host);
    };
  }

  const collect = (msgs: Array<{ type: string; [k: string]: unknown }>): void => {
    for (const ev of msgs) {
      if (ev.type === "correction") {
        const state = ev.state as { phase: string; issue: Record<string, unknown> | null };
        if (state.phase === "error" && state.issue) {
          corrections.push({ ...state.issue, atSeconds: offset / 16000 });
          collect(host.correct("dismiss") as Array<{ type: string; [k: string]: unknown }>);
        }
      } else if (ev.type === "correction_note") {
        notes.push({ ...(ev.issue as Record<string, unknown>), atSeconds: offset / 16000 });
      } else if (ev.type === "word_progress") {
        const key = `${ev.surah}:${ev.ayah}`;
        if (cursorOrder[cursorOrder.length - 1] !== key) cursorOrder.push(key);
      } else if (ev.type === "debug") {
        const data = (ev.data ?? {}) as Record<string, unknown>;
        events.push({ type: ev.event, ...data, ...(DIAG ? { t: offset / 16000 } : {}) });
      }
    }
  };

  for (let i = 0; i < pcm.length; i += CHUNK) {
    offset = Math.min(pcm.length, i + CHUNK);
    collect(await host.feed(pcm.subarray(i, offset)));
    if (DIAG) {
      const tr = hostInternals.engine.tracker;
      const rate = tr?.costRate() ?? null;
      const c = hostInternals.engine.corpus;
      const w = tr ? tr.cursorWordIndex : -1;
      diag.push([offset / 16000, hostInternals.engine.state === "tracking" ? 1 : 0, w,
        w >= 0 ? c.wordSurah[w] : 0, w >= 0 ? c.wordAyah[w] : 0, w >= 0 ? c.wordInAyah[w] : 0,
        tr?.lost ? 1 : 0, rate === null ? null : Math.round(rate * 1000) / 1000]);
    }
  }
  const preStop = process.env.ZIPFORMER_DIAG_TRACK === "1" ? readTrack() : null;
  collect(await host.stop());
  const postStop = process.env.ZIPFORMER_DIAG_TRACK === "1" ? readTrack() : null;
  if (LP_MODE === "record") {
    if (process.env.ZIPFORMER_SKIP_LP !== "1") saveLp(key);
    if (encCalls.length) {
      mkdirSync(LP_CACHE, { recursive: true });
      const per = encCalls[0]!.length; const b = Buffer.alloc(8 + encCalls.length * per * 4);
      b.writeUInt32LE(encCalls.length, 0); b.writeUInt32LE(per, 4);
      encCalls.forEach((c, i) => Buffer.from(c.buffer, c.byteOffset, c.byteLength).copy(b, 8 + i * per * 4));
      writeFileSync(path.join(LP_CACHE, `${key}.enc.f32`), b);
    }
  }

  const tallies = host.tallies;
  const verses: BridgedAyahTally[] = host.verses;

  let fallback = host.lastFallback;
  if (FALLBACK && verses.length === 0 && fallback) {
    const words = ayahWordCount(host, fallback.surah, fallback.ayah);
    verses.push({
      surah: fallback.surah,
      ayah: fallback.ayah,
      ok: words,
      unsure: 0,
      wrong: 0,
      skipped: 0,
      pending: 0,
      words,
      firstSeen: 0,
    });
    events.push({ type: "fallback", ...fallback });
  }

  return {
    verses,
    fallback,
    all: tallies,
    cursorOrder,
    corrections,
    notes,
    ...(TRACE ? { trace } : {}),
    ...(DIAG ? { diag, seen } : {}),
    ...(preStop ? { track: preStop } : {}),
    ...(postStop ? { trackPost: postStop } : {}),
    ...(key ? { lpKey: key } : {}),
    transcript: host.transcript,
    events,
    state: host.engineState,
    decodeMs: Math.round(performance.now() - t0),
  };
}

async function main(): Promise<void> {
  const host = await createHost();
  const rl = createInterface({ input: process.stdin });
  process.stdout.write(JSON.stringify({
    ready: true,
    model: MODEL,
    minWordFraction: MIN_FRAC,
    allowGaps: ALLOW_GAPS,
    gapMaxWords: GAP_MAX_WORDS,
    tailSeconds: TAIL_SECONDS,
    okDistance: CONFIG.okDistance,
    unsureDistance: CONFIG.unsureDistance,
    searchDecisiveDistance: CONFIG.searchDecisiveDistance,
  }) + "\n");

  for await (const line of rl) {
    if (!line.trim()) continue;
    type Req = { id?: number; pcm: string; mode?: string; expected?: { surah: number; ayah: number; ayahEnd?: number } | null };
    let req: Req;
    try {
      req = JSON.parse(line) as Req;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      process.stdout.write(JSON.stringify({ error: `bad json: ${msg}` }) + "\n");
      continue;
    }
    try {
      const buf = readFileSync(req.pcm);
      const pcm = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
      const key = LP_MODE ? createHash("sha1").update(buf).digest("hex").slice(0, 16) : "";
      host.setExpected(req.expected ?? null);
      const res = await recognize(host, pcm, req.mode, key);
      process.stdout.write(JSON.stringify({ id: req.id, ...res }) + "\n");
    } catch (e) {
      process.stdout.write(JSON.stringify({
        id: req.id,
        error: e instanceof Error ? e.stack ?? e.message : String(e),
      }) + "\n");
    }
  }
}

void main();
